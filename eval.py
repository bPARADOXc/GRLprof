import argparse
import importlib.util
import json
import os
import re
from pathlib import Path
# python eval.py --dataset webqsp --input_file checkpoints/webqsp/experiment/eval_results/webqsp_checkpoint_eval --quiet
# python eval.py --dataset webqsp --experiment grlprof-llama-3-8b-instruct-E5-base-webqsp-projector --split validation
def _load_eval_utils():
    """Load the sibling utils.py, not the utils/ package."""
    utils_path = Path(__file__).with_name("utils.py")
    spec = importlib.util.spec_from_file_location("llaga_eval_utils", utils_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eval_utils = _load_eval_utils()


def _mean(values):
    return float(sum(values) / len(values)) if values else 0.0


def _alias_file_path(filename, alias_dir=None):
    candidates = []
    if alias_dir:
        candidates.append(Path(alias_dir).expanduser() / filename)
    env_alias_dir = os.environ.get("GRLPROF_ALIAS_DIR")
    if env_alias_dir:
        candidates.append(Path(env_alias_dir).expanduser() / filename)
    candidates.extend([
        Path(__file__).resolve().parent / "data" / "aliases" / filename,
        Path(__file__).resolve().parent / "cope_alias" / filename,
        Path.cwd() / "cope_alias" / filename,
        Path("../../VoG/cope_alias") / filename,
    ])
    for path in candidates:
        if path.is_file():
            return path
    searched = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Missing evaluation alias resource {filename}. Searched:\n  {searched}\n"
        "Set --alias_dir or GRLPROF_ALIAS_DIR to the original cope_alias directory."
    )


def _load_alias_data(dataset, alias_dir=None):
    type_field = ''
    aname_dict = {}
    alias_dict = {}
    add_ans_alias_dict = {}

    if dataset == 'cwq':
        type_field = 'compositionality_type'
        with open(_alias_file_path('cwq_aname_dict.json', alias_dir), 'r', encoding='utf-8') as f:
            aname_dict = json.load(f)
        with open(_alias_file_path('CWQ_aliase_data31158.json', alias_dir), 'r', encoding='utf-8') as f:
            alias_dict = json.load(f)
        with open(_alias_file_path('ComplexWebQuestions_test_wans.json', alias_dir), 'r', encoding='utf-8') as f:
            q_all_list = json.load(f)
            for q_item in q_all_list:
                ans_list = []
                for ans_item in q_item['answers']:
                    if ans_item['answer']:
                        ans_list.append(ans_item['answer'])
                    else:
                        ans_list.append(ans_item['answer_id'])
                    if 'aliases' in ans_item.keys():
                        ans_list += ans_item['aliases']

                add_ans_alias_dict[q_item['question']] = ans_list

    elif dataset == 'webqsp':
        alias_path = _alias_file_path('WQSP_aliase_data.json', alias_dir)
        if alias_path.is_file():
            with open(alias_path, 'r', encoding='utf-8') as f:
                alias_dict = json.load(f)

    return type_field, aname_dict, alias_dict, add_ans_alias_dict


def _question_text(data, question_string):
    return data.get(question_string) or data.get('question') or data.get('RawQuestion') or data.get('id', '')


def _result_is_empty(response):
    return len(eval_utils.split_prediction_answers(response)) == 0


def _evaluate_checkpoint_record(data, alias_dict):
    answer_groups = eval_utils.expand_answer_alias_groups(eval_utils.extract_gold_answers(data), alias_dict)
    answers = eval_utils.flatten_answer_groups(answer_groups)
    response = eval_utils.extract_prediction_answers(data)
    return response, answers, answer_groups, data


def _legacy_gold_answers(dataset, ori_data):
    if dataset == 'webqsp':
        answer_list = []
        for parse in ori_data.get("Parses", []):
            for answer in parse.get("Answers", []):
                if answer.get("EntityName") is None:
                    answer_list.append(answer.get("AnswerArgument"))
                else:
                    answer_list.append(answer.get("EntityName"))
        return eval_utils.normalize_answer_list(answer_list)

    if dataset == 'webquestions':
        return eval_utils.normalize_answer_list(ori_data.get("answers", []))

    return []


def _evaluate_legacy_record(
    dataset,
    question_string,
    data,
    ground_truth_datas,
    aname_dict,
    alias_dict,
    add_ans_alias_dict,
):
    answers, ori_data = eval_utils.align(
        dataset,
        question_string,
        data,
        ground_truth_datas,
        aname_dict,
        alias_dict,
        add_ans_alias_dict,
    )
    response = eval_utils.extract_prediction_answers(data)
    raw_gold_answers = _legacy_gold_answers(dataset, ori_data)
    answer_groups = None
    if raw_gold_answers:
        answer_groups = eval_utils.expand_answer_alias_groups(raw_gold_answers, alias_dict)
    return response, answers, answer_groups, ori_data


def _infer_checkpoint_and_split(path):
    name = Path(path).name
    split = "validation" if "_validation_" in name else "test" if "_test_" in name else ""
    if name.startswith("final_"):
        checkpoint = "final"
    else:
        checkpoint = ""
        match = re.search(r"(checkpoint-\d+)", name)
        if match:
            checkpoint = match.group(1)
    return checkpoint, split


def _checkpoint_sort_key(path):
    checkpoint, split = _infer_checkpoint_and_split(path)
    if checkpoint == "final":
        checkpoint_key = (1, float("inf"))
    elif checkpoint.startswith("checkpoint-"):
        checkpoint_key = (0, int(checkpoint.split("-")[-1]))
    else:
        checkpoint_key = (0, -1)
    return checkpoint_key, split, Path(path).name


def _resolve_eval_root(eval_root):
    candidates = [Path(eval_root), Path("checkpoints/webqsp"), Path("checkpoint/webqsp")]
    for path in candidates:
        if path.is_dir():
            return path
    return candidates[0]


def _candidate_eval_dirs(base_path):
    return [
        base_path,
        base_path / "webqsp_checkpoint_eval",
        base_path / "webqsp-checkpoint",
        base_path / "eval_results" / "webqsp_checkpoint_eval",
        base_path / "eval_results" / "webqsp-checkpoint",
        base_path / "eval-results" / "webqsp_checkpoint_eval",
        base_path / "eval-results" / "webqsp-checkpoint",
    ]


def _has_eval_detail_json(path):
    return path.is_dir() and any("details_extracted" in item.name for item in path.glob("*.json"))


def _resolve_checkpoint_eval_dir(experiment_dir, checkpoint_eval_dir, eval_root):
    if checkpoint_eval_dir:
        bases = [Path(checkpoint_eval_dir).expanduser()]
    elif experiment_dir:
        experiment_path = Path(experiment_dir).expanduser()
        bases = [experiment_path]
        if not experiment_path.is_absolute():
            bases.append(_resolve_eval_root(eval_root) / experiment_dir)
    else:
        raise ValueError("Batch mode needs --experiment_dir or --checkpoint_eval_dir.")

    seen = set()
    for base in bases:
        for candidate in _candidate_eval_dirs(base):
            candidate = candidate.resolve() if candidate.exists() else candidate
            if candidate in seen:
                continue
            seen.add(candidate)
            if _has_eval_detail_json(candidate):
                return candidate

    searched = []
    for base in bases:
        searched.extend(str(path) for path in _candidate_eval_dirs(base))
    raise FileNotFoundError("No eval json directory found. Searched:\n  " + "\n  ".join(searched))


def _select_batch_files(eval_dir, split, json_file=None, include_backups=False):
    if json_file:
        requested = Path(json_file).expanduser()
        candidates = [requested]
        if not requested.is_absolute():
            candidates.extend([eval_dir / requested, eval_dir / requested.name])
        for candidate in candidates:
            if candidate.is_file():
                return [candidate]
        raise FileNotFoundError(f"Requested json file not found: {json_file}")

    files = []
    for path in eval_dir.glob("*.json"):
        name = path.name
        if name == "summary.json":
            continue
        if not include_backups and ".backup." in name:
            continue
        if "details_extracted" not in name:
            continue
        if split != "all" and f"_{split}_" not in name:
            continue
        files.append(path)
    return sorted(files, key=_checkpoint_sort_key)


def evaluate_file(dataset, input_file, limit=None, quiet=False, print_summary=True, alias_dir=None):
    checkpoint_details_mode = eval_utils.is_existing_json_file(input_file)

    if checkpoint_details_mode:
        ground_truth_datas = None
        question_string = "question"
        output_datas = eval_utils.read_checkpoint_eval_details(input_file)
        loaded_type_field, aname_dict, alias_dict, add_ans_alias_dict = _load_alias_data(dataset, alias_dir)
        type_field = loaded_type_field if output_datas and loaded_type_field in output_datas[0] else ''
    else:
        ground_truth_datas, question_string, output_datas = eval_utils.prepare_dataset_for_eval(dataset, input_file)
        type_field, aname_dict, alias_dict, add_ans_alias_dict = _load_alias_data(dataset, alias_dir)

    if limit is not None:
        output_datas = output_datas[:limit]

    total_precision = 0
    total_recall = 0
    total_f1 = 0
    num_questions = 0
    count_q = {}
    right_q = {}

    num_right = 0
    num_error = 0
    num_empty = 0
    error_question = []
    right_question = []

    call_num_list = []
    time_list = []
    token_num_list = {
        "input": [],
        "output": [],
        "total": []
    }

    for data in output_datas:
        if checkpoint_details_mode:
            response, answers, answer_groups, ori_data = _evaluate_checkpoint_record(data, alias_dict)
        else:
            response, answers, answer_groups, ori_data = _evaluate_legacy_record(
                dataset,
                question_string,
                data,
                ground_truth_datas,
                aname_dict,
                alias_dict,
                add_ans_alias_dict,
            )

        question = _question_text(data, question_string)
        if not quiet:
            print(question, response, answers)

        if 'time' in data.keys():
            call_num_list.append(data.get('call_num', 0))
            time_list.append(data.get('time', 0))
            token_num_list['input'].append(data.get('input_token', 0))
            token_num_list['output'].append(data.get('output_token', 0))
            token_num_list['total'].append(data.get('total_token', 0))

        if type_field:
            q_type = ori_data[type_field]
            if q_type not in count_q.keys():
                count_q[q_type] = 0
            count_q[q_type] += 1

        f1, precision, recall = eval_utils.calculate_f1(response, answers, answer_groups=answer_groups)
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        num_questions += 1

        is_correct = eval_utils.exact_match(response, answers)
        if _result_is_empty(response):
            num_empty += 1

        if is_correct:
            if type_field:
                q_type = ori_data[type_field]
                if q_type not in right_q.keys():
                    right_q[q_type] = 0
                right_q[q_type] += 1
            num_right += 1
            right_question.append(question)
            if not quiet:
                print(f"Correct: {response} | {answers}")
                print(f"P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}")
        else:
            num_error += 1
            error_question.append(question)
            if not quiet:
                print(f"INcorrect: {response} | {answers}")
                print(f"P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}")

    total_count = len(output_datas)
    exact_match = float(num_right / total_count) if total_count else 0.0
    avg_precision = total_precision / num_questions if num_questions else 0.0
    avg_recall = total_recall / num_questions if num_questions else 0.0
    avg_f1 = total_f1 / num_questions if num_questions else 0.0

    checkpoint, split = _infer_checkpoint_and_split(input_file)
    metrics = {
        "file": str(input_file),
        "name": Path(input_file).name,
        "checkpoint": checkpoint,
        "split": split,
        "all": total_count,
        "evaluated": num_questions,
        "exact_match": exact_match,
        "right": num_right,
        "error": num_error,
        "empty": num_empty,
        "average_precision": avg_precision,
        "average_recall": avg_recall,
        "average_f1": avg_f1,
    }

    if count_q:
        metrics["count_by_type"] = count_q
        metrics["right_by_type"] = right_q
        metrics["accuracy_by_type"] = {
            k: (right_q[k] / v if k in right_q else 0) for k, v in count_q.items()
        }

    if call_num_list:
        metrics["api_count"] = len(call_num_list)
        metrics["mean_call_num"] = _mean(call_num_list)
        metrics["mean_time"] = _mean(time_list)
        metrics["mean_tokens"] = {
            t_type: _mean(nu_l) for t_type, nu_l in token_num_list.items()
        }

    if print_summary:
        _print_single_summary(metrics)

    return metrics


def _print_single_summary(metrics):
    print(metrics["right"])
    print("All: ", metrics["all"])
    print("Evaluated: ", metrics["evaluated"])
    print("Exact Match: {}".format(metrics["exact_match"]))
    print("right: {}, error: {}, empty: {}".format(metrics["right"], metrics["error"], metrics["empty"]))
    print(f"Average Precision: {metrics['average_precision']:.4f}")
    print(f"Average Recall: {metrics['average_recall']:.4f}")
    print(f"Average F1: {metrics['average_f1']:.4f}")

    if "count_by_type" in metrics:
        count_q = metrics["count_by_type"]
        right_q = metrics["right_by_type"]
        print(sorted(count_q.items(), key=lambda x:x[0]))
        print(sorted(right_q.items(), key=lambda x:x[0]))
        for k, value in metrics["accuracy_by_type"].items():
            print(k, value)

    if "api_count" in metrics:
        print(metrics["api_count"])
        print('call num:', metrics["mean_call_num"])
        print('time:', metrics["mean_time"])
        for t_type, value in metrics["mean_tokens"].items():
            print(t_type, value)


def _print_batch_summary(metrics_list):
    if not metrics_list:
        print("No matching eval json files found.")
        return

    headers = ["checkpoint", "split", "EM", "F1", "P", "R", "right", "error", "empty", "all", "file"]
    rows = []
    for metrics in metrics_list:
        rows.append([
            metrics["checkpoint"] or "-",
            metrics["split"] or "-",
            f"{metrics['exact_match']:.4f}",
            f"{metrics['average_f1']:.4f}",
            f"{metrics['average_precision']:.4f}",
            f"{metrics['average_recall']:.4f}",
            str(metrics["right"]),
            str(metrics["error"]),
            str(metrics["empty"]),
            str(metrics["all"]),
            metrics["name"],
        ])

    widths = [
        max(len(headers[col_idx]), *(len(row[col_idx]) for row in rows))
        for col_idx in range(len(headers))
    ]
    print("  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))


def run_batch(args):
    if args.json_file and not args.experiment_dir and not args.checkpoint_eval_dir:
        json_path = Path(args.json_file).expanduser()
        if not json_path.is_file():
            raise FileNotFoundError(f"Requested json file not found: {args.json_file}")
        eval_dir = json_path.parent
        files = [json_path]
    else:
        eval_dir = _resolve_checkpoint_eval_dir(
            args.experiment_dir,
            args.checkpoint_eval_dir,
            args.eval_root,
        )
        files = _select_batch_files(
            eval_dir,
            args.split,
            json_file=args.json_file,
            include_backups=args.include_backups,
        )

    metrics_list = []
    for path in files:
        metrics = evaluate_file(
            args.dataset,
            path,
            limit=args.limit,
            quiet=not args.print_details,
            print_summary=False,
            alias_dir=args.alias_dir,
        )
        metrics_list.append(metrics)

    _print_batch_summary(metrics_list)

    if metrics_list and not args.no_summary_file:
        if args.summary_file:
            summary_path = Path(args.summary_file)
        else:
            suffix = Path(args.json_file).stem if args.json_file else args.split
            summary_path = eval_dir / f"eval_py_summary_{suffix}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(metrics_list, f, ensure_ascii=False, indent=2)
        print(f"Saved summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cwq", help="choose the dataset.")
    parser.add_argument("--output_file", type=str, default="VoG_cwq_gpt-3.5-turbo", help="legacy output name or a json eval details file.")
    parser.add_argument("--input_file", type=str, default=None, help="json eval details file, e.g. final_validation_details_extracted.json.")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N records.")
    parser.add_argument("--quiet", action="store_true", help="do not print per-question details.")
    parser.add_argument("--experiment_dir", "--experiment-dir", "--experiment", default=None, help="Experiment name/path under checkpoints/webqsp for batch eval.")
    parser.add_argument("--checkpoint_eval_dir", "--checkpoint-eval-dir", default=None, help="Direct path to eval_results/webqsp_checkpoint_eval.")
    parser.add_argument("--eval_root", "--eval-root", default="checkpoints/webqsp", help="Root containing WebQSP experiment directories.")
    parser.add_argument("--split", choices=["validation", "test", "all"], default="validation", help="Batch split selector.")
    parser.add_argument("--json_file", "--json-file", default=None, help="One json file to evaluate in batch mode; may be absolute or relative to checkpoint eval dir.")
    parser.add_argument("--include_backups", "--include-backups", action="store_true", help="Include *.backup.json files in batch mode.")
    parser.add_argument("--summary_file", "--summary-file", default=None, help="Where to write batch metrics json.")
    parser.add_argument("--no_summary_file", "--no-summary-file", action="store_true", help="Do not write batch metrics json.")
    parser.add_argument("--print_details", "--print-details", action="store_true", help="Print per-question details in batch mode.")
    parser.add_argument(
        "--alias_dir",
        "--alias-dir",
        default=None,
        help="Directory containing the original WebQSP/CWQ evaluation alias JSON files.",
    )

    args = parser.parse_args()

    if args.experiment_dir or args.checkpoint_eval_dir or args.json_file:
        run_batch(args)
        return

    input_file = args.input_file or args.output_file
    evaluate_file(
        args.dataset,
        input_file,
        limit=args.limit,
        quiet=args.quiet,
        print_summary=True,
        alias_dir=args.alias_dir,
    )


if __name__ == '__main__':
    main()


# Examples:
# python eval.py --dataset cwq --output_file VoG_cwq_gpt-4
# python eval.py --dataset webqsp --input_file checkpoints/webqsp/experiment/eval_results/webqsp_checkpoint_eval/final_test_details_extracted.json --quiet
# python eval.py --dataset webqsp --experiment grlprof-llama-3-8b-instruct-gte-modernbert-base-webqsp-projector --split validation
# python eval.py --dataset webqsp --experiment checkpoints/webqsp/webqsp_checkpoint_eval_no_graph/webqsp_checkpoint_eval_no_graph --split test
# python eval.py --dataset webqsp --checkpoint_eval_dir checkpoints/webqsp/grlprof-.../eval_results --json_file final_validation_details_extracted.json
