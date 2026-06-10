import json
from pathlib import Path

def _resolve_output_path(file_path):
    path = Path(file_path)
    if path.is_file():
        return path

    vog_jsonl_path = Path("../VoG") / f"{file_path}.jsonl"
    if vog_jsonl_path.is_file():
        return vog_jsonl_path

    if path.suffix:
        return path
    return vog_jsonl_path


def is_existing_json_file(file_path):
    path = Path(file_path)
    return path.is_file() and path.suffix == ".json"


def read_output(file_path, question_string):
    answered_dict = {}
    file_path = _resolve_output_path(file_path)

    if file_path.suffix == ".json":
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        if isinstance(data, dict):
            data = data.get("details", data.get("records", data.get("results", data.get("data", []))))
        return data

    with open(file_path, 'r', encoding='utf-8') as file:
        start_index = 0  # Set the starting index here
        for i, line in enumerate(file):
            if i < start_index:
                continue
            data = json.loads(line)
            answered_dict[data[question_string]] = data

    answered_list = list(answered_dict.values())
    return answered_list


def read_checkpoint_eval_details(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    if isinstance(data, dict):
        for key in ("details", "records", "results", "data"):
            if isinstance(data.get(key), list):
                return data[key]
        raise ValueError(f"Unsupported checkpoint eval json object keys: {sorted(data.keys())}")

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of eval records in {file_path}, got {type(data).__name__}")

    return data

def prepare_dataset_for_eval(dataset_name, output_file):
    if dataset_name == 'cwq':
        with open('../data/cwq.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'webqsp':

        with open('../data/WebQSP.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'RawQuestion'
        # question_string = 'question'

    elif dataset_name == 'webquestions':
        with open('../data/WebQuestions.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    else:
        print("dataset not found, you should pick from {cwq, webqsp, webquestions}.")
        exit(-1)

    output_datas = read_output(output_file, question_string)


    return datas, question_string, output_datas


def align(dataset_name, question_string, data, ground_truth_datas, aname_dict, alias_dict, add_ans_alias_dict):
    answer_list= []
    origin_data = [j for j in ground_truth_datas if j[question_string] == data[question_string]][0]
    if dataset_name == 'cwq':
        add_data = aname_dict[data[question_string]]
        add_ans_alias_data = add_ans_alias_dict[data[question_string]]
        add_data += add_ans_alias_data
        if 'answers' in origin_data:
            answers = origin_data["answers"]
        else:
            answers = origin_data["answer"]
        if answers not in add_data:
            add_data.append(answers)
        
        answer_list = add_data
        alias_list = []
        for x in answer_list:
            if x in alias_dict.keys():
                alias_list += alias_dict[x]
        
        answer_list = list(set(answer_list)|set(alias_list))
    elif dataset_name == 'webquestions':
        answer_list = origin_data["answers"]

    elif dataset_name == 'webqsp':
        answers = origin_data["Parses"]
        for answer in answers:
            for name in answer['Answers']:
                if name['EntityName'] == None:
                    answer_list.append(name['AnswerArgument'])
                else:
                    answer_list.append(name['EntityName'])
        alias_list = []
        for x in answer_list:
            if x in alias_dict.keys():
                alias_list += alias_dict[x]
        
        answer_list = list(set(answer_list)|set(alias_list))
        alias_list = []
        for x in answer_list:
            if x in alias_dict.keys():
                alias_list += alias_dict[x]
        
        answer_list = list(set(answer_list)|set(alias_list))

    return list(set(answer_list)), origin_data
    

def _flatten_answer_values(value):
    if value is None:
        return []
    if isinstance(value, dict):
        if "answers" in value:
            return _flatten_answer_values(value["answers"])
        flattened = []
        for item in value.values():
            flattened.extend(_flatten_answer_values(item))
        return flattened
    if isinstance(value, (list, tuple, set)):
        flattened = []
        for item in value:
            flattened.extend(_flatten_answer_values(item))
        return flattened
    return [str(value)]


def split_prediction_answers(prediction):
    parts = []
    for item in _flatten_answer_values(prediction):
        for part in str(item).split(','):
            part = part.strip()
            if part:
                parts.append(part)
    return parts


def normalize_answer_list(answers):
    return [answer.strip() for answer in _flatten_answer_values(answers) if str(answer).strip()]


def _unique_preserve_order(values):
    seen = set()
    unique_values = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique_values.append(value)
    return unique_values


def parse_prediction_raw(prediction_raw):
    if prediction_raw is None:
        return []
    if not isinstance(prediction_raw, str):
        return _flatten_answer_values(prediction_raw)

    text = prediction_raw.strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return split_prediction_answers(text)

    return _flatten_answer_values(parsed)


def extract_prediction_answers(data):
    if "prediction_answers_recovered" in data:
        return normalize_answer_list(data["prediction_answers_recovered"])
    if "results" in data:
        return data["results"]
    if "prediction_raw" in data:
        return parse_prediction_raw(data["prediction_raw"])
    if "prediction" in data:
        return data["prediction"]
    return []


def extract_gold_answers(data):
    if "gold_raw" in data:
        return normalize_answer_list(data["gold_raw"])
    if "answers" in data:
        return normalize_answer_list(data["answers"])
    if "answer" in data:
        return normalize_answer_list(data["answer"])
    return []


def expand_answers_with_aliases(answers, alias_dict, rounds=2):
    answer_list = _unique_preserve_order(normalize_answer_list(answers))
    if not alias_dict:
        return answer_list

    for _ in range(rounds):
        alias_list = []
        for answer in answer_list:
            if answer in alias_dict.keys():
                alias_list += alias_dict[answer]
        answer_list = _unique_preserve_order(answer_list + normalize_answer_list(alias_list))

    return answer_list


def expand_answer_alias_groups(answers, alias_dict, rounds=2):
    groups = []
    for answer in _unique_preserve_order(normalize_answer_list(answers)):
        groups.append(expand_answers_with_aliases([answer], alias_dict, rounds=rounds))
    return groups


def flatten_answer_groups(answer_groups):
    flattened = []
    for group in answer_groups or []:
        flattened.extend(normalize_answer_list(group))
    return _unique_preserve_order(flattened)


def exact_match(response, answers):
    response_parts = split_prediction_answers(response)
    answer_list = normalize_answer_list(answers)
    for part in response_parts:
        clean_result = part.strip().replace(" ","").lower()
        for answer in answer_list:
            clean_answer = answer.strip().replace(" ","").lower()
            if clean_result == clean_answer or clean_result in clean_answer or clean_answer in clean_result:
                return True
    return False

def calculate_f1(prediction, answers, answer_groups=None):

    prediction_parts = split_prediction_answers(prediction)
    answer_list = normalize_answer_list(answers)
    if not prediction_parts or not answer_list:
        return 0, 0, 0
    matched = 0
    p_matched=0

    recall_units = answer_groups if answer_groups else [[answer] for answer in answer_list]
    for answer_group in recall_units:
        if exact_match(prediction_parts, answer_group):
            matched += 1

    for part in prediction_parts:
        if exact_match(part,answer_list):
            p_matched+=1

    precision = p_matched / len(prediction_parts) if len(prediction_parts)>0 else 0
    recall = matched / len(recall_units) if len(recall_units)>0 else 0
    if precision + recall == 0:
        return 0, precision, recall
    else:
        return 2 * precision * recall / (precision + recall), precision, recall
