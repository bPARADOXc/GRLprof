#!/usr/bin/env python3
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from requests.exceptions import ProxyError

# Example:
# DASHSCOPE_API_KEY=... python tools/semantic_match_with_qwen.py \
#   checkpoints/webqsp/experiment/eval_results/webqsp_checkpoint_eval/final_test_details_extracted.json \
#   --proxy-mode direct --batch-size 5 --max-workers 8 --max-tokens 768
DEFAULT_MODEL = "qwen3.5-plus"
DEFAULT_ENDPOINT = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
)
DEFAULT_BATCH_SIZE = 5
DEFAULT_MAX_WORKERS = 8
DEFAULT_TIMEOUT = 180
DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_TOKENS = 768
METRIC_FIELDS = ("precision", "recall", "f1", "hit_at_1", "true_positive")
TRUNCATED_PARSE_MODES = ("truncated_answers_field", "truncated_list")

SYSTEM_PROMPT = """Evaluate WebQSP answer overlap semantically.

For each sample, prediction_answers and gold_answers may both contain multiple answers.
Count one-to-one semantic matches between the two answer sets. A prediction matches a gold answer if a human would accept them as the same answer for the question: aliases, abbreviations, harmless wording changes, singular/plural, year vs full date when the question only asks the year, and acceptable location granularity. Do not match different entities or merely related places/facts. Each prediction and each gold answer can be used at most once.

Let true_positive be the number of matched pairs. Then use exactly these formulas:
precision = true_positive / prediction_count if prediction_count > 0 else 0.0
recall = true_positive / gold_count if gold_count > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
hit_at_1 = 1.0 if true_positive > 0 else 0.0
Special case: if prediction_count == 0 and gold_count == 0, then precision = recall = f1 = hit_at_1 = 1.0.

Return only one JSON list in the same order. Each sample must output one JSON object. Each object must be:
{"precision": number, "recall": number, "f1": number, "hit_at_1": number, "true_positive": integer}
Use compact JSON on a single line. Do not pretty-print. Keep decimals short.

Example 1:
question: what else did ben franklin invent
prediction_answers: ["Bifocal glasses", "Lightning rod"]
gold_answers: ["Lightning rod", "Glass harmonica", "Bifocals", "Franklin stove"]
output json:
{"precision": 1.0, "recall": 0.5, "f1": 0.6666666667, "hit_at_1": 1.0, "true_positive": 2}

Example 2:
question: what timezone is sweden
prediction_answers: ["Central European Time", "Eastern European Time"]
gold_answers: ["Central European Time Zone"]
output json:
{"precision": 0.5, "recall": 1.0, "f1": 0.6666666667, "hit_at_1": 1.0, "true_positive": 1}

No markdown. No explanation.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Use qwen3.5-plus to compute semantic precision/recall/F1/hit@1 "
            "for extracted WebQSP samples."
        )
    )
    parser.add_argument(
        "input",
        help="Path to the extracted JSON file.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("QWEN_MODEL", DEFAULT_MODEL),
        help="Model name. Default: qwen3.5-plus",
    )
    parser.add_argument(
        "--endpoint",
        default=os.getenv("DASHSCOPE_CHAT_ENDPOINT", DEFAULT_ENDPOINT),
        help="DashScope OpenAI-compatible chat completions endpoint.",
    )
    parser.add_argument(
        "--proxy-mode",
        choices=["auto", "env", "direct"],
        default="auto",
        help="HTTP proxy mode: auto falls back from env proxy to direct, env only uses environment proxies, direct ignores them. Default: auto",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of samples per request.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Maximum number of batches to request in parallel. Default: 4",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Only process the first N pending items.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Sleep time between batch submissions. Default: 0.0",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Retry times for a failed batch.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum output tokens for each batch request. Default: 768",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Default: 0.0",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Disable Qwen hybrid thinking mode.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-evaluate all records, even if metric fields already exist.",
    )
    parser.add_argument(
        "--only-truncated-prediction-raw",
        action="store_true",
        help="Only process samples whose prediction_raw is truncated but recoverable.",
    )
    parser.add_argument(
        "--only-generation-failed",
        action="store_true",
        help=(
            "Only re-evaluate records whose semantic_match_generation_failed "
            "field is explicitly true. Can be combined with "
            "--only-truncated-prediction-raw."
        ),
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a backup file before in-place modification.",
    )
    return parser


def get_api_key() -> str:
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("BAILIAN_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set DASHSCOPE_API_KEY or BAILIAN_API_KEY first."
        )
    return api_key


def load_items(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as infile:
        items = json.load(infile)
    if not isinstance(items, list):
        raise ValueError("Input JSON must be a list.")
    return items


def save_items(path: Path, items: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as outfile:
        json.dump(items, outfile, ensure_ascii=False, indent=2)


def maybe_backup(path: Path) -> Optional[Path]:
    backup_path = path.with_name("%s.backup.json" % path.stem)
    if backup_path.exists():
        return backup_path
    shutil.copy2(str(path), str(backup_path))
    return backup_path


def ensure_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def normalize_answer_text(answer: str) -> str:
    answer = str(answer).strip()
    answer = " ".join(answer.split())
    return answer.rstrip(".;:,")


def normalize_answer_list(answers: Any) -> List[str]:
    answers = ensure_list(answers)
    normalized = []
    seen = set()
    for answer in answers:
        value = normalize_answer_text(answer)
        key = value.lower()
        if not value or key in seen:
            continue
        seen.add(key)
        normalized.append(value)
    return normalized


def iter_json_like_candidates(text: str) -> List[str]:
    candidates = []

    def add(candidate: str) -> None:
        candidate = candidate.strip()
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    stripped = strip_code_fence(text)
    add(text)
    add(stripped)

    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        add(stripped[first_brace : last_brace + 1])

    first_bracket = stripped.find("[")
    last_bracket = stripped.rfind("]")
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        add(stripped[first_bracket : last_bracket + 1])

    return candidates


def try_parse_json_like(candidate: str) -> Any:
    try:
        return json.loads(candidate), "json.loads"
    except Exception:
        pass

    sanitized = re.sub(r",(\s*[}\]])", r"\1", candidate)
    if sanitized != candidate:
        try:
            return json.loads(sanitized), "json.loads_sanitized"
        except Exception:
            pass

    try:
        return ast.literal_eval(candidate), "ast.literal_eval"
    except Exception:
        pass

    if sanitized != candidate:
        try:
            return ast.literal_eval(sanitized), "ast.literal_eval_sanitized"
        except Exception:
            pass

    return None, None


def extract_answers_from_payload(payload: Any) -> Optional[List[str]]:
    if isinstance(payload, dict):
        if "answers" in payload:
            answers = payload["answers"]
        elif "answer" in payload:
            answers = payload["answer"]
        else:
            return None
    elif isinstance(payload, list):
        answers = payload
    elif isinstance(payload, str):
        answers = [payload]
    else:
        return None

    if isinstance(answers, list):
        return [str(item) for item in answers]
    if answers is None:
        return []
    return [str(answers)]


def extract_quoted_strings(text: str) -> List[str]:
    values = []
    for match in re.finditer(r'"((?:\\.|[^"\\])*)"', text):
        token = match.group(1)
        try:
            values.append(json.loads('"%s"' % token))
        except Exception:
            values.append(token)
    return values


def recover_answers_from_truncated_json(raw_text: str) -> Any:
    stripped = strip_code_fence(raw_text)

    answers_key_match = re.search(r'"answers"\s*:\s*\[', stripped)
    if answers_key_match is not None:
        list_body = stripped[answers_key_match.end() :]
        answers = normalize_answer_list(extract_quoted_strings(list_body))
        if answers:
            return answers, "truncated_answers_field"

    first_bracket = stripped.find("[")
    if first_bracket != -1:
        list_body = stripped[first_bracket + 1 :]
        answers = normalize_answer_list(extract_quoted_strings(list_body))
        if answers:
            return answers, "truncated_list"

    return None, None


def parse_prediction_answers_and_mode(prediction_raw: Any) -> Any:
    raw_text = "" if prediction_raw is None else str(prediction_raw).strip()
    if not raw_text:
        return [], "empty"

    for candidate in iter_json_like_candidates(raw_text):
        payload, parse_mode = try_parse_json_like(candidate)
        if payload is None:
            continue
        answers = extract_answers_from_payload(payload)
        if answers is None:
            continue
        return normalize_answer_list(answers), parse_mode

    recovered_answers, parse_mode = recover_answers_from_truncated_json(raw_text)
    if recovered_answers is not None:
        return recovered_answers, parse_mode

    return normalize_answer_list([strip_code_fence(raw_text)]), "plain_text_fallback"


def parse_prediction_answers(prediction_raw: Any) -> List[str]:
    answers, _ = parse_prediction_answers_and_mode(prediction_raw)
    return answers


def parse_gold_answers(gold_raw: Any) -> List[str]:
    if isinstance(gold_raw, str):
        try:
            decoded = json.loads(gold_raw)
        except Exception:
            return normalize_answer_list([gold_raw]) if gold_raw.strip() else []
        return normalize_answer_list(decoded)
    return normalize_answer_list(gold_raw)


def build_batch_payload(batch_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    payload = []
    for item in batch_items:
        prediction_answers, prediction_parse_mode = parse_prediction_answers_and_mode(
            item.get("prediction_raw")
        )
        gold_answers = parse_gold_answers(item.get("gold_raw"))
        payload.append(
            {
                "id": item["id"],
                "question": item["question"],
                "prediction_answers": prediction_answers,
                "prediction_parse_mode": prediction_parse_mode,
                "gold_answers": gold_answers,
                "prediction_count": len(prediction_answers),
                "gold_count": len(gold_answers),
            }
        )
    return payload


def build_messages(batch_payload: List[Dict[str, Any]], attempt: int) -> List[Dict[str, str]]:
    retry_note = ""
    if attempt > 1:
        retry_note = (
            "\nPrevious attempt was invalid. This time, return only a JSON list of "
            "objects with the required metric fields. Use compact one-line JSON only. No explanation, no markdown, no extra text."
        )

    user_prompt = (
        "Judge the following samples.\n"
        "Remember: output only one compact JSON list of metric objects in the same order as input."
        "%s\nSamples:%s"
        % (
            retry_note,
            json.dumps(batch_payload, ensure_ascii=False, separators=(",", ":")),
        )
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def extract_content(response_json: Dict[str, Any]) -> str:
    choices = response_json.get("choices") or []
    if not choices:
        raise ValueError("API response does not contain choices.")

    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict):
                if "text" in part and part["text"] is not None:
                    texts.append(str(part["text"]))
                elif "content" in part and part["content"] is not None:
                    texts.append(str(part["content"]))
        return "\n".join(texts).strip()
    raise ValueError("Unable to extract assistant content from API response.")


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def find_json_array_candidate(text: str) -> str:
    stripped = strip_code_fence(text)
    if stripped.startswith("[") and stripped.endswith("]"):
        return stripped

    match = re.search(r"\[[\s\S]*\]", stripped)
    if match:
        return match.group(0)

    raise ValueError("Could not find a JSON array in model output: %s" % text)


def repair_unclosed_json(text: str) -> str:
    stripped = strip_code_fence(text).strip()
    if not stripped:
        return stripped

    result = []
    stack = []
    in_string = False
    escaped = False

    for ch in stripped:
        result.append(ch)
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch in "[{":
            stack.append(ch)
        elif ch == "]":
            if stack and stack[-1] == "[":
                stack.pop()
        elif ch == "}":
            if stack and stack[-1] == "{":
                stack.pop()

    if in_string:
        result.append('"')

    while stack:
        opener = stack.pop()
        result.append("]" if opener == "[" else "}")

    return "".join(result)


def parse_metric_objects_by_pattern(
    text: str, expected_len: int
) -> Optional[List[Dict[str, Any]]]:
    num = r"-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    pattern = re.compile(
        r'\{\s*"precision"\s*:\s*(?P<precision>%s)\s*,\s*"recall"\s*:\s*(?P<recall>%s)\s*,\s*"f1"\s*:\s*(?P<f1>%s)\s*,\s*"hit_at_1"\s*:\s*(?P<hit_at_1>%s)\s*,\s*"true_positive"\s*:\s*(?P<true_positive>%s)\s*\}?'
        % (num, num, num, num, num)
    )
    matches = list(pattern.finditer(strip_code_fence(text)))
    if len(matches) != expected_len:
        return None

    parsed = []
    for match in matches:
        parsed.append(
            {
                "precision": float(match.group("precision")),
                "recall": float(match.group("recall")),
                "f1": float(match.group("f1")),
                "hit_at_1": float(match.group("hit_at_1")),
                "true_positive": int(round(float(match.group("true_positive")))),
            }
        )
    return parsed


def to_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except Exception:
        raise ValueError("Field %s is not numeric: %r" % (field_name, value))


def to_int(value: Any, field_name: str) -> int:
    try:
        value_float = float(value)
    except Exception:
        raise ValueError("Field %s is not numeric: %r" % (field_name, value))
    value_int = int(round(value_float))
    if abs(value_float - value_int) > 1e-6:
        raise ValueError("Field %s is not an integer: %r" % (field_name, value))
    return value_int


def compute_metrics_from_counts(true_positive: int, pred_count: int, gold_count: int) -> Dict[str, Any]:
    if pred_count == 0 and gold_count == 0:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "hit_at_1": 1.0,
            "true_positive": 0,
        }

    precision = float(true_positive) / pred_count if pred_count > 0 else 0.0
    recall = float(true_positive) / gold_count if gold_count > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    hit_at_1 = 1.0 if true_positive > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hit_at_1": hit_at_1,
        "true_positive": true_positive,
    }


def coerce_metric_list(
    value: Any, batch_payload: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("Model output is not a list.")
    if len(value) != len(batch_payload):
        raise ValueError(
            "Model output length mismatch: expected %d, got %d"
            % (len(batch_payload), len(value))
        )

    result = []
    for idx, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError("Model output item is not a JSON object: %r" % (item,))
        for field_name in METRIC_FIELDS:
            if field_name not in item:
                raise ValueError("Missing field %s in model output item %d" % (field_name, idx))

        true_positive = to_int(item["true_positive"], "true_positive")
        pred_count = batch_payload[idx]["prediction_count"]
        gold_count = batch_payload[idx]["gold_count"]
        max_tp = min(pred_count, gold_count)
        if true_positive < 0 or true_positive > max_tp:
            raise ValueError(
                "true_positive out of range for item %d: %d not in [0, %d]"
                % (idx, true_positive, max_tp)
            )

        metrics = compute_metrics_from_counts(true_positive, pred_count, gold_count)

        # Parse and lightly validate model-returned metrics, but trust the local recomputation.
        model_precision = to_float(item["precision"], "precision")
        model_recall = to_float(item["recall"], "recall")
        model_f1 = to_float(item["f1"], "f1")
        model_hit_at_1 = to_float(item["hit_at_1"], "hit_at_1")
        if any(number < 0 for number in (model_precision, model_recall, model_f1, model_hit_at_1)):
            raise ValueError("Negative metric value in model output item %d" % idx)

        result.append(metrics)
    return result


def parse_model_output(
    text: str, batch_payload: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    candidates = []
    raw_text = text.strip()
    if raw_text:
        candidates.append(raw_text)

    repaired = repair_unclosed_json(text)
    if repaired and repaired not in candidates:
        candidates.append(repaired)

    try:
        candidate = find_json_array_candidate(text)
        if candidate not in candidates:
            candidates.append(candidate)
    except Exception:
        pass

    last_error = None
    for candidate_text in candidates:
        try:
            parsed = json.loads(candidate_text)
            return coerce_metric_list(parsed, batch_payload)
        except Exception as exc:
            last_error = exc

    regex_parsed = parse_metric_objects_by_pattern(text, len(batch_payload))
    if regex_parsed is not None:
        return coerce_metric_list(regex_parsed, batch_payload)

    raise ValueError("Failed to parse model output: %s" % last_error)


def request_batch(
    endpoint: str,
    api_key: str,
    model: str,
    batch_payload: List[Dict[str, Any]],
    temperature: float,
    timeout: int,
    max_retries: int,
    max_tokens: int,
    enable_thinking: bool,
    proxy_mode: str,
) -> List[Dict[str, Any]]:
    last_error = None
    headers = {
        "Authorization": "Bearer %s" % api_key,
        "Content-Type": "application/json",
    }

    for attempt in range(1, max_retries + 1):
        messages = build_messages(batch_payload, attempt)
        request_body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": 0.1,
            "max_tokens": max_tokens,
            "stream": False,
            "enable_thinking": enable_thinking,
        }
        try:
            request_modes = ["env", "direct"] if proxy_mode == "auto" else [proxy_mode]
            for mode_index, request_mode in enumerate(request_modes):
                session = requests.Session()
                session.trust_env = request_mode == "env"
                try:
                    response = session.post(
                        endpoint,
                        headers=headers,
                        json=request_body,
                        timeout=timeout,
                    )
                    response.raise_for_status()
                    response_json = response.json()
                    content = extract_content(response_json)
                    return parse_model_output(content, batch_payload)
                except ProxyError as exc:
                    last_error = exc
                    if proxy_mode == "auto" and request_mode == "env" and mode_index == 0:
                        print(
                            "Proxy request failed, retrying this batch with direct connection...",
                            file=sys.stderr,
                        )
                        continue
                    raise
                finally:
                    session.close()
        except Exception as exc:
            last_error = exc
            if attempt == max_retries:
                break
            wait_seconds = min(8, 2 ** (attempt - 1))
            print(
                "Batch request failed on attempt %d/%d: %s. Retrying in %.1fs..."
                % (attempt, max_retries, exc, wait_seconds),
                file=sys.stderr,
            )
            time.sleep(wait_seconds)

    raise RuntimeError("Batch request failed after retries: %s" % last_error)


def collect_pending_indices(
    items: List[Dict[str, Any]],
    force: bool,
    max_items: Optional[int],
    only_truncated_prediction_raw: bool,
    only_generation_failed: bool,
) -> List[int]:
    indices = []
    for idx, item in enumerate(items):
        _, prediction_parse_mode = parse_prediction_answers_and_mode(
            item.get("prediction_raw")
        )
        if (
            only_truncated_prediction_raw
            and prediction_parse_mode not in TRUNCATED_PARSE_MODES
        ):
            continue
        missing_metric = any(field_name not in item for field_name in METRIC_FIELDS)
        generation_failed = item.get("semantic_match_generation_failed")
        if only_generation_failed and generation_failed is not True:
            continue
        if (
            force
            or missing_metric
            or generation_failed is True
        ):
            indices.append(idx)
    if max_items is not None:
        return indices[:max_items]
    return indices


def shorten_error_message(error: Exception, limit: int = 500) -> str:
    text = str(error).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def build_batch_jobs(
    items: List[Dict[str, Any]], pending_indices: List[int], batch_size: int
) -> List[Dict[str, Any]]:
    jobs = []
    total_pending = len(pending_indices)
    total_batches = (total_pending + batch_size - 1) // batch_size
    for batch_number, start in enumerate(range(0, total_pending, batch_size), start=1):
        batch_indices = pending_indices[start : start + batch_size]
        batch_items = [items[idx] for idx in batch_indices]
        jobs.append(
            {
                "batch_number": batch_number,
                "total_batches": total_batches,
                "batch_indices": batch_indices,
                "batch_ids": [str(item["id"]) for item in batch_items],
                "batch_payload": build_batch_payload(batch_items),
            }
        )
    return jobs


def process_batch_job(
    job: Dict[str, Any],
    endpoint: str,
    api_key: str,
    model: str,
    temperature: float,
    timeout: int,
    max_retries: int,
    max_tokens: int,
    enable_thinking: bool,
    proxy_mode: str,
) -> Dict[str, Any]:
    metrics_list = request_batch(
        endpoint=endpoint,
        api_key=api_key,
        model=model,
        batch_payload=job["batch_payload"],
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
        max_tokens=max_tokens,
        enable_thinking=enable_thinking,
        proxy_mode=proxy_mode,
    )
    result = dict(job)
    result["metrics_list"] = metrics_list
    return result


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.is_file():
        raise FileNotFoundError("Input file not found: %s" % input_path)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.max_workers <= 0:
        raise ValueError("--max-workers must be positive.")
    if args.sleep_seconds < 0:
        raise ValueError("--sleep-seconds must be non-negative.")

    if args.batch_size > 5:
        print(
            "Warning: batch_size=%d may increase output truncation risk; reduce it if parsing keeps failing."
            % args.batch_size,
            file=sys.stderr,
        )

    items = load_items(input_path)
    total_truncated_prediction_raw = sum(
        1
        for item in items
        if parse_prediction_answers_and_mode(item.get("prediction_raw"))[1]
        in TRUNCATED_PARSE_MODES
    )
    pending_indices = collect_pending_indices(
        items,
        args.force,
        args.max_items,
        args.only_truncated_prediction_raw,
        args.only_generation_failed,
    )
    if not pending_indices:
        print(
            "No pending items to process. total_truncated_prediction_raw=%d"
            % total_truncated_prediction_raw
        )
        return

    api_key = get_api_key()
    if not args.no_backup:
        backup_path = maybe_backup(input_path)
        print("Backup created at %s" % backup_path)
    total_pending = len(pending_indices)
    batch_jobs = build_batch_jobs(items, pending_indices, args.batch_size)
    total_batches = len(batch_jobs)
    print(
        "Processing %d items in %d batches with model %s, max_workers=%d, total_truncated_prediction_raw=%d"
        % (
            total_pending,
            total_batches,
            args.model,
            args.max_workers,
            total_truncated_prediction_raw,
        )
    )

    processed = 0
    enable_thinking = not args.disable_thinking
    future_to_job = {}

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        for job in batch_jobs:
            print(
                "Submitting batch %d/%d ids=%s"
                % (
                    job["batch_number"],
                    job["total_batches"],
                    ",".join(job["batch_ids"]),
                )
            )
            future = executor.submit(
                process_batch_job,
                job,
                args.endpoint,
                api_key,
                args.model,
                args.temperature,
                args.timeout,
                args.max_retries,
                args.max_tokens,
                enable_thinking,
                args.proxy_mode,
            )
            future_to_job[future] = job
            if args.sleep_seconds > 0 and len(future_to_job) < total_batches:
                time.sleep(args.sleep_seconds)

        for future in as_completed(future_to_job):
            job = future_to_job[future]
            batch_indices = job["batch_indices"]
            try:
                result = future.result()
                metrics_list = result["metrics_list"]
                for idx, metrics, payload_item in zip(
                    batch_indices, metrics_list, job["batch_payload"]
                ):
                    for field_name in METRIC_FIELDS:
                        items[idx][field_name] = metrics[field_name]
                    items[idx]["prediction_parse_mode"] = payload_item[
                        "prediction_parse_mode"
                    ]
                    items[idx]["prediction_answers_recovered"] = payload_item[
                        "prediction_answers"
                    ]
                    items[idx]["semantic_match_generation_failed"] = False
                    items[idx].pop("semantic_match_error", None)
                    items[idx].pop("semantic_match", None)

                save_items(input_path, items)

                processed += len(batch_indices)
                print(
                    "Completed batch %d/%d. Saved %d/%d items. Metrics=%s"
                    % (
                        job["batch_number"],
                        job["total_batches"],
                        processed,
                        total_pending,
                        json.dumps(metrics_list, ensure_ascii=False),
                    )
                )
            except Exception as exc:
                error_message = shorten_error_message(exc)
                for idx in batch_indices:
                    _, prediction_parse_mode = parse_prediction_answers_and_mode(
                        items[idx].get("prediction_raw")
                    )
                    items[idx]["prediction_parse_mode"] = prediction_parse_mode
                    items[idx]["semantic_match_generation_failed"] = True
                    items[idx]["semantic_match_error"] = error_message
                    items[idx].pop("semantic_match", None)
                    for field_name in METRIC_FIELDS:
                        items[idx].pop(field_name, None)

                save_items(input_path, items)

                processed += len(batch_indices)
                print(
                    "Completed batch %d/%d with failure. Saved %d/%d items. Marked %d items as generation_failed. Error=%s"
                    % (
                        job["batch_number"],
                        job["total_batches"],
                        processed,
                        total_pending,
                        len(batch_indices),
                        error_message,
                    ),
                    file=sys.stderr,
                )

    successful_items = [
        item
        for item in items
        if not item.get("semantic_match_generation_failed")
        and all(field_name in item for field_name in METRIC_FIELDS)
    ]
    success_count = len(successful_items)
    failed_count = sum(
        1 for item in items if item.get("semantic_match_generation_failed") is True
    )
    avg_precision = (
        sum(float(item["precision"]) for item in successful_items) / success_count
        if success_count
        else None
    )
    avg_recall = (
        sum(float(item["recall"]) for item in successful_items) / success_count
        if success_count
        else None
    )
    avg_f1 = (
        sum(float(item["f1"]) for item in successful_items) / success_count
        if success_count
        else None
    )
    avg_hit_at_1 = (
        sum(float(item["hit_at_1"]) for item in successful_items) / success_count
        if success_count
        else None
    )
    avg_true_positive = (
        sum(float(item["true_positive"]) for item in successful_items) / success_count
        if success_count
        else None
    )
    print(
        "Done. success_count: %d, generation_failed: %d, total_truncated_prediction_raw: %d, avg_precision: %s, avg_recall: %s, avg_f1: %s, avg_hit_at_1: %s, avg_true_positive: %s"
        % (
            success_count,
            failed_count,
            total_truncated_prediction_raw,
            "None" if avg_precision is None else "%.6f" % avg_precision,
            "None" if avg_recall is None else "%.6f" % avg_recall,
            "None" if avg_f1 is None else "%.6f" % avg_f1,
            "None" if avg_hit_at_1 is None else "%.6f" % avg_hit_at_1,
            "None" if avg_true_positive is None else "%.6f" % avg_true_positive,
        )
    )


if __name__ == "__main__":
    main()
