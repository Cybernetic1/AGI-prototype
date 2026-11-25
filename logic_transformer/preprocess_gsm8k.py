"""
Loader hook for GSM8K-like files. Expect JSONL with {id, question, answer, cot}.
This script also provides a tiny proposition extractor for the LT bootstrap.
"""
import json
import random
import re
from pathlib import Path


ADD_HINTS = {
    "add", "plus", "more", "total", "altogether", "buy", "bought", "gets",
    "got", "receive", "receives", "received", "join", "gave", "gives",
}
SUB_HINTS = {
    "left", "remain", "remains", "gave away", "spent", "loss", "lose",
    "lost", "take away", "difference", "fewer", "less",
}
CONTENT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "did",
    "for", "from", "has", "have", "how", "if", "in", "is", "it", "many",
    "much", "of", "on", "or", "the", "to", "what", "when", "where", "who",
    "why", "with", "would", "will", "can", "could", "should", "there",
    "their", "they", "them", "then", "than", "into", "over", "under",
}

MATH_HINTS = {
    "factor", "factors", "product", "multiply", "multiplied", "times",
    "double", "twice", "divide", "divided", "quotient", "ratio", "fraction",
    "perimeter", "area", "volume", "radius", "diameter", "square", "squared",
    "root", "roots", "solve", "solving", "equation", "equations", "denominator",
    "numerator", "asymptote", "asymptotes", "polynomial", "graph", "system",
}


def _detect_operation(text: str) -> str:
    lower = text.lower()
    if any(h in lower for h in SUB_HINTS):
        return "sub"
    if any(h in lower for h in ADD_HINTS):
        return "add"
    return "unknown"


def extract_propositions(text: str, source: str = "text"):
    lower = str(text).lower()
    numbers = re.findall(r"-?\d+(?:\.\d+)?", lower)
    words = [
        word
        for word in re.findall(r"[a-z]+", lower)
        if word not in CONTENT_STOPWORDS and len(word) > 2
    ]
    formulas = []
    for raw_formula in re.findall(r"\$([^$]+)\$", str(text)):
        normalized = re.sub(r"\s+", " ", raw_formula.strip())
        if normalized:
            formulas.append(normalized)
    for line in str(text).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "=" in stripped or "\\boxed" in stripped:
            normalized = re.sub(r"\s+", " ", stripped)
            formulas.append(normalized)
    operation = _detect_operation(lower)
    props = []

    for i, num in enumerate(numbers[:6]):
        props.append({
            "pred": f"{source}:number",
            "args": [f"n{i}", num],
            "numeric_value": float(num),
        })

    for i, word in enumerate(words[:6]):
        props.append({
            "pred": f"{source}:word",
            "args": [f"w{i}", word],
            "numeric_value": None,
        })

    for i, formula in enumerate(formulas[:4]):
        props.append({
            "pred": f"{source}:formula",
            "args": [f"f{i}", formula[:96]],
            "numeric_value": None,
        })

    for i, hint in enumerate(sorted(MATH_HINTS & set(words))[:4]):
        props.append({
            "pred": f"{source}:mathhint",
            "args": [f"h{i}", hint],
            "numeric_value": None,
        })

    if operation != "unknown":
        props.append({
            "pred": f"{source}:operation",
            "args": [operation],
        })

    if len(numbers) >= 2 and operation in {"add", "sub"}:
        props.append({
            "pred": f"{source}:candidate",
            "args": [operation, numbers[0], numbers[1]],
            "numeric_value": None,
        })

    if not props:
        props.append({
            "pred": f"{source}:text",
            "args": [lower[:32] or "<empty>"],
            "numeric_value": None,
        })

    return props


def build_example_props(question: str, cot: str = "", meta=None):
    props = extract_propositions(question, source="question") + extract_propositions(cot, source="cot")
    meta = meta or {}
    subject = str(meta.get("subject", "")).strip().lower()
    level = str(meta.get("level", "")).strip().lower()
    problem_type = str(meta.get("type", "")).strip().lower()
    if subject:
        props.append({"pred": "meta:subject", "args": [subject], "numeric_value": None})
    if level:
        props.append({"pred": "meta:level", "args": [level], "numeric_value": None})
    if problem_type:
        props.append({"pred": "meta:type", "args": [problem_type], "numeric_value": None})
    return props


def load_tinystories_file(path: str | Path) -> list[str]:
    stories = Path(path).read_text(encoding="utf-8").split("---STORY_SEPARATOR---")
    return [story.strip() for story in stories if story.strip()]


def split_story_sentences(story: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", str(story or "").strip())
    return [
        piece.strip()
        for piece in re.split(r"(?<=[.!?])\s+", normalized)
        if len(piece.strip()) >= 8
    ]


def build_continuation_pairs(
    stories: list[str],
    *,
    validation_ratio: float = 0.1,
    seed: int = 0,
    context_sentences: int = 2,
) -> list[dict]:
    if not stories:
        return []
    story_ids = list(range(len(stories)))
    rng = random.Random(seed)
    rng.shuffle(story_ids)
    validation_count = max(1, int(len(story_ids) * validation_ratio))
    split_ids = {
        "validation": set(story_ids[:validation_count]),
        "train": set(story_ids[validation_count:]),
    }
    sentence_rows: dict[str, list[tuple[int, list[str]]]] = {"train": [], "validation": []}
    for split, ids in split_ids.items():
        for story_id in sorted(ids):
            sentences = split_story_sentences(stories[story_id])
            if len(sentences) >= 2:
                sentence_rows[split].append((story_id, sentences))

    pairs: list[dict] = []
    for split in ("train", "validation"):
        candidates = [
            sentence
            for _, sentences in sentence_rows[split]
            for sentence in sentences[1:]
        ]
        for story_id, sentences in sentence_rows[split]:
            for target_index in range(1, len(sentences)):
                start = max(0, target_index - context_sentences)
                context = " ".join(sentences[start:target_index])
                true_next = sentences[target_index]
                negative = true_next
                local_negatives = [
                    sentence
                    for index, sentence in enumerate(sentences)
                    if index != target_index and index > 0
                ]
                negative_pool = local_negatives if local_negatives and rng.random() < 0.8 else candidates
                for _ in range(24):
                    proposal = rng.choice(negative_pool)
                    if proposal != true_next:
                        negative = proposal
                        break
                true_is_a = rng.random() < 0.5
                pairs.append(
                    {
                        "id": f"tiny-cont-{split}-{story_id}-{target_index}",
                        "task": "semantic_continuation",
                        "split": split,
                        "story_id": story_id,
                        "target_sentence_index": target_index,
                        "context": context,
                        "candidate_a": true_next if true_is_a else negative,
                        "candidate_b": negative if true_is_a else true_next,
                        "label": 0 if true_is_a else 1,
                    }
                )
    return pairs


def prepare_tinystories_continuation_file(
    source_path: str | Path,
    output_path: str | Path,
    *,
    max_stories: int = 10_000,
    validation_ratio: float = 0.1,
    seed: int = 0,
) -> list[dict]:
    stories = load_tinystories_file(source_path)
    if max_stories > 0:
        stories = stories[:max_stories]
    pairs = build_continuation_pairs(
        stories,
        validation_ratio=validation_ratio,
        seed=seed,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    return pairs


def extract_gsm8k_arithmetic(cot: str):
    """
    Parse the final annotated arithmetic step from GSM8K-style reasoning text.
    Returns a dict with op, left, right, result, or None if parsing fails.
    """
    text = str(cot)
    annotations = re.findall(r"<<([^<>]+)>>", text)
    if not annotations:
        return None

    expr = annotations[-1].strip()
    if "=" in expr:
        expr, result = expr.split("=", 1)
    else:
        result = None

    expr = expr.replace(" ", "")
    match = re.match(r"^(-?\d+(?:\.\d+)?)([+\-*/x])(-?\d+(?:\.\d+)?)$", expr)
    if not match:
        return None

    left = match.group(1)
    op = match.group(2)
    right = match.group(3)

    op_map = {"+": "add", "-": "sub", "*": "mul", "x": "mul", "/": "div"}
    return {
        "op": op_map.get(op, "unknown"),
        "left": left,
        "right": right,
        "result": result.strip() if result is not None else None,
        "expr": expr,
    }


def extract_gsm8k_steps(cot: str):
    """
    Parse all annotated arithmetic steps from GSM8K-style reasoning text.
    Returns a list of dicts in order of appearance.
    """
    text = str(cot)
    annotations = re.findall(r"<<([^<>]+)>>", text)
    steps = []
    for ann in annotations:
        expr = ann.strip()
        if "=" in expr:
            expr, result = expr.split("=", 1)
        else:
            result = None
        expr = expr.replace(" ", "")
        match = re.match(r"^(-?\d+(?:\.\d+)?)([+\-*/x])(-?\d+(?:\.\d+)?)$", expr)
        if not match:
            continue
        left = match.group(1)
        op = match.group(2)
        right = match.group(3)
        op_map = {"+": "add", "-": "sub", "*": "mul", "x": "mul", "/": "div"}
        steps.append({
            "op": op_map.get(op, "unknown"),
            "left": left,
            "right": right,
            "result": result.strip() if result is not None else None,
            "expr": expr,
        })
    return steps


def extract_math_steps(cot: str):
    """
    Parse generic MATH-style derivation steps from worked solutions.
    The parser prefers explicit boxed answers and equation lines, then
    keeps numeric results in order of appearance.
    """
    text = str(cot)
    steps = []

    def add_step(expr: str, result):
        result_text = result.strip() if result is not None else None
        if result_text is None:
            return
        if not re.search(r"-?\d+(?:\.\d+)?", result_text):
            return
        if steps and steps[-1].get("result") == result_text:
            return
        steps.append({
            "op": "unknown",
            "left": None,
            "right": None,
            "result": result_text,
            "expr": expr.strip(),
        })

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        boxed = re.findall(r"\\boxed\{([^{}]+)\}", stripped)
        if boxed:
            add_step(stripped, boxed[-1])
            continue

        if "=" in stripped or "\\approx" in stripped or "\\Rightarrow" in stripped:
            numbers = re.findall(r"-?\d+(?:\.\d+)?", stripped)
            if numbers:
                add_step(stripped, numbers[-1])

    if not steps:
        boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
        if boxed and re.search(r"-?\d+(?:\.\d+)?", boxed[-1]):
            add_step(text, boxed[-1])

    return steps

def load_jsonl(path):
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples

if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'synthetic_gsm8k_demo.jsonl'
    ex = load_jsonl(path)
    print('Loaded', len(ex))
    print(ex[0])
    print(build_example_props(ex[0]["question"], ex[0].get("cot", "")))
