#!/usr/bin/env bash

set -Eeuo pipefail

DATA_DIR="$PWD/data"
DATASETS="all"
FORCE=0
PYTHON_BIN="${PYTHON:-python3}"

usage() {
  cat <<'EOF'
Usage: ./download_all_in_one.sh [options]

Download/prepare the default all-in-one HiCache benchmark datasets under one
data directory. Run this from extern/sglang/benchmark/hicache unless you pass
--data-dir explicitly.

Options:
  --data-dir DIR       Dataset root. Defaults to ./data in the current dir.
  --datasets LIST     Comma-separated: all,loogle,sharegpt,narrativeqa,reviewmt.
                       Defaults to all.
  --force             Re-download/re-convert even if outputs already exist.
  -h, --help          Show this help.

Expected outputs:
  data/LooGLE/data/longdep_qa.jsonl
  data/ShareGPT_V3_unfiltered_cleaned_split.json
  data/narrativeqa_long_context.json
  data/reviewmt_sharegpt.json
EOF
}

log() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

warn() {
  printf '[warn] %s\n' "$*" >&2
}

die() {
  printf '[error] %s\n' "$*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir)
      [[ $# -ge 2 ]] || die "--data-dir requires a value"
      DATA_DIR="$2"
      shift 2
      ;;
    --datasets)
      [[ $# -ge 2 ]] || die "--datasets requires a value"
      DATASETS="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

DATA_DIR="$(mkdir -p "$DATA_DIR" && cd "$DATA_DIR" && pwd)"
need_cmd curl
need_cmd "$PYTHON_BIN"

selected() {
  local name="$1"
  [[ "$DATASETS" == "all" ]] && return 0
  [[ ",$DATASETS," == *",$name,"* ]]
}

valid_json() {
  local path="$1"
  [[ -s "$path" ]] || return 1
  "$PYTHON_BIN" - "$path" <<'PY' >/dev/null 2>&1
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as fin:
    json.load(fin)
PY
}

valid_jsonl() {
  local path="$1"
  [[ -s "$path" ]] || return 1
  "$PYTHON_BIN" - "$path" <<'PY' >/dev/null 2>&1
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as fin:
    for idx, line in enumerate(fin):
        line = line.strip()
        if line:
            json.loads(line)
        if idx > 10:
            break
PY
}

backup_invalid() {
  local path="$1"
  [[ -e "$path" ]] || return 0
  local backup="${path}.invalid.$(date '+%Y%m%d-%H%M%S')"
  warn "$path exists but failed validation; moving it to $backup"
  mv "$path" "$backup"
}

download_url() {
  local url="$1"
  local dest="$2"
  mkdir -p "$(dirname "$dest")"
  local part="${dest}.part"
  log "download $(basename "$dest")"
  curl -L --fail --retry 3 --retry-delay 2 --continue-at - -o "$part" "$url"
  mv "$part" "$dest"
}

download_loogle() {
  local repo="$DATA_DIR/LooGLE"
  local target="$repo/data/longdep_qa.jsonl"
  if [[ "$FORCE" -eq 0 ]] && valid_jsonl "$target"; then
    log "skip LooGLE: $target already exists"
    return 0
  fi

  need_cmd git
  need_cmd unzip

  if [[ -e "$repo" && ! -d "$repo/.git" && "$FORCE" -eq 0 ]]; then
    warn "$repo exists but is not a git checkout and $target is missing/invalid"
    warn "move it aside or rerun with --force"
    return 1
  fi

  if [[ -e "$repo" && ! -d "$repo/.git" && "$FORCE" -eq 1 ]]; then
    mv "$repo" "${repo}.invalid.$(date '+%Y%m%d-%H%M%S')"
  fi

  if [[ ! -d "$repo/.git" ]]; then
    local tmp="$DATA_DIR/.LooGLE.partial"
    if [[ -e "$tmp" ]]; then
      mv "$tmp" "${tmp}.old.$(date '+%Y%m%d-%H%M%S')"
    fi
    log "clone LooGLE metadata"
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/bigainlco/LooGLE "$tmp"
    mv "$tmp" "$repo"
  fi

  if ! git -C "$repo" lfs install --local >/dev/null 2>&1; then
    warn "git-lfs is not available; install git-lfs before downloading LooGLE data.zip"
    return 1
  fi

  log "fetch LooGLE data.zip via git-lfs"
  if ! git -C "$repo" lfs pull --include "data.zip" --exclude ""; then
    warn "LooGLE git-lfs pull failed. If Hugging Face asks for auth, run:"
    warn "  hf auth login --force --add-to-git-credential"
    return 1
  fi

  log "unzip LooGLE data.zip"
  unzip -o "$repo/data.zip" -d "$repo" >/dev/null

  if valid_jsonl "$target"; then
    log "ready LooGLE: $target"
    return 0
  fi
  warn "LooGLE finished but $target is missing or invalid"
  return 1
}

download_sharegpt() {
  local target="$DATA_DIR/ShareGPT_V3_unfiltered_cleaned_split.json"
  if [[ "$FORCE" -eq 0 ]] && valid_json "$target"; then
    log "skip ShareGPT: $target already exists"
    return 0
  fi
  [[ -e "$target" ]] && backup_invalid "$target"
  download_url \
    "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json" \
    "$target"
  valid_json "$target" || return 1
  log "ready ShareGPT: $target"
}

convert_narrativeqa() {
  local target="$DATA_DIR/narrativeqa_long_context.json"
  if [[ "$FORCE" -eq 0 ]] && valid_json "$target"; then
    log "skip NarrativeQA: $target already exists"
    return 0
  fi
  [[ -e "$target" ]] && backup_invalid "$target"

  log "prepare NarrativeQA with Hugging Face datasets"
  "$PYTHON_BIN" - "$target" <<'PY'
import json
import sys

target = sys.argv[1]

try:
    from datasets import load_dataset
except Exception as exc:
    print(
        "Python package 'datasets' is required for NarrativeQA. "
        "Install it in this env, e.g. `uv pip install datasets`, then rerun.",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc


def textify(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        pieces = [textify(item) for item in value]
        return "\n".join(piece for piece in pieces if piece).strip()
    if isinstance(value, dict):
        for key in ("text", "value", "content", "summary", "document"):
            piece = textify(value.get(key))
            if piece:
                return piece
        pieces = [textify(item) for item in value.values()]
        return "\n".join(piece for piece in pieces if piece).strip()
    return str(value).strip()


def doc_text(row):
    doc = row.get("document")
    if isinstance(doc, dict):
        for key in ("text", "summary", "content", "document"):
            piece = textify(doc.get(key))
            if piece:
                return piece
    for key in ("context", "story", "document", "text", "summary"):
        piece = textify(row.get(key))
        if piece:
            return piece
    return ""


def question_text(row):
    for key in ("question", "query"):
        piece = textify(row.get(key))
        if piece:
            return piece
    return ""


def answer_text(row):
    for key in ("answers", "answer", "reference_answer"):
        value = row.get(key)
        if isinstance(value, list) and value:
            piece = textify(value[0])
        else:
            piece = textify(value)
        if piece:
            return piece
    return ""


last_error = None
dataset = None
for name in ("deepmind/narrativeqa", "narrativeqa"):
    for split in ("test", "validation", "train"):
        try:
            dataset = load_dataset(name, split=split, trust_remote_code=True)
            print(f"loaded {name} split={split}", file=sys.stderr)
            break
        except Exception as exc:
            last_error = exc
    if dataset is not None:
        break

if dataset is None:
    print(f"failed to load NarrativeQA: {last_error}", file=sys.stderr)
    raise SystemExit(1)

contexts = []
context_index = {}
queries = []

for row in dataset:
    context = doc_text(row)
    question = question_text(row)
    answer = answer_text(row)
    if not context or not question or not answer:
        continue
    key = row.get("document", {}).get("id") if isinstance(row.get("document"), dict) else ""
    key = key or context[:200]
    if key not in context_index:
        context_index[key] = len(contexts)
        contexts.append(context)
    queries.append(
        {
            "context": context_index[key],
            "question": "\n\nQuestion: " + question,
            "reference_answer": answer,
        }
    )

if not contexts or not queries:
    print("NarrativeQA conversion produced no examples", file=sys.stderr)
    raise SystemExit(1)

with open(target, "w", encoding="utf-8") as fout:
    json.dump({"contexts": contexts, "queries": queries}, fout, ensure_ascii=False)

print(f"wrote {len(contexts)} contexts and {len(queries)} queries to {target}", file=sys.stderr)
PY
  valid_json "$target" || return 1
  log "ready NarrativeQA: $target"
}

reviewmt_asset_url() {
  local api_json="$DATA_DIR/.reviewmt_releases.json"
  if [[ -n "${REVIEWMT_URL:-}" ]]; then
    printf '%s\n' "$REVIEWMT_URL"
    return 0
  fi
  if curl -L --fail --retry 3 --retry-delay 2 \
    -o "${api_json}.part" \
    "https://api.github.com/repos/chengtan9907/ReviewMT/releases" >/dev/null 2>&1; then
    mv "${api_json}.part" "$api_json"
    "$PYTHON_BIN" - "$api_json" <<'PY' || true
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as fin:
    releases = json.load(fin)

priority = (
    "reviewmt_test.json",
    "iclr_test_data.json",
    "nature_test_data.json",
)

assets = []
for release in releases:
    for asset in release.get("assets", []):
        name = asset.get("name", "")
        url = asset.get("browser_download_url", "")
        if name and url and name.endswith((".json", ".jsonl", ".zip")):
            assets.append((name, url))

for wanted in priority:
    for name, url in assets:
        if name == wanted:
            print(url)
            raise SystemExit(0)

for name, url in assets:
    if "test" in name.lower() and name.endswith((".json", ".jsonl")):
        print(url)
        raise SystemExit(0)
PY
    return 0
  fi

  printf '%s\n' \
    "https://github.com/chengtan9907/ReviewMT/releases/download/ReviewMT_plus/reviewmt_test.json"
}

convert_reviewmt() {
  local target="$DATA_DIR/reviewmt_sharegpt.json"
  local raw="$DATA_DIR/reviewmt_raw.json"
  if [[ "$FORCE" -eq 0 ]] && valid_json "$target"; then
    log "skip ReviewMT: $target already exists"
    return 0
  fi
  [[ -e "$target" ]] && backup_invalid "$target"

  if [[ "$FORCE" -eq 1 || ! -s "$raw" ]]; then
    local url
    url="$(reviewmt_asset_url)"
    if [[ -z "$url" ]]; then
      warn "could not discover a ReviewMT release asset"
      return 1
    fi
    download_url "$url" "$raw"
  fi

  log "convert ReviewMT to ShareGPT-style JSON"
  "$PYTHON_BIN" - "$raw" "$target" <<'PY'
import json
import sys

raw_path, target = sys.argv[1], sys.argv[2]

with open(raw_path, "r", encoding="utf-8") as fin:
    data = json.load(fin)

if isinstance(data, dict):
    for key in ("data", "examples", "samples", "conversations", "dialogs"):
        if isinstance(data.get(key), list):
            data = data[key]
            break

if not isinstance(data, list):
    print("ReviewMT raw file is not a list-like JSON dataset", file=sys.stderr)
    raise SystemExit(1)


def textify(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        pieces = [textify(item) for item in value]
        return "\n".join(piece for piece in pieces if piece).strip()
    if isinstance(value, dict):
        for key in ("value", "content", "text", "paper", "review", "decision", "response"):
            piece = textify(value.get(key))
            if piece:
                return piece
        pieces = [textify(item) for item in value.values()]
        return "\n".join(piece for piece in pieces if piece).strip()
    return str(value).strip()


def role_to_from(role, idx):
    role = str(role or "").lower()
    if role in {"human", "user", "author", "reviewer"}:
        return "human"
    if role in {"gpt", "assistant", "system", "chair", "decision"}:
        return "gpt"
    return "human" if idx % 2 == 0 else "gpt"


def normalize_turns(row):
    if not isinstance(row, dict):
        return []
    if {"instruction", "input", "output"} & set(row):
        paper = textify(row.get("input"))
        instruction = textify(row.get("instruction"))
        output = textify(row.get("output"))
        history = row.get("history") or []
        out = []

        for idx, pair in enumerate(history):
            if not isinstance(pair, list) or len(pair) < 2:
                continue
            prompt = textify(pair[0])
            answer = textify(pair[1])
            if not prompt or not answer:
                continue
            if idx == 0 and paper:
                prompt = prompt + "\n\nPaper:\n" + paper
            out.append({"from": "human", "value": prompt})
            out.append({"from": "gpt", "value": answer})

        final_prompt_parts = [part for part in (instruction, paper) if part]
        final_prompt = "\n\n".join(final_prompt_parts)
        if final_prompt and output:
            out.append({"from": "human", "value": final_prompt})
            out.append({"from": "gpt", "value": output})
        return out

    for key in ("conversations", "conversation", "messages", "dialogue", "dialog", "turns"):
        turns = row.get(key)
        if isinstance(turns, list):
            out = []
            for idx, turn in enumerate(turns):
                if isinstance(turn, str):
                    value = turn.strip()
                    role = ""
                elif isinstance(turn, dict):
                    value = textify(turn)
                    role = (
                        turn.get("from")
                        or turn.get("role")
                        or turn.get("speaker")
                        or turn.get("name")
                    )
                else:
                    value = textify(turn)
                    role = ""
                if value:
                    out.append({"from": role_to_from(role, idx), "value": value})
            return out

    ordered_keys = (
        "paper",
        "abstract",
        "submission",
        "review",
        "reviews",
        "rebuttal",
        "response",
        "meta_review",
        "decision",
    )
    out = []
    for key in ordered_keys:
        value = textify(row.get(key))
        if value:
            out.append({"from": role_to_from(key, len(out)), "value": value})
    return out


converted = []
for idx, row in enumerate(data):
    turns = normalize_turns(row)
    if len(turns) < 2:
        continue
    if len(turns) % 2 == 1:
        turns = turns[:-1]
    if not turns:
        continue
    turns[0]["from"] = "human"
    for turn_idx, turn in enumerate(turns):
        turn["from"] = "human" if turn_idx % 2 == 0 else "gpt"
    converted.append({"id": str(row.get("id", idx)) if isinstance(row, dict) else str(idx), "conversations": turns})

if not converted:
    print("ReviewMT conversion produced no ShareGPT-style conversations", file=sys.stderr)
    raise SystemExit(1)

with open(target, "w", encoding="utf-8") as fout:
    json.dump(converted, fout, ensure_ascii=False)

print(f"wrote {len(converted)} conversations to {target}", file=sys.stderr)
PY
  valid_json "$target" || return 1
  log "ready ReviewMT: $target"
}

FAILURES=0

log "data dir: $DATA_DIR"

if selected loogle; then
  download_loogle || FAILURES=$((FAILURES + 1))
fi
if selected sharegpt; then
  download_sharegpt || FAILURES=$((FAILURES + 1))
fi
if selected narrativeqa; then
  convert_narrativeqa || FAILURES=$((FAILURES + 1))
fi
if selected reviewmt; then
  convert_reviewmt || FAILURES=$((FAILURES + 1))
fi

cat <<EOF

Dataset layout:
  LooGLE:      $DATA_DIR/LooGLE/data/longdep_qa.jsonl
  ShareGPT:    $DATA_DIR/ShareGPT_V3_unfiltered_cleaned_split.json
  NarrativeQA: $DATA_DIR/narrativeqa_long_context.json
  ReviewMT:    $DATA_DIR/reviewmt_sharegpt.json

Use with:
  python3 bench_all_in_one.py --data-dir "$DATA_DIR" --workloads strata ...
EOF

if [[ "$FAILURES" -ne 0 ]]; then
  warn "$FAILURES dataset step(s) failed. Fix the messages above and rerun; completed outputs will be skipped."
  exit 1
fi

log "all selected datasets are ready"
