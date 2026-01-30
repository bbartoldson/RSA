"""
Evaluate API-based models on FrontierScience benchmark using RSA aggregation.
Adapted from eval_loop.py with minimal changes for API-based inference.

Usage:
    python eval_frontierscience_v3.py \
        --model claude-3-5-haiku-20241022 \
        --domain bio \
        --n-samples 10 \
        --loops 3 \
        --k 4 \
        --population 8
"""

from typing import List, Dict, Any, Optional, Callable
import argparse
import json
import os
import re
import random
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import pickle

# Optional rich library for verbose output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.live import Live
    from rich.text import Text
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# --------------------- verbose logging ---------------------

class VerboseLogger:
    """Rich console logger for real-time evaluation feedback."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled and RICH_AVAILABLE
        if self.enabled:
            self.console = Console()
        self.running_scores: List[float] = []

    def warn_no_rich(self):
        if not RICH_AVAILABLE:
            print("[Warning] --verbose requires 'rich' library. Install with: pip install rich")

    def loop_header(self, loop_idx: int, n_problems: int, population: int):
        if not self.enabled:
            return
        self.running_scores = []
        self.console.rule(f"[bold blue]Loop {loop_idx}[/bold blue]", style="blue")
        self.console.print(f"  Problems: {n_problems} | Population: {population}")

    def show_problem(self, problem_idx: int, problem: Dict, extracted_answers: List[str], scores: List[float]):
        """Show extracted answers and scores for a single problem."""
        if not self.enabled:
            return

        self.running_scores.extend(scores)
        running_mean = sum(self.running_scores) / len(self.running_scores)
        n_correct = sum(1 for s in scores if s >= 1.0)

        # Problem header
        subject = problem.get('subject', 'unknown')
        task_id = problem.get('task_group_id', 'unknown')[:20]
        self.console.print(f"\n[dim]Problem {problem_idx + 1}[/dim] [cyan]{subject}[/cyan] ({task_id}...)")

        # Show ground truth (truncated)
        gt = problem.get('gt', '')[:100]
        self.console.print(f"  [dim]GT:[/dim] {gt}{'...' if len(problem.get('gt', '')) > 100 else ''}")

        # Create table of extracted answers
        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
        table.add_column("#", style="dim", width=3)
        table.add_column("Score", width=6)
        table.add_column("Extracted Answer", overflow="ellipsis", max_width=70)

        # Show up to 4 answers (to keep output manageable)
        for i, (ans, score) in enumerate(zip(extracted_answers[:4], scores[:4])):
            score_str = "[green]1.0[/green]" if score >= 1.0 else "[red]0.0[/red]"
            ans_display = (ans or "[none]")[:70]
            table.add_row(str(i + 1), score_str, ans_display)

        if len(extracted_answers) > 4:
            table.add_row("...", "", f"(+{len(extracted_answers) - 4} more)")

        self.console.print(table)

        # Running stats
        self.console.print(
            f"  [bold]This problem:[/bold] {n_correct}/{len(scores)} correct | "
            f"[bold]Running mean:[/bold] {running_mean:.3f}"
        )

    def loop_summary(self, loop_idx: int, metrics: Dict):
        """Show summary table at end of loop."""
        if not self.enabled:
            return

        table = Table(title=f"Loop {loop_idx} Summary", show_header=True, header_style="bold green")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        table.add_row("Mean Accuracy", f"{metrics.get('mean_acc_k', 0):.4f}")
        table.add_row("Pass@k", f"{metrics.get('mean_pass_at_k', 0):.4f}")
        table.add_row("Majority Vote", f"{metrics.get('mean_majority_acc', 0):.4f}")
        table.add_row("Avg Response Len", f"{metrics.get('mean_length', 0):.0f} chars")

        self.console.print()
        self.console.print(table)
        self.console.print()


# --------------------- API client setup ---------------------

def load_api_keys(key_file: str = "api_key.txt") -> Dict[str, str]:
    """Load API keys from file. Format: KEY_NAME=value (one per line)."""
    keys = {}
    path = Path(key_file)
    if path.exists():
        for line in path.read_text().strip().split("\n"):
            line = line.strip()
            if line and "=" in line and not line.startswith("#"):
                name, value = line.split("=", 1)
                keys[name.strip()] = value.strip()
    return keys


def get_api_client(model: str, openai_api_key: Optional[str] = None, anthropic_api_key: Optional[str] = None):
    """Return appropriate client based on model name."""
    if model.startswith("claude"):
        import anthropic
        return anthropic.Anthropic(api_key=anthropic_api_key), "anthropic"
    else:
        import openai
        return openai.OpenAI(api_key=openai_api_key), "openai"


def get_grader_client(openai_api_key: Optional[str] = None):
    """Return OpenAI client for grading (always uses OpenAI for GPT 5.2)."""
    import openai
    return openai.OpenAI(api_key=openai_api_key)


def call_api(client, client_type: str, model: str, prompt: str, max_tokens: int, temperature: float, retries: int = 3) -> str:
    """Call API and return response text with retry logic."""
    for attempt in range(retries):
        try:
            if client_type == "anthropic":
                response = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            else:
                response = client.chat.completions.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"API error: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


# --------------------- helpers ---------------------

def load_latest_loop_file(dir_path):
    dir_path = Path(dir_path)
    pattern = re.compile(r"loop_(\d+)\.pkl$")

    max_i = -1
    latest_file = None

    for file in dir_path.iterdir():
        if file.is_file():
            match = pattern.match(file.name)
            if match:
                i = int(match.group(1))
                if i > max_i:
                    max_i = i
                    latest_file = file

    if latest_file is None:
        raise FileNotFoundError("No loop_{i}.pkl files found in directory")

    with open(latest_file, "rb") as f:
        data = pickle.load(f)

    return data, max_i, latest_file


def _append_metrics_to_json(path: str, entry: dict):
    """Append `entry` to a JSON array file at `path` (create if needed)."""
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = [data]
        else:
            data = []
    except Exception:
        data = []
    data.append(entry)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# --------------------- answer extraction ---------------------

def extract_frontierscience_answer(response: str) -> Optional[str]:
    """Extract answer from 'FINAL ANSWER: ...' pattern.

    The grader should ONLY see this extracted answer, not the full reasoning chain.
    """
    # Try "FINAL ANSWER:" pattern (case insensitive)
    match = re.search(r'FINAL\s*ANSWER\s*[:\s]\s*(.+)', response, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: try "Answer:" at end of response
    match = re.search(r'\bAnswer\s*[:\s]\s*(.+?)$', response, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    return None


# --------------------- prompt building ---------------------

def aggregate_prompt(question: str, candidate_answers: List[str], task: str, loops_left: int) -> str:
    if task == 'frontierscience':
        problem_kind = 'scientific problem'
        format_hint = 'FINAL ANSWER: <your answer>'
    else:
        raise ValueError(f"Unknown task: {task}")

    parts = []
    if len(candidate_answers) == 1:
        parts.append(
            f"You are given a {problem_kind} and a candidate solution. "
            "The candidate may be incomplete or contain errors. "
            "Refine this trajectory and produce an improved, higher-quality solution. "
            "If it is entirely wrong, attempt a new strategy. "
            f"End with the final result as {format_hint}.\n"
        )
    else:
        parts.append(
            f"You are given a {problem_kind} and several candidate solutions. "
            "Some candidates may be incorrect or contain errors. "
            "Aggregate the useful ideas and produce a single, high-quality solution. "
            "Reason carefully; if candidates disagree, choose the correct path. If all are incorrect, then attempt a different strategy. "
            f"End with the final result as {format_hint}.\n"
        )

    parts.append("Problem:\n")
    parts.append(question.strip() + "\n")

    if len(candidate_answers) == 1:
        parts.append("Candidate solution (may contain mistakes):\n")
        ans_str = (candidate_answers[0] or "").strip()
        parts.append(f"---- Candidate ----\n{ans_str}\n")
        parts.append(
            f"Now refine the candidate into an improved solution. Provide clear reasoning and end with {format_hint}."
        )
    else:
        parts.append("Candidate solutions (may contain mistakes):\n")
        for i, ans in enumerate(candidate_answers, 1):
            ans_str = (ans or "").strip()
            parts.append(f"---- Solution {i} ----\n{ans_str}\n")
        parts.append(
            f"Now write a single improved solution. Provide clear reasoning and end with {format_hint}."
        )

    return "\n".join(parts)


def build_base_prompt(problem: str) -> str:
    """Build the initial prompt for a FrontierScience problem."""
    return (
        "You are an expert scientist. Solve the following problem step by step, "
        "showing your reasoning clearly. At the end, provide your final answer "
        "in the format: FINAL ANSWER: <your answer>\n\n"
        f"Problem:\n{problem}"
    )


def build_prompt(question: str, candidate_answers: Optional[List[str]], task: str, loops_left: int) -> str:
    if candidate_answers is not None:
        prompt = aggregate_prompt(question, candidate_answers, task, loops_left)
    else:
        prompt = build_base_prompt(question)
    return prompt


# --------------------- aggregation prompt visualization ---------------------

def truncate_text(text: str, max_chars: int = 100) -> str:
    """Show first and last max_chars of text."""
    if len(text) <= max_chars * 2 + 20:
        return text
    return f"{text[:max_chars]}  ...  {text[-max_chars:]}"


def visualize_aggregation_example(
    question: str,
    candidate_answers: List[str],
    response: str,
    output_dir: str,
    loop_idx: int,
    loops: int,
) -> None:
    """Save full aggregation example and print truncated version to console."""

    # Build the full aggregation prompt
    agg_prompt = aggregate_prompt(question, candidate_answers, "frontierscience", loops-loop_idx-1)

    # Save full version to file
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, f"agg_example_loop_{loop_idx}.txt")
    with open(full_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("AGGREGATION PROMPT EXAMPLE\n")
        f.write("=" * 80 + "\n\n")
        f.write("--- FULL AGGREGATION PROMPT ---\n\n")
        f.write(agg_prompt)
        f.write("\n\n")
        f.write("--- RESPONSE TO AGGREGATION PROMPT ---\n\n")
        f.write(response)
        f.write("\n")

    # Print truncated version to console
    print("\n" + "=" * 60)
    print(f"AGGREGATION EXAMPLE (Loop {loop_idx}) - Full saved to: {full_path}")
    print("=" * 60)

    print("\n[PROBLEM]")
    print(truncate_text(question, 150))

    print(f"\n[CANDIDATE ANSWERS] ({len(candidate_answers)} candidates)")
    for i, cand in enumerate(candidate_answers, 1):
        print(f"  Candidate {i}: {truncate_text(cand, 100)}")

    print("\n[RESPONSE TO AGG PROMPT]")
    print(truncate_text(response, 200))
    print("=" * 60 + "\n")


# --------------------- evaluation (LLM grader) ---------------------

GRADER_MODEL = "gpt-5.2"

GRADER_PROMPT = """
You are grading an attempted answer to a science olympiad problem. You will be given the problem, attempted answer, and reference answer. Evaluate the solution against the provided reference solution, ensuring it is complete and matches the reference solution. Pay close attention to detail and grade it strictly, but fairly.
The reference answer is either a single number or expression in latex formatting, a chemical formula, a compound name, or a phrase referring to a specific name, entity, or method. Mark the attempted answer as correct if it fully matches the reference answer or is otherwise equivalent (e.g., an equivalent algebraic expression, a numerical number within 1 decimal place rounding of the reference answer (e.g., 6.69 ≈ 6.7), an equivalent name for a compound/formula, equivalent when accounting for units, etc.). Mark it as incorrect if it is not equivalent to the reference answer.
***
The problem: {problem}
***
The reference answer: {ground_truth}
***
The attempted answer: {prediction}
***
First, think step-by-step about whether the attempted answer matches the reference answer. If the attempted answer is correct, write ”VERDICT: CORRECT” in the last line of your response, with no other text or formatting. If it is incorrect, write ”VERDICT: INCORRECT”.
"""


def score_with_llm(grader_client, problem: str, prediction: str, ground_truth: str) -> float:
    """Score a prediction using GPT 5.2 as grader. Returns 1.0 or 0.0."""
    prompt = GRADER_PROMPT.format(
        problem=problem,
        ground_truth=ground_truth,
        prediction=prediction
    )

    for attempt in range(3):
        try:
            response = grader_client.responses.create(
                model=GRADER_MODEL,
                input=prompt,
                reasoning={"effort": "high"}
            )
            verdict = response.output_text.strip()
            # Parse "VERDICT: CORRECT" or "VERDICT: INCORRECT"
            if "VERDICT: CORRECT" in verdict.upper():
                return 1.0
            else:
                return 0.0
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"Grading failed: {e}, defaulting to 0.0")
                return 0.0
    return 0.0


def evaluate_k_answers_frontierscience(
    grader_client,
    problem: str,
    k_answers: List[str],
    gt: str,
    max_workers: int = 32
) -> Dict[str, Any]:
    """Evaluate multiple candidates using LLM grader.

    IMPORTANT: Extracts answers first, then grades only the extracted answers.
    The grader should never see the full reasoning chain.
    """
    # Extract answers from full responses
    extracted = [extract_frontierscience_answer(a) or "" for a in k_answers]

    # Grade only the extracted answers (parallel)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(score_with_llm, grader_client, problem, e, gt) for e in extracted]
        scores = [f.result() for f in futures]

    mean_acc = float(sum(scores) / max(1, len(scores)))
    pass_at_k = float(1.0 if any(s >= 1.0 for s in scores) else 0.0)

    # Majority vote (simple: most common extracted answer)
    clusters: List[Dict[str, Any]] = []
    for e, s in zip(extracted, scores):
        placed = False
        for c in clusters:
            if e == c["rep"]:  # Exact string match for simplicity
                c["count"] += 1
                c["score"] = max(c["score"], s)  # Track if any in cluster is correct
                placed = True
                break
        if not placed:
            clusters.append({"rep": e, "count": 1, "score": s})

    if clusters:
        best = max(clusters, key=lambda c: c["count"])
        majority_vote = float(best["score"])
    else:
        majority_vote = 0.0

    return {
        "pred_accuracies": [float(s) for s in scores],
        "extracted_answers": extracted,
        "mean_acc": mean_acc,
        "pass_at_k": pass_at_k,
        "majority_vote_correct": majority_vote
    }


# --------------------- data loading ---------------------

def parse_sample_range(sample_spec: Optional[str]) -> tuple:
    """Parse sample specification like '10', '4:10', or ':10'.

    Returns (start, end) tuple. End is exclusive (Python slice style).
    Examples:
        '10' -> (0, 10)      # first 10 samples
        '4:10' -> (4, 10)    # samples 4-9 (0-indexed)
        '4:' -> (4, None)    # samples 4 onwards
        ':10' -> (0, 10)     # first 10 samples
    """
    if sample_spec is None:
        return (0, None)

    sample_spec = str(sample_spec)
    if ':' in sample_spec:
        parts = sample_spec.split(':')
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else None
        return (start, end)
    else:
        # Just a number means first N samples
        return (0, int(sample_spec))


def load_frontierscience(domain: Optional[str] = None, sample_range: Optional[str] = None, seed: int = 1234) -> List[Dict]:
    """Load FrontierScience dataset with optional filtering.

    Args:
        domain: Filter to specific domain (bio, physics, chem)
        sample_range: Sample specification - int for first N, or 'start:end' for range (0-indexed)
        seed: Random seed (for reproducibility tracking)
    """
    ds = load_dataset("openai/frontierscience", split="test")

    data = []
    for i, row in enumerate(ds):
        data.append({
            "problem": row["problem"],
            "answer": row["answer"],
            "subject": row["subject"],
            "task_group_id": row["task_group_id"],
            "original_idx": i,
        })

    # Filter by domain if specified
    if domain:
        domain_lower = domain.lower()
        domain_map = {"bio": "biology", "chem": "chemistry", "physics": "physics"}
        target = domain_map.get(domain_lower, domain_lower)
        data = [d for d in data if d["subject"].lower() == target]

    # Apply sample range (after domain filter)
    start, end = parse_sample_range(sample_range)
    data = data[start:end]

    return data


# --------------------- main ---------------------

def generate_candidates(A, M, R):
    """Sample R candidates from pool A to create M aggregation batches."""
    if A is None:
        return [None for _ in range(M)]
    return [random.sample(A, R) for _ in range(M)]


def reshape_list(lst, K):
    return [lst[i:i+K] for i in range(0, len(lst), K)]


def run(
    client,
    client_type: str,
    model: str,
    grader_client,
    k: int,
    population: int,
    data: List[Dict],
    task: str,
    max_tokens: int,
    temperature: float,
    max_workers: int = 32,
    logger: Optional[VerboseLogger] = None,
    loop_idx: int = 0,
    output_dir: Optional[str] = None,
    loops: int = 1,
) -> tuple:
    """Run one loop of generation + evaluation."""
    if logger:
        logger.loop_header(loop_idx, len(data), population)

    # Build all prompts
    requests = []
    # Store one example for visualization (first problem, first aggregation)
    agg_example_candidates = None
    agg_example_question = None

    for problem in data:
        prompt_text = problem['orig_prompt']
        candidate_answers = generate_candidates(problem['candidates'], population, k)

        # Capture first aggregation example for visualization
        if agg_example_candidates is None and candidate_answers[0] is not None:
            agg_example_candidates = candidate_answers[0]
            agg_example_question = prompt_text

        for candidates in candidate_answers:
            request = build_prompt(prompt_text, candidates, task, loops-loop_idx-1)
            requests.append((problem, request))

    # Generate responses in parallel
    print(f"Generating {len(requests)} responses...")

    def call_one(req_tuple):
        _, prompt = req_tuple
        return call_api(client, client_type, model, prompt, max_tokens, temperature)

    all_responses = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(call_one, req) for req in requests]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
            pass  # Just for progress bar
        all_responses = [f.result() for f in futures]

    print(f"Sample response:\n{all_responses[0][:500]}...")

    # Calculate response lengths
    response_lengths = [len(r) for r in all_responses]  # Character count since no tokenizer
    median = np.percentile(response_lengths, 50)
    q25 = np.percentile(response_lengths, 25)
    q75 = np.percentile(response_lengths, 75)
    mean_response_length = sum(response_lengths) / max(1, len(response_lengths))

    # Reshape responses back to per-problem
    all_responses = reshape_list(all_responses, population)

    for problem, responses in zip(data, all_responses):
        problem['candidates'] = responses

    # Visualize one aggregation example (if this is an aggregation loop)
    if agg_example_candidates is not None and output_dir:
        # Get the first response for the first problem (corresponds to agg_example_candidates)
        visualize_aggregation_example(
            question=agg_example_question,
            candidate_answers=agg_example_candidates,
            response=all_responses[0][0],  # First problem, first response
            output_dir=output_dir,
            loop_idx=loop_idx,
            loops=loops,
        )

    # Evaluate
    mean_acc_list: List[float] = []
    pass_at_k_list: List[float] = []
    majority_acc_list: List[float] = []

    print("Evaluating...")
    use_tqdm = not (logger and logger.enabled)  # Don't use tqdm if verbose logging
    iterator = tqdm(enumerate(zip(data, all_responses)), total=len(data), desc="Grading") if use_tqdm else enumerate(zip(data, all_responses))

    for idx, (problem, responses) in iterator:
        gt = problem['gt']
        orig_prompt = problem['orig_prompt']

        perf_metric = evaluate_k_answers_frontierscience(
            grader_client, orig_prompt, responses[:], gt
        )
        # Save extracted answers and scores for debugging/reproducibility
        problem['extracted_answers'] = perf_metric['extracted_answers']
        problem['scores'] = perf_metric['pred_accuracies']

        mean_acc_list.append(perf_metric['mean_acc'])
        pass_at_k_list.append(perf_metric['pass_at_k'])
        majority_acc_list.append(perf_metric['majority_vote_correct'])

        # Verbose logging: show extracted answers and scores
        if logger:
            logger.show_problem(idx, problem, perf_metric['extracted_answers'], perf_metric['pred_accuracies'])

    metrics_dict = {
        "n_samples": len(mean_acc_list),
        "k": k,
        "mean_acc_k": sum(mean_acc_list) / max(1, len(mean_acc_list)),
        "mean_pass_at_k": sum(pass_at_k_list) / max(1, len(pass_at_k_list)),
        "mean_majority_acc": sum(majority_acc_list) / max(1, len(majority_acc_list)),
        "mean_length": mean_response_length,
        "median_length": median,
        "q25_length": q25,
        "q75_length": q75,
    }

    # Verbose logging: show loop summary
    if logger:
        logger.loop_summary(loop_idx, metrics_dict)

    return data, json.dumps(metrics_dict, indent=2)


def loop(
    model_name: str,
    loops: int,
    k: int,
    population: int,
    domain: Optional[str],
    sample_range: Optional[str],
    output_dir: str,
    max_tokens: int,
    temperature: float,
    seed: int,
    resume: bool,
    openai_api_key: Optional[str],
    anthropic_api_key: Optional[str],
    max_workers: int,
    verbose: bool = False,
):
    # Setup verbose logger
    logger = VerboseLogger(enabled=verbose)
    if verbose and not RICH_AVAILABLE:
        logger.warn_no_rich()

    # Setup API clients
    client, client_type = get_api_client(model_name, openai_api_key, anthropic_api_key)
    grader_client = get_grader_client(openai_api_key)
    print(f"Using model: {model_name} ({client_type})")
    print(f"Using {GRADER_MODEL} as grader")

    # Load dataset
    print("Loading FrontierScience dataset...")
    ds = load_frontierscience(domain=domain, sample_range=sample_range, seed=seed)
    range_info = f" (samples {sample_range})" if sample_range else ""
    domain_info = f" (domain={domain})" if domain else ""
    print(f"Loaded {len(ds)} samples{domain_info}{range_info}")

    task = "frontierscience"

    # Control RNG for candidate sampling
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, f'k_{k}_N_{population}_seed_{seed}.json')
    if not resume:
        if os.path.exists(metrics_path):
            os.remove(metrics_path)
    checkpoints_path = os.path.join(output_dir, f'checkpoints/k_{k}_N_{population}_seed_{seed}')
    os.makedirs(checkpoints_path, exist_ok=True)

    if resume:
        try:
            data, start_loop_idx, _ = load_latest_loop_file(checkpoints_path)
            print(f'Resuming from loop: {start_loop_idx + 1}')
        except FileNotFoundError:
            print('Checkpoint not found; starting fresh')
            data = [
                {
                    'orig_prompt': row['problem'],
                    'subject': row['subject'],
                    'task_group_id': row['task_group_id'],
                    'gt': row['answer'],
                    'candidates': None,
                }
                for row in ds
            ]
            start_loop_idx = -1
    else:
        data = [
            {
                'orig_prompt': row['problem'],
                'subject': row['subject'],
                'task_group_id': row['task_group_id'],
                'gt': row['answer'],
                'candidates': None,
            }
            for row in ds
        ]
        start_loop_idx = -1

    for loop_idx in range(start_loop_idx + 1, loops):
        if not (logger and logger.enabled):
            print(f"\n=== Loop {loop_idx} ===")
        data, metrics = run(
            client=client,
            client_type=client_type,
            model=model_name,
            grader_client=grader_client,
            k=k,
            population=population,
            data=data,
            task=task,
            max_tokens=max_tokens,
            temperature=temperature,
            max_workers=max_workers,
            logger=logger,
            loop_idx=loop_idx,
            output_dir=output_dir,
            loops=loops
        )

        # Save checkpoint
        with open(os.path.join(checkpoints_path, f'loop_{loop_idx}.pkl'), 'wb') as file:
            pickle.dump(data, file)

        print(f"Loop {loop_idx} metrics: {metrics}")
        metrics_dict = json.loads(metrics)

        out_entry = {
            "n_samples": metrics_dict.get("n_samples", None),
            "k": k,
            "population": population,
            "loop": loop_idx,
            "task": task,
            "domain": domain,
            "mean_acc_k": metrics_dict["mean_acc_k"],
            "mean_pass_at_k": metrics_dict["mean_pass_at_k"],
            "mean_majority_acc": metrics_dict["mean_majority_acc"],
            "mean_length": metrics_dict["mean_length"],
            "median_length": metrics_dict["median_length"],
            "q25_length": metrics_dict["q25_length"],
            "q75_length": metrics_dict["q75_length"],
        }

        _append_metrics_to_json(metrics_path, out_entry)
        print(f"Appended metrics for loop {loop_idx} to {metrics_path}")

    # Save question IDs for reproducibility
    question_ids = [d['task_group_id'] for d in data]
    ids_path = os.path.join(output_dir, f'question_ids_k_{k}_N_{population}_seed_{seed}.json')
    with open(ids_path, 'w') as f:
        json.dump({"question_ids": question_ids, "domain": domain, "n_samples": len(data)}, f, indent=2)
    print(f"Saved question IDs to {ids_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Model name (e.g., gpt-4o, claude-3-5-haiku-20241022)")
    ap.add_argument("--domain", choices=["bio", "biology", "physics", "chem", "chemistry"],
                    help="Filter to domain")
    ap.add_argument("--n-samples", type=str, help="Sample spec: N for first N, or 'start:end' for range (0-indexed, e.g., '4:10')")
    ap.add_argument("--output", default="./eval/frontierscience", help="Output directory")
    ap.add_argument("--k", type=int, default=4, help="Candidates to sample for aggregation")
    ap.add_argument("--population", type=int, default=16, help="Candidates per problem per loop")
    ap.add_argument("--loops", type=int, default=10, help="Number of RSA loops")
    ap.add_argument("--max-tokens", type=int, default=4096, help="Max tokens per response")
    ap.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    ap.add_argument("--seed", type=int, default=1234, help="Random seed")
    ap.add_argument("--resume", action='store_true', default=False, help="Resume from checkpoint")
    ap.add_argument("--max-workers", type=int, default=32, help="Max parallel API calls")
    ap.add_argument("--verbose", "-v", action='store_true', help="Show live extracted answers and scores (requires 'rich')")

    # API keys (default to env vars via api_key.txt)
    ap.add_argument("--openai-api-key", help="OpenAI API key")
    ap.add_argument("--anthropic-api-key", help="Anthropic API key")
    args = ap.parse_args()

    # Load API keys: CLI args > api_key.txt > env vars
    file_keys = load_api_keys()
    openai_key = args.openai_api_key or file_keys.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    anthropic_key = args.anthropic_api_key or file_keys.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")

    loop(
        model_name=args.model,
        loops=args.loops,
        k=args.k,
        population=args.population,
        domain=args.domain,
        sample_range=args.n_samples,
        output_dir=os.path.join(args.output, args.model.replace('/', '_')),
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
        resume=args.resume,
        openai_api_key=openai_key,
        anthropic_api_key=anthropic_key,
        max_workers=args.max_workers,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
