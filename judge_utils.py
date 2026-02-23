"""
Shared grading utilities for FrontierScience benchmark.

Provides LLM-based grading using GPT 5.2 and Gemini, with a debate
tie-breaking mechanism. Imported by eval_frontierscience.py, judge.py,
and debate.py.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional



# --------------------- constants ---------------------

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
First, think step-by-step about whether the attempted answer matches the reference answer. If the attempted answer is correct, write "VERDICT: CORRECT" in the last line of your response, with no other text or formatting. If it is incorrect, write "VERDICT: INCORRECT".
"""

GEMINI_GRADER_MODEL = "gemini-3.1-pro-preview"

DEBATE_PROMPT = """You previously graded an attempted answer to a science olympiad problem. Another independent grader DISAGREED with your verdict. Please reconsider carefully and make a final decision.

Here is the original grading prompt you were given:
===
{original_judge_prompt}
===

Your original verdict: {gpt_verdict}

A different grader disagreed with you. Here is their reasoning:
---
{gemini_reasoning}
---

Please reconsider the attempted answer against the reference answer, taking into account the other grader's reasoning. First, write a detailed explanation of your reasoning. Then, if the attempted answer is correct, write "VERDICT: CORRECT" in the last line of your response. If it is incorrect, write "VERDICT: INCORRECT"."""

GEMINI_REBUTTAL_PROMPT = """You previously graded an attempted answer to a science olympiad problem. Another independent grader DISAGREED with your verdict and provided the following reasoning after seeing your analysis.

Here is the original grading prompt:
===
{original_judge_prompt}
===

Your original verdict: {gemini_verdict}

The other grader's response after considering your reasoning:
---
{gpt_reconsideration}
---

Please reconsider the attempted answer against the reference answer, taking into account the other grader's counter-argument. First, write a detailed explanation of your reasoning. Then, if the attempted answer is correct, write "VERDICT: CORRECT" in the last line of your response. If it is incorrect, write "VERDICT: INCORRECT"."""

GPT_FINAL_PROMPT = """You are making a final grading decision on an attempted answer to a science olympiad problem, after a multi-round debate with another grader.

Here is the original grading prompt:
===
{original_judge_prompt}
===

Debate history:
1. Your original verdict: {gpt_verdict}
2. Other grader's original verdict: {gemini_verdict} (with reasoning below)
3. Your first reconsideration (after seeing the other grader's reasoning)
4. The other grader's rebuttal (after seeing your reconsideration)

--- Other grader's rebuttal ---
{gemini_rebuttal}
---

This is your FINAL decision. Consider all arguments carefully. First, write a detailed explanation of your reasoning. Then, if the attempted answer is correct, write "VERDICT: CORRECT" in the last line of your response. If it is incorrect, write "VERDICT: INCORRECT"."""


# --------------------- client factories ---------------------

def get_grader_client(openai_api_key: Optional[str] = None):
    """Return OpenAI client for grading (always uses OpenAI for GPT 5.2)."""
    import openai
    return openai.OpenAI(api_key=openai_api_key)


def get_gemini_grader_client(google_api_key: Optional[str] = None):
    """Return Google GenAI client for Gemini grading."""
    from google import genai
    return genai.Client(api_key=google_api_key)


def _is_retryable(e: Exception) -> bool:
    """Check if an exception is a retryable error (rate limit 429, overloaded 503, etc.)."""
    s = str(e).lower()
    return any(tok in s for tok in ("429", "503", "rate", "throttl", "too many", "resource_exhausted", "unavailable", "overloaded"))


# --------------------- scoring functions ---------------------

def score_with_llm(grader_client, problem: str, prediction: str, ground_truth: str) -> tuple[Optional[float], str]:
    """Score a prediction using GPT 5.2 as grader.

    Returns:
        (score, reasoning) tuple where:
        - score is 1.0 (correct), 0.0 (incorrect), or None (grading failed)
        - reasoning is the grader's full response text, or empty string if grading failed
    """
    prompt = GRADER_PROMPT.format(
        problem=problem,
        ground_truth=ground_truth,
        prediction=prediction
    )

    attempt = 0
    while True:
        try:
            response = grader_client.responses.create(
                model=GRADER_MODEL,
                input=prompt,
                reasoning={"effort": "high"}
            )
            reasoning = response.output_text.strip()
            # Parse "VERDICT: CORRECT" or "VERDICT: INCORRECT"
            if "VERDICT: CORRECT" in reasoning.upper():
                return (1.0, reasoning)
            elif "VERDICT: INCORRECT" in reasoning.upper():
                return (0.0, reasoning)
            break  # parsed neither verdict — fall through
        except Exception as e:
            if _is_retryable(e):
                print(f"Retryable error: {e}. Waiting 30s...")
                time.sleep(30)
            elif attempt < 2:
                time.sleep(2 ** attempt)
                attempt += 1
            else:
                print(f"Grading failed: {e}, returning None")
                return (None, "")
    return (None, "")


def score_with_gemini(gemini_client, problem: str, prediction: str, ground_truth: str) -> tuple[Optional[float], str]:
    """Score a prediction using Gemini 3 Flash as grader with high reasoning effort.

    Returns:
        (score, reasoning) tuple where:
        - score is 1.0 (correct), 0.0 (incorrect), or None (grading failed)
        - reasoning is the grader's full response text, or empty string if grading failed
    """
    from google.genai import types

    prompt = GRADER_PROMPT.format(
        problem=problem,
        ground_truth=ground_truth,
        prediction=prediction
    )

    attempt = 0
    while True:
        try:
            response = gemini_client.models.generate_content(
                model=GEMINI_GRADER_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level="high")
                ),
            )
            reasoning = response.text.strip()
            if "VERDICT: CORRECT" in reasoning.upper():
                return (1.0, reasoning)
            elif "VERDICT: INCORRECT" in reasoning.upper():
                return (0.0, reasoning)
            break  # parsed neither verdict — fall through
        except Exception as e:
            if _is_retryable(e):
                print(f"Retryable error: {e}. Waiting 30s...")
                time.sleep(30)
            elif attempt < 2:
                time.sleep(2 ** attempt)
                attempt += 1
            else:
                print(f"Gemini grading failed: {e}, returning None")
                return (None, "")
    return (None, "")


def _parse_verdict(text: str) -> Optional[float]:
    """Parse VERDICT from grader response text."""
    upper = text.upper()
    if "VERDICT: CORRECT" in upper:
        return 1.0
    elif "VERDICT: INCORRECT" in upper:
        return 0.0
    return None


def _call_grader_with_retries(grader_client, prompt: str, retries: int = 3) -> Optional[str]:
    """Call GPT grader with retries. Returns response text or None on failure."""
    attempt = 0
    while True:
        try:
            response = grader_client.responses.create(
                model=GRADER_MODEL,
                input=prompt,
                reasoning={"effort": "high"}
            )
            return response.output_text.strip()
        except Exception as e:
            if _is_retryable(e):
                print(f"Retryable error: {e}. Waiting 30s...")
                time.sleep(30)
            elif attempt < retries - 1:
                time.sleep(2 ** attempt)
                attempt += 1
            else:
                print(f"GPT grader call failed after {retries} attempts: {e}")
                return None


def _call_gemini_with_retries(gemini_client, prompt: str, retries: int = 3) -> Optional[str]:
    """Call Gemini grader with retries. Returns response text or None on failure."""
    from google.genai import types
    attempt = 0
    while True:
        try:
            response = gemini_client.models.generate_content(
                model=GEMINI_GRADER_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level="high")
                ),
            )
            return response.text.strip()
        except Exception as e:
            if _is_retryable(e):
                print(f"Retryable error: {e}. Waiting 30s...")
                time.sleep(30)
            elif attempt < retries - 1:
                time.sleep(2 ** attempt)
                attempt += 1
            else:
                print(f"Gemini grader call failed after {retries} attempts: {e}")
                return None


def score_with_debate(grader_client, gemini_client, problem: str, prediction: str, ground_truth: str, debate_rounds: int = 1) -> tuple[Optional[float], str]:
    """Score using debate: Gemini and GPT 5.2 grade independently, GPT 5.2 breaks ties.

    Steps:
    1. Get Gemini's grade and reasoning
    2. Get GPT 5.2's grade and reasoning
    3. If they agree, return the agreed grade
    4. If they disagree, GPT 5.2 reconsiders given Gemini's reasoning (round 1)
    5. If debate_rounds=2: Gemini rebuts GPT's reconsideration, then GPT makes final decision

    Args:
        debate_rounds: Number of debate rounds when graders disagree (default 1).
            1 = GPT reconsiders after seeing Gemini's reasoning (current behavior)
            2 = adds Gemini rebuttal + GPT final decision

    Returns:
        (score, reasoning) tuple with combined reasoning from the debate
    """
    # Step 1 & 2: Grade with both models in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        gemini_future = executor.submit(score_with_gemini, gemini_client, problem, prediction, ground_truth)
        gpt_future = executor.submit(score_with_llm, grader_client, problem, prediction, ground_truth)

    gemini_score, gemini_reasoning = gemini_future.result()
    gpt_score, gpt_reasoning = gpt_future.result()

    # If either grader failed, raise so the user knows immediately
    if gemini_score is None:
        raise RuntimeError(f"Debate grader failed: Gemini returned None (API error). Cannot proceed with debate.")
    if gpt_score is None:
        raise RuntimeError(f"Debate grader failed: GPT returned None (API error). Cannot proceed with debate.")

    # Step 3: If they agree, return the agreed grade
    if gemini_score == gpt_score:
        verdict = "CORRECT" if gpt_score == 1.0 else "INCORRECT"
        combined = (
            f"[debate: both graders agree - {verdict}]\n"
            f"--- GPT 5.2 reasoning ---\n{gpt_reasoning}\n"
            f"--- Gemini reasoning ---\n{gemini_reasoning}"
        )
        return (gpt_score, combined)

    # Step 4: They disagree - GPT 5.2 reconsiders with Gemini's reasoning
    gpt_verdict = "CORRECT" if gpt_score == 1.0 else "INCORRECT"
    gemini_verdict = "CORRECT" if gemini_score == 1.0 else "INCORRECT"
    original_judge_prompt = GRADER_PROMPT.format(
        problem=problem,
        ground_truth=ground_truth,
        prediction=prediction,
    )
    debate_prompt = DEBATE_PROMPT.format(
        original_judge_prompt=original_judge_prompt,
        gpt_verdict=gpt_verdict,
        gemini_reasoning=gemini_reasoning,
    )

    # Round 1: GPT reconsiders given Gemini's reasoning
    gpt_reconsideration = _call_grader_with_retries(grader_client, debate_prompt)
    if gpt_reconsideration is None:
        print("Debate round 1 failed, returning GPT's original score")
        return (gpt_score, f"[debate: round 1 failed, using gpt original]\n{gpt_reasoning}")

    if debate_rounds < 2:
        # Single round: GPT's reconsideration is the final decision
        final_score = _parse_verdict(gpt_reconsideration)
        combined = (
            f"[debate: graders disagreed, GPT={gpt_verdict}, Gemini={gemini_verdict}]\n"
            f"--- GPT 5.2 original reasoning ---\n{gpt_reasoning}\n"
            f"--- Gemini reasoning ---\n{gemini_reasoning}\n"
            f"--- GPT 5.2 final decision ---\n{gpt_reconsideration}"
        )
        return (final_score, combined)

    # Round 2: Gemini rebuts GPT's reconsideration
    gemini_rebuttal_prompt = GEMINI_REBUTTAL_PROMPT.format(
        original_judge_prompt=original_judge_prompt,
        gemini_verdict=gemini_verdict,
        gpt_reconsideration=gpt_reconsideration,
    )
    gemini_rebuttal = _call_gemini_with_retries(gemini_client, gemini_rebuttal_prompt)
    if gemini_rebuttal is None:
        # Gemini rebuttal failed, fall back to round 1 result
        print("Debate round 2 (Gemini rebuttal) failed, using GPT round 1 decision")
        final_score = _parse_verdict(gpt_reconsideration)
        combined = (
            f"[debate: graders disagreed, GPT={gpt_verdict}, Gemini={gemini_verdict}, Gemini rebuttal failed]\n"
            f"--- GPT 5.2 original reasoning ---\n{gpt_reasoning}\n"
            f"--- Gemini reasoning ---\n{gemini_reasoning}\n"
            f"--- GPT 5.2 round 1 decision ---\n{gpt_reconsideration}"
        )
        return (final_score, combined)

    # Round 2: GPT makes final decision after seeing Gemini's rebuttal
    gpt_final_prompt = GPT_FINAL_PROMPT.format(
        original_judge_prompt=original_judge_prompt,
        gpt_verdict=gpt_verdict,
        gemini_verdict=gemini_verdict,
        gemini_rebuttal=gemini_rebuttal,
    )
    gpt_final = _call_grader_with_retries(grader_client, gpt_final_prompt)
    if gpt_final is None:
        print("Debate round 2 (GPT final) failed, using GPT round 1 decision")
        final_score = _parse_verdict(gpt_reconsideration)
        combined = (
            f"[debate: graders disagreed, GPT={gpt_verdict}, Gemini={gemini_verdict}, GPT final failed]\n"
            f"--- GPT 5.2 original reasoning ---\n{gpt_reasoning}\n"
            f"--- Gemini reasoning ---\n{gemini_reasoning}\n"
            f"--- GPT 5.2 round 1 decision ---\n{gpt_reconsideration}\n"
            f"--- Gemini rebuttal ---\n{gemini_rebuttal}"
        )
        return (final_score, combined)

    final_score = _parse_verdict(gpt_final)
    combined = (
        f"[debate: 2-round, GPT={gpt_verdict}, Gemini={gemini_verdict}]\n"
        f"--- GPT 5.2 original reasoning ---\n{gpt_reasoning}\n"
        f"--- Gemini original reasoning ---\n{gemini_reasoning}\n"
        f"--- GPT 5.2 round 1 reconsideration ---\n{gpt_reconsideration}\n"
        f"--- Gemini rebuttal ---\n{gemini_rebuttal}\n"
        f"--- GPT 5.2 final decision ---\n{gpt_final}"
    )
    return (final_score, combined)


# --------------------- shared helpers ---------------------

def load_api_keys(key_file: str = "api_key.txt") -> dict[str, str]:
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


def load_frontierscience_problems() -> dict[int, str]:
    """Load FrontierScience dataset and return mapping of original_idx -> problem text."""
    from datasets import load_dataset
    ds = load_dataset("openai/frontierscience", split="test")
    problems = {}
    for i, row in enumerate(ds):
        # Skip research questions (same filter as eval_frontierscience.py)
        if not row["answer"].startswith("Points"):
            problems[i] = row["problem"]
    return problems
