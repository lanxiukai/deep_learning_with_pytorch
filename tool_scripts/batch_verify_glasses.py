#!/usr/bin/env python3
"""Batch verify glasses dataset using OpenRouter vision API (qwen3.7-plus)."""

import asyncio
import aiohttp
import base64
import json
import os
import sys
from pathlib import Path

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "qwen/qwen3.7-plus"
CONCURRENCY = 15  # parallel API calls

DATA_DIR = Path(__file__).parent.parent / "data" / "glasses"
RESULTS_FILE = Path(__file__).parent.parent / "data" / "glasses_misclassified.json"


def encode_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


async def check_image(session: aiohttp.ClientSession, path: Path, sem: asyncio.Semaphore) -> tuple:
    """Check one image: returns (rel_path, predicted_glasses_bool, error_msg_or_None)"""
    async with sem:
        try:
            b64 = encode_image(path)
            payload = {
                "model": MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": "Is this person wearing glasses? Answer ONLY 'yes' or 'no'."
                            }
                        ]
                    }
                ],
                "max_tokens": 5,
                "temperature": 0,
            }
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            }
            async with session.post(API_URL, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                data = await resp.json()
                if "error" in data:
                    return (str(path.relative_to(DATA_DIR)), None, f"API error: {data['error']}")
                content = data["choices"][0]["message"]["content"].strip().lower()
                is_glasses = "yes" in content
                return (str(path.relative_to(DATA_DIR)), is_glasses, None)
        except Exception as e:
            return (str(path.relative_to(DATA_DIR)), None, str(e))


async def process_batch(paths: list[Path], batch_label: str):
    """Process a batch of images in parallel."""
    sem = asyncio.Semaphore(CONCURRENCY)
    results = []
    errors = []

    connector = aiohttp.TCPConnector(limit=CONCURRENCY, limit_per_host=CONCURRENCY)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [check_image(session, p, sem) for p in paths]
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            rel_path, is_glasses, err = await coro
            if err:
                errors.append((rel_path, err))
            else:
                results.append((rel_path, is_glasses))
            if (i + 1) % 50 == 0:
                print(f"  [{batch_label}] Progress: {i+1}/{len(paths)}", flush=True)

    return results, errors


def main():
    if not API_KEY:
        print("ERROR: OPENROUTER_API_KEY not set")
        sys.exit(1)

    # Collect all images
    g_dir = DATA_DIR / "G"
    nog_dir = DATA_DIR / "NoG"

    g_paths = sorted(g_dir.glob("*.png"))
    nog_paths = sorted(nog_dir.glob("*.png"))

    print(f"G/ images: {len(g_paths)}")
    print(f"NoG/ images: {len(nog_paths)}")
    print(f"Total: {len(g_paths) + len(nog_paths)}")
    print(f"Concurrency: {CONCURRENCY}")
    print()

    # Load existing results if any
    existing = {}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            existing = json.load(f)
        print(f"Loaded {len(existing)} existing results from {RESULTS_FILE}")

    # Filter out already processed
    g_to_process = [p for p in g_paths if str(p.relative_to(DATA_DIR)) not in existing]
    nog_to_process = [p for p in nog_paths if str(p.relative_to(DATA_DIR)) not in existing]

    print(f"G/ to process: {len(g_to_process)} (skipped {len(g_paths) - len(g_to_process)})")
    print(f"NoG/ to process: {len(nog_to_process)} (skipped {len(nog_paths) - len(nog_to_process)})")

    if not g_to_process and not nog_to_process:
        print("All images already processed. Computing final report...")
    else:
        # Process NoG first (smaller), then G
        for paths, label in [(nog_to_process, "NoG"), (g_to_process, "G")]:
            if not paths:
                continue
            print(f"\nProcessing {label}/ ({len(paths)} images)...")
            results, errors = asyncio.run(process_batch(paths, label))

            # Save incrementally
            for rel_path, is_glasses in results:
                existing[rel_path] = is_glasses
            with open(RESULTS_FILE, "w") as f:
                json.dump(existing, f, indent=2)
            print(f"  [{label}] Done. Results: {len(results)}, Errors: {len(errors)}")
            if errors:
                for p, e in errors[:10]:
                    print(f"    ERROR: {p}: {e}")
                if len(errors) > 10:
                    print(f"    ... and {len(errors) - 10} more errors")

    # Final report
    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)

    misclassified = []
    G_wrong = []  # in G/ but predicted no glasses
    NoG_wrong = []  # in NoG/ but predicted yes glasses

    for rel_path, is_glasses in existing.items():
        if is_glasses is None:
            continue  # skip errors
        actual_folder = "G" if rel_path.startswith("G/") else "NoG"
        expected_glasses = actual_folder == "G"
        if is_glasses != expected_glasses:
            misclassified.append(rel_path)
            if actual_folder == "G":
                G_wrong.append(rel_path)
            else:
                NoG_wrong.append(rel_path)

    print(f"\nTotal images processed: {len(existing)}")
    print(f"Misclassified: {len(misclassified)}")
    print(f"  In G/ but NOT wearing glasses: {len(G_wrong)}")
    print(f"  In NoG/ but wearing glasses: {len(NoG_wrong)}")

    if G_wrong:
        print(f"\nG/ → should be in NoG/ ({len(G_wrong)}):")
        for p in G_wrong[:30]:
            print(f"  {p}")
        if len(G_wrong) > 30:
            print(f"  ... and {len(G_wrong) - 30} more")

    if NoG_wrong:
        print(f"\nNoG/ → should be in G/ ({len(NoG_wrong)}):")
        for p in NoG_wrong[:30]:
            print(f"  {p}")
        if len(NoG_wrong) > 30:
            print(f"  ... and {len(NoG_wrong) - 30} more")

    # Save detailed report
    report = {
        "total_processed": len(existing),
        "misclassified_count": len(misclassified),
        "G_wrong_count": len(G_wrong),
        "NoG_wrong_count": len(NoG_wrong),
        "G_wrong": G_wrong,
        "NoG_wrong": NoG_wrong,
    }
    report_path = Path(__file__).parent.parent / "data" / "glasses_misclassified_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to {report_path}")

    return misclassified


if __name__ == "__main__":
    main()
