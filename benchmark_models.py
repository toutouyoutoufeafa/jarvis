"""Benchmark: compare local models on French chat + tool-calling JSON."""
import asyncio
import json
import re
import time

import httpx

URL = "http://localhost:11434/api/chat"

CASES = [
    ("chat_fr", "Quelle est la capitale de la France ? Reponds en une phrase."),
    ("memory", None),  # handled separately
    ("tool_json", (
        "Utilisateur: Cree un fichier hello.py\n"
        "Reponds UNIQUEMENT avec ce JSON: "
        '{"type":"tool","tool":"fs_write","args":{"path":"hello.py","content":"print(1)"}}'
    )),
    ("reasoning", "Combien font 17 x 23 ? Reponds juste par le nombre."),
    ("style", "Explique ce qu'est un API REST en 2 phrases, niveau debutant."),
]


async def ask(client: httpx.AsyncClient, model: str, messages: list[dict]) -> tuple[str, float]:
    t0 = time.perf_counter()
    r = await client.post(URL, json={
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.7, "num_predict": 512},
    }, timeout=120)
    r.raise_for_status()
    dt = time.perf_counter() - t0
    return r.json()["message"]["content"], dt


async def run_model(model: str) -> None:
    print(f"\n{'='*60}\nMODEL: {model}\n{'='*60}")
    results = {}
    async with httpx.AsyncClient() as client:
        # warm up
        await ask(client, model, [{"role": "user", "content": "OK"}])

        for name, prompt in CASES:
            if name == "memory":
                msgs = [
                    {"role": "user", "content": "Je m'appelle Tony, retiens-le."},
                    {"role": "assistant", "content": "Je retiens, Tony."},
                    {"role": "user", "content": "Comment je m'appelle ?"},
                ]
                content, dt = await ask(client, model, msgs)
                ok = "tony" in content.lower()
                results[name] = (dt, ok, content)
                print(f"[{name:10}] {dt:5.2f}s  memory_ok={ok}  -> {content[:70]!r}")
                continue

            content, dt = await ask(client, model, [{"role": "user", "content": prompt}])
            if name == "tool_json":
                m = re.search(r"\{.*\}", content, re.DOTALL)
                ok = False
                if m:
                    try:
                        obj = json.loads(m.group())
                        ok = obj.get("type") == "tool" and obj.get("tool") == "fs_write"
                    except Exception:
                        ok = False
                results[name] = (dt, ok, content)
                print(f"[{name:10}] {dt:5.2f}s  json_ok={ok}  -> {content[:70]!r}")
            elif name == "reasoning":
                ok = "391" in content
                results[name] = (dt, ok, content)
                print(f"[{name:10}] {dt:5.2f}s  math_ok={ok}  -> {content[:70]!r}")
            else:
                results[name] = (dt, True, content)
                print(f"[{name:10}] {dt:5.2f}s            -> {content[:90]!r}")

    total_time = sum(v[0] for v in results.values())
    score = sum(1 for v in results.values() if v[1])
    print(f"\n>> {model}: score {score}/{len(results)}  total {total_time:.2f}s")
    return results


async def main():
    models = ["nemotron-3-nano:4b", "qwen2.5:3b"]
    summary = {}
    for m in models:
        try:
            summary[m] = await run_model(m)
        except Exception as e:
            print(f"FAILED {m}: {e}")

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for m, res in summary.items():
        score = sum(1 for v in res.values() if v[1])
        tt = sum(v[0] for v in res.values())
        print(f"{m:22} score={score}/{len(res)}  total={tt:.2f}s")


asyncio.run(main())
