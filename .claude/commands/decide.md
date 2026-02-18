Log a research decision to `docs/decisions.jsonl`.

The user will describe a decision they've made or need to make. Your job:

1. If the user describes a decision already made, extract the details and log it.
2. If the user is weighing options, help them think through it, then log the result once they choose.

Gather these fields (ask if unclear):
- **title**: short description of the decision
- **context**: what prompted it
- **options**: what was on the table (name + one-line detail each)
- **chosen**: which option won
- **rationale**: why — this is the most important field
- **references**: relevant files, URLs, or papers (optional)

Then append a single JSON line to `docs/decisions.jsonl` with today's date. Use the Bash tool to append:

```
echo '{"date": "...", "title": "...", ...}' >> docs/decisions.jsonl
```

After appending, confirm the decision was logged and read back the entry.

Schema (one JSON object per line, no trailing commas):
```json
{
  "date": "YYYY-MM-DD",
  "title": "Short imperative description",
  "context": "What prompted this decision",
  "options": [{"name": "A", "detail": "..."}, {"name": "B", "detail": "..."}],
  "chosen": "A",
  "rationale": "Why A over B",
  "references": ["file.py", "https://..."]
}
```

Rules:
- One line per decision. Valid JSONL — each line must be valid JSON.
- Keep rationale honest and concise. One to three sentences.
- References are optional but encouraged. Use relative paths from the repo root.
- If the decision supersedes a previous one, mention it in the rationale.
- Do not modify existing lines. Append only.

The input from the user is: $ARGUMENTS
