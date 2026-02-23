---
name: advise
description: Search past decisions, failures, and experiment logs for relevant context before starting a task. Use this before any significant implementation or experiment to avoid repeating mistakes.
argument-hint: [topic]
allowed-tools: Read, Grep, Glob
---

# Advise: Search Past Context

Before starting work on "$ARGUMENTS", search the project's knowledge base for relevant past experience.

## Search locations (in order)

1. **Decision log** — `docs/decisions.jsonl`
   Search for decisions related to the topic. Each line is a JSON object with: date, title, context, options, chosen, rationale, references.

2. **Known failures** — `docs/failures.md`
   Check if there are documented failure patterns related to the task.

3. **Experiment reports** — `docs/experiments/*.md`
   Look for past experiments that produced relevant findings.

4. **SOTA analysis** — `docs/analysis/*.md`
   Check if the SOTA review covers this topic.

5. **v2 plan** — `docs/v2-plan.md`
   Check if this task is already planned and what the requirements are.

6. **Related work** — `docs/related-work-positioning.md`
   Check how other systems handle this.

## Output format

Present findings as:

### Relevant decisions
- [date] Title: rationale (from decisions.jsonl)

### Known failures to avoid
- Failure pattern: what went wrong and why (from failures.md)

### Related experiments
- Experiment N: key finding relevant to this task

### Recommendations
- Based on past context, here's what to consider for this task
- Specific pitfalls to avoid
- Approaches that worked before

If no relevant context is found, say so clearly — absence of past context is also useful information.
