---
name: retrospective
description: After completing a significant task or experiment, extract lessons learned and update the project knowledge base. Captures what worked, what failed, and what to remember for next time.
argument-hint: [task-description]
disable-model-invocation: true
allowed-tools: Read, Edit, Write, Bash, Grep, Glob
---

# Retrospective: Capture Lessons Learned

After completing "$ARGUMENTS", extract insights and update the project knowledge base.

## Steps

1. **Gather context**: Read recent changes, results, and any errors encountered during the task. Look at:
   - Recent git commits: `git log --oneline -10`
   - Modified files: `git diff --name-only HEAD~5`
   - Any error logs in `logs/`
   - Result files if experiment-related

2. **Extract lessons** in these categories:

   **What worked** — approaches, tools, or patterns that were effective
   **What failed** — approaches that were tried and didn't work, with WHY
   **Surprises** — unexpected findings or behaviors
   **For next time** — specific recommendations for similar future tasks

3. **Update knowledge base**:

   **If there are failed approaches**: Append to `docs/failures.md` following the existing format:
   ```markdown
   ### Title
   **When**: Date
   **What happened**: Description
   **Root cause**: Why it failed
   **Fix/workaround**: What to do instead
   ```

   **If there's a decision**: Append to `docs/decisions.jsonl` using the established schema.

   **If there are experiment findings**: Consider whether `docs/findings.md` needs updating.

4. **Summarize** the retrospective in 3-5 bullet points for the user.

## Rules

- Be specific. "It was hard" is not useful. "XGBoost training fails with >100k samples on MPS due to memory limit" is useful.
- Include exact error messages or config values when relevant.
- Failed attempts with details are MORE valuable than successes — they prevent repeated mistakes.
- Don't update files unless there's genuinely new information to capture.
