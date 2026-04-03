---
title: Email Triage Server
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# Email Triage OpenEnv Environment v2.0

Meta PyTorch OpenEnv Hackathon x Scaler School of Technology submission.

This project simulates a real enterprise workflow: triaging inbound emails for spam, business intent, urgency, ownership team, and the best response template. It is designed for agent learning and evaluation with deterministic tasks, graded rewards, and a standard reset/step/state interface.

## Why this environment

Email triage is a high-impact, real-world automation problem seen in support, finance, HR, legal, and sales operations. Unlike toy tasks, this environment includes ambiguity, adversarial messages, constrained actions, and partial-information tradeoffs that force agents to make practical decisions.

## OpenEnv compliance

This repository includes:

- typed Pydantic models for Observation, Action, Reward
- canonical API: reset(), step(), state()
- deterministic benchmark tasks: easy, medium, hard
- openenv.yaml metadata and schema definitions
- baseline inference.py at repository root

Core files:

- [email_triage_env/server/app.py](email_triage_env/server/app.py)
- [email_triage_env/openenv_env.py](email_triage_env/openenv_env.py)
- [openenv.yaml](openenv.yaml)
- [inference.py](inference.py)

## Observation space

Each step returns one email observation:

```json
{
  "id": "string",
  "subject": "string",
  "sender": "string",
  "body": "string",
  "body_hidden": false,
  "difficulty": "easy|medium|hard|mixed"
}
```

In partial-info mode, `body` is hidden until `reveal_body` is called.

## Action space

Agent action is a 5-field classification object:

```json
{
  "is_spam": true,
  "category": "billing|support|sales|hr|general",
  "priority": "urgent|normal|low",
  "department": "finance|customer_support|sales|human_resources|legal|none",
  "response_template": "billing_info|billing_escalation|contract_escalation|password_reset_guide|hr_policy_info|interview_scheduling|enterprise_demo_request|spam_discard|no_reply_needed|automated_no_action"
}
```

## Reward function

Per-step reward is clamped to [0.0, 1.0].

- weighted correctness:
  - is_spam: 0.30
  - category: 0.20
  - priority: 0.20
  - department: 0.15
  - response_template: 0.15
- partial-info penalty:
  - reveal_body: -0.05 per email when used
- difficulty bonus (if base >= 0.7):
  - easy: +0.00
  - medium: +0.10
  - hard: +0.20
  - mixed: +0.05

This gives dense trajectory feedback while discouraging unnecessary reveals.

## Benchmark tasks and graders

Deterministic tasks with fixed seeds and scoring in [0.0, 1.0]:

1. `email_triage_easy_v1`
2. `email_triage_medium_v1`
3. `email_triage_hard_v1`

All tasks are exposed through [email_triage_env/server/app.py](email_triage_env/server/app.py) and [openenv.yaml](openenv.yaml). Grading is deterministic because dataset sampling is seed-controlled and reward computation is purely programmatic.

Dataset sizes:

- easy: 8 emails
- medium: 7 emails
- hard: 7 emails
- total unique emails: 22

## API and tools

Core endpoints:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /tasks`
- `GET /health`

MCP tool endpoints:

- `POST /tools/get_current_email`
- `POST /tools/reveal_body`
- `POST /tools/classify_email`
- `POST /tools/get_available_options`
- `POST /tools/get_episode_statistics`
- `POST /tools/get_leaderboard`

## Setup

1. Install dependencies.

```bash
pip install -r requirements.txt
```

2. Run the environment server.

```bash
python main.py
```

3. Useful local URLs.

- dashboard: http://localhost:8000/dashboard
- docs: http://localhost:8000/docs
- health: http://localhost:8000/health

## Baseline inference script (required)

Run [inference.py](inference.py) from repository root.

Required environment variables:

- `API_BASE_URL`: OpenAI-compatible endpoint
- `MODEL_NAME`: model identifier
- `HF_TOKEN`: Hugging Face/API token required by submission infra
- `OPENAI_API_KEY`: OpenAI client API key

Optional environment variable:

- `ENV_BASE_URL` (default: `http://localhost:8000`)

Run command:

```bash
python inference.py
```

Structured stdout format emitted by inference:

- `[START] task=... env=... model=...`
- `[STEP] step=... action=... reward=... done=... error=...`
- `[END] success=... steps=... score=... rewards=[...]`

## Baseline reproducibility

- task order is fixed: easy -> medium -> hard
- each task uses fixed seed: 101, 202, 303
- temperature is fixed to 0 for model calls
- fallback policy is deterministic when model output is invalid

## Baseline scores

The script computes a score per task and reports it in each `[END]` record. The final reproducible baseline is the mean of the three task scores.

## Docker and Hugging Face Spaces

Build and run:

```bash
docker build -t email-triage-env .
docker run -p 8000:8000 email-triage-env
```

This repo is configured as a Docker Space with the frontmatter in this README. Tag the Space with `openenv` before submission.

## Pre-submission validation checklist

1. Docker builds and server responds on `/health`.
2. `openenv validate` passes from repository root.
3. [inference.py](inference.py) runs with required env vars and emits strict logs.
4. All three tasks produce deterministic scores in [0.0, 1.0].
5. Total inference runtime is under 20 minutes on 2 vCPU, 8 GB RAM.

## Project structure

```text
email_triage_env/
  __init__.py
  client.py
  emails.py
  openenv_env.py
  server/
    __init__.py
    app.py
main.py
demo.py
inference.py
openenv.yaml
requirements.txt
Dockerfile
README.md
```
