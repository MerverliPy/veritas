# Contributing to veritas

Thanks for considering a contribution. This is a small, focused research
tool; the guidelines below keep setup and testing simple.

## Setup

Python **3.12** is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

cp .env.example .env          # then add DEEPSEEK_API_KEY (DeepSeek API)
```

`TAVILY_API_KEY` is optional — the keyless web engines (DuckDuckGo, Wikipedia,
arXiv, Hacker News, GitHub) work without it.

## Running the tests

```bash
pytest
```

The suite is hermetic: it runs fully offline with a scripted FakeLLM and
never touches the network.

You can also smoke-check the pipeline offline (no API key needed):

```bash
veritas run "offline demo" --surfaces web --fake
```

## Pull requests

- Keep changes minimal and focused; one concern per PR.
- Add or update tests for behaviour changes.
- Update `CHANGELOG.md` under an *Unreleased* heading when your change is
  user-visible.
