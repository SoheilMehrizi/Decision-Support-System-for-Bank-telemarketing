## Contributing Guidelines

Thank you for your interest in contributing! Please follow the steps below.

### Workflow
- Fork the repo and create a feature branch: `feat/<short-description>` or `fix/<short-description>`.
- Use Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, etc.
- Open a PR with a clear description, context, and checklist.

### Development Setup
- Python 3.10+
- Create a virtual environment and install deps:
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip -r requirements.txt
```
- Ensure PostgreSQL is running and set `DATABASE_URL`.
- Run the API from `deployment/app`:
```bash
export DATABASE_URL="postgresql+psycopg2://postgres:password@localhost:5432/mydb"
export PYTHONPATH="$(pwd)/..:${PYTHONPATH}"
uvicorn main:app --reload
```

### Testing
- Add tests under `tests/` and run:
```bash
pytest -q
```

### Code Style
- Prefer clear names and small, focused functions.
- Add docstrings for non-trivial logic.

### Security
- Do not commit secrets. Externalize JWT secrets and DB credentials.

### Reporting Issues
- Include steps to reproduce, expected vs actual behavior, logs, and environment details.

