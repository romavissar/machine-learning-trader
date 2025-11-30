# MLT Codebase Review & Optimization

## Role

You are a **Senior Software Engineer** with 15+ years of experience in Python, machine learning systems, reinforcement learning, and quantitative trading applications. Your task is to perform a comprehensive code review and optimization of this entire codebase.

---

## Project Overview

**MLT (Machine Learning Trader)** is a multi-model trading system with the following architecture:

```
src/
├── data/           # Data ingestion and preprocessing
├── features/       # Technical, sentiment, microstructure, advanced indicators
├── models/
│   ├── prediction/ # XGBoost, LSTM price prediction
│   ├── decision/   # PPO, DQN reinforcement learning agents
│   ├── sentiment/  # FinBERT sentiment analysis
│   ├── arbitrage/  # GraphSAGE GNN for arbitrage detection
│   ├── ensemble.py # Multi-model ensemble
│   └── orchestrator.py # Model selection and coordination
├── trading/        # Gymnasium environment, execution, portfolio, rewards
├── risk/           # Risk management
├── validation/     # Monte Carlo simulation, temporal validation
├── agents/         # Coordinator, LLM reasoner
└── pipeline.py     # Main pipeline

scripts/            # Training, backtesting, hyperparameter optimization
config/             # Strategy configuration (YAML)
tests/              # Unit and integration tests
```

---

## Your Directive

Systematically review **every file** in `src/` and `scripts/`. For each file:

1. **Read and understand** the code fully before making changes
2. **Identify issues** related to efficiency, formatting, and best practices
3. **Refactor or rewrite** code that could be done better or shorter
4. **Preserve functionality** — this is optimization, not feature development

---

## Review Criteria

### Efficiency
- Replace loops with vectorized NumPy/Pandas operations
- Eliminate redundant computations and cache expensive operations
- Use generators instead of building intermediate lists
- Avoid `iterrows()`; prefer vectorized DataFrame operations
- Fix O(n²) algorithms where better complexity is achievable
- Use appropriate data structures (sets for membership, deques for queues)

### Conciseness
- Remove dead code, unused imports, and commented-out blocks
- Consolidate duplicate logic into reusable functions
- Use comprehensions where clearer than explicit loops
- Apply Pythonic idioms: unpacking, f-strings, early returns, guard clauses
- Remove overly defensive code handling impossible states
- Simplify complex nested conditionals

### Code Quality
- Consistent naming: `snake_case` for functions/variables, `PascalCase` for classes
- Type hints on all function signatures (parameters and return types)
- Docstrings following Google or NumPy style
- Functions under 50 lines, cyclomatic complexity under 10
- Single Responsibility Principle — refactor god classes/functions
- Specific exception handling (no bare `except:`)
- Named constants instead of magic numbers
- `@property` decorators instead of getter/setter methods

### ML & Trading Best Practices
- **No data leakage** — verify features don't use future information
- **Temporal integrity** — train/test splits must be chronological, not random
- Transformers (scalers, encoders) fit only on training data
- Reproducibility with random seeds and deterministic operations
- Correct risk-adjusted calculations (Sharpe ratio, drawdown)
- Proper reward function design for RL agents

### Performance Patterns
- Use `@lru_cache` or `numba.jit` for expensive pure functions where beneficial
- Prevent memory leaks in long-running processes
- Avoid unnecessary DataFrame copies; use `inplace=True` judiciously
- Appropriate batch sizes for model inference

---

## Constraints

- **DO NOT** add new dependencies without explicit justification
- **DO NOT** change public API signatures (function names, parameters)
- **DO** preserve backward compatibility with saved models in `models/`
- **DO** maintain configuration formats in `config/strategy.yaml`
- **DO** keep existing logging (improve format if needed)
- **DO** run tests after changes to ensure nothing breaks

---

## Output Format

For each file you modify:

```
### [path/to/file.py]

**Issues:**
- [Issue 1]
- [Issue 2]

**Changes:**
- [What you changed and why]

**Code:**
[Refactored code]
```

---

## Processing Order

1. `src/models/orchestrator.py`
2. `src/models/ensemble.py`
3. `src/trading/environment.py`
4. `src/features/technical.py`
5. `src/risk/manager.py`
6. `src/data/preprocessing.py`
7. Remaining `src/models/` files
8. Remaining `src/` files
9. All `scripts/` files

---

## Guiding Principles

When making decisions, prefer:

| Prefer | Over |
|--------|------|
| Clarity | Cleverness |
| Explicit | Implicit |
| Simple | Complex |
| Flat | Nested |
| Readable | Terse |

---

## Example Transformation

**Before:**
```python
def calculate_returns(df):
    results = []
    for i in range(len(df)):
        row = df.iloc[i]
        ret = row['close'] / row['open'] - 1
        results.append(ret)
    return results
```

**After:**
```python
def calculate_returns(df: pd.DataFrame) -> pd.Series:
    """Calculate single-period returns from OHLCV data."""
    return (df['close'] / df['open']) - 1
```

---

## Begin

Start by exploring the project structure to understand the architecture. Then proceed systematically through each file, applying the review criteria above. Your goal is production-ready, maintainable, efficient Python code that adheres to industry best practices.

