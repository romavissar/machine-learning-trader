# MLT Trading Strategy: Multi-Model Architecture

## Overview

This document describes how the MLT system's models work together to generate trading signals. The architecture follows a **layered decision framework** where models can operate independently or in concert, with selection based on market conditions, confidence levels, and risk constraints.

---

## Model Inventory

| Model | Category | Purpose | Output |
|-------|----------|---------|--------|
| **XGBoost** | Prediction | Price direction classification (Up/Down/Hold) | Class probabilities [0,1,2] |
| **LSTM** | Prediction | Sequence-based price prediction | Continuous price forecast |
| **PPO** | Decision | Reinforcement learning position sizing | Action (buy/sell/hold) |
| **DQN** | Decision | Q-learning based trade execution | Action (discrete) |
| **FinBERT** | Sentiment | Financial news sentiment analysis | {positive, negative, neutral} scores |
| **GraphSAGE** | Arbitrage | Cross-exchange arbitrage detection | Opportunity cycles with profit % |
| **Ensemble** | Meta | Combines XGBoost, LightGBM, RandomForest | Consensus signal with confidence |

---

## Architecture: Layered Decision Framework

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LAYER 1: SIGNAL GENERATION                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│   │   XGBoost   │    │    LSTM     │    │   FinBERT   │    │  GraphSAGE  │ │
│   │  Direction  │    │  Forecast   │    │  Sentiment  │    │  Arbitrage  │ │
│   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘ │
│          │                  │                  │                  │        │
│          ▼                  ▼                  ▼                  ▼        │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                    ENSEMBLE / META-LEARNER                          │  │
│   │          (VotingEnsemble / StackingEnsemble / Confidence)           │  │
│   └──────────────────────────────┬──────────────────────────────────────┘  │
│                                  │                                         │
└──────────────────────────────────┼─────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LAYER 2: DECISION/EXECUTION                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                      PPO / DQN AGENT                                 │  │
│   │   Receives: Signal + Features + Portfolio State                     │  │
│   │   Outputs:  Position Size + Timing                                  │  │
│   └──────────────────────────────┬──────────────────────────────────────┘  │
│                                  │                                         │
└──────────────────────────────────┼─────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LAYER 3: RISK MANAGEMENT                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│   │  Position   │    │  Stop-Loss  │    │  Volatility │    │   Drawdown  │ │
│   │   Limits    │    │  (ATR-based)│    │  Monitoring │    │    Check    │ │
│   └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘ │
│                                                                             │
│                          FINAL TRADE EXECUTION                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Best-Fit Model Selection

### Decision Framework

Model selection is determined by a **scoring system** that evaluates each model/combination against current market conditions:

```python
best_fit_score = (
    historical_accuracy * 0.30 +      # Out-of-sample performance
    regime_match * 0.25 +             # Alignment with current market regime
    confidence_level * 0.20 +         # Model's prediction confidence
    recency_weight * 0.15 +           # How recently model was retrained
    diversity_bonus * 0.10            # Ensemble disagreement premium
)
```

### Selection Criteria by Scenario

| Market Condition | Primary Model(s) | Rationale |
|-----------------|------------------|-----------|
| **Trending Market** (ADX > 25) | XGBoost + PPO | Tree models excel at capturing momentum; PPO optimizes entry/exit |
| **Ranging/Choppy** (ADX < 20) | Ensemble (Confidence) | Requires consensus; reduces false signals |
| **High Volatility** (Vol > 2σ) | LSTM + Risk Override | Sequential models better at regime shifts; tighter risk limits |
| **News-Driven Event** | FinBERT → XGBoost | Sentiment leads; prediction validates |
| **Cross-Exchange Divergence** | GraphSAGE (standalone) | Pure arbitrage opportunity; no prediction needed |
| **Low Confidence (all models < 60%)** | HOLD / Reduce Exposure | No clear signal = no action |

### Automated Selection Logic

```
IF arbitrage_opportunity.profit_pct > MIN_ARB_THRESHOLD:
    EXECUTE arbitrage (bypass prediction layer)
    
ELSE IF news_event_detected AND sentiment_confidence > 0.8:
    signal = FinBERT.analyze(news)
    IF signal.sentiment != "neutral":
        weight_sentiment = 0.6, weight_prediction = 0.4
        
ELSE IF market_regime == "trending":
    USE XGBoost (single model, high confidence threshold: 0.7)
    
ELSE:
    USE Ensemble (ConfidenceEnsemble with threshold: 0.5)
    REQUIRE minimum 2/3 model agreement
```

---

## Single vs Multiple Model Usage

### When to Use a SINGLE Model

| Scenario | Model | Threshold |
|----------|-------|-----------|
| Clear trend + high conviction | XGBoost | Confidence > 75% |
| Pure arbitrage opportunity | GraphSAGE | Profit > 0.1% after fees |
| RL-optimized execution | PPO | During trained market regime |
| Breaking news (< 1hr old) | FinBERT | Sentiment score > 0.85 |

**Advantages:**
- Lower latency (single inference)
- Clearer attribution for performance analysis
- Avoids "design by committee" signals

### When to Use MULTIPLE Models (Ensemble)

| Scenario | Configuration | Agreement Required |
|----------|--------------|-------------------|
| Uncertain/ranging markets | VotingEnsemble | ≥ 2/3 models agree |
| High-stakes positions | StackingEnsemble | Meta-model confidence > 0.6 |
| Model disagreement | ConfidenceEnsemble | Weighted by individual confidence |
| Regime uncertainty | Full stack + sentiment | All signals aligned |

**Ensemble Configurations:**

```yaml
# Conservative (High Agreement Required)
ensemble:
  type: voting
  voting_method: hard
  min_agreement: 0.67
  
# Balanced (Confidence-Weighted)
ensemble:
  type: confidence
  confidence_threshold: 0.5
  fallback: majority_vote
  
# Aggressive (Meta-Learning)
ensemble:
  type: stacking
  meta_model: logistic_regression
  use_probabilities: true
  passthrough_features: false
```

---

## Model Interaction Patterns

### Pattern 1: Sequential Pipeline (Default)

```
FinBERT → XGBoost → PPO → RiskManager → Execute
   │          │        │
   │          │        └── Position sizing
   │          └── Direction prediction
   └── Sentiment context feature
```

**Use When:** Normal market conditions, medium-term signals

### Pattern 2: Parallel Ensemble

```
     ┌── XGBoost ──┐
     │             │
Data ├── LightGBM ─┼── VotingEnsemble → PPO → Execute
     │             │
     └── RandomForest
```

**Use When:** Low individual model confidence, ranging markets

### Pattern 3: Gated Execution

```
                    ┌─ YES ─→ GraphSAGE → Execute Arbitrage
                    │
Arbitrage Check? ───┤
                    │
                    └─ NO ──→ Standard Pipeline
```

**Use When:** Always running in background; arbitrage takes priority

### Pattern 4: Sentiment Override

```
                         ┌─ Strong Signal ─→ Increase Position Weight
                         │
News Event → FinBERT ────┼─ Weak Signal ───→ Standard Pipeline
                         │
                         └─ Conflicting ───→ Reduce Position / HOLD
```

**Use When:** Major news events, earnings, regulatory announcements

---

## Risk Integration

### Pre-Trade Risk Checks

Every signal passes through the RiskManager before execution:

```python
def validate_trade(signal, portfolio_state, risk_limits):
    checks = {
        'position_limit': risk.check_position_limit(
            proposed_position, 
            max_position=risk_limits['max_position']
        ),
        'volatility_ok': not risk.monitor_volatility(
            recent_returns, 
            threshold=risk_limits['max_volatility']
        ),
        'drawdown_ok': risk.max_drawdown(equity_curve) < risk_limits['max_drawdown'],
        'halt_check': not risk.should_halt_trading(
            current_drawdown, 
            current_vol, 
            risk_limits
        ),
    }
    return all(checks.values()), checks
```

### Risk Limits by Model Type

| Model Type | Max Position | Stop-Loss | Max Drawdown | Notes |
|------------|-------------|-----------|--------------|-------|
| XGBoost (single) | 1.0 | 2× ATR | 15% | Standard limits |
| Ensemble (consensus) | 1.0 | 2× ATR | 15% | Same as single |
| PPO (RL) | 0.5 | 3× ATR | 10% | Tighter: RL can overfit |
| FinBERT (sentiment-only) | 0.3 | 1.5× ATR | 5% | Very conservative |
| GraphSAGE (arbitrage) | Per-opportunity | N/A | N/A | Risk-free by design |

### Dynamic Risk Adjustment

```python
# Adjust risk based on model confidence
if model_confidence < 0.6:
    effective_max_position *= 0.5  # Half position on low confidence
    
# Tighten during high volatility
if current_volatility > 2 * historical_volatility:
    stop_loss_multiplier = 1.5  # Tighter stop
    max_drawdown_limit *= 0.7   # Lower tolerance
    
# Regime-based adjustment
if market_regime == 'crisis':
    halt_trading = True  # Override all signals
```

---

## Validation Framework

### Temporal Validation (CRITICAL)

All models use **walk-forward validation** to prevent lookahead bias:

```
Timeline: ═══════════════════════════════════════════════════════►
                                                                  
Split 1:  [====TRAIN====]--gap--[TEST]                           
Split 2:  [======TRAIN======]--gap--[TEST]                       
Split 3:  [========TRAIN========]--gap--[TEST]                   
Split 4:  [==========TRAIN==========]--gap--[TEST]               

✓ Train data is ALWAYS before test data
✓ Gap (embargo) prevents feature leakage
✓ Expanding window captures all available history
```

### Validation Protocol

| Model | Validation Method | Retrain Frequency | Embargo Gap |
|-------|------------------|-------------------|-------------|
| XGBoost | Walk-forward (expanding) | 30 days | 1 hour |
| LSTM | Walk-forward (rolling) | 14 days | 6 hours |
| PPO | Episodic + holdout | 90 days | 24 hours |
| Ensemble | Nested CV + temporal | 30 days | 1 hour |
| FinBERT | Pre-trained (frozen) | Never | N/A |

### Performance Monitoring

```python
validation_metrics = {
    'out_of_sample_accuracy': 0.0,  # Must exceed 0.52 for XGBoost
    'out_of_sample_sharpe': 0.0,    # Must exceed 0.5 for deployment
    'regime_stability': 0.0,        # Consistent across market regimes
    'max_drawdown': 0.0,            # Must stay under limits
    'win_rate': 0.0,                # Track for reporting
}

# Model health check
def should_use_model(model_name, recent_performance):
    if recent_performance['sharpe_30d'] < 0:
        return False  # Disable underperforming model
    if recent_performance['max_drawdown_30d'] > LIMITS['max_drawdown']:
        return False  # Risk limit breach
    return True
```

---

## Operational Decision Tree

```
START
  │
  ▼
┌─────────────────────────────────────────┐
│ 1. Check for arbitrage opportunities    │
│    (GraphSAGE continuous monitoring)    │
└─────────────────┬───────────────────────┘
                  │
          Opportunity Found?
                  │
         YES     │     NO
          │      │      │
          ▼      │      ▼
    ┌─────────┐  │  ┌─────────────────────────────────────────┐
    │ Execute │  │  │ 2. Check for news events (FinBERT)     │
    │ Arb     │  │  └─────────────────┬───────────────────────┘
    └─────────┘  │                    │
                 │            Strong Sentiment?
                 │                    │
                 │           YES     │     NO
                 │            │      │      │
                 │            ▼      │      ▼
                 │      ┌─────────┐  │  ┌─────────────────────────────────────────┐
                 │      │ Weight  │  │  │ 3. Determine market regime (ADX, Vol)  │
                 │      │ Signal  │  │  └─────────────────┬───────────────────────┘
                 │      │ +60%    │  │                    │
                 │      └────┬────┘  │          Trending? │ Ranging?
                 │           │       │                    │
                 │           ▼       │      ┌─────────────┴─────────────┐
                 │    ┌─────────────────────┴──┐                        │
                 │    │                        │                        ▼
                 │    ▼                        ▼                  ┌───────────┐
                 │  ┌───────────┐        ┌───────────┐           │  Ensemble │
                 │  │  XGBoost  │        │  Ensemble │           │ Consensus │
                 │  │  (Single) │        │ (Voting)  │           │ Required  │
                 │  └─────┬─────┘        └─────┬─────┘           └─────┬─────┘
                 │        │                    │                       │
                 │        └────────────────────┴───────────────────────┘
                 │                             │
                 │                             ▼
                 │               ┌─────────────────────────────┐
                 │               │ 4. PPO position sizing       │
                 │               └─────────────┬───────────────┘
                 │                             │
                 │                             ▼
                 │               ┌─────────────────────────────┐
                 │               │ 5. Risk Manager validation   │
                 │               │    - Position limits         │
                 │               │    - Volatility check        │
                 │               │    - Drawdown check          │
                 │               └─────────────┬───────────────┘
                 │                             │
                 │                      Passes Risk?
                 │                             │
                 │                    YES     │     NO
                 │                     │      │      │
                 │                     ▼      │      ▼
                 │               ┌─────────┐  │  ┌─────────┐
                 │               │ EXECUTE │  │  │  HOLD   │
                 │               └─────────┘  │  └─────────┘
                 │                            │
                 └────────────────────────────┘
```

---

## Configuration

### Default Model Weights (strategy.yaml)

```yaml
model_selection:
  # Trending market weights
  trending:
    xgboost: 0.7
    lstm: 0.2
    sentiment: 0.1
    
  # Ranging market weights  
  ranging:
    xgboost: 0.33
    lightgbm: 0.33
    random_forest: 0.34
    
  # Event-driven weights
  news_event:
    sentiment: 0.6
    xgboost: 0.3
    lstm: 0.1

confidence_thresholds:
  single_model_required: 0.75
  ensemble_min_agreement: 0.67
  position_scaling_threshold: 0.60

risk_limits:
  max_position: 1.0
  max_drawdown: 0.15
  max_volatility: 0.05
  stop_loss_atr_multiplier: 2.0
```

---

## Summary

| Question | Answer |
|----------|--------|
| **How is best-fit decided?** | Scoring system: historical accuracy (30%) + regime match (25%) + confidence (20%) + recency (15%) + diversity (10%) |
| **Single or multiple models?** | Single for high-confidence trending markets; ensemble for uncertainty/ranging |
| **When is each model used?** | XGBoost (trending), Ensemble (ranging), FinBERT (news), GraphSAGE (arbitrage), PPO (sizing) |
| **How is risk considered?** | Every trade passes through RiskManager; position limits, stop-loss, volatility, and drawdown checks |
| **How is validation done?** | Walk-forward temporal validation only; no shuffled CV; mandatory embargo gaps |

---

*Last updated: November 2025*

