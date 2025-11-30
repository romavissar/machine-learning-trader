"""LLM Reasoning Agent for market analysis and decision support."""
import json
import re
from typing import Optional
from transformers import pipeline


class LLMReasoningAgent:
    """Uses LLMs to provide market reasoning and trading suggestions."""

    MODELS = {"finbert": "ProsusAI/finbert", "opt": "facebook/opt-1.3b", "gpt4": "gpt-4"}

    def __init__(self, model_choice: str = "finbert"):
        self.model_choice = model_choice
        model_id = self.MODELS.get(model_choice, self.MODELS["finbert"])
        if model_choice == "finbert":
            self.pipe = pipeline("sentiment-analysis", model=model_id)
        elif model_choice == "opt":
            self.pipe = pipeline("text-generation", model=model_id, max_new_tokens=128)
        else:
            self.pipe = None  # GPT-4 requires OpenAI client

    def analyze_market_context(self, news: list, technicals: dict, sentiment: dict) -> str:
        """Synthesize market data into actionable reasoning."""
        prompt = f"News: {news[:3]}\nTechnicals: RSI={technicals.get('rsi')}, MACD={technicals.get('macd')}\nSentiment: {sentiment}"
        if self.model_choice == "finbert":
            results = self.pipe(news[:5] if news else ["neutral"])
            scores = [f"{r['label']}({r['score']:.2f})" for r in results]
            return f"Sentiment: {', '.join(scores)}. Technicals: {technicals}"
        if self.model_choice == "opt":
            out = self.pipe(f"Market analysis:\n{prompt}\nConclusion:")[0]["generated_text"]
            return out.split("Conclusion:")[-1].strip()[:300]
        return f"Analysis pending: {prompt[:200]}"

    def suggest_action(self, context: str, current_position: Optional[dict] = None) -> dict:
        """Suggest trading action with confidence score."""
        if self.model_choice == "finbert":
            result = self.pipe(context[:512])[0]
            action_map = {"positive": "BUY", "negative": "SELL", "neutral": "HOLD"}
            return {"action": action_map.get(result["label"], "HOLD"), "confidence": round(result["score"], 2)}
        if self.model_choice == "opt":
            prompt = f"Position: {current_position}\nContext: {context[:256]}\nAction (BUY/SELL/HOLD) with confidence:"
            raw = self.pipe(prompt)[0]["generated_text"]
            return self._parse_action(raw)
        return {"action": "HOLD", "confidence": 0.5, "reason": "model_unavailable"}

    def _parse_action(self, raw: str) -> dict:
        """Reliably parse LLM output to structured action dict."""
        try:
            if match := re.search(r'\{[^}]+\}', raw):
                parsed = json.loads(match.group())
                if "action" in parsed:
                    return {"action": parsed["action"].upper(), "confidence": float(parsed.get("confidence", 0.5))}
        except (json.JSONDecodeError, ValueError):
            pass
        action = "BUY" if "buy" in raw.lower() else ("SELL" if "sell" in raw.lower() else "HOLD")
        conf_match = re.search(r'(\d\.\d+)', raw)
        return {"action": action, "confidence": float(conf_match.group(1)) if conf_match else 0.5}

    def explain_decision(self, action: str, factors: dict) -> str:
        """Generate human-readable explanation for trading decision."""
        factor_strs = [f"{k.replace('_', ' ')}: {v}" for k, v in list(factors.items())[:5]]
        rationale = {"BUY": "bullish signals detected", "SELL": "bearish indicators present", "HOLD": "mixed signals"}
        return f"Recommendation: {action.upper()} — {rationale.get(action.upper(), 'awaiting confirmation')}. Factors: {', '.join(factor_strs)}."

