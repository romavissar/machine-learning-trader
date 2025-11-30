from xgboost import XGBClassifier
import joblib


class PriceDirectionModel:
    def __init__(self, **kwargs):
        params = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "num_class": 3,
            "tree_method": "hist",
        }
        params.update(kwargs)
        self._params = params
        self.model = XGBClassifier(**params)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, path):
        joblib.dump({"params": self._params, "model": self.model}, path)

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        obj = cls(**data.get("params", {}))
        obj.model = data["model"]
        return obj

