# src/modules/model_trainer.py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

class ModelTrainer:
    def __init__(self, model_name="logistic_regression"):
        self.model_name = model_name
        self.model = self._init_model()

    def _init_model(self):
        if self.model_name == "logistic_regression":
            return LogisticRegression(max_iter=1000)
        elif self.model_name == "random_forest":
            return RandomForestClassifier()
        elif self.model_name == "gbm":
            return GradientBoostingClassifier()
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def train(self, X_train, y_train, param_grid=None, search_type="grid"):
        if param_grid:
            if search_type == "grid":
                search = GridSearchCV(self.model, param_grid, cv=2, scoring="roc_auc")
            else:
                search = RandomizedSearchCV(self.model, param_grid, cv=2, scoring="roc_auc", n_iter=10)
            search.fit(X_train, y_train)
            self.model = search.best_estimator_
            return search
        else:
            self.model.fit(X_train, y_train)
            return self.model

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:,1]
