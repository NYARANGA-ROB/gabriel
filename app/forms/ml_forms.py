"""Machine learning training forms."""

from flask_wtf import FlaskForm
from wtforms import BooleanField, FloatField, IntegerField, SelectField, StringField, SubmitField
from wtforms.validators import DataRequired, NumberRange, Optional


class TrainModelForm(FlaskForm):
    """Configure and launch model training."""

    processed_dataset_id = SelectField(
        "Processed dataset",
        validators=[DataRequired()],
        coerce=int,
    )
    model_type = SelectField(
        "Model type",
        validators=[DataRequired()],
        coerce=str,
    )
    use_smote_data = BooleanField("Use SMOTE training data (if available)", default=False)

    # Logistic Regression
    lr_c = FloatField("C", default=1.0, validators=[Optional(), NumberRange(min=0.0001)])
    lr_max_iter = IntegerField("Max iterations", default=1000, validators=[Optional(), NumberRange(min=50)])

    # Decision Tree
    dt_max_depth = IntegerField("Max depth", default=6, validators=[Optional(), NumberRange(min=1)])
    dt_min_samples_split = IntegerField(
        "Min samples split", default=2, validators=[Optional(), NumberRange(min=2)]
    )

    # Random Forest
    rf_n_estimators = IntegerField("Estimators", default=100, validators=[Optional(), NumberRange(min=10)])
    rf_max_depth = IntegerField("Max depth", default=8, validators=[Optional(), NumberRange(min=1)])

    # XGBoost
    xgb_n_estimators = IntegerField("Estimators", default=100, validators=[Optional(), NumberRange(min=10)])
    xgb_max_depth = IntegerField("Max depth", default=5, validators=[Optional(), NumberRange(min=1)])
    xgb_learning_rate = FloatField("Learning rate", default=0.1, validators=[Optional(), NumberRange(min=0.01)])

    # Neural Network
    ann_hidden_layers = StringField("Hidden layers", default="64,32", validators=[Optional()])
    ann_max_iter = IntegerField("Max epochs", default=300, validators=[Optional(), NumberRange(min=50)])
    ann_learning_rate = FloatField(
        "Learning rate", default=0.001, validators=[Optional(), NumberRange(min=0.00001)]
    )

    submit = SubmitField("Train model")

    def extract_parameters(self) -> dict:
        """Map form values to estimator parameters for the selected model."""
        model_type = self.model_type.data
        if model_type == "logistic_regression":
            return {"C": self.lr_c.data, "max_iter": self.lr_max_iter.data}
        if model_type == "decision_tree":
            return {
                "max_depth": self.dt_max_depth.data,
                "min_samples_split": self.dt_min_samples_split.data,
            }
        if model_type == "random_forest":
            return {
                "n_estimators": self.rf_n_estimators.data,
                "max_depth": self.rf_max_depth.data,
            }
        if model_type == "xgboost":
            return {
                "n_estimators": self.xgb_n_estimators.data,
                "max_depth": self.xgb_max_depth.data,
                "learning_rate": self.xgb_learning_rate.data,
            }
        if model_type == "neural_network":
            return {
                "hidden_layer_sizes": self.ann_hidden_layers.data,
                "max_iter": self.ann_max_iter.data,
                "learning_rate_init": self.ann_learning_rate.data,
            }
        return {}

    def apply_parameters(self, model_type: str, parameters: dict) -> None:
        """Populate form fields from a stored parameter set."""
        self.model_type.data = model_type
        if model_type == "logistic_regression":
            self.lr_c.data = parameters.get("C", 1.0)
            self.lr_max_iter.data = parameters.get("max_iter", 1000)
        elif model_type == "decision_tree":
            self.dt_max_depth.data = parameters.get("max_depth", 6)
            self.dt_min_samples_split.data = parameters.get("min_samples_split", 2)
        elif model_type == "random_forest":
            self.rf_n_estimators.data = parameters.get("n_estimators", 100)
            self.rf_max_depth.data = parameters.get("max_depth", 8)
        elif model_type == "xgboost":
            self.xgb_n_estimators.data = parameters.get("n_estimators", 100)
            self.xgb_max_depth.data = parameters.get("max_depth", 5)
            self.xgb_learning_rate.data = parameters.get("learning_rate", 0.1)
        elif model_type == "neural_network":
            hidden = parameters.get("hidden_layer_sizes", "64,32")
            if isinstance(hidden, (list, tuple)):
                hidden = ",".join(str(size) for size in hidden)
            self.ann_hidden_layers.data = hidden
            self.ann_max_iter.data = parameters.get("max_iter", 300)
            self.ann_learning_rate.data = parameters.get("learning_rate_init", 0.001)
