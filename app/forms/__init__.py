"""WTForms definitions."""

from app.forms.auth_forms import LoginForm, ProfileForm, RegistrationForm
from app.forms.dataset_forms import DatasetUploadForm
from app.forms.eda_forms import EDAForm
from app.forms.history_forms import PredictionHistoryFilterForm
from app.forms.ml_forms import TrainModelForm
from app.forms.prediction_forms import PatientPredictionForm
from app.forms.preprocessing_forms import PreprocessingForm

__all__ = [
    "DatasetUploadForm",
    "EDAForm",
    "LoginForm",
    "PatientPredictionForm",
    "PredictionHistoryFilterForm",
    "PreprocessingForm",
    "ProfileForm",
    "RegistrationForm",
    "TrainModelForm",
]
