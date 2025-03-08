from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def get_estimator():
    """Define preprocessing and a regression model for legal assistance cost prediction."""
    
    numerical_features = [
        "Nombre foyers NDUR", "Nombre foyers ARS", "Nombre personnes ARS",
        "Montant total ARS", "Nombre foyers ASF", "Nombre foyers AJPA",
        "Nombre foyers AJPP", "Nombre foyers ALF", "Nombre foyers RSO",
        "Nombre personnes RSO", "Montant total RSO", "Nombre foyers NDURINT", "Nombre foyers AMI", "Montant total AMI",
        "Montant total ADI", "Nombre foyers CDI", "Montant total CDI"
    ]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
        ],
        remainder="drop"
    )

    model = XGBRegressor(n_estimators=1000, random_state=42, objective='reg:squarederror')

    pipeline = make_pipeline(preprocessor, model)

    return pipeline
