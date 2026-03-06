import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, roc_auc_score


def load_data(path):
    df = pd.read_csv(path)
    return df


def prepare_features(df):

    features = df[
        [
            'price',
            'freight_value',
            'delivery_days',
            'shipping_days',
            'product_weight_g',
            'product_volume',
            'payment_installments',
            'seller_state',
        ]
    ]

    target = df['late_delivery']

    features = pd.get_dummies(
        features,
        columns=['seller_state'],
        drop_first=True
    )

    features = features.fillna(features.median(numeric_only=True))

    return features, target


def train_models(X_train, y_train):

    log_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        class_weight="balanced"
    )

    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3
    )

    log_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)

    return log_model, rf_model, gb_model


def build_ensemble(log_model, rf_model, gb_model, X_train, y_train):

    voting_model = VotingClassifier(
        estimators=[
            ('log', log_model),
            ('rf', rf_model),
            ('gb', gb_model)
        ],
        voting='soft'
    )

    voting_model.fit(X_train, y_train)

    return voting_model


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))


def main():

    df = load_data("data/marketplace_analytics.csv")

    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    log_model, rf_model, gb_model = train_models(X_train, y_train)

    ensemble_model = build_ensemble(
        log_model,
        rf_model,
        gb_model,
        X_train,
        y_train
    )

    evaluate_model(ensemble_model, X_test, y_test)

    joblib.dump(
        ensemble_model,
        "models/late_delivery_model.pkl"
    )


if __name__ == "__main__":
    main()