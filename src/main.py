from . import data_handler, config
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import optuna
from optuna.samplers import TPESampler
from dvclive.optuna import DVCLiveCallback
import joblib

train_X, train_y = data_handler.get_embedded_data(config.TRAIN_DATASET_PATH)
test_X, test_y = data_handler.get_embedded_data(config.TEST_DATASET_PATH)
val_X, val_y = data_handler.get_embedded_data(config.VAL_DATASET_PATH)


def print_study_info(_study):
    print(f"Finished trials: {len(_study.trials)}")
    print(f"Best trial:{_study.best_trial.value}")
    for key, value in _study.best_trial.params.items():
        print(f"{key}: {value}")


def objective(trial):
    model = RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 2000, step=50),
        criterion=trial.suggest_categorical("criterion", ['gini', 'entropy', 'log_loss']),
        max_depth=trial.suggest_int("max_depth", 2, 15, step=1),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 10, step=1),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10, step=1),
        random_state=1337
    )

    model.fit(train_X, train_y)
    pred_y = model.predict(test_X)
    score = accuracy_score(test_y, pred_y.flatten().tolist())
    trial.set_user_attr("accuracy_score", score)
    return score


sampler = TPESampler(seed=1337)
study = optuna.create_study(study_name="RandomForest", direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=75, timeout=10000, callbacks=[DVCLiveCallback()])

joblib.dump(study, "reports/study.pkl")
study = joblib.load("reports/study.pkl")
print_study_info(study)

model = RandomForestClassifier(**study.best_trial.params, verbose=False)
model.fit(train_X, train_y)
joblib.dump(model, "models/latest.joblib")

y_pred = model.predict(test_X)
report = classification_report(test_y, y_pred.flatten().tolist())
print(f"Results with best params:", report)
