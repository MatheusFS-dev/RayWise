# ---------------------------------------------------------------------------- #
#                      Authored by Matheus Ferreira Silva                      #
#                           github.com/MatheusFS-dev                           #
# ---------------------------------------------------------------------------- #

import optuna

def load_study_from_db(db_path: str, study_name: str) -> optuna.study.Study:
    """
    Loads an Optuna study from a SQLite database.

    Args:
        db_path (str): Path to the SQLite database file.
        study_name (str): Name of the study to retrieve.

    Returns:
        optuna.study.Study: Loaded Optuna study object.
    """
    storage = f"sqlite:///{db_path}"
    study = optuna.load_study(study_name=study_name, storage=storage)
    return study

def get_best_trial(study: optuna.study.Study) -> dict:
    """
    Retrieves the best trial from the given Optuna study.

    Args:
        study (optuna.study.Study): The Optuna study object.

    Returns:
        dict: A dictionary containing the best trial's results, including:
            - trial_number (int): The best trial number.
            - value (float): The best objective value.
            - params (dict): The best hyperparameters.
            - user_attrs (dict): User-defined attributes.
            - system_attrs (dict): System-defined attributes.
    """
    best_trial = study.best_trial
    return {
        "trial_number": best_trial.number,
        "value": best_trial.value,
        "params": best_trial.params,
        "user_attrs": best_trial.user_attrs,
        "system_attrs": best_trial.system_attrs
    }

def display_best_trial(best_trial: dict):
    """
    Displays the best trial's results in a structured format.

    Args:
        best_trial (dict): Dictionary containing the best trial's results.
    """
    print("\nBest Trial Details")
    print("-" * 40)
    print(f"Trial Number: {best_trial['trial_number']}")
    print(f"Objective Value: {best_trial['value']}")
    print("Best Parameters:")
    for param, value in best_trial["params"].items():
        print(f"  {param}: {value}")
    if best_trial["user_attrs"]:
        print("User Attributes:")
        for attr, value in best_trial["user_attrs"].items():
            print(f"  {attr}: {value}")
    if best_trial["system_attrs"]:
        print("System Attributes:")
        for attr, value in best_trial["system_attrs"].items():
            print(f"  {attr}: {value}")

def main():
    """
    Main function to fetch the best trial from an Optuna study.
    """
    db_path = "/media/matheus/SSD/Projects/Fluid-Antenna-Channel-Estimation/felipe/results/lstm_10_ports/study_SNR_events_W1.0_U1_N100_kappa1.0e-16_mu1.0_m50.0.db"
    study_name = "study_SNR_events_W1.0_U1_N100_kappa1.0e-16_mu1.0_m50.0"

    try:
        study = load_study_from_db(db_path, study_name)
        best_trial = get_best_trial(study)
        display_best_trial(best_trial)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
