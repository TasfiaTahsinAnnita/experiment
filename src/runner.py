import yaml
import numpy as np
from data_loader import load_data
from models import get_model
from explainers import shap_explain
from perturbations import add_noise
from metrics import spearman_stability, top_k_jaccard

def run_experiment(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    results = []

    for seed in config["seeds"]:
        X_train, X_test, y_train, y_test = load_data(
            config["data_path"],
            config["target"],
            seed
        )

        model = get_model(config["model"], seed)
        model.fit(X_train, y_train)

        base_exp = shap_explain(model, X_train, X_test)

        X_test_noisy = add_noise(X_test, config["noise"])
        noisy_exp = shap_explain(model, X_train, X_test_noisy)

        stability = spearman_stability(base_exp, noisy_exp)
        jaccard = top_k_jaccard(base_exp, noisy_exp)

        results.append((stability, jaccard))

        print(f"Seed {seed}: Spearman={stability:.3f}, Jaccard={jaccard:.3f}")

    print("Average Stability:", np.mean([r[0] for r in results]))
    print("Average Jaccard:", np.mean([r[1] for r in results]))

if __name__ == "__main__":
    run_experiment("configs/experiment.yaml")
