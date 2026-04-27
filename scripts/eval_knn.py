import argparse
from pathlib import Path
import csv
import h5py
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


def load_h5(path):
    with h5py.File(path, "r") as f:
        print(f"{path.name} keys: {list(f.keys())}")
        embeddings = f["embeddings"][:]
        labels = f["labels"][:]
    return embeddings, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_dir", required=True)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 10, 20])
    parser.add_argument("--output_csv", default="results/baseline_knn_val.csv")
    args = parser.parse_args()

    emb_dir = Path(args.embeddings_dir)

    train_files = sorted(emb_dir.glob("*embeddings_train.h5"))
    val_files = sorted(emb_dir.glob("*embeddings_val.h5"))

    if len(train_files) != 1 or len(val_files) != 1:
        raise ValueError(
            f"Expected exactly one train and one val embedding file. "
            f"Found train={train_files}, val={val_files}"
        )

    train_path = train_files[0]
    val_path = val_files[0]

    print(f"Loading train embeddings from: {train_path}")
    print(f"Loading val embeddings from: {val_path}")

    X_train, y_train = load_h5(train_path)
    X_val, y_val = load_h5(val_path)

    print(f"Raw X_train shape: {X_train.shape}")
    print(f"Raw X_val shape: {X_val.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_val shape: {y_val.shape}")

    # Match the linear-probe evaluation: non-learned global average pooling.
    if X_train.ndim == 4:
        X_train = X_train.mean(axis=(-2, -1))
        X_val = X_val.mean(axis=(-2, -1))

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)

    # Normalize labels using train statistics only.
    label_scaler = StandardScaler()
    y_train_norm = label_scaler.fit_transform(y_train)
    y_val_norm = label_scaler.transform(y_val)

    # Normalize features for distance-based kNN.
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_val_scaled = feature_scaler.transform(X_val)

    print("\nBaseline kNN validation results")
    print("--------------------------------")
    
    results = []

    for k in args.ks:
        knn = KNeighborsRegressor(n_neighbors=k, weights="distance")
        knn.fit(X_train_scaled, y_train_norm)
        preds = knn.predict(X_val_scaled)

        mse_alpha = mean_squared_error(y_val_norm[:, 0], preds[:, 0])
        mse_zeta = mean_squared_error(y_val_norm[:, 1], preds[:, 1])
        mse_avg = mean_squared_error(y_val_norm, preds)
        
        results.append({
             "k": k,
             "alpha_mse": mse_alpha,
             "zeta_mse": mse_zeta,
             "avg_mse": mse_avg,
        })

        print(f"k={k}")
        print(f"  alpha MSE: {mse_alpha:.6f}")
        print(f"  zeta  MSE: {mse_zeta:.6f}")
        print(f"  avg   MSE: {mse_avg:.6f}")

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["k", "alpha_mse", "zeta_mse", "avg_mse"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved kNN results to: {output_path}")

if __name__ == "__main__":
    main()
