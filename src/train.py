"""
Training Pipeline for Sleep Stage Classification Models
Author: Vaibhav Mistry

Executes end-to-end dataset loading, class imbalance compensation,
model compilation, training with callbacks, and performance evaluation.
"""

import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from preprocess import load_multi_subject_dataset
from models import build_cnn_bilstm, build_cnn_baseline
from evaluate import evaluate_predictions, plot_confusion_matrix, plot_learning_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Train Sleep Stage Classification Deep Learning Models")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to directory with Sleep-EDF files")
    parser.add_argument("--model_type", type=str, choices=["cnn_bilstm", "cnn_baseline"], default="cnn_bilstm",
                        help="Model architecture to train")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save models and figures")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n==========================================")
    print(f"Loading Polysomnography Dataset from: {args.data_dir}")
    print(f"==========================================")
    X, y = load_multi_subject_dataset(args.data_dir)
    print(f"Dataset Shape: {X.shape}, Total Epochs: {len(y)}")

    # Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train set: {X_train.shape[0]} epochs | Test set: {X_test.shape[0]} epochs")

    # Handle Severe Sleep Stage Class Imbalance with Balanced Weights
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))
    print(f"Calculated Class Weights: {class_weight_dict}")

    # Build Model
    input_shape = (X_train.shape[1], X_train.shape[2])
    if args.model_type == "cnn_bilstm":
        model = build_cnn_bilstm(input_shape=input_shape, num_classes=5, learning_rate=args.lr)
    else:
        model = build_cnn_baseline(input_shape=input_shape, num_classes=5, learning_rate=args.lr)

    model.summary()

    # Callbacks
    model_save_path = os.path.join(args.output_dir, f"{args.model_type}_best.h5")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-5),
        ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    print(f"\nTraining {args.model_type} for {args.epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks
    )

    # Save Learning Curves
    plot_learning_curves(
        history,
        title=f"{args.model_type.upper()} Performance",
        save_path=os.path.join(args.output_dir, f"{args.model_type}_learning_curves.png")
    )

    # Evaluation
    print("\nRunning Evaluation on Test Set...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    metrics = evaluate_predictions(y_test, y_pred)
    print(f"\nFinal Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Final Weighted F1 Score: {metrics['weighted_f1']:.4f}")
    print(f"Final Macro F1 Score: {metrics['macro_f1']:.4f}")
    print("\nClassification Report:\n", metrics['report'])

    # Save Confusion Matrix
    plot_confusion_matrix(
        y_test, y_pred,
        title=f"{args.model_type.upper()} Confusion Matrix",
        save_path=os.path.join(args.output_dir, f"{args.model_type}_confusion_matrix.png")
    )
    print(f"Results and plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
