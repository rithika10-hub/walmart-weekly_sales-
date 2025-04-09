import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import SalesANN

def evaluate_model(model_path="models/sales_ann_model.pth"):
    # Load test data
    transformer = DataTransformation()
    X_train, X_test, y_train, y_test = transformer.run()

    input_size = X_test.shape[1]
    model = SalesANN(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Predict
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        predictions = model(X_test_tensor)

    preds = predictions.numpy()
    true = y_test.values.reshape(-1, 1)

    # Metrics
    mse = mean_squared_error(true, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true, preds)

    print("\n📊 Model Evaluation Metrics (Re-Evaluated):")
    print(f"  ✅ MSE:  {mse:.2f}")
    print(f"  📉 RMSE: {rmse:.2f}")
    print(f"  📍 MAE:  {mae:.2f}")

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(true[:100], label="Actual", marker='o')
    plt.plot(preds[:100], label="Predicted", marker='x')
    plt.title("📈 Actual vs Predicted Weekly Sales")
    plt.xlabel("Samples")
    plt.ylabel("Sales")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    evaluate_model()
