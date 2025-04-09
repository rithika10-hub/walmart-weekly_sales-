import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.components.data_transformation import DataTransformation

# Define a simple ANN model
class SalesANN(nn.Module):
    def __init__(self, input_size):
        super(SalesANN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

# Train the model
def train_model(X_train, y_train, X_test, y_test, epochs=100, lr=0.001):
    input_size = X_train.shape[1]
    model = SalesANN(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"🧪 Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32)
        predictions = model(X_test_tensor)

        preds = predictions.numpy()
        true = y_test_tensor.numpy()

        mse = mean_squared_error(true, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true, preds)

        print("\n📊 Model Evaluation Metrics:")
        print(f"  ✅ MSE:  {mse:.2f}")
        print(f"  📉 RMSE: {rmse:.2f}")
        print(f"  📍 MAE:  {mae:.2f}")

    # ✅ Make sure models directory exists
    os.makedirs("models", exist_ok=True)

    # ✅ Save the trained model
    torch.save(model.state_dict(), "models/sales_ann_model.pth")
    print("✅ Model saved to models/sales_ann_model.pth")

if __name__ == "__main__":
    transformer = DataTransformation()
    X_train, X_test, y_train, y_test = transformer.run()
    train_model(X_train, y_train, X_test, y_test, epochs=100, lr=0.001)
