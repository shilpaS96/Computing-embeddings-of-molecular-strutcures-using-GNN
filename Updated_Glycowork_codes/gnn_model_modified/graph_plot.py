import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from pathlib import Path

# Check what backend matplotlib is using
print("Backend:", matplotlib.get_backend())

'''
# Example data
test_losses = [0.1, 0.08, 0.05, 0.03]
epoch = len(test_losses) - 1

test_acc = [0.122, 0.98, 0.05, 0.03]

# Create subplots
fig, ax = plt.subplots(nrows=2, ncols=1)

# Plot on the first subplot
plt.subplot(2, 1, 1)
plt.plot(range(epoch + 1), test_losses)
plt.title("Test Losses")
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.subplot(2, 1, 2)
plt.plot(range(epoch + 1), test_acc)
plt.title("Test Accuracies")
plt.xlabel("Epochs")
plt.ylabel("accuracy")

# Display the plot
plt.tight_layout()
# plt.show()  
# Save the plot to a file since the display is not available
# plt.savefig('output_plot.png')

model_path = Path('plots')
model_path.mkdir(parents = True, exist_ok=True)
conv_type = 'SageConv'

output_path = f'./plots/output_{conv_type}_plot.png'  # Relative path

# Save the plot to the specified path
plt.savefig(output_path)
print("Plot saved as 'output_plot.png'")
'''


import matplotlib.pyplot as plt

# Example data for two models over epochs
epochs_model1 = range(1, 11)  # 10 epochs for model 1
model1_accuracy = [0.7, 0.72, 0.75, 0.78, 0.80, 0.82, 0.84, 0.85, 0.87, 0.89]

epochs_model2 = range(1, 16)  # 15 epochs for model 2
model2_accuracy = [0.68, 0.70, 0.73, 0.76, 0.78, 0.79, 0.82, 0.83, 0.85, 0.88, 0.89, 0.90, 0.91, 0.91, 0.92]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs_model1, model1_accuracy, label="Model 1 Accuracy", color='b')
plt.plot(epochs_model2, model2_accuracy, label="Model 2 Accuracy", color='r')

# Labels and Title
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison (Different Epochs)")
plt.legend()

# Display Plot
#plt.show()

plt.savefig('output_plot.png')

model_path = Path('plots')
model_path.mkdir(parents = True, exist_ok=True)
conv_type = 'SageConv'

output_path = f'./plots/test_output_{conv_type}_plot.png'  # Relative path

# Save the plot to the specified path
plt.savefig(output_path)
print("Plot saved as 'output_plot.png'")
