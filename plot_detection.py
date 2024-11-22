import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

total = 1000

# Data
resolutions = ['80x60', '160x120', '320x240', '640x480']
median_values = [1000, 959, 283, 342]
without_median_values = [1000, 958, 283, 342]

# Normalize the values to be out of 100
median_values = [(value / total) * 100 for value in median_values]
without_median_values = [(value / total) * 100 for value in without_median_values]

# Initialize Seaborn
sns.set(style="whitegrid")

# Create a DataFrame for easier plotting with Seaborn
data = pd.DataFrame({
    'Resolution': resolutions * 2,
    'Detection Value': median_values + without_median_values,
    'Type': ['With Median'] * len(resolutions) + ['Without Median'] * len(resolutions)
})

# Plotting
plt.figure(figsize=(12, 8))
palette = sns.color_palette("husl", 2)  # Use a more sophisticated color palette
sns.lineplot(data=data, x='Resolution', y='Detection Value', hue='Type', marker='o', linewidth=2.5, palette=palette)

# Adding titles and labels
plt.title('Missed Detection with and without Median Filter', fontsize=20, weight='bold')
plt.xlabel('Resolution', fontsize=16, weight='bold')
plt.ylabel('Detection Value (out of 100)', fontsize=16, weight='bold')

# Enhancing the plot visually
plt.xticks(fontsize=14, weight='bold')
plt.yticks(fontsize=14, weight='bold')
plt.legend(title='Type', title_fontsize='15', fontsize='13', loc='upper right', frameon=True, shadow=True)
plt.grid(True, linestyle='--', linewidth=0.7)
plt.tight_layout()

# Show plot
plt.show()
