import matplotlib.pyplot as plt

def plot_metric(metric_data, title, xlabel, ylabel, filename='metric.png'):
    plt.figure()
    for descriptor, data in metric_data.items():
        if len(data['transformation']) == len(data['values']):
            plt.plot(data['transformation'], data['values'], label=descriptor)
        else:
            print(f"Skipping plot for {descriptor} due to mismatched data lengths")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

def plot_all_metrics(metrics_data, descriptors, filename='repeat.png'):
    plot_metric(
        {desc: {'transformation': metrics_data[desc]['rotation'], 'values': metrics_data[desc]['repeatability']['rotation']}
         for desc in descriptors},
        title="Repeatability vs Rotation",
        xlabel="Rotation (degrees)",
        ylabel="Repeatability",
        filename=filename
    )


def plot_distance_distribution(distances, descriptor, angle):
    plt.figure()
    plt.hist(distances, bins=50, alpha=0.75)
    plt.title(f'Match Distance Distribution - {descriptor} at {angle} degrees')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(f'distance_distribution_{descriptor}_{angle}.png')
    plt.close()