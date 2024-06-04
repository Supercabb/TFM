import numpy as np

import matplotlib.pyplot as plt

FILE="mean_scores_RL.txt"
postTitle="RL PauliFeatureMap"

FILE="mean_scores_ZZ.txt"
postTitle="ZZFeatureMap"

FILE="mean_scores_Z.txt"
postTitle="ZFeatureMap"

FILE="mean_scores_Quantums.txt"
postTitle="QSVC Comparation"

metrics_dict = {}


with open(FILE, 'r') as f:
    for line in f:
        parts = line.split(':')
        value = float(parts[1].strip())  # Convert the value to a float


        key_parts = parts[0].split()
        type_alg = key_parts[-1]
        stadistic = key_parts[1]
        metric = key_parts[0]

        # If the main key is not in the dictionary, add it
        if metric not in metrics_dict:
            metrics_dict[metric] = {}

        # Add the subkey-value pair to the dictionary under the main key
        if type_alg not in metrics_dict[metric]:
            metrics_dict[metric][type_alg] = {}
        
        metrics_dict[metric][type_alg][stadistic] = value


for metric, type_algs in metrics_dict.items():
    
    for type_alg in type_algs.keys():

        # Calculate the mean and standard deviation
        mean =  metrics_dict[metric][type_alg]["Mean"]
        std = metrics_dict[metric][type_alg]["Std"]

        plt.errorbar(type_alg, mean, std, linestyle='None', marker='o')
        # Plot the mean and standard deviation
        #plt.plot([mean, mean], [0, 1], 'r-', label='Mean')
        #plt.fill_betweenx([0, 1], mean - std, mean + std, color='gray', alpha=0.3, label='Standard Deviation')
        #plt.legend()

        # Add labels and title
        #plt.xlabel("Type of Algorithm")
        plt.xlabel("FeatureMap")
        plt.ylabel(metric)
        plt.title(f'{postTitle} {metric}')

        # Show the plot
    plt.savefig(f"{postTitle.replace(' ','_')}_{metric}.png")
    plt.clf()  # Clear the plot for the next iteration