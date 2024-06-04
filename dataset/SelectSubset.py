import json
import numpy as np
import cv2
import json
import numpy as np
import cv2
import glob
import pickle
import coreset
import random
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def main():
    
    #SubsetFromAllDataset(5000)
    #JoinAllClassesSubset()
    GenerateCoreset(20)
    ReductFeatures(5)

def plot_features(features_list, labels, weights, filenamePlot):
    features = np.array(features_list)
    features = features.reshape(-1, 224)

    # Perform t-SNE on the training features
    tsne = TSNE(n_components=2, perplexity=19)
    train_features_tsne = tsne.fit_transform(features)
    # Convert labels to numbers
    labels_set = set(labels)

    label_numbers = []
    for i in range(len(labels)):
        label_numbers.append(list(labels_set).index(labels[i]))
    

    # Plot the projected data
    scatter = plt.scatter(train_features_tsne[:, 0], train_features_tsne[:, 1], c=label_numbers, s=weights)

    handles, _ = scatter.legend_elements(prop='colors')
    plt.legend(handles, labels_set)

    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('Data Projection using t-SNE')

    plt.savefig(filenamePlot)
    plt.clf()


def ReductFeatures(numDimensions:int):
    data_vectors=np.load(open("dataset_hyper_subset_all_classes.npy", "rb"))
    pca = PCA(n_components=numDimensions)
    data_vectors_pca = pca.fit_transform(data_vectors)

    data_spectral_coreset=pickle.load(open("dataset_hyper_coreset.pkl", "rb"))
    for key,value in data_spectral_coreset.items():
        for data in value:
            data["spectral"]=pca.transform(data["spectral"].reshape(1,-1))[0]
               
    print("Explained Variance PCA: "+str(pca.explained_variance_ratio_.sum()))

    #pickle.dump(data_spectral_coreset, open("dataset_hyper_coreset_pca.pkl", "wb"))

def GenerateCoreset(numSamplesCoreset:int):
    data_vectors=np.load(open("dataset_hyper_subset_all_classes.npy", "rb"))
    mean_minus=np.mean(data_vectors, axis=0)
    data_vectors = data_vectors - mean_minus
    np.random.shuffle(data_vectors)
    B = coreset.get_bestB(data_vectors, num_runs=10, k=4)

    cost_whole = coreset.kmeans_cost(data_vectors, coreset_vectors=data_vectors)
    print('cost_whole is %s' % cost_whole)
    coreset_vectors, coreset_weights = coreset.BFL16(data_vectors, B=B, m=numSamplesCoreset)

    cost_coreset = coreset.kmeans_cost(data_vectors, coreset_vectors=coreset_vectors, sample_weight=coreset_weights)
    print('cost_coreset is %s' % cost_coreset)

    #Search classes coreset
    data_spectral=pickle.load(open("dataset_hyper_few.pkl", "rb"))
    data_spectral_coreset={}

    features_list=[]
    labels=[]
    weights=[]

    for data_core, weight_core in zip(coreset_vectors, coreset_weights):
        Found=False
        for key,value in data_spectral.items():            
            if ([np.array_equal(data_core,x-mean_minus) for x in value].count(True)==1):
                index=[np.array_equal(data_core,x-mean_minus) for x in value].index(True)

                if(key not in data_spectral_coreset):
                    data_spectral_coreset[key]=[]

                data_spectral_coreset[key].append({"weight":weight_core, "spectral":value[index]})
                Found=True

                features_list.append(value[index])
                labels.append(key)
                weights.append(weight_core)

                break

        if (Found==False):
            print("Error, not found data_core")
            

    plot_features(features_list, labels,weights, "plot_coreset.png")
    pickle.dump(data_spectral_coreset, open("dataset_hyper_coreset.pkl", "wb"))


def SubsetFromAllDataset(n:int):
    data_spectral=pickle.load(open("dataset_hyper.pkl", "rb"))

    for key,value in data_spectral.items():

        value_subset = random.sample(value, n)
        data_spectral[key]=value_subset

    pickle.dump(data_spectral, open("dataset_hyper_subset.pkl", "wb"))


          

def JoinAllClassesSubset():
    data_spectral=pickle.load(open("dataset_hyper_few.pkl", "rb"))

    data_spectral_all_classes=np.empty((0,224))

    for value in data_spectral.values():
        for spectral_data in value:
            data_spectral_all_classes=np.vstack((data_spectral_all_classes, spectral_data))

    print(len(data_spectral_all_classes))

    np.save(open("dataset_hyper_subset_all_classes.npy","wb"), data_spectral_all_classes, allow_pickle=True)




if __name__ == "__main__":
    main()