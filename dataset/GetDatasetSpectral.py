import json
import numpy as np
import cv2
import json
import numpy as np
import cv2
import glob
import pickle
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MaxAbsScaler
from scipy.signal import savgol_filter
from scipy.signal import detrend

def snv(input_data):
    # DEFINE A NEW ARRAY AND POPULATE IT WITH THE CORRECTED DATA
    output_data = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
    # APPLY CORRECTION
        output_data[i,:] = (input_data[i,:] - np.mean(input_data[i,:])) / np.std(input_data[i,:])
    return output_data

def plot_hyperspectral(features_list, label):

    x = [range(224)]

    for i in range(0, len(features_list)):
        plt.plot(features_list[i])

    plt.savefig(label+".png")
    plt.clf()


def plot_features(features_list, labels, filenamePlot):
    features = np.array(features_list)
    features = features.reshape(-1, 224)

    # Perform t-SNE on the training features
    tsne = TSNE(n_components=2, n_jobs=15)
    train_features_tsne = tsne.fit_transform(features)
    # Convert labels to numbers
    labels_set = set(labels)

    label_numbers = []
    for i in range(len(labels)):
        label_numbers.append(list(labels_set).index(labels[i]))
    

    # Plot the projected data
    scatter = plt.scatter(train_features_tsne[:, 0], train_features_tsne[:, 1], c=label_numbers)

    handles, _ = scatter.legend_elements(prop='colors')
    plt.legend(handles, labels_set)

    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('Data Projection using t-SNE')

    plt.savefig(filenamePlot)
    plt.clf()


sizeCeldilla=8
selectedCategories=["background", "film", "cardboard"]

PATH = "./TFM/spectralwaste_segmentation"
jsonMeta = json.load(open(PATH + "/meta.json"))
print(jsonMeta)

image_folder = PATH + "/hyper/train"
labels_folder = PATH + "/labels_hyper_lt/train"
image_files = glob.glob(image_folder + "/*.tiff")  # Modify the file extension if needed

image_foder_val = PATH + "/hyper/val"
labels_folder_val = PATH + "/labels_hyper_lt/val"
image_files_val = glob.glob(image_foder_val + "/*.tiff")

#image_files.extend(image_files_val)

random.shuffle(image_files)

shape_hyper=(224, 256, 256)

dataset={}

for category in jsonMeta["categories"]:
    dataset[category]=[]

total=len(image_files)
actual=0

features_list=[]
labels=[]

countsForCat={}
    


for k in selectedCategories:
    countsForCat[k]=0



for image_file in image_files:
    # Process each image here
    _, image_hyper = cv2.imreadmulti(image_file, flags=cv2.IMREAD_ANYDEPTH)
    image_label = cv2.imread(labels_folder + "/" + image_file.split("/")[-1].replace(".tiff", ".png"), cv2.IMREAD_GRAYSCALE)
    """
    if(image_label is None):
        image_label = cv2.imread(labels_folder_val + "/" + image_file.split("/")[-1].replace(".tiff", ".png"), cv2.IMREAD_GRAYSCALE)
    """

    print("Processing image ", actual, " of ", total)
    actual+=1

    image_show_celdillas = (cv2.cvtColor(image_hyper[10], cv2.COLOR_GRAY2RGB)/256).astype(np.uint8)
    print(image_hyper[10].shape)
    print(image_show_celdillas.shape)
   


    for j in range(shape_hyper[1]//sizeCeldilla):
        for i in range(shape_hyper[2]//sizeCeldilla):

            
            #Miro todos los pixeles en un celda de 4x4 que sean iguales
            Equals=True
            for k in range(sizeCeldilla):
                for l in range(sizeCeldilla):
                    if(image_label[j*sizeCeldilla+k,i*sizeCeldilla+l]!=image_label[j*sizeCeldilla,i*sizeCeldilla]):
                        Equals=False
                        break

            if(Equals==False):
                continue



            pixel_label=image_label[j*sizeCeldilla,i*sizeCeldilla]
            label_class=jsonMeta["categories"][pixel_label]

            if(label_class not in selectedCategories):
                continue

            """
            if label_class in countsForCat:
                if label_class=="background" and countsForCat[label_class] > min(countsForCat.values()):
                    continue
            """

            
            if label_class in countsForCat:
                if countsForCat[label_class] > min(countsForCat.values()):
                    continue
            

            spectral_data=np.zeros(shape_hyper[0])
            numMean=0
            for k in range(shape_hyper[0]):
                for l in range(sizeCeldilla//2):
                    for m in range(sizeCeldilla//2):
                        spectral_data[k]+=image_hyper[k][j*sizeCeldilla+l*2+1][i*sizeCeldilla+m*2+1]
                        numMean+=1                    
                        image_show_celdillas[j*sizeCeldilla+l*2+1][i*sizeCeldilla+m*2+1]=[0,0,255]
                        
                        
            
            spectral_data=spectral_data/numMean
            
            dataset[label_class].append(spectral_data)
            countsForCat[label_class]=countsForCat.get(label_class, 0)+1
            print(countsForCat)

            #cv2.imwrite("image_"+str(actual)+"_celdilla.png", image_show_celdillas)

            #print("Pixel label: ", pixel_label)


    #if(actual%50==0):
    #    pickle.dump(dataset, open("dataset_hyper.pkl", "wb"))

"""
minSamples=min(countsForCat.values())
for key,value in dataset.items():
    print("0. Category ", key, " has ", len(dataset[key]), " samples")
    if(len(value)>minSamples):
        dataset[key]=random.sample(value, minSamples)

    print("1. Category ", key, " has ", len(dataset[key]), " samples")
"""

for key,value in dataset.items():
    if(len(value)==0):
        continue
    
    value = np.array(value)
    value = value.reshape(-1, 224)

    #Max normalization
    for k in range(value.shape[0]):
        value[k]=value[k]/np.max(value[k])


    value=savgol_filter(value, 6, 4, mode='nearest')
    value = detrend(value, type='linear')
    value=snv(value)

    for k in range(value.shape[0]):
        features_list.append(value[k])
        labels.append(key)

    dataset[key]=list(value)
    
    plot_hyperspectral(value, key)





plot_features(features_list, labels, "getspectral_plot.png")


pickle.dump(dataset, open("dataset_hyper_few.pkl", "wb"))
  



