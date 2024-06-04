from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.datasets import ad_hoc_data
import matplotlib.pyplot as plt
import numpy as np
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, PauliFeatureMap
from qiskit_aer.primitives import Sampler
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.svm import SVC
from qiskit_machine_learning.algorithms import QSVC
import pickle
from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.manifold import TSNE
from qiskit_aer import StatevectorSimulator, UnitarySimulator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score,  f1_score
from qiskit import QuantumCircuit, transpile


class qSVMExecutor:
    def __init__(self, fullDataset, objectiveClasses) -> None:
        self.fullDataset = fullDataset
        self.objectiveClasses = objectiveClasses
        
        self.train_features=[]
        self.test_features=[]
        self.train_labels=[]
        self.test_labels=[]
        self.train_weights=[]
        self.test_weights=[]
        self.size_train=0
        self.num_features_dataset=0

        self.CreateTrainValDataset()

    def GetNumFeatures(self)->int:
        return self.num_features_dataset
    
    def GetNumTrainSamples(self)->int:
        return self.size_train

    def CreateTrainValDataset(self, trainSize=0.5):
        data=[]
        weights=[]
        labels=[]

  
        for key,value in self.fullDataset.items():
            print(key+" "+str(len(value)))
            for val in value:
                data.append(val["spectral"])
                weights.append(val["weight"])
                if(key not in self.objectiveClasses):
                    labels.append(key)
                else:
                    labels.append("resto")
        
        self.size_train=int(len(data)*trainSize)
        #Generate list of not repeated random indexes
        random_indexes = np.random.choice(len(data), self.size_train, replace=False)
        self.train_features=[data[i] for i in random_indexes]
        self.train_labels=[labels[i] for i in random_indexes]
        self.train_weights=[weights[i] for i in random_indexes]
        self.test_features=[data[i] for i in range(len(data)) if i not in random_indexes]
        self.test_labels=[labels[i] for i in range(len(data)) if i not in random_indexes]
        self.test_weights=[weights[i] for i in range(len(data)) if i not in random_indexes]

        self.num_features_dataset=self.train_features[0].shape[0]

    def GetNTrainSamples(self, numSamples)->np.ndarray:
        array_samples=np.array(self.train_features)
        return array_samples[:numSamples,].astype(np.float32)
        
    def TrainClassicSVM(self, kernel_type)->SVC:
        clf = SVC(kernel=kernel_type)
        clf.fit(self.train_features, self.train_labels, sample_weight=self.train_weights)
        return clf
    
    def CreateFeatureMap(self, reps, paulis_list:list[str], entanglement="full", alpha=2)->PauliFeatureMap:
        print("Creating feature map with "+str(reps)+" reps and paulis "+str(paulis_list)+" and entanglement "+entanglement+" and alpha "+str(round(alpha,2)))
        feature_map = PauliFeatureMap(feature_dimension=self.num_features_dataset, reps=reps, paulis=paulis_list, entanglement=entanglement, alpha=alpha)
        return feature_map
    
    def CreateQuantumKernel(self, quantum_feature_map: PauliFeatureMap)->FidelityQuantumKernel:
        sampler = Sampler(backend_options={"device": "GPU"})
        fidelity = ComputeUncompute(sampler=sampler)
        quantum_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=quantum_feature_map)
        return quantum_kernel

    
    def TrainQSVM(self, featureMap)->QSVC:
        quantumKernel = self.CreateQuantumKernel(featureMap)
        qsvc = QSVC(quantum_kernel=quantumKernel)
        qsvc.fit(self.train_features, self.train_labels, sample_weight=self.train_weights)
        return qsvc
    
    def TestModel(self, model, title:str, prefix_filaname_fig:str)->dict:
        predictions = np.empty(0)
        for xtest in self.test_features:
            pred=model.predict([xtest])
            predictions=np.append(predictions, pred)

        return self.PrintMetrics(predictions, model.classes_, self.test_labels, title, prefix_filaname_fig)
            

    def MatrixFromFeatureMap(self, feature_map:PauliFeatureMap)->np.ndarray:
        circuit=QuantumCircuit(feature_map.num_qubits)
        circuit.append(feature_map, range(feature_map.num_qubits))
        
        simulator = UnitarySimulator(device='GPU')
        circuit = transpile(circuit, backend=simulator)
    
        #Los parametros son las features de un ejemplo aleatorio del dataset de entrenamiento
        randomTrainSample=self.train_features[np.random.randint(0, len(self.train_features))]

        dict_parameters = {}
        for param in circuit.parameters:
            dict_parameters[param] = randomTrainSample[len(dict_parameters)]

        circuit.assign_parameters(dict_parameters, inplace = True)
            

        job = simulator.run(circuit, shots=1)
        result = job.result()
        matrix=result.get_unitary(circuit)

        return matrix.to_matrix().real.astype(np.float32).flatten()



    def PrintMetrics(self, predictions, classes, test_labels, title, prefix_filaname:str, plotMetrics=False)->dict:

        metrics = {}
        
        score = precision_score(test_labels, predictions, average='macro')
        recall = recall_score(test_labels, predictions, average='macro')
        f1 = f1_score(test_labels, predictions, average='macro')


        if(plotMetrics):
            cm = confusion_matrix(test_labels, predictions, labels=classes)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
            disp.plot()
            plt.suptitle("F1 "+title+": "+'{:.2f}'.format(f1.round(2)), fontsize=14, fontweight='bold')
            plt.savefig(prefix_filaname+title.replace(" ", "_")+"_confusion_matrix.png")
            plt.clf()
            plt.close()

        metrics["precision"]=score
        metrics["recall"]=recall
        metrics["f1"]=f1
        
        #print("Precision "+title+": "+str(score))
        #print("Recall "+title+": "+str(recall))
        #print("F1 "+title+": "+str(f1))



        return metrics
        
        