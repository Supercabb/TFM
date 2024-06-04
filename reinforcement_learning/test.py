from qSVMExecutor import qSVMExecutor
import pickle
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, PauliFeatureMap

dataset_spectralWaste_coreset_pca=pickle.load(open("dataset_hyper_coreset_pca.pkl", "rb"))
qSVMExec=qSVMExecutor(dataset_spectralWaste_coreset_pca,["film"])


qSVMExec.CreateTrainValDataset()
svm_linear=qSVMExec.TrainClassicSVM("linear")
svm_rbf=qSVMExec.TrainClassicSVM("rbf")

featureMap=qSVMExec.CreateFeatureMap(2, ["ZX"])
svm_quantum=qSVMExec.TrainQSVM(featureMap)

pfeature_map = PauliFeatureMap(feature_dimension=5, reps=1, alpha=0.10, entanglement="reverse_linear", paulis=["ZZ","ZX","IY"])
pfeature_map.decompose(reps=1).draw("mpl").savefig("PauliFeatureMap_rep1_rew0_45.png")


print("SVM Linear")
print(qSVMExec.TestModel(svm_linear, "SVM Linear", ""))
print("-----------------")
      
print("SVM RBF")
print(qSVMExec.TestModel(svm_rbf, "SVM RBF", ""))
print("-----------------")

print("QSVM")
print(qSVMExec.TestModel(svm_quantum, "QSVM", ""))
print("-----------------")  


