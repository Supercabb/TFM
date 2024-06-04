import numpy as np
import gymnasium as gym
from qSVMExecutor import qSVMExecutor
from itertools import combinations, product

class OptimizeFeatureMap(gym.Env):
    
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, qSvmExecutor, render_mode='ansi'):
         self.render_mode = render_mode
         self.numMaxRepeats=4
         self.NumMaxPaulisStringsSize=3
         self.MaxOrderPaulis=3
         self.MaxNumSamplesObervations=10
         self.NumDivisionsAlphaRot=32

         self.allScores={}
         self.allScores["lineal"]={}
         self.allScores["lineal"]["precision"]=[]
         self.allScores["lineal"]["recall"]=[]
         self.allScores["lineal"]["f1"]=[]

         self.allScores["rbf"]={}
         self.allScores["rbf"]["precision"]=[]
         self.allScores["rbf"]["recall"]=[]
         self.allScores["rbf"]["f1"]=[]

         self.allScores["quantum"]={}
         self.allScores["quantum"]["precision"]=[]
         self.allScores["quantum"]["recall"]=[]
         self.allScores["quantum"]["f1"]=[]         

        
        
         self.MapIntToPaulis={0:""}

         for i in range(self.NumMaxPaulisStringsSize):
            c = list(product(["X","Y","Z","I"], repeat=i+1))
            for j in range(len(c)):
                if(all([c[j][k]=="I" for k in range(len(c[j]))])):
                    continue

                self.MapIntToPaulis[len(self.MapIntToPaulis)]="".join(c[j])

         print("Paulis: "+str(self.MapIntToPaulis))

         self.entaglementTypes={0:"linear", 1:"full", 2:"circular", 3:"reverse_linear", 4:"sca"}

         actionsSpaceVec=np.array([self.numMaxRepeats]+[len( self.entaglementTypes)]+[self.NumDivisionsAlphaRot]+([len(self.MapIntToPaulis.keys())]*self.MaxOrderPaulis), dtype=np.uint8)
         print("Action Space: "+str(actionsSpaceVec))

         self.action_space = gym.spaces.MultiDiscrete(actionsSpaceVec, dtype=np.uint8)

        

         self.qSvmExecutor=qSvmExecutor
         self.actual_feature_map=None
         self.actual_qSVM=None

         self.NumSamplesObervations=min(self.MaxNumSamplesObervations, self.qSvmExecutor.GetNumTrainSamples())

         #self.observation_space = gym.spaces.Box(-np.inf, np.inf, ((2**self.qSvmExecutor.GetNumFeatures())*(2**self.qSvmExecutor.GetNumFeatures()),), dtype=np.float32)
         self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.NumSamplesObervations,self.qSvmExecutor.GetNumFeatures()), dtype=np.float32)

         self.numTotalTrain=0
         self.numTotalEpisode=0
         self.log = ''

    def reset(self, seed=None, options=None):
         
         self.qSvmExecutor.CreateTrainValDataset()
         
         self.actual_feature_map=self.qSvmExecutor.CreateFeatureMap(2, ["Z"])
         self.numTotalEpisode=0
        
         #observations=self.qSvmExecutor.MatrixFromFeatureMap(self.actual_feature_map)
         observations=self.qSvmExecutor.GetNTrainSamples(self.NumSamplesObervations)
         return observations, {}
    
    def actionToString(self, action):
        numReps=action[0]+1
        entaglementType=self.entaglementTypes[action[1]]
        alphaRot=action[2]/(self.NumDivisionsAlphaRot-1)*np.pi

        paulis_list=[]
        pauli_str=""

        for i in range(3, len(action)):
            if(action[i]!=0):
                paulis_list.append(self.MapIntToPaulis[action[i]])

        return str(numReps)+","+entaglementType+","+str(alphaRot)+","+str(paulis_list)

    def step(self, action):
        numReps=action[0]+1
        entaglementType=self.entaglementTypes[action[1]]
        alphaRot=action[2]/(self.NumDivisionsAlphaRot-1)*np.pi

        paulis_list=[]
        pauli_str=""

        for i in range(3, len(action)):
            if(action[i]!=0):
                paulis_list.append(self.MapIntToPaulis[action[i]])

        self.actual_feature_map=self.qSvmExecutor.CreateFeatureMap(numReps, paulis_list, entaglementType, alphaRot)
        #observations=self.qSvmExecutor.MatrixFromFeatureMap(self.actual_feature_map)
        observations=self.qSvmExecutor.GetNTrainSamples(self.NumSamplesObervations)

        self.actual_qSVM=self.qSvmExecutor.TrainQSVM(self.actual_feature_map)
        metrics_quantum=self.qSvmExecutor.TestModel(self.actual_qSVM, "Training iteration "+str(self.numTotalTrain),"")
        self.log+=str(metrics_quantum)+"\n"

        print("SVM Quantum")
        print(metrics_quantum)
        print("-----------------")

        linear_svm=self.qSvmExecutor.TrainClassicSVM("linear")
        rbf_svm=self.qSvmExecutor.TrainClassicSVM("rbf")

        metrics_lin=self.qSvmExecutor.TestModel(linear_svm, "SVM Linear", "")
        print("SVM Linear")
        print(metrics_lin)
        print("-----------------")

        metrics_rbf=self.qSvmExecutor.TestModel(rbf_svm, "SVM RBF", "")
        print("SVM RBF")
        print(metrics_rbf)
        print("-----------------")

        reward=metrics_quantum["f1"]-metrics_rbf["f1"]


        self.allScores["lineal"]["precision"].append(metrics_lin["precision"])
        self.allScores["lineal"]["recall"].append(metrics_lin["recall"])
        self.allScores["lineal"]["f1"].append(metrics_lin["f1"])

        self.allScores["rbf"]["precision"].append(metrics_rbf["precision"])
        self.allScores["rbf"]["recall"].append(metrics_rbf["recall"])
        self.allScores["rbf"]["f1"].append(metrics_rbf["f1"])


        self.allScores["quantum"]["precision"].append(metrics_quantum["precision"])
        self.allScores["quantum"]["recall"].append(metrics_quantum["recall"])
        self.allScores["quantum"]["f1"].append(metrics_quantum["f1"])

        with open("mean_scores_RL.txt", "w") as file:
            for key in self.allScores.keys():
                precision_Mean=np.mean(self.allScores[key]["precision"])
                precision_Std=np.std(self.allScores[key]["precision"])
                recall_Mean=np.mean(self.allScores[key]["recall"])
                precision_Std=np.std(self.allScores[key]["recall"])
                f1_Mean=np.mean(self.allScores[key]["f1"])
                f1_Stdev=np.std(self.allScores[key]["f1"])
                print("Precision Mean "+key+": "+str(precision_Mean))
                print("Precision Std "+key+": "+str(precision_Std))
                print("Recall Mean "+key+": "+str(recall_Mean))
                print("Recall Std "+key+": "+str(precision_Std))
                print("F1 Mean "+key+": "+str(f1_Mean))
                print("F1 Std "+key+": "+str(f1_Stdev))
                file.write(f"Precision Mean {key}: {precision_Mean}\n")
                file.write(f"Precision Std {key}: {precision_Std}\n")
                file.write(f"Recall Mean {key}: {recall_Mean}\n")
                file.write(f"Recall Std {key}: {precision_Std}\n")
                file.write(f"F1 Mean {key}: {f1_Mean}\n")
                file.write(f"F1 Std {key}: {f1_Stdev}\n")

        done=False
        
        #if(self.numTotalEpisode>=20):
            #done=True
        
        #Si la recompensa es negativa, terminamos el episodio
        if(reward<0) or (self.numTotalEpisode>=50):
            done=True

        self.numTotalTrain+=1
        self.numTotalEpisode+=1

        print("Episode "+str(self.numTotalEpisode)+" Reward "+str(reward))

        with open("trainlog.csv", "a") as f:
            f.write(str(self.numTotalTrain)+","+str(reward)+"\n")

        return observations, reward, done, False, {}
    
    def close(self):
        pass
        
    def render(self, mode='ansi'):
        print(self.log)
        self.log = ''


