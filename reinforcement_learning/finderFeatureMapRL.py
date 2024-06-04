from stable_baselines3 import A2C
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import os
import gymnasium as gym
import pickle
from qSVMExecutor import qSVMExecutor
from envFeatureMapRL import OptimizeFeatureMap

FILE_MODEL="ppo_opt_featureMap1.zip"
FILE_MODEL_TEST="ppo_opt_featureMap.zip"
NUM_CPU_TRAIN=1

def main():
    """
    if(os.path.exists("trainlog.csv")):
        os.remove("trainlog.csv")
    """

    dataset_spectralWaste_coreset_pca=pickle.load(open("dataset_hyper_coreset_pca.pkl", "rb"))

    qSVMExec=qSVMExecutor(dataset_spectralWaste_coreset_pca,["film"])

    #optimizer=OptimizeFeatureMap(qSVMExecutor)

    gym.envs.register(
    id='opt_featureMap',
    entry_point='envFeatureMapRL:OptimizeFeatureMap',
    kwargs={'qSvmExecutor':qSVMExec}
    )

    test()
    #train()
    #test_env()

def test():

    NUM_BEST_ACTIONS=5
    opt_featureMap = make_vec_env('opt_featureMap', n_envs=NUM_CPU_TRAIN, env_kwargs={'render_mode': 'ansi'})
    opt_featureMap = VecNormalize(opt_featureMap, norm_obs=True, norm_reward=True, gamma=0)

    dataset_spectralWaste_coreset_pca=pickle.load(open("dataset_hyper_coreset_pca.pkl", "rb"))
    qSVMExec=qSVMExecutor(dataset_spectralWaste_coreset_pca,["film"])
    opt_featureMap_test=OptimizeFeatureMap(qSVMExec)


    model = PPO.load(FILE_MODEL_TEST, env=opt_featureMap)

    done = False
    total_reward=0
    

    actions={}

    NumIterations=100
    actualIt=0
    while(actualIt<NumIterations):
        observation = opt_featureMap.reset()
        action, _states = model.predict(observation, deterministic=True)
            
        observation, reward, done, _ = opt_featureMap.step(action)
        print("Reward: "+str(opt_featureMap.unnormalize_reward(reward))+" for action: "+str(action[0]))

        #Me quedo con las mejores 5
        for act in action:
            if(tuple(act) not in actions):
                if(len(actions)<NUM_BEST_ACTIONS):
                    actions[tuple(act)]=opt_featureMap.unnormalize_reward(reward)
                    print("Action: "+opt_featureMap_test.actionToString(act)+" with reward: "+str(reward))
                else:
                    if(opt_featureMap.unnormalize_reward(reward)>min(actions.values())): 
                        actions.pop(min(actions, key=actions.get))
                        actions[tuple(act)]=opt_featureMap.unnormalize_reward(reward)
                        print("Action: "+opt_featureMap_test.actionToString(act)+" with reward: "+str(reward))
                        
                       

        actualIt+=1
        print("Iteration: "+str(actualIt)+"/"+str(NumIterations))
        
    for act, reward in actions.items():
        print(opt_featureMap_test.actionToString(act)+","+str(reward))

def train():
 
    # Create the vectorized environment
    opt_featureMap = make_vec_env('opt_featureMap', n_envs=NUM_CPU_TRAIN, env_kwargs={'render_mode': 'ansi'})
    #opt_threshold = DummyVecEnv([lambda: opt_threshold])
    opt_featureMap = VecNormalize(opt_featureMap, norm_obs=True, norm_reward=True, gamma=0)
    #opt_featureMap = VecNormalize(opt_featureMap, norm_obs=True, norm_reward=True)
   
    opt_featureMap.reset()    
    
    #model = PPO("MlpPolicy", opt_featureMap, verbose=1)
    model = PPO("MlpPolicy", opt_featureMap, verbose=1, gamma=0)
    #model = A2C("MlpPolicy", opt_featureMap, verbose=1, gamma=0)
    model.learn(total_timesteps=100000, log_interval=10, progress_bar=True)
    model.save(FILE_MODEL)

def test_env():
    opt_featureMap = gym.make('opt_featureMap', render_mode='ansi')
    opt_featureMap = DummyVecEnv([lambda: opt_featureMap])
    opt_featureMap = VecNormalize(opt_featureMap, norm_obs=True, norm_reward=False,
                    gamma=0)    
    opt_featureMap.reset() 
    done = False
    total_reward=0
    observation = opt_featureMap.reset()

    action=[1, 0, 2, 0, 0, 0]

    while(done==False):
        #action=[0, np.random.randint(5), 0, 0, 0]
        observation, reward, done, _ = opt_featureMap.step([action])
        total_reward+=reward

if __name__ == "__main__":
    main()