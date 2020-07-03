import gym
import time
import argparse

from stable_baselines import DDPG
from stable_baselines.common.evaluation import evaluate_policy

parser = argparse.ArgumentParser()
parser.add_argument("--load_path", help="Indica la ruta en la que se encuentra el modelo a cargar")
parser.add_argument("--env", help="entorno gym que utiliza el modelo a cargar")
args = parser.parse_args()

env=gym.make(args.env)
model = DDPG.load(args.load_path)

# Evaluate the agent
#mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Probando el agente entrenado
obs = env.reset()
reward_episodio=0
while(True):
	#Paramos un poco el tiempo para que de tiempo a visualizarlo
    time.sleep(0.05)
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    reward_episodio+=rewards
    if(dones):
    	print("Recompensa episodio: "+str(reward_episodio))
    	reward_episodio=0
    	env.reset()
    else:
    	env.render()