import os
import argparse
import gym
import numpy as np

from stable_baselines.bench import Monitor

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.results_plotter import load_results, ts2xy

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, save_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = save_dir
        self.best_mean_reward = -np.inf

    #def _init_callback(self) -> None:
    #    # Create folder if needed
    #    if self.save_path is not None:
    #        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True

#Tratamiento de parámetros del script
parser = argparse.ArgumentParser()
parser.add_argument("--num_timesteps", type=float, default=1e6, help="Indica el numero de timesteps que va a tener el entrenamiento, por defecto 1e6")
parser.add_argument("--save_path", help="Indica la ruta y el nombre del archivo en el que se va a almacenar el modelo.pkl")
parser.add_argument("--log_path", help="Indica la ruta en la que se almacena los logs del proceso de aprendizaje")
parser.add_argument("--env", help="entorno gym que utiliza el modelo")
parser.add_argument("--seed", type=int, help="Semilla con la que se inicia el entrenamiento del modelo")
args = parser.parse_args()

env = gym.make(args.env)
# Creamos la carpeta para los logs
if args.log_path:
	log_dir = args.log_path
	os.makedirs(log_dir, exist_ok=True)
	# los logs serán guardados en log_dir/monitor.csv
	env = Monitor(env, log_dir)

if args.save_path:
	save_dir=args.save_path

# El ruido para DDPG
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise,seed=args.seed)

callback=SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir, save_dir=save_dir)

model.learn(total_timesteps=args.num_timesteps,callback=callback)