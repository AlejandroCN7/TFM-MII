from baselines.common import plot_util as pu
import matplotlib.pyplot as plt
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--log_path", help="Indica la ruta en la que se encuentra los archivos a monitorizar (busca subcarpetas), en caso de no indicarse comienza a sacar gráficas predefinidas")
args = parser.parse_args()

if(args.log_path):
	results = pu.load_results(args.log_path)
	plt.plot(results[0].progress['misc/total_timesteps'], pu.smooth(results[0].progress['loss/policy_entropy'],radius=20),label="semilla 113")
	plt.plot(results[1].progress['misc/total_timesteps'], pu.smooth(results[1].progress['loss/policy_entropy'],radius=20),label="semilla 32")
	plt.title("Entropía PPO2 Super Mario Bros (100 millones de interacciones)")
	plt.xlabel("Recompensas")
	plt.ylabel("Pasos transcurridos episodios")
	plt.legend()
	plt.show()
else:
	#MOUNTAIN CAR V0

	#DQN

	results = pu.load_results('./mountain/logs/DQN/')

	plt.plot(results[0].progress['episodes'], results[0].progress['mean 100 episode reward'],label="semilla 3")
	plt.plot(results[1].progress['episodes'], results[1].progress['mean 100 episode reward'],label="semilla 13")
	plt.plot(results[2].progress['episodes'], results[2].progress['mean 100 episode reward'],label="semilla 113")

	plt.title("Media de recompensas obtenidas durante entrenamiento DQN MountainCar-v0")
	plt.xlabel("episodios")
	plt.ylabel("Media recompensa")
	plt.legend()
	plt.show()

	plt.plot(results[0].progress['episodes'], results[0].progress['% time spent exploring'])
	plt.title("Porcentaje de tiempo empleado exploración MountainCar-v0 DQN")
	plt.xlabel("Tiempo(%)")
	plt.ylabel("Episodios")
	plt.show()

	pu.plot_results(results, average_group=True, split_fn=lambda _: '')
	plt.title("Resumen de entrenamiento DQN MountainCar-v0")
	plt.ylabel('recompensas')
	plt.xlabel('timesteps')
	plt.show()

	#DDPG

	results = pu.load_results('./mountain/logs/DDPG/')

	plt.plot(np.cumsum(results[0].monitor['l']), pu.smooth(results[0].monitor['r'],radius=10),label="semilla 3")
	plt.plot(np.cumsum(results[1].monitor['l']), pu.smooth(results[1].monitor['r'],radius=10),label="semilla 13")
	plt.plot(np.cumsum(results[2].monitor['l']), pu.smooth(results[2].monitor['r'],radius=10),label="semilla 113")

	plt.title("Entrenamientos MountainCar-v0 DDPG")
	plt.xlabel("pasos")
	plt.ylabel("recompensas")
	plt.show()


	pu.plot_results(results, average_group=True, split_fn=lambda _: '')
	plt.title("Resumen DDPG MountainCar-v0")
	plt.ylabel('recompensas')
	plt.xlabel('timesteps')
	plt.show()

	#PPO2 y A2C

	results = pu.load_results('./mountain/logs/A2C/')
	plt.plot(results[0].progress['total_timesteps'], results[0].progress['policy_entropy'],label="A2C")
	results = pu.load_results('./mountain/logs/PPO2/')
	plt.plot(results[0].progress['misc/total_timesteps'], results[0].progress['loss/policy_entropy'],label="PPO2")
	plt.title("Entropía de política A2C y PPO2 para MountainCar-v0")
	plt.ylabel('Entropía')
	plt.xlabel('pasos (t)')
	plt.legend()
	plt.show()

	#Comparativa todos modelos
	results = pu.load_results('./mountain/logs/')
	pu.plot_results(results, average_group=True, split_fn=lambda _: '')
	plt.title("Comparativa modelos MountainCar-v0")
	plt.ylabel('recompensas')
	plt.xlabel('pasos (t)')
	plt.show()

	#Super Mario Bros 

	#DQN
	results = pu.load_results('./mario/logs/DQN/mariolevel11-52/')
	plt.plot(results[0].progress['episodes'], results[0].progress['mean 100 episode reward'])
	plt.title("Recompensas DQN Super Mario Bros")
	plt.ylabel('Media recompensas')
	plt.xlabel('Episodios')
	plt.show()

	plt.plot(results[0].progress['episodes'], results[0].progress['% time spent exploring'])
	plt.title("Tiempo empleado explorando DQN Super Mario Bros")
	plt.ylabel('Tiempo explorando(%)')
	plt.xlabel('Episodios')
	plt.show()

	#A2C y PPO2
	results_a2c = pu.load_results('./mario/logs/A2C/mariolevel11-84/')
	results_ppo2 = pu.load_results('./mario/logs/PPO2/mariolevel11-77/')
	plt.plot(results_a2c[0].progress['total_timesteps'], results_a2c[0].progress['eprewmean'],label="A2C")	
	plt.plot(results_ppo2[0].progress['misc/total_timesteps'], results_ppo2[0].progress['eprewmean'],label="PPO2")
	plt.title("Media recompensas Super Mario Bros PPO2 y A2C")
	plt.ylabel('Media de recompensas por episodio')
	plt.xlabel('Pasos (t)')
	plt.legend()
	plt.show()

	
	plt.plot(results_a2c[0].progress['total_timesteps'], results_a2c[0].progress['eplenmean'],label="A2C")
	plt.plot(results_ppo2[0].progress['misc/total_timesteps'], results_ppo2[0].progress['eplenmean'],label="PPO2")
	plt.title("Media de pasos por episodios A2C y PPO2 Super Mario Bros")
	plt.ylabel('Media de pasos por episodio')
	plt.xlabel('Pasos (t)')
	plt.legend()
	plt.show()


	plt.plot(results_a2c[0].progress['eprewmean'],results_a2c[0].progress['eplenmean'],'o',markersize=2,label="A2C")
	plt.plot(results_ppo2[0].progress['eprewmean'],results_ppo2[0].progress['eplenmean'],'o',markersize=2,label="PPO2")
	plt.title("Nube de puntos media recompensas/ media pasos por episodios entre A2C y PPO2 Super Mario Bros")
	plt.ylabel('Media de pasos por episodio')
	plt.xlabel('Media de recompensas')
	plt.legend()
	plt.show()


	plt.plot(results_a2c[0].progress['total_timesteps'], pu.smooth(results_a2c[0].progress['policy_entropy'],radius=20),label="A2C")
	plt.plot(results_ppo2[0].progress['misc/total_timesteps'], pu.smooth(results_ppo2[0].progress['loss/policy_entropy'],radius=20),label="PPO2")
	plt.title("Entropía PPO2 y A2C Super Mario Bros")
	plt.ylabel('Entropía')
	plt.xlabel('Pasos (t)')
	plt.legend()
	plt.show()


	#Desarrollo PPO2
	results_113 = pu.load_results('./mario/logs/PPO2/mariolevel11-113/')
	results_32 = pu.load_results('./mario/logs/PPO2/mariolevel11-32/')
	plt.plot(results_113[0].progress['misc/total_timesteps'], results_113[0].progress['eprewmean'],label="Semilla 113")
	plt.plot(results_32[0].progress['misc/total_timesteps'], results_32[0].progress['eprewmean'],label="Semilla 32")
	plt.title("Media recompensas Super Mario Bros PPO2 (100 millones de interacciones)")
	plt.ylabel('Media de recompensas por episodio')
	plt.xlabel('Pasos (t)')
	plt.legend()
	plt.show()

	plt.plot(results_113[0].progress['misc/total_timesteps'], pu.smooth(results_113[0].progress['loss/policy_entropy'],radius=20),label="Semilla 113")
	plt.plot(results_32[0].progress['misc/total_timesteps'], pu.smooth(results_32[0].progress['loss/policy_entropy'],radius=20),label="Semilla 32")
	plt.title("Entropía PPO2 Super Mario Bros (100 millones de interacciones)")
	plt.ylabel('Entropía')
	plt.xlabel('Pasos (t)')
	plt.legend()
	plt.show()


#results = pu.load_results('./prueba/A2C/')
#plt.plot(results[0].progress['total_timesteps'], pu.smooth(results[0].progress['policy_entropy'],radius=10),label="A2C")

#results = pu.load_results('./prueba/PPO2/')
#plt.plot(results[0].progress['total_timesteps'], results[0].progress['eprewmean']/results[0].progress['eplenmean'])
#plt.plot(results[0].progress['misc/total_timesteps'],results[0].progress['eprewmean'])
#plt.plot(results[0].progress['misc/total_timesteps'], pu.smooth(results[0].progress['loss/policy_entropy'],radius=10),label="PPO2")
#plt.title("Media de recompensas PPO2 Super Mario Bros")
#plt.xlabel('Paso (t)')
#plt.ylabel('Recompensa')
#plt.xticks(np.arange(0,1e8,0.1e8))
#plt.ylim(400,2200)
#plt.show()

#plt.title("Entropía PPO2 A2C")
#plt.xlabel('Paso (t)')
#plt.ylabel('Entropía')
#plt.legend()
#plt.yticks(np.arange(0,2201,200))
#plt.ylim(400,2200)
#plt.show()