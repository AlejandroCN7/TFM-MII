from baselines.common import plot_util as pu
import matplotlib.pyplot as plt
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--log_path", help="Indica la ruta en la que se encuentra los archivos a monitorizar (busca subcarpetas)")
args = parser.parse_args()
labels=["DDPG","DQN", "PPO2"]
if(args.log_path):
	results = pu.load_results(args.log_path)
	'''for r in results:
		plt.plot(r.progress.episodes, r.progress['% time spent exploring'], label='semilla '+str(semilla[i]))
		i+=1
	fig,axs=plt.subplots(3,1)
	fig.tight_layout()
	i=0
	for r in results:

		grafica=axs[i]
		grafica.set_ylabel('media recompensa')
		grafica.set_xlabel('episodios')
		grafica.set_title('DQN (semilla '+str(semilla[i])+')')
		grafica.set_ylim(-200, -90)

		grafica.plot(r.progress.episodes, r.progress['mean 100 episode reward'])	
		#plt.show()
		i+=1
	
	i=0
	for r in results:
		plt.plot(np.cumsum(r.monitor.l), pu.smooth(r.monitor.r, radius=10), label=labels[i])
		i+=1'''
	pu.plot_results(results, average_group=True, split_fn=lambda _: '')
	plt.title("Comparativa modelos")
	plt.ylabel('recompensas')
	plt.xlabel('timesteps')
	#plt.yticks(np.arange(-200,-90,10))
	plt.legend(labels)
	plt.show()

else:
	print("Tiene que especificar la ruta de logs en --log_path")