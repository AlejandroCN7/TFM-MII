from baselines.common import plot_util as pu
import matplotlib.pyplot as plt
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--log_path", help="Indica la ruta en la que se encuentra los archivos a monitorizar (busca subcarpetas)")
args = parser.parse_args()

if(args.log_path):
	semilla=[3,13,113]
	results = pu.load_results(args.log_path)
	i=0
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
		i+=1'''
	for r in results:
		plt.plot(r.progress.episodes, r.progress['mean 100 episode reward'], label="semilla "+str(semilla[i]))
		i+=1
	plt.title("Media de recompensas obtenidas durante entrenamiento")
	plt.ylabel('media recompensa')
	plt.xlabel('episodios')
	plt.yticks(np.arange(-200,-90,10))
	plt.xticks(np.arange(0,22001,2000))
	plt.legend()
	plt.show()

else:
	print("Tiene que especificar la ruta de logs en --log_path")