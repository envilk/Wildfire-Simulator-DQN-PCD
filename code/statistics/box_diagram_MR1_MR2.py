import matplotlib.pyplot as plt
import numpy as np

# Datos de ejemplo
num_datasets = 3  # Número de conjuntos de datos
DQN_MR1_data = [  # genetico_coste_data
    [32.84082794189453, 29.87889289855957, 31.10034942626953, 29.204153060913086, 29.58131980895996, 34.3010368347168,
     31.48097038269043, 28.71280288696289, 28.4394474029541, 31.242218017578125, 27.103805541992188, 35.89271926879883,
     26.653980255126953, 29.224912643432617, 33.22491455078125, 30.363325119018555, 30.62629508972168,
     30.100345611572266, 25.792390823364258, 35.44290542602539, 31.23529052734375, 35.91350173950195, 32.7093391418457,
     31.667818069458008, 28.64359474182129, 29.692041397094727, 31.110727310180664, 29.861587524414062,
     23.913496017456055, 31.18338966369629],
    [32.52075958251953, 31.946365356445312, 35.23183250427246, 31.916954040527344, 32.80276966094971, 35.84256553649902,
     35.73874473571777, 31.399651527404785, 34.366777420043945, 32.349477767944336, 33.41349411010742,
     31.747406005859375, 30.745676040649414, 33.1297550201416, 32.079580307006836, 31.02075958251953, 32.0553674697876,
     33.89965343475342, 29.19895839691162, 33.66608810424805, 31.795849800109863, 30.455016136169434, 33.21799278259277,
     32.19723033905029, 31.178202629089355, 31.57785129547119, 30.904841423034668, 29.538068771362305,
     31.231837272644043, 33.847747802734375],
    [33.53748448689779, 33.1407101949056, 32.968859354654946, 32.45098304748535, 31.69088617960612, 33.408302307128906,
     30.807380040486652, 33.49826431274414, 31.243366241455078, 28.57093620300293, 31.6159184773763, 32.53517723083496,
     33.08650143941244, 31.964242299397785, 32.40599568684896, 33.498270670572914, 31.359858830769856,
     32.12918217976888, 32.42214393615723, 31.851214090983074, 32.08880869547526, 33.21683883666992, 33.39676856994629,
     33.189160664876304, 33.469435373942055, 33.49365234375, 32.99653752644857, 32.820072174072266, 31.801615397135418,
     32.08996264139811]
]

PCD_MR1_data = [  # milp_coste_data
    [33.50865173339844, 31.35294532775879, 34.43944549560547, 29.384077072143555, 30.37369728088379, 33.612464904785156,
     34.000003814697266, 35.858131408691406, 29.640138626098633, 29.15224838256836, 32.53633499145508,
     30.356395721435547, 31.86504554748535, 28.332178115844727, 30.87196922302246, 30.463666915893555,
     35.235294342041016, 29.6401309967041, 34.667808532714844, 34.27680969238281, 32.826988220214844,
     31.162635803222656, 30.179929733276367, 32.25605010986328, 30.51211166381836, 36.006927490234375,
     29.53632926940918, 35.62284469604492, 29.532873153686523, 30.273361206054688],
    [33.200687408447266, 33.1003475189209, 29.59861183166504, 34.30623245239258, 33.92387676239014, 33.16263198852539,
     33.5899658203125, 32.961941719055176, 33.487887382507324, 33.09169292449951, 33.3442907333374, 34.10554313659668,
     32.56228446960449, 33.3148775100708, 33.3840856552124, 32.14532470703125, 34.49308204650879, 29.50692081451416,
     32.45156478881836, 32.83217430114746, 33.0692024230957, 33.463669776916504, 33.704145431518555, 33.02595138549805,
     33.97231674194336, 31.958478927612305, 34.24221420288086, 32.8615837097168, 35.10726737976074, 32.44290351867676],
    [32.550170262654625, 34.16147486368815, 32.73010381062826, 31.4163761138916, 31.534027735392254, 32.79931195576986,
     31.936564763387043, 32.27681414286295, 33.258365631103516, 32.15801493326823, 31.301045099894207,
     32.18800481160482, 30.911189397176106, 31.457900365193684, 32.51210721333822, 33.34256362915039,
     32.990769704182945, 32.907727559407554, 32.02191607157389, 33.40484364827474, 31.547862370808918,
     31.811995188395183, 33.52133433024088, 33.03690719604492, 33.94233322143555, 33.73933029174805, 32.64359982808431,
     32.03574879964193, 31.786619186401367, 31.1337947845459]
]

DQN_MR2_data = [  # genetico_timespan_data
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [11.0, 24.0, 6.0, 27.0, 14.0, 6.0, 7.0, 16.0, 11.0, 8.0, 16.0, 9.0, 7.0, 8.0, 13.0, 21.0, 9.0, 26.0, 25.0, 91.0,
     7.0, 13.0, 6.0, 29.0, 9.0, 10.0, 15.0, 23.0, 21.0, 13.0],
    [35.0, 30.0, 37.0, 42.0, 31.0, 31.0, 37.0, 34.0, 30.0, 64.0, 18.0, 38.0, 30.0, 25.0, 37.0, 29.0, 69.0, 30.0, 29.0,
     31.0, 66.0, 41.0, 32.0, 30.0, 22.0, 28.0, 21.0, 22.0, 27.0, 102.0]
]

PCD_MR2_data = [  # milp_timespan_data
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [8.0, 7.0, 12.0, 10.0, 10.0, 8.0, 10.0, 8.0, 10.0, 6.0, 6.0, 10.0, 6.0, 9.0, 9.0, 6.0, 8.0, 10.0, 6.0, 7.0, 8.0,
     7.0, 6.0, 12.0, 9.0, 9.0, 8.0, 10.0, 10.0, 10.0],
    [27.0, 23.0, 23.0, 20.0, 22.0, 23.0, 26.0, 21.0, 27.0, 23.0, 20.0, 21.0, 23.0, 25.0, 23.0, 21.0, 23.0, 26.0, 21.0,
     24.0, 22.0, 25.0, 31.0, 22.0, 28.0, 27.0, 22.0, 21.0, 21.0, 24.0]
]

# Configuración del gráfico
fig, axs = plt.subplots(1, 1, figsize=(16, 8))

# Crear un segundo eje y para timespan en el lado derecho
ax2 = axs.twinx()

# Posiciones para los boxplots
positions_DQN_MR1 = np.arange(1, 3 * num_datasets * 2, step=6)
positions_PCD_MR1 = np.arange(2, 3 * num_datasets * 2 + 1, step=6)

# Boxplots para el MR1 en el eje izquierdo
bp1_DQN = axs.boxplot(DQN_MR1_data, positions=positions_DQN_MR1, labels=[f' ' for i in range(num_datasets)], widths=0.4,
                      patch_artist=True)
bp1_PCD = axs.boxplot(PCD_MR1_data, positions=positions_PCD_MR1, labels=[f' ' for i in range(num_datasets)], widths=0.4,
                      patch_artist=True)

axs.set_ylabel('MR1', fontsize=24, color='tab:blue')
axs.set_ylim([10, 40])  # Ajusta el rango según tus datos

# Colorea los boxplots de coste para DQN
colors_DQN_MR1 = ['lightblue'] * num_datasets
for box, color in zip(bp1_DQN['boxes'], colors_DQN_MR1):
    box.set(color='blue', facecolor=color)

# Colorea los boxplots de coste para PCD
colors_PCD_MR1 = ['lightblue'] * num_datasets
for box, color in zip(bp1_PCD['boxes'], colors_PCD_MR1):
    box.set(color='blue', facecolor=color)

# Posiciones para los boxplots de timespan
positions_DQN_MR2 = np.arange(3, 3 * num_datasets * 2, step=6)
positions_PCD_MR2 = np.arange(4, 3 * num_datasets * 2 + 1, step=6)

# Boxplots para timespan en el lado derecho
bp2_DQN = ax2.boxplot(DQN_MR2_data, positions=positions_DQN_MR2, labels=[f' ' for i in range(num_datasets)], widths=0.4,
                      patch_artist=True)
bp2_PCD = ax2.boxplot(PCD_MR2_data, positions=positions_PCD_MR2, labels=[f' ' for i in range(num_datasets)], widths=0.4,
                      patch_artist=True)

ax2.set_ylabel('MR2', fontsize=24, color='tab:red')
ax2.set_ylim([0, 70])  # Ajusta el rango según tus datos

# Colorea los boxplots de timespan para DQN
colors_DQN_MR2 = ['lightsalmon'] * num_datasets
for box, color in zip(bp2_DQN['boxes'], colors_DQN_MR2):
    box.set(color='red', facecolor=color)

# Colorea los boxplots de timespan para PCD
colors_PCD_MR2 = ['lightsalmon'] * num_datasets
for box, color in zip(bp2_PCD['boxes'], colors_PCD_MR2):
    box.set(color='red', facecolor=color)

for i in range(0, 30, 6):
    ax2.text(i + 1, -3, 'DQN', ha='center', va='center', fontsize=18, color='black')
    ax2.text(i + 2, -3, 'PCD', ha='center', va='center', fontsize=18, color='black')
    ax2.text(i + 3, -3, 'DQN', ha='center', va='center', fontsize=18, color='black')
    ax2.text(i + 4, -3, 'PCD', ha='center', va='center', fontsize=18, color='black')

for i in range(0, 30, 6):
    ax2.text(i + 1.5, -6, 'MR1', ha='center', va='center', fontsize=18, color='blue')
    ax2.text(i + 3.5, -6, 'MR2', ha='center', va='center', fontsize=18, color='red')

for i in range(0, 30, 6):
    if i == 0:
        ax2.text(i + 2.5, -9, '1 UAV', ha='center', va='center', fontsize=22, color='black')
    elif i == 6:
        ax2.text(i + 2.5, -9, '2 UAV', ha='center', va='center', fontsize=22, color='black')
    elif i == 12:
        ax2.text(i + 2.5, -9, '3 UAV', ha='center', va='center', fontsize=22, color='black')

# Añadir título superior
title = 'NormalConditions-metrics'
# plt.suptitle(title, fontsize=20, y=0.92)

# Ajusta el tamaño de las etiquetas de los ejes
axs.tick_params(axis='both', labelsize=18)
ax2.tick_params(axis='both', labelsize=18)

plt.savefig(title + '.pdf', format='pdf')
plt.show()
