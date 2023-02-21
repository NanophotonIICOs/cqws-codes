import os
import matplotlib.pyplot as plt
import matplotlib

def load_style():
    stylelib_folder = os.path.join(matplotlib.get_configdir(), 'stylelib')
    if not os.path.exists(stylelib_folder):
        os.makedirs(stylelib_folder)

    filename = os.path.join(stylelib_folder, 'qws_solver_style.mplstyle')
    with open(filename, 'w') as f:
        f.write('figure.figsize : 7, 13\n')
        f.write('text.usetex : True\n')
        f.write('axes.labelsize : 15\n')
        f.write('legend.frameon : False\n')
        f.write('xtick.labelsize : 13\n')
        f.write('ytick.labelsize:  13\n')
        f.write('axes.linewidth : 2\n')
        f.write('xtick.minor.visible : True\n')
        f.write('xtick.major.size : 7\n')
        f.write('xtick.minor.size : 3.5\n')
        f.write('xtick.major.width : 2\n')
        f.write('xtick.minor.width : 2\n')
        f.write('xtick.direction : in\n')
        f.write('ytick.minor.visible : True\n')
        f.write('ytick.major.size : 7\n')
        f.write('ytick.minor.size : 3.5\n')
        f.write('ytick.major.width : 2\n')
        f.write('ytick.minor.width : 2\n')
        f.write('ytick.direction : in\n')

    # Check if the file has a valid format
    with open(filename, 'r') as f:
        contents = f.readlines()

    if 'figure.figsize :' not in contents[0]:
        print(f"The 'figure.figsize' parameter is not in the correct format in {filename}.")
        print("Please modify the file to fix this parameter.")
        return

    # If everything is OK, load the style
    plt.style.reload_library()
    plt.style.use('qws_solver_style')
    print("Custom style created and loaded successfully.")
    
# llamado a la función load_style() al final del script
load_style()
