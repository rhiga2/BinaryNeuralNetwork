import matplotlib
import matplotlib.pyplot as plt

# Inverse Blues colormap
cdict = {
    'red':   ((0.0,  1.0, 1.0), (1.0,  0.0, 0.0)),
    'green': ((0.0,  1.0, 1.0), (1.0,  .15, .15)),
    'blue':  ((0.0,  1.0, 1.0), (1.0,  0.4, 0.4)),
    'alpha': ((0.0,  0.0, 0.0), (1.0,  1.0, 1.0))}
plt.register_cmap(name='InvBlueA', data=cdict)

# Customize my figure style
plt.rc('figure', figsize=(8,4), dpi=96, facecolor='#FFFFFF00', autolayout=False)
plt.rc('lines', linewidth=1)
plt.rc('axes', axisbelow=True, titlesize=10, titleweight=500,
   labelsize=9, labelweight=400, linewidth=0.5, facecolor='#FFFFFF00')
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('axes.spines', right=False, top=False)
plt.rc('grid', linestyle='-', linewidth=0.5)

# Get a decent figure font
matplotlib.font_manager._rebuild()
plt.rc('font', family='Avenir Next LT Pro', weight=400, size=9)

# Light colors
plt.rc('axes', edgecolor='#404040')
plt.rc('grid', color='#DDDDDD')
plt.rc('xtick', color='#222222')
plt.rc('ytick', color='#222222')
plt.rc('text', color='#222222')
plt.rc('image', cmap='InvBlueA')
plt.rc('legend', facecolor='#FFFFFF55', framealpha=0.5)
