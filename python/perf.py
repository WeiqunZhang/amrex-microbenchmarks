#!/usr/bin/env python3

from numpy import *
import matplotlib.pyplot as plt

# ncell=256, max_grid_size=64

tests =       ['mb'    , 'cb'    , 'br'    , 'dsum'  , 'max'   ,  'scn'  , 'jac'   , 'aos'   , 'gsrb', 'parser']
thip  = array([3.21e-04, 6.90e-04, 2.46e-04, 2.61e-04, 2.80e-04, 1.49e-03, 8.57e-04, 2.66e-03, 3.94e-04, 2.54e-3])
tcuda = array([3.38e-04, 9.70e-04, 2.73e-04, 2.28e-04, 2.30e-04, 1.24e-03, 8.51e-04, 2.41e-03, 4.08e-04, 1.64e-3])

x = arange(len(tests))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, tcuda/tcuda, width, color='#76B900', label='NVIDIA')
rects2 = ax.bar(x + width/2, tcuda/thip , width, color='#ED1C24', label='AMD')

plt.ylim(0,1.45)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Performance')
ax.set_title('Microbenchmark')
ax.set_xticks(x)
ax.set_xticklabels(tests)
ax.legend()

#ax.bar_label(rects1, padding=3)
#ax.bar_label(rects2, padding=3)
#ax.bar_label(rects3, padding=3)

fig.tight_layout()

plt.show()

