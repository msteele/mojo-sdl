#!/usr/bin/env python3

import sys
import matplotlib as mpl

cmap_name = 'afmhot'
if len(sys.argv) > 1:
    cmap_name = sys.argv[1]

cmap8bit = mpl.colormaps[cmap_name].resampled(256)

print("alias vec3 = StaticTuple[3]")
print("alias cmap = StaticTuple[256] (")
for i in range(256):
    c = cmap8bit(i)
    print(f'vec3({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)}),')
print(')')
