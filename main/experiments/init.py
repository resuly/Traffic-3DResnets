import os,sys,glob,shutil

lists = glob.glob('./*/*.log')
lists += glob.glob('./*/*.tar')
lists += glob.glob('./*/*.pk')
lists += glob.glob('./*/*weights.json')

for f in lists:
    os.remove(f)

