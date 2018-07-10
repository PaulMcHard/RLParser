import os

dirname = os.getcwd()
print(dirname)
files = []
for file in os.listdir(dirname+'/data/'):
    if file.endswith(".DAT"):
        files.append(file)
        print(file)
