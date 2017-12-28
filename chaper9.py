import os

t = open("1.txt", "a+",encoding='UTF-8')
for eachline in t:
    print(eachline,)
t.write("python"+"\n")
print(t.tell())
print(os.stat("1.txt").st_size)
print(os.getcwd())
print(os.listdir(os.getcwd()))
os.chdir("c:/")
print(os.getcwd())
print(os.listdir(os.getcwd()))
