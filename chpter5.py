f=open("1.txt")
for i in f:
    len([word for i in f for word in i.split()])
    print(i)