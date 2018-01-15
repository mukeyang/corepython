import  os
a="hello"
b=[x**2 for x in  range(8) if not x%2]
print("1213")
a='''nidhka\ngbdg'''
print(a)
adict={"host":"earth"}
print(b[-1:1])
print((a*2)[3:-1])
x,y,z=1,2,3
fname=""
def addMe(x):
    return x+x


class FooClass(object):
    version=0.1

    def __init__(self,name="yang") -> None:
        self.name=name;super().__init__()
    def showname(self):
        print("name is",self. __class__.__name__)


fc=FooClass()
print(addMe(3))
ls=os.linesep
while True:
    if os.path.exists(fname):
        print(fname,"already exists")
    else:
        break
all=[]
while True:
    entry=input()
    if entry==".":
        break
    else:
        all.append(entry)
fobj=open(fname)
fobj.writelines([x for x in all])
fobj.close()

