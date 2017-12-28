# def log(func):
#     def wrapper(*args,**kw):
#         print("call",func.__name__)
#         return func(*args,**kw)
#     return wrapper
# def text(con):
#     def decorate(func):
#         def wrapper(*args,**kw):
#             print(con,func.__name__)
#             return func(*args,**kw)
#         return  wrapper
#     return decorate
# @text("123")
# def foo():
#     bar()
#     return  1
#
def bar():
    print(2)
# print(foo())
class tt:
    def __init__(self,name) -> None:
        self.name=name


class student(tt):
    name="er"

    def __init__(self, name) -> None:
        super().__init__(name)


st=student("1")
st.s=bar
st.s()
print(student.__bases__)
class Fib(object):
    def __init__(self) -> None:
        super().__init__()
        self.a,self.b=0,1
    def __iter__(self):
        return self
    def __next__(self):
        self.a,self.b=self.b,self.a+self.b
        if self.a>10:
            raise StopIteration()
        return self.a
    def __getitem__(self, item):
        if isinstance(item,int):
            a,b,=1,1
            for i in range(item):
                a,b=b,a+b
            return a
        if isinstance(item, slice):
            start=item.start
            end=item.stop
            if start is None:
                start=0
            a,b=1,1
            L=[]
            for x in range(end):
                if x>=start:
                    L.append(a)
                a,b=b,a+b
            return L

for i in Fib():
    print(i)
print(Fib()[2:6])