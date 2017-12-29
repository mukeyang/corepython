class P(object):
    a=1


    def foo(self):
        print("P()")
class C(P):
    a=2
    def foo(self):
        super().foo()

# class W(P,C):
#     pass


# c=C("c")
# c.foo()
# P.foo(c)
# print(C.__mro__)
# print(W.__mro__)
class P1(object):
    def foo(self):
        print("P1")
    def __f(self):
        print("p1")
class P2(object):
    def foo(self):
        print("p2")

    def __f(self):
        print("p1")
    def bar(self):
        print("p2 bar2")
class C1(P1,P2):
    pass
class C2(P1,P2):
    def bar(self):
        print("C2 bar")
class GC(C1,C2):
    pass
print(GC.__mro__)
class RoundFloatManual(object):
    def __init__(self,val) -> None:
        assert  isinstance(val,float),"value must be a float"
        self.value=val

    def __str__(self) -> str:
        return "{:.2f}".format(self.value)
class Timer60(object):
    def __init__(self,hr,min) -> None:
        self.hr=hr
        self.min=min
    def __str__(self):
        return "{}:{}".format(self.hr,self.min)
    __repr__=__str__
    def __add__(self, other):

        return self.__class__(self.hr+other.hr,self.min+other.min)
    def __iadd__(self, other):
        self.hr+=other.hr
        self.min+=other.min
        return self
time=Timer60(3,20)
print(time)
time1=Timer60(3,10)

print(time+time1)
p1=P1()
p2=P2()
def f(P1):
    P1.__f()
f(p1)
f(p2)

