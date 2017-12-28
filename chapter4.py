# from  string import  Template
# print("{1}{0}{1}".format("si","yang"))
# s=Template("it is ${name} who are ${tear}")
# print(s.substitute(name="12",tear="12"))
# print(type("123"))
#
# a="hello"
# print(a.count("h"))
# print(a.rjust(10).lstrip())
# print((a*2)[-1:1:- 1])
# print(oct(8 ))
# s="abcdef"
# for a,b in zip(a,s):
#     print(a,b)
# for i in range(-1,-len(s),-1):
#     print(s[:i])
# for i in  reversed(s):
#     print(i)
# list=[1,2,3,1,2]
# print(max(list))
# del  list[1]
# list.pop()
#
# list.remove(1)
# print(list)
# tuple=(1,2,3)
dict1={}.fromkeys(("x", "y"), -1)
print(dict1)
for i in dict1.items():
    print(i)
s=set("hsadhkjla")
c=set("rt")
a=[123,23]
b=[1,3]
x,y=4,3
smaller=x if x<y else y