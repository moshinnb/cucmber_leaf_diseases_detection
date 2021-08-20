s="ELECTROBIT"


s=s[::-1]
print(s)
s=[1,2,3]
s.insert(3,5)
print(s)
a="abcd"
b="bcmda"

res=map(lambda x,y:x==y,sorted(a),sorted(b))
for i in res:
    if(i==False):
        print(i)
        print("not equal")

#print("equal")
#b=sorted(b)
#print(a)
#print(b)

s="mohsin bepari"
m=""
for i in s.split(" "):
    m+=i.capitalize()
print(m)
