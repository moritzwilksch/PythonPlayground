# %%
def isMirror(x,y):
    x, y = int(x), int(y)
    mirrorList = [(6,9), (1,1), (0,0), (8,8)]
    return (x,y) in mirrorList or (y,x) in mirrorList

def isUsdNumber(x):
    x = str(x)
    if True in list(map(lambda x: int(x) in [2,3,4,5,7], x)):
        return False
    for i in range(1, round(len(x)/2 + 0.1)+1):
        if not isMirror(x[i-1], x[-i]):
            return False
    return True


def upsidedown(x,y):
    l = []
    x, y = int(x), int(y)
    for i in range(x,y):
        if isUsdNumber(i):
            l.append(i)
    print("LIST = ", l)
    return len(l)



print(isUsdNumber(1961))
print(isUsdNumber(88))
print(isUsdNumber(25))

print(upsidedown('0','10'))
print(upsidedown('6','25'))
print(upsidedown('10','100'))
print(upsidedown('100','1000'))
print(upsidedown('100000','12345678900000000'))
# %%
