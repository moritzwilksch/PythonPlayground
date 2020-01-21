def sieb(n,i=2,s = ()):
    print(f"CALL: sieb(n={n}, i={i}, s={s})")
    if s == ():
        s = list(range(1,n+1))
    if i > n:
        return s 
    else:
        s = list(filter(lambda x: (x== i) or (x%i!=0),s))
        i += 1
        print(f"RETURN sieb(n={n}, i={i}, s={s})")
        return sieb(n,i,s)

print(sieb(25))