import sys
if len(sys.argv) !=3 :
  print "usage:" + sys.argv[0] + "   bla bla"
  exit
# go
mm=0.
n=0
ave=0
file1 = open(sys.argv[1])
file2 = open(sys.argv[2])
for l1 in file1.readlines() :
    l2 = file2.readline()
    a1 =l1.split('(')
    a2 = l2.split('(')
    if len(a1)<2 :
        continue
    p1 = a1[1].split(',')
    p1[2] = p1[2].rstrip(') \n')
    p2 = a2[1].split(',')
    p2[2] = p2[2].rstrip(') \n')
    f1 = [float(i) for i in p1]
    f2 = [float(i) for i in p2]
    diff = []
    for i in range(len(f1)):
        diff.append(f1[i]-f2[i])
    for x in diff:
        mm = max(mm,abs(x))
        if (x!=0) :
            n+=1
            ave+=abs(x)
print mm, n, ave/n
               
