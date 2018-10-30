import sys

def func(var):
	out = var.split('/')
	print (out[27])
	return out[27]

def betterfunc(var):
	ar = var.split(':')
	for x in range (0,len(ar)):
		if(ar[x].find('boost') >= 0):
			out = ar[x]
	out = out.replace('/include','')
	print (out)
	return out

betterfunc(sys.argv[1])
