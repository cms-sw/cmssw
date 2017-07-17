#! /usr/bin/env python
import pydablooms
CAPACITY=5000
ERROR_RATE=float(1)/CAPACITY
bloom = pydablooms.Dablooms(capacity=CAPACITY, error_rate=ERROR_RATE,filepath='bloom.bin')

f = open('classes.txt','r')
g = open('classnames.txt','w')
for line in f :
	fields = line.split("'")
	if fields[0] == 'class ' :
		g.write(fields[1]+'\n')

f.close()
g.close()
h = open('classnames.txt','rb')
i = 0
for line in h:
	bloom.add(line.rstrip(), i)
	i += 1

bloom.flush()

