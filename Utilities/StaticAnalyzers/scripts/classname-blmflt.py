#! /usr/bin/env python
import pydablooms
CAPACITY=5000
ERROR_RATE=float(1)/CAPACITY
BYTES=2
bloom = pydablooms.Dablooms(capacity=CAPACITY, error_rate=ERROR_RATE,filepath='bloom.bin')

f = open('classes.txt')

for line in f :
	fields = line.split("'")
	if fields[0] == 'class ' :
		bloom.add(fields[1],BYTES)
		bloom.check(fields[1])
f.close()
