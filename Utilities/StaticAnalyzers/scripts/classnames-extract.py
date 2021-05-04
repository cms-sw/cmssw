#! /usr/bin/env python

f = open('classes.txt','r')
g = open('classnames.txt','w')
for line in f :
	fields = line.split("'")
	if fields[0] == 'class ' :
		g.write(fields[1]+'\n')

f.close()
g.close()
