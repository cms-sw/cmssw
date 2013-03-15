#! /usr/bin/env python
import re
warning = re.compile("warning: function")
tab = re.compile("\s+")
topfunc = re.compile("::produce$|::analyze$|::filter$")
paths = re.compile(".*?\s*src/([A-Z].*?/[A-z].*?)(/.*?):(.*?):(.*?)")
from collections import defaultdict

gets = defaultdict(list)
calls = defaultdict(list)

f = open('log.txt')
lines=[]

for line in f:
	if warning.search(line):
		line = line.strip();
#		print line
		fields = line.split("\'")
#		print fields
		if fields[2] == ' calls ' :
#			print fields[3]+" here\n"
			if fields[3].strip() not in calls[fields[1]]:
				calls[fields[1]].append(fields[3].strip())
		else : 
#			print fields[3]+" not\n"
			if fields[3].strip() not in gets[fields[1]]:
				gets[fields[1]].append(fields[3].strip())
		if topfunc.search(fields[1]):
			dirs = paths.match(fields[0])
			filename = dirs.group(1)+dirs.group(2)
			line = filename+";"+fields[1]
			if line not in lines:
				lines.append(line)

f.close()


lines.sort()

import pprint
#pprint.pprint(gets)
#pprint.pprint(calls)
	
def funcprint(str,nspaces):
	"This prints out the get and calls of a function"
	print "".join((nspaces * "\t")+"function:\t"+str) 
	for l in gets[str]:
#		print l
		lf = l.split(" edm::Handle ")
#		print lf
		if len(lf) == 2 :
			print "".join(((nspaces+1) * "\t")+"acceses:\t"+lf[1].strip())
			print "".join(((nspaces+2) * "\t")+"label:\t"+lf[0].strip())
		else :
			print "".join(((nspaces+1) * "\t")+"acceses:\t"+l.strip())
	for call in calls[str]:
		print "".join(((nspaces+1) * "\t")+"calls:\t"+call.strip())
		if call != str: funcprint(call,(nspaces+2))
	return

for line in lines:
	fields = line.split(";")
#	print fields
	print "Package and filename: "+fields[0]
#	print fields[1]
	funcprint(fields[1],1)
	print "\n"
