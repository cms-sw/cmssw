#! /usr/bin/env python
import re
warning = re.compile("warning: function")
tab = re.compile("\s+")
handle = re.compile("edm\:\:Handle")
topfunc = re.compile("::produce$|::analyze$|::filter$")
paths = re.compile(".*?\s*src/([A-Z].*?/[A-z].*?)/.*")
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
		if handle.search(fields[3]):
			gets[fields[1]].append(fields[3].strip())
		else : 
			calls[fields[1]].append(fields[3].strip())
		if topfunc.search(fields[1]):
			line = fields[0]+";"+fields[1]
			lines.append(line)
#			print line
f.close()


lines.sort()

#import pprint
#pprint.pprint(gets)
#pprint.pprint(calls)
	
def funcprint(str,nspaces):
	"This prints out the get and calls of a function"
	print "".join((nspaces * "\t")+"function:\t"+str) 
	for l in gets[str]:
		lf = l.split("edm::Handle")
		print "".join(((nspaces+1) * "\t")+"acceses:\t"+lf[1].strip())
		print "".join(((nspaces+2) * "\t")+"label:\t"+lf[0].strip())
	for call in calls[str]:
		print "".join(((nspaces+1) * "\t")+"calls:\t"+call.strip())
		if call != str: funcprint(call,(nspaces+2))
	return

for line in lines:
	fields = line.split(";")
#	print fields
	dirs = paths.match(fields[0])
	print "Package: "+dirs.group(1)
	funcprint(fields[1],1)
	print "\n"
