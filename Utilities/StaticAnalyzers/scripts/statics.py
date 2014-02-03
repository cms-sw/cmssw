#! /usr/bin/env python
import re
warning = re.compile("^function ")
tab = re.compile("\s+")
topfunc = re.compile("::produce\(|::analyze\(|::filter\(")
keyword = re.compile("calls|overrides|variable")
paths = re.compile(".*?\s*src/([A-Z].*?/[A-z].*?)(/.*?):(.*?):(.*?)")
from collections import defaultdict

gets = defaultdict(list)
calls = defaultdict(list)
stack = list()

f = open('db.txt')

for line in f :
#	print line
	fields = line.split("\'")
#	print fields
	if keyword.search(line) :
		if fields[2] == ' calls function ' :
			if fields[1].strip() not in calls[fields[3].strip()]:
				calls[fields[3].strip()].append(fields[1].strip())
		if fields[2] == ' overrides function ' :
			if fields[3].strip() not in calls[fields[1].strip()]:
				calls[fields[1].strip()].append(fields[3].strip())		
		if fields[2] == ' static variable ' :
				if fields[1].strip() not in gets[fields[3].strip()]:
					gets[fields[3].strip()].append(fields[1].strip())

f.close()

import pprint
#pprint.pprint(gets)
#pprint.pprint(calls)
	
def callstack(str):
#	print str
	if calls[str] : 
#		print calls[str]
		for call in calls[str]:
			if call not in stack : 
        			stack.append(call)
				callstack(call.strip())
	return

funcs=[]

for key in gets: 
	for get in gets[key]:
		func = get+" # "+key,
		del stack[:]
		callstack(get.strip())
		stack.sort()
		for item in  stack:
			func += item,
#			func += "\t"+item,
	
	if func not in funcs:
		funcs.append(func)
	

funcs.sort()
#pprint.pprint(funcs)

for func in funcs:
#	print func
	for field in func:
#		print field
		if topfunc.search(field):
			print func[0]+ " # "+ field
#			print field + " # "+ func[0]
#			print field 
		
