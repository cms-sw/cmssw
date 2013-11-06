#! /usr/bin/env python
import re
warning = re.compile("^function ")
tab = re.compile("\s+")
topfunc = re.compile("::produce\(|::analyze\(|::filter\(")
edmns = re.compile("(edm::ED|edm::one|edm::stream|edm::global)(Producer|Analyzer|Filter)")
keyword = re.compile("calls|overrides|variable")
paths = re.compile(".*?\s*src/([A-Z].*?/[A-z].*?)(/.*?):(.*?):(.*?)")
from collections import defaultdict

gets = defaultdict(list)
calls = defaultdict(list)

f = open('db.txt')

for line in f :
	fields = line.split("'")
	if keyword.search(line) :
		if fields[2] == ' calls function ' :
			if fields[1] not in calls[fields[3]]:
				calls[fields[3]].append(fields[1])
		if fields[2] == ' overrides function ' :
			if fields[3] not in calls[fields[1]] and not edmns.search(fields[3]) :
				calls[fields[1]].append(fields[3])		
		if fields[2] == ' static variable ' :
				if fields[1] not in gets[fields[3]]:
					gets[fields[3]].append(fields[1])

f.close()

import pdb
	
def callstack(str):
	for call in calls[str]:
		if call not in stack:
			stack.add(call)
			callstack(call)
	return

funcs=[]

for key in gets: 
	for get in gets[key]:
		func = get+" # "+key
		stack = set()
		callstack(get)
		for item in sorted(stack):
			func += " # "+item
		funcs.append(func)


for func in sorted(set(funcs)):
	fields = func.split("#")
	for field in fields:
		if topfunc.search(field):
			print fields[0]+ " # "+ fields[1]+ " # " + field
		
