#! /usr/bin/env python
import re
warning = re.compile("^function ")
tab = re.compile("\s+")
topfunc = re.compile("::produce\(|::analyze\(|::filter\(")
edmns = re.compile("::ED(Producer|Analyzer|Filter)")
keyword = re.compile("calls|overrides|variable")
paths = re.compile(".*?\s*src/([A-Z].*?/[A-z].*?)(/.*?):(.*?):(.*?)")
from collections import defaultdict

gets = defaultdict(set)
callby = defaultdict(set)
calls = defaultdict(set)

f = open('db.txt')

for line in f :
	fields = line.split("'")
	if keyword.search(line) :
		if fields[2] == ' calls function ' :
			if fields[1] not in callby[fields[3]]:
				callby[fields[3]].add(fields[1])
			if fields[3] not in calls[fields[1]]:
				calls[fields[1]].add(fields[3])
		if fields[2] == ' overrides function ' :
			if fields[3] not in callby[fields[1]] and not edmns.search(fields[3]) :
				callby[fields[1]].add(fields[3])
		if fields[2] == ' static variable ' :
				if fields[1] not in gets[fields[3]]:
					gets[fields[3]].add(fields[1])

f.close()

def stackup(str):
	for call in callby[str]:
		if call not in stack:
			stack.add(call)
			stackup(call)
	return

funcs=defaultdict(list)

for key in gets: 
	for get in gets[key]:
		func = get+" # "+key
		stack = set()
		stack.add(get)
		stackup(get)
		funcs[func].append(sorted(stack))


for func in sorted(funcs.keys()):
	for fields in funcs[func]:
		for field in fields:
			if topfunc.search(field) :
				print func + " # " + field
		
