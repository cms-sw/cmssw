#! /usr/bin/env python
import re
warning = re.compile("^function ")
tab = re.compile("\s+")
topfunc = re.compile("::produce\(|::analyze\(|::filter\(")
edmns = re.compile("::ED(Producer|Analyzer|Filter)")
keyword = re.compile("calls|overrides|variable|edmplugin")
paths = re.compile(".*?\s*src/([A-Z].*?/[A-z].*?)(/.*?):(.*?):(.*?)")
from collections import defaultdict

gets = defaultdict(set)
callby = defaultdict(set)
calls = defaultdict(set)
plugins = set()

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
		if fields[0].strip() == 'edmplugin type':
			plugins.add(fields[1])
f.close()

def stackup(str):
	for call in callby[str]:
		if call not in stack:
			stack.append(call)
			stackup(call)
	return


funcs=defaultdict(list)

for key in gets: 
	for get in gets[key]:
		func = get+"#"+key
		stack = list()
		stack.append(get)
		stackup(get)
		funcs[func].append(stack)

import copy

for func in sorted(funcs.keys()):
	get,var = func.split("#")
	clone = copy.deepcopy(funcs[func])
	for fields in clone:
		found = ""
		while fields:
			field = fields.pop()
			if topfunc.search(field) and not found:
				fqn = topfunc.split(field)[0]
				if fqn in plugins:
					print "Non-const static var '"+var+"' is accessed in call stack '"+field+"->",
					found = field
			if field in calls[found] and found :
				print field+"->",
				found = field
			if field == get and found :
				print field 

for func in sorted(funcs.keys()):
	get,var = func.split("#")
	clone = copy.deepcopy(funcs[func])
	for fields in clone: 
		found = ""
		while fields:
			field = fields.pop()
			if topfunc.search(field) and not found:
				fqn = topfunc.split(field)[0]
				if fqn in plugins:
					print "In call stack '"+field+"->",
					found = field
			if field in calls[found] and found :
				print field+"->",
				found = field
			if field == get and found :
				print field + "' non-const static var '"+var+"' is accessed."

