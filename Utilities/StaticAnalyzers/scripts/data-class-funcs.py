#! /usr/bin/env python
import re
datacl = re.compile("^class ")
mbcl = re.compile("(base|data) class")
farg = re.compile("\(\w+\)")
nsep = re.compile("\:\:")
topfunc = re.compile("::produce\(|::analyze\(|::filter\(::beginLuminosityBlock\(|::beginRun\(")
onefunc = re.compile("edm::one::ED(Producer|Filter|Analyzer)Base::(produce|filter|analyze)")
getfunc = re.compile("edm::eventsetup::EventSetupRecord::get\(class (.*)&\)")
handle = re.compile("edm::ES.*Handle<(class|struct) (.*)>")
statics = set()
toplevelfuncs = set()
onefuncs = set()
classes = set()
dataclassfuncs = set()
badclasses = set()
esdclasses = set()

import networkx as nx
G=nx.DiGraph()
H=nx.DiGraph()


f = open('classes.txt.dumperft')
for line in f:
	if datacl.search(line) :
		classname = re.sub('\n','',re.sub(datacl,'',line))
		classes.add(classname)
f.close()

f = open('classes.txt.inherits')

for line in f:
	if datacl.search(line) :
		classname = re.sub('\n','',re.sub(datacl,'',line))
		classes.add(classname)
f.close()

f = open('functions.txt')
for line in f:
	fields = line.split("'")
	nspace = fname.split(fields[1])
	classname = nspace[-1]
	if classname in classes :
		name = classname[-1]
		badclasses.add(name)
f.close()

f = open('classes.txt.dumperall.sorted')
for line in f :
	if mbcl.search(line) :
		fields = line.split("'")
	if fields[2] == ' member data class ':
		H.add_edge(fields[1],fields[3])
	if fields[2] == ' templated member data class ':
		H.add_edge(fields[1],fields[3])
	if fields[2] == ' base class ':
		H.add_edge(fields[1],fields[3])

f.close()



f = open('db.txt')

for line in f :
	fields = line.split("'")
	if fields[2] == ' calls function ' :
		G.add_edge(fields[1],fields[3])
		funcname = farg.split(fields[3])[0]
		if getfunc.search(fields[3]) :
			dataclassfuncs.add(fields[3])
		if topfunc.search(fields[1]):
			toplevelfuncs.add(fields[1])
	if fields[2] == ' overrides function ' :
		G.add_edge(fields[1],fields[3])
	if fields[2] == ' static variable ' :
		G.add_edge(fields[1],fields[3])
		statics.add(fields[3])
f.close()



for tfunc in toplevelfuncs:
	for key in G[tfunc].keys():
		if onefunc.search(key):
			onefuncs.add(tfunc)
			break


for dataclassfunc in dataclassfuncs:
	m = getfunc.match(dataclassfunc)
	n = handle.match(m.group(1))
	if n : esdclass = n.group(2)
	else : esdclass = "None"
	esdclasses.add(esdclass)

objtree = nx.shortest_path(H)	

for esdclass in esdclasses:
	for badclass in badclasses:
		if esdclass in badclass:
			print esdclass+" is a flagged class"
		else:
			if H.has_node(badclass) and H.has_node(esdclass) and nx.has_path(H,esdclass, badclass) :
				print esdclass+" contains a flagged class "+badclass+" in objtree \'",
				for objt in objtree[esdclass][badclass]:
					print objt+";",
				print "\'."
		
	

paths = nx.shortest_path(G)
for dataclassfunc in dataclassfuncs:
	for tfunc in toplevelfuncs:
		if nx.has_path(G,tfunc,dataclassfunc):	
			m = getfunc.match(dataclassfunc)
			n = handle.match(m.group(1))
			if n : esdclass = n.group(2)
			else : esdclass = "None"
			print "Event setup data "+esdclass+" is accessed in call stack \'",
			for path in paths[tfunc][dataclassfunc]:
				print path+"; ",
			print "'."
