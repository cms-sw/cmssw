#! /usr/bin/env python
import re
stdcl = re.compile("^std::")
datacl = re.compile("^class ")
bfunc = re.compile("^function ")
mbcl = re.compile("(base|data) class")
farg = re.compile("(.*)\(\w+\)")
nsep = re.compile("\:\:")
topfunc = re.compile("::(produce|analyze|filter|beginLuminosityBlock|beginRun)\(")
baseclass = re.compile("edm::(one::|stream::|global::)?ED(Producer|Filter|Analyzer)(Base)?")
getfunc = re.compile("edm::eventsetup::EventSetupRecord::get\((.*)&\) const")
handle = re.compile("(.*),?class edm::ES(.*)Handle<(.*)>")
statics = set()
toplevelfuncs = set()
onefuncs = set()
dataclassfuncs = set()
virtfuncs = set()
virtclasses = set()
badfuncs = set()
badclasses = set()
esdclasses = set()
dataclasses = set()
flaggedclasses = set()
memberclasses = set()
derivedclasses = set()

import networkx as nx
G=nx.DiGraph()
H=nx.DiGraph()


f = open('classes.txt.dumperall')
for line in f :
	if mbcl.search(line) :
		fields = line.split("'")
		if fields[2] == ' member data class ':
			H.add_edge(fields[1],fields[3],kind=fields[2])
		if fields[2] == ' templated member data class ':
			H.add_edge(fields[1],fields[3],kind=' member data class ')
			H.add_edge(fields[1],fields[5],kind=' templated member data class type ')
		if fields[2] == ' base class ':
			H.add_edge(fields[1],fields[3],kind=fields[2])
f.close()

f = open('function-calls-db.txt')

for line in f :
	if not bfunc.search(line) : continue
	fields = line.split("'")
	if fields[2] == ' calls function ' :
		G.add_edge(fields[1],fields[3],kind=' calls function ')
		if getfunc.search(fields[3]) :
			dataclassfuncs.add(fields[3])
			m = getfunc.match(fields[3])
			n = handle.match(m.group(1))
			if n : o = n.group(3)
			else : o = m.group(1)
			p = re.sub("class ","",o)
			dataclass = re.sub("struct ","",p)
			dataclasses.add(dataclass)
	if fields[2] == ' overrides function ' :
		if baseclass.search(fields[3]) :
			G.add_edge(fields[1],fields[3],kind=' overrides function ')
			if topfunc.search(fields[3]) : toplevelfuncs.add(fields[1])
		else : G.add_edge(fields[3],fields[1], kind=' calls override function ')
	if fields[2] == ' static variable ' :
		G.add_edge(fields[1],fields[3],kind=' static variable ')
		statics.add(fields[3])
f.close()

for n,nbrdict in H.adjacency_iter():
	for nbr,eattr in nbrdict.items():
		if nbr in dataclasses and 'kind' in eattr and eattr['kind'] == ' base class '  :
			dataclasses.add(n)

for dclass in sorted(dataclasses):
	if dclass in H:
		for n in nx.dfs_preorder_nodes(H,dclass):
			memberclasses.add(n)			

for n,nbrdict in H.adjacency_iter():
	for nbr,eattr in nbrdict.items():
		if nbr in memberclasses and 'kind' in eattr and eattr['kind'] == ' base class '  :
			memberclasses.add(n)

for mclass in sorted(memberclasses):
	dataclasses.add(mclass)

for dclass in sorted(dataclasses):
	print "class '"+dclass+"'"


