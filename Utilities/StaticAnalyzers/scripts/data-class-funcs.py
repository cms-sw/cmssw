#! /usr/bin/env python
import re
datacl = re.compile("^class ")
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
import networkx as nx
G=nx.DiGraph()
H=nx.DiGraph()

f = open('classes.txt.dumperft')
for line in f:
       if datacl.search(line) :
               classname = line.split("'")[1]
               esdclasses.add(classname)
f.close()

f = open('classes.txt.inherits')
for line in f:
       if datacl.search(line) :
               classname = line.split("'")[1]
               esdclasses.add(classname)
f.close()


f = open('class-checker.txt')
for line in f:
	if mbcl.search(line):
		fields = line.split("'")
		classname = fields[1]
		funcname = fields[3]
		badclasses.add(classname)
		badfuncs.add(funcname)
f.close()


f = open('classes.txt.dumperall')
for line in f :
	if mbcl.search(line) :
		fields = line.split("'")
		if fields[2] == ' member data class ':
			H.add_edge(fields[1],fields[3],kind=fields[2])
		if fields[2] == ' templated member data class ':
			H.add_edge(fields[1],fields[3],kind=fields[2])
		if fields[2] == ' base class ':
			H.add_edge(fields[1],fields[3],kind=fields[2])
f.close()


f = open('db.txt')

for line in f :
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


for n,nbrdict in G.adjacency_iter():
	for nbr,eattr in nbrdict.items():
		if n in badfuncs or nbr in badfuncs :
			if 'kind' in eattr and eattr['kind'] == ' overrides function '  :
				print "'"+n+"'"+eattr['kind']+"'"+nbr+"'"
				virtfuncs.add(nbr)
print

for n,nbrdict in H.adjacency_iter():
	for nbr,eattr in nbrdict.items():
		if n in badclasses and 'kind' in eattr and eattr['kind'] == ' base class '  :
			virtclasses.add(nbr)


for n,nbrdict in H.adjacency_iter():
	for nbr,eattr in nbrdict.items():
		if nbr in dataclasses and 'kind' in eattr and eattr['kind'] == ' base class '  :
			dataclasses.add(n)

print "flagged functions found by checker"
for dfunc in sorted(badfuncs) : 
	print dfunc
print

print "flagged classes found by checker "
for dclass in sorted(badclasses) :
	print dclass
print

print "flagged classes found by checker union get" 
for dclass in sorted(dataclasses.intersection(badclasses)) :
	print dclass
print

print "flagged classes found by checker union dumper" 
for dclass in sorted(esdclasses.intersection(badclasses)) :
	print dclass
print

print "classes inheriting from flagged classes"
for dclass in sorted(virtclasses):
	print dclass
print

print "functions overridden by flagged functions"
for dfunc in sorted(virtfuncs):
	print dfunc
print


for badclass in sorted(badclasses):
	print "Event setup data class '"+badclass+"' is flagged."
	flaggedclasses.add(badclass)
print

for virtclass in sorted(virtclasses):
	print "Event setup data class '"+virtclass+"' is flagged because inheriting class is flagged"
	flaggedclasses.add(virtclass)
print

for badclass in sorted(badclasses):
	for dataclass in sorted(dataclasses):
		if H.has_node(badclass) and H.has_node(dataclass):
			if nx.has_path(H,dataclass, badclass) :
				print "Event setup data class '"+dataclass+"' contains or inherits from flagged class '"+badclass+"'."
				flaggedclasses.add(dataclass)
			
print


for dataclassfunc in sorted(dataclassfuncs):
	for tfunc in sorted(toplevelfuncs):
		if nx.has_path(G,tfunc,dataclassfunc):
			m = getfunc.match(dataclassfunc)
			n = handle.match(m.group(1))
			if n : o = n.group(3)
			else : o = m.group(1)
			p = re.sub("class ","",o)
			dataclass = re.sub("struct ","",p)
			for flaggedclass in sorted(flaggedclasses):
				if re.search(flaggedclass,dataclass) :
					print "Flagged event setup data class '"+dataclass+"' is accessed in call stack '",
					path = nx.shortest_path(G,tfunc,dataclassfunc)
					for p in path:
						print p+"; ",
					print "' ",
					for key in  G[tfunc].keys() :
						if 'kind' in G[tfunc][key] and G[tfunc][key]['kind'] == ' overrides function '  :
							print "'"+tfunc+"'"+G[tfunc][key]['kind']+"'"+key+"'",
					print ""
					print "In call stack '",
					path = nx.shortest_path(G,tfunc,dataclassfunc)
					for p in path:
						print p+"; ",
					print "' flagged event setup data class '"+dataclass+"' is accessed. ",
					for key in  G[tfunc].keys() :
						if 'kind' in G[tfunc][key] and G[tfunc][key]['kind'] == ' overrides function '  :
							print "'"+tfunc+"'"+G[tfunc][key]['kind']+"'"+key+"'",
					print ""

