#! /usr/bin/env python
import re
topfunc = re.compile("::produce\(|::analyze\(|::filter\(::beginLuminosityBlock\(|::beginRun\(")
onefunc = re.compile("edm::one::ED(Producer|Filter|Analyzer)Base::(produce|filter|analyze|beginLuminosityBlock|beginRun)")
farg = re.compile("\(.*\)")
statics = set()
toplevelfuncs = set()
onefuncs = set()

import networkx as nx
G=nx.DiGraph()

f = open('db.txt')

for line in f :
	fields = line.split("'")
	if fields[2] == ' calls function ' :
		G.add_edge(fields[1],fields[3],kind=fields[2])
		if topfunc.search(fields[1]):
			toplevelfuncs.add(fields[1])
	if fields[2] == ' overrides function ' :
		G.add_edge(fields[1],fields[3],kind=fields[2])
	if fields[2] == ' static variable ' :
		G.add_edge(fields[1],fields[3],kind=fields[2])
		statics.add(fields[3])
f.close()


paths = nx.shortest_path(G)

for static in statics:
	for tfunc in toplevelfuncs:
		if nx.has_path(G,tfunc,static): 
			print "Non-const static variable \'"+re.sub(farg,"()",static)+"\' is accessed in call stack \'",
			for path in paths[tfunc][static]:
				if not path == static : print re.sub(farg,"()",path)+"; ",
			print "\'. ",
			for key in  G[tfunc].keys() :
				if 'kind' in G[tfunc][key] and G[tfunc][key]['kind'] == ' overrides function '  :
					print "'"+tfunc+"'"+G[tfunc][key]['kind']+"'"+key+"'"
			print ""
			print "In call stack '",
			for path in paths[tfunc][static]:
				if not path == static : print re.sub(farg,"()",path)+"; ",
				else : print "\' non-const static variable \'"+re.sub(farg,"()",static)+"\' is accessed. ",
			for key in  G[tfunc].keys() :
				if 'kind' in G[tfunc][key] and G[tfunc][key]['kind'] == ' overrides function '  :
					print "'"+tfunc+"'"+G[tfunc][key]['kind']+"'"+key+"'"
			print ""

