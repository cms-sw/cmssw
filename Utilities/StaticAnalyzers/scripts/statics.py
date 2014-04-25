#! /usr/bin/env python
import re
topfunc = re.compile("::(produce|analyze|filter|beginLuminosityBlock|beginRun)\(")
baseclass = re.compile("edm::(one::|stream::|global::)?ED(Producer|Filter|Analyzer)(Base)?")
farg = re.compile("\(.*\)")
statics = set()
toplevelfuncs = set()

import networkx as nx
G=nx.DiGraph()

f = open('db.txt')

for line in f :
	fields = line.split("'")
	if fields[2] == ' calls function ' :
		G.add_edge(fields[1],fields[3],kind=fields[2])
	if fields[2] == ' overrides function ' :
		if baseclass.search(fields[3]) :
			if topfunc.search(fields[3]) : toplevelfuncs.add(fields[1])
			G.add_edge(fields[1],fields[3],kind=' overrides function ')
		else :
			G.add_edge(fields[3],fields[1],kind=' calls function ')
	if fields[2] == ' static variable ' :
		G.add_edge(fields[1],fields[3],kind=' static variable ')
		statics.add(fields[3])
f.close()


for static in statics:
	for tfunc in toplevelfuncs:
		if nx.has_path(G,tfunc,static): 
			path = nx.shortest_path(G,tfunc,static)
			print "Non-const static variable \'"+re.sub(farg,"()",static)+"\' is accessed in call stack ",
			print " \'",
			for p in path :			
				print re.sub(farg,"()",p)+"; ",
			print " \'. ",
			for key in  G[tfunc].keys() :
				if 'kind' in G[tfunc][key] and G[tfunc][key]['kind'] == ' overrides function '  :
					print "'"+re.sub(farg,"()",tfunc)+"'"+G[tfunc][key]['kind']+"'"+re.sub(farg,"()",key)+"'",
			print ""
			path = nx.shortest_path(G,tfunc,static)
			print "In call stack ' ",
			for p in path:
				print re.sub(farg,"()",p)+"; ",
			print "\'",
			print " non-const static variable \'"+re.sub(farg,"()",static)+"\' is accessed. ",
			for key in  G[tfunc].keys() :
				if 'kind' in G[tfunc][key] and G[tfunc][key]['kind'] == ' overrides function '  :
					print "'"+re.sub(farg,"()",tfunc)+"'"+G[tfunc][key]['kind']+"'"+re.sub(farg,"()",key)+"'",
			print

