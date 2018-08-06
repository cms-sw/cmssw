#! /usr/bin/env python
from __future__ import print_function
import re
topfunc = re.compile("::(produce|analyze|filter|beginLuminosityBlock|beginRun|beginStream)\(")
baseclass = re.compile("edm::(one::|stream::|global::)?ED(Producer|Filter|Analyzer)(Base)?")
farg = re.compile("\(.*\)")
toplevelfuncs = set()
epfunc = re.compile("TGraph::(.*)\(.*\)")
skipfunc = re.compile("TGraph::IsA\(.*\)")
epfuncs=set()

import networkx as nx
G=nx.DiGraph()

f = open('function-calls-db.txt')

for line in f :
	fields = line.split("'")
	if fields[2] == ' calls function ' :
		if not skipfunc.search(line) : 
			G.add_edge(fields[1],fields[3],kind=fields[2])
			if epfunc.search(fields[3])  : epfuncs.add(fields[3])
	if fields[2] == ' overrides function ' :
		if baseclass.search(fields[3]) :
			if topfunc.search(fields[3]) : toplevelfuncs.add(fields[1])
			G.add_edge(fields[1],fields[3],kind=' overrides function ')
		else :
			if not skipfunc.search(line) : 
				G.add_edge(fields[3],fields[1],kind=' calls override function ')
				if epfunc.search(fields[1]) : epfuncs.add(fields[1])
f.close()

for epfunc in sorted(epfuncs): print(epfunc)
print()

for epfunc in epfuncs:
	for tfunc in toplevelfuncs:
		if nx.has_path(G,tfunc,epfunc) : 
			path = nx.shortest_path(G,tfunc,epfunc)
			print("Call stack \'", end=' ')
			for p in path :			
				print(re.sub(farg,"()",p)+"; ", end=' ')
			print(" \'. ")

