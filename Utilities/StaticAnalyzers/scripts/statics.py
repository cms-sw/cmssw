#! /usr/bin/env python
import re
topfunc = re.compile("::produce\(|::analyze\(|::filter\(")
statics = set()
toplevelfuncs = set()

import networkx as nx
G=nx.DiGraph()

f = open('db.txt')

for line in f :
	fields = line.split("'")
	if fields[2] == ' calls function ' :
		G.add_edge(fields[1],fields[3])
		if topfunc.search(fields[1]):
			toplevelfuncs.add(fields[1])
	if fields[2] == ' overrides function ' :
		G.add_edge(fields[1],fields[3])
	if fields[2] == ' static variable ' :
		G.add_edge(fields[1],fields[3])
		statics.add(fields[3])
f.close()

paths = nx.shortest_path(G)

for static in statics:
	for tfunc in toplevelfuncs:
		if nx.has_path(G,tfunc,static):
			print "Non-const static variable \'"+static+"\' is accessed in call stack \'",
			for path in paths[tfunc][static]:
				if not path == static : print path+"; ",
			print "'."
			print "In call stack '",
			for path in paths[tfunc][static]:
				if not path == static : print path+"; ",
				else : print "\' non-const static variable \'"+static+"\' is accessed."

