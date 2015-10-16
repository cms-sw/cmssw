#! /usr/bin/env python
import re
stdcl = re.compile("^std::(.*)[^>]$")
stdptr = re.compile("(.*)_ptr$")
datacl = re.compile("^class ")
bfunc = re.compile("^function ")
mbcl = re.compile("(base|data) class")
farg = re.compile("(.*)\(\w+\)")
nsep = re.compile("\:\:")
topfunc = re.compile("::(produce|analyze|filter|beginLuminosityBlock|beginRun)\(")
rootclass = re.compile("T(H1|Tree|Enum|DataType|Class|Branch|Named|File)")
baseclass = re.compile("edm::(one::|stream::|global::)?ED(Producer|Filter|Analyzer)(Base)?")
globalclass = re.compile("edm::global::ED(Producer|Filter|Analyzer)(Base)?")
getfunc = re.compile("edm::eventsetup::EventSetupRecord::get\<.*\>\((.*)&\) const")
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
globalclasses = set()

import networkx as nx
G=nx.DiGraph()
H=nx.DiGraph()
I=nx.DiGraph()


f = open('classes.txt.dumperall')
for line in f :
	if mbcl.search(line) :
		fields = line.split("'")
		if fields[2] == ' member data class ':
			if not stdcl.search(fields[2]) : H.add_edge(fields[1],fields[3],kind=fields[2])
		if fields[2] == ' templated member data class ':
			H.add_edge(fields[1],fields[5],kind=fields[3])
		if fields[2] == ' base class ':
			H.add_edge(fields[1],fields[3],kind=fields[2])
			I.add_edge(fields[3],fields[1],kind=' derived class')
			if globalclass.search(fields[3]): 
				globalclasses.add(fields[1])
				print "edm::global class '"+fields[1]+"'"
f.close()


visited = set()
nodes = sorted(globalclasses)
for node in nodes:
	if node in visited:
		continue
	visited.add(node)			
	if node in H : stack = [(node,iter(H[node]))]
	if node in I :
		Q=nx.dfs_preorder_nodes(I,node)
		for q in Q:
			if q in H : 
				stack.append(  ( q, iter( H[q] ) ) )
	while stack:
		parent,children = stack[-1]
		try:
			child = next(children)
			if globalclass.search(child): visited.add(child)
			if rootclass.search(child): visited.add(child)
			print "class '"+parent+"'"
			if child not in visited:
				visited.add(child)
				if not stdcl.search(child): 
					print "\tclass '"+child+"'"
					stack.append( ( child, iter( H[child] ) ) )
					kind=H[parent][child]['kind']
					#print parent, kind, child 
					if stdptr.search(kind):
						if child in I :
							Q=nx.dfs_preorder_nodes(I,child)
							for q in Q :
								print "\t\tclass '"+q+"'"
								if q in H : 
									stack.append(  ( q, iter( H[q] ) ) )
		except StopIteration:
			stack.pop()
         
