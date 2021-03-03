#! /usr/bin/env python
from __future__ import print_function
import re
topfunc = re.compile(r"::(accumulate|acquire|startingNewLoop|duringLoop|endOfLoop|beginOfJob|endOfJob|produce|analyze|filter|beginLuminosityBlock|beginRun|beginStream|streamBeginRun|streamBeginLuminosityBlock|streamEndRun|streamEndLuminosityBlock|globalBeginRun|globalEndRun|globalBeginLuminosityBlock|globalEndLuminosityBlock|endRun|endLuminosityBlock)\(")

baseclass = re.compile(r"edm::(one::|stream::|global::)?(ED(Producer|Filter|Analyzer|(IterateNTimes|NavigateEvents)?Looper)(Base)?|impl::(ExternalWork|Accumulator))")
farg = re.compile(r"\(.*?\)")
tmpl = re.compile(r'<.*?>')
toplevelfuncs = set()
epfunc = re.compile(r"edm::eventsetup::EventSetupRecord::get<.*>\(.*\)")
skipfunc = re.compile(r"TGraph::IsA\(.*\)")
epfuncs=set()

import networkx as nx
G=nx.DiGraph()

f = open('function-calls-db.txt')

for line in f :
	fields = line.split("'")
        if len(fields) < 3:
            continue
	if fields[2] == ' calls function ' :
		if not skipfunc.search(line) : 
			G.add_edge(fields[1],fields[3],kind=fields[2])
			if epfunc.search(fields[3])  :
                            epfuncs.add(fields[3])
	if fields[2] == ' overrides function ' :
		if baseclass.search(fields[3]) :
			if topfunc.search(fields[3]) : 
                            toplevelfuncs.add(fields[1])
			G.add_edge(fields[1],fields[3],kind=' overrides function ')
		else :
			if not skipfunc.search(line) : 
				G.add_edge(fields[3],fields[1],kind=' calls override function ')
				if epfunc.search(fields[1]) : epfuncs.add(fields[1])
f.close()

#for epfunc in sorted(epfuncs): print(epfunc)
print("Callstacks for top level functions calling EventSetupRecord::get<>()")
print()

callstacks=set()
for tfunc in sorted(toplevelfuncs):
        for epfunc in sorted(epfuncs):
		if G.has_node(tfunc) and G.has_node(epfunc) and nx.has_path(G,tfunc,epfunc) : 
			path = nx.shortest_path(G,tfunc,epfunc)
			cs=""
                        previous=str("")
			for p in path :
                            stripped=re.sub(farg,"()",p)
                            if previous != stripped:
                                cs+=stripped+"; "
                                previous = stripped
                        callstacks.add(cs)
                        break

for cs in sorted(callstacks): print(cs)
