#! /usr/bin/env python
from __future__ import print_function
import re
import yaml

topfunc = re.compile(r"::(accumulate|acquire|startingNewLoop|duringLoop|endOfLoop|beginOfJob|endOfJob|produce|analyze|filter|beginLuminosityBlock|beginRun|beginStream|streamBeginRun|streamBeginLuminosityBlock|streamEndRun|streamEndLuminosityBlock|globalBeginRun|globalEndRun|globalBeginLuminosityBlock|globalEndLuminosityBlock|endRun|endLuminosityBlock)\(")

baseclass = re.compile(r"edm::(one::|stream::|global::)?(ED(Producer|Filter|Analyzer|(IterateNTimes|NavigateEvents)?Looper)(Base)?|impl::(ExternalWork|Accumulator))")
farg = re.compile(r"\(.*?\)")
tmpl = re.compile(r'<.*?>')
toplevelfuncs = set()
epfuncre = re.compile(r"edm::eventsetup::EventSetupRecord::get<.*>\(.*\)")
skipfunc = re.compile(r"TGraph::IsA\(.*\)")
epfuncs=set()

import networkx as nx
G=nx.DiGraph()

#g = open('module_to_package.txt')
#module2package=dict()
#for line in g:
#    fields = line.strip().split(';')
#    if len(fields) <2:
#        continue
#    module2package.setdefault(fields[1], []).append(fields[0])
#
#i = open('module_to_package.yaml', 'w')
#yaml.dump(module2package, i)
#i.close()

h = open('module_to_package.yaml', 'r')
module2package=yaml.load(h, Loader=yaml.FullLoader)

with open('function-calls-db.txt') as f:
   for line in f :
	fields = line.split("'")
        if len(fields) < 3:
            continue
	if fields[2] == ' calls function ' :
		if not skipfunc.search(line) : 
			G.add_edge(fields[1],fields[3],kind=fields[2])
			if epfuncre.search(fields[3])  :
                            epfuncs.add(fields[3])
	if fields[2] == ' overrides function ' :
		if baseclass.search(fields[3]) :
			if topfunc.search(fields[3]) : 
                            toplevelfuncs.add(fields[1])
			G.add_edge(fields[1],fields[3],kind=' overrides function ')
		else :
			if not skipfunc.search(line) : 
				G.add_edge(fields[3],fields[1],kind=' calls override function ')
				if epfuncre.search(fields[1]) : epfuncs.add(fields[1])

    

callstacks=set()
for tfunc in toplevelfuncs:
        for epfunc in epfuncs:
		if G.has_node(tfunc) and G.has_node(epfunc) and nx.has_path(G,tfunc,epfunc) : 
			path = nx.shortest_path(G,tfunc,epfunc)
			cs=str("")
                        previous=str("")
			for p in path :
                            if epfuncre.search(p): break
                            stripped=re.sub(farg,"()",p)
                            if previous != stripped:
                                cs+=' '+stripped+";"
                                previous = stripped
                        callstacks.add(cs)
                        break


report=dict()
for key in sorted(module2package.keys()):
   for value in sorted(module2package[key]):
       vre=re.compile(' %s::.*();' % value)
       for cs in sorted(callstacks):
           if vre.search(cs):
               report.setdefault(key, {}).setdefault(value, []).append(cs)
r=open('eventsetuprecord-get.yaml', 'w')
yaml.dump(report,r)
