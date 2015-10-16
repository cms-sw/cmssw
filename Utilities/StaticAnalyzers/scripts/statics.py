#! /usr/bin/env python
import re
topfunc = re.compile("::(produce|analyze|filter|beginLuminosityBlock|beginRun)\(")
baseclass = re.compile("edm::(one::|stream::|global::)?ED(Producer|Filter|Analyzer)(Base)?")
farg = re.compile("\(.*\)")
fre = re.compile("function")

statics = set()
toplevelfuncs = set()
skipfunc = re.compile("(edm::(LuminosityBlock::|Run::|Event::)getBy(Label|Token))|(fwlite::|edm::EDProductGetter::getIt|edm::Event::|edm::eventsetup::EventSetupRecord::get|edm::eventsetup::DataProxy::getImpl|edm::EventPrincipal::unscheduledFill|edm::ServiceRegistry::get|edm::eventsetup::EventSetupRecord::getImplementation|edm::eventsetup::EventSetupRecord::getFromProxy|edm::eventsetup::DataProxy::get|edm::serviceregistry::ServicesManager::MakerHolder::add)")
skipfuncs=set()

import networkx as nx
G=nx.DiGraph()

f = open('db.txt')

for line in f :
	if fre.search(line):
		fields = line.split("'")
		if fields[2] == ' calls function ':
			if skipfunc.search(line) : skipfuncs.add(line)
			else : G.add_edge(fields[1],fields[3],kind=fields[2])
		if fields[2] == ' overrides function ' :
			if baseclass.search(fields[3]) :
				if topfunc.search(fields[3]) : toplevelfuncs.add(fields[1])
				G.add_edge(fields[1],fields[3],kind=fields[2])
			else :
				if skipfunc.search(line) : skipfuncs.add(line)
				else : G.add_edge(fields[3],fields[1],kind=' calls function ')
		if fields[2] == ' static variable ' :
			G.add_edge(fields[1],fields[3],kind=' static variable ')
			statics.add(fields[3])
		if fields[2] == ' known thread unsafe function ' :
			G.add_edge(fields[1],fields[3],kind=' known thread unsafe function ')
			statics.add(fields[3])
f.close()
for tfunc in sorted(toplevelfuncs):
	for static in sorted(statics): 
		if nx.has_path(G,tfunc,static): 
			path = nx.shortest_path(G,tfunc,static)

			print "Non-const static variable \'"+re.sub(farg,"()",static)+"' is accessed in call stack '",
			for i in range(0,len(path)-1) :			
				print re.sub(farg,"()",path[i])+G[path[i]][path[i+1]]['kind'],
			print re.sub(farg,"()",path[i+1])+"' ,",
			for key in  G[tfunc].keys() :
				if 'kind' in G[tfunc][key] and G[tfunc][key]['kind'] == ' overrides function '  :
					print "'"+re.sub(farg,"()",tfunc)+"' overrides '"+re.sub(farg,"()",key)+"'",
			print

			print "In call stack ' ",
			for i in range(0,len(path)-1) :			
				print re.sub(farg,"()",path[i])+G[path[i]][path[i+1]]['kind'],
			print re.sub(farg,"()",path[i+1])+"' is accessed ,",
			for key in  G[tfunc].keys() :
				if 'kind' in G[tfunc][key] and G[tfunc][key]['kind'] == ' overrides function '  :
					print "'"+re.sub(farg,"()",tfunc)+"' overrides '"+re.sub(farg,"()",key)+"'",
			print

