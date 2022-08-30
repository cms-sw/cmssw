#! /usr/bin/env python3
from __future__ import print_function
import fileinput
import networkx as nx
from builtins import range
import re
topfunc = re.compile(r"::(produce|analyze|filter|beginLuminosityBlock|beginRun|beginStream|streamBeginRun|streamBeginLuminosityBlock|streamEndRun|streamEndLuminosityBlock|globalBeginRun|globalEndRun|globalBeginLuminosityBlock|globalEndLuminosityBlock|endRun|endLuminosityBlock)\(")
baseclass = re.compile(
    "edm::(one::|stream::|global::)?ED(Producer|Filter|Analyzer)(Base)?")
farg = re.compile(r"\(.*\)")
fre = re.compile("function")

statics = set()
toplevelfuncs = set()
skipfunc = re.compile(r"(edm::(LuminosityBlock::|Run::|Event::|Principal::)getBy(Label|Token))|(fwlite::|edm::EDProductGetter::getIt|edm::Event::|edm::eventsetup::EventSetupRecord::get|edm::eventsetup::DataProxy::getImpl|edm::EventPrincipal::unscheduledFill|edm::ServiceRegistry::get|edm::eventsetup::EventSetupRecord::getImplementation|edm::eventsetup::EventSetupRecord::getFromProxy|edm::eventsetup::DataProxy::get|edm::serviceregistry::ServicesManager::MakerHolder::add|(cond::service::PoolDBOutputService::(writeOne|appendSinceTime|tagInfo))|edm::EventProcessor::|edm::SubProcess::)")
skipfuncs = set()

G = nx.DiGraph()

for line in fileinput.input(files=('function-statics-db.txt', 'function-calls-db.txt')):
    if fre.search(line):
        fields = line.split("'")
        if topfunc.search(fields[1]) and not baseclass.search(fields[1]):
            toplevelfuncs.add(fields[1])
        if fields[2] == ' calls function ':
            if skipfunc.search(line):
                skipfuncs.add(line)
            else:
                G.add_edge(fields[1], fields[3], kind=fields[2])
        if fields[2] == ' overrides function ':
            if baseclass.search(fields[3]):
                if topfunc.search(fields[3]):
                    toplevelfuncs.add(fields[1])
                G.add_edge(fields[1], fields[3], kind=fields[2])
            else:
                if skipfunc.search(line):
                    skipfuncs.add(line)
                else:
                    G.add_edge(fields[3], fields[1], kind=' calls function ')
        if fields[2] == ' static variable ':
            G.add_edge(fields[1], fields[3], kind=fields[2])
            statics.add(fields[3])
        if fields[2] == ' known thread unsafe function ':
            G.add_edge(fields[1], fields[3],
                       kind=' known thread unsafe function ')
            statics.add(fields[3])
fileinput.close()

for tfunc in sorted(toplevelfuncs):
    for static in sorted(statics):
        if G.has_node(tfunc) and G.has_node(static) and nx.has_path(G, tfunc, static):
            path = nx.shortest_path(G, tfunc, static)

            if 'kind' in G[path[len(path)-2]][path[len(path)-1]] and G[path[len(path)-2]][path[len(path)-1]]['kind'] == ' static variable ':
                for key in G[tfunc].keys():
                    if 'kind' in G[tfunc][key] and G[tfunc][key]['kind'] == ' overrides function ':
                        print("Non-const static variable \'"+re.sub(farg, "()",
                                                                    static)+"' is accessed in call stack '", end=' ')
                        for i in range(0, len(path)-1):
                            print(re.sub(farg, "()", path[i]) +
                                  G[path[i]][path[i+1]]['kind'], end=' ')
                        print(re.sub(farg, "()", path[i+1])+"' ,", end=' ')
                        print("'"+re.sub(farg, "()", tfunc)+"' overrides '" +
                              re.sub(farg, "()", key)+"'", end=' ')
                        print()

                        print("In call stack ' ", end=' ')
                        for i in range(0, len(path)-1):
                            print(re.sub(farg, "()", path[i]) +
                                  G[path[i]][path[i+1]]['kind'], end=' ')
                        print(re.sub(farg, "()", path[i+1])+"' is accessed ,", end=' ')
                        print("'"+re.sub(farg, "()", tfunc)+"' overrides '" +
                              re.sub(farg, "()", key)+"'", end=' ')
                        print()

            else:
                for key in G[tfunc].keys():
                    if 'kind' in G[tfunc][key] and G[tfunc][key]['kind'] == ' overrides function ' and not (key.startswith('edm::one') and 'TFileService::' in static):
                        print("Known thread unsafe function '"+re.sub(farg, "()",
                                                                       static)+"' is called in call stack '", end=' ')
                        for i in range(0, len(path)-1):
                            print(re.sub(farg, "()", path[i]) +
                                  G[path[i]][path[i+1]]['kind'], end=' ')
                        print(re.sub(farg, "()", path[i+1])+"' ,", end=' ')
                        print("'"+re.sub(farg, "()", tfunc)+"' overrides '" +
                                  re.sub(farg, "()", key)+"'", end=' ')
                        print()

                        print("In call stack ' ", end=' ')
                        for i in range(0, len(path)-1):
                            print(re.sub(farg, "()", path[i]) +
                                  G[path[i]][path[i+1]]['kind'], end=' ')
                        print(re.sub(farg, "()", path[i+1])+"' known thread unsafe function '"+re.sub(farg, "()", static)+"' is called, ", end=' ')
                        print("'"+re.sub(farg, "()", tfunc)+"' overrides '" +
                                      re.sub(farg, "()", key)+"'", end=' ')
                        print()
