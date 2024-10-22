#! /usr/bin/env python3
from __future__ import print_function
import networkx as nx
import re
stdcl = re.compile("^std::(.*)[^>]$")
stdptr = re.compile("(.*)_ptr$")
datacl = re.compile("^class ")
bfunc = re.compile("^function ")
mbcl = re.compile("(base|data) class")
farg = re.compile(r"(.*)\(\w+\)")
nsep = re.compile(r"\:\:")
topfunc = re.compile(
    r"::(produce|analyze|filter|beginLuminosityBlock|beginRun|beginStream)\(")
baseclass = re.compile(
    "edm::(one::|stream::|global::)?ED(Producer|Filter|Analyzer)(Base)?")
getfunc = re.compile(
    r"edm::eventsetup::EventSetupRecord::get\<.*\>\((.*)&\) const")
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

Gdg = nx.DiGraph()
Hdg = nx.DiGraph()
Idg = nx.DiGraph()


f = open('classes.txt.dumperall')
for line in f:
    if mbcl.search(line):
        fields = line.split("'")
        if fields[2] == ' member data class ':
            if not stdcl.search(fields[2]):
                Hdg.add_edge(fields[1], fields[3], kind=fields[2])
        if fields[2] == ' templated member data class ':
            Hdg.add_edge(fields[1], fields[5], kind=fields[3])
        if fields[2] == ' base class ':
            Hdg.add_edge(fields[1], fields[3], kind=fields[2])
            Idg.add_edge(fields[3], fields[1], kind=' derived class')
f.close()

f = open('function-calls-db.txt')

for line in f:
    if not bfunc.search(line):
        continue
    fields = line.split("'")
    if fields[2] == ' calls function ':
        Gdg.add_edge(fields[1], fields[3], kind=' calls function ')
        if getfunc.search(fields[3]):
            dataclassfuncs.add(fields[3])
            m = getfunc.match(fields[3])
            n = handle.match(m.group(1))
            if n:
                o = n.group(3)
            else:
                o = m.group(1)
            p = re.sub("class ", "", o)
            dataclass = re.sub("struct ", "", p)
            dataclasses.add(dataclass)
    if fields[2] == ' overrides function ':
        if baseclass.search(fields[3]):
            Gdg.add_edge(fields[1], fields[3], kind=' overrides function ')
            if topfunc.search(fields[3]):
                toplevelfuncs.add(fields[1])
        else:
            Gdg.add_edge(fields[3], fields[1], kind=' calls override function ')
    if fields[2] == ' static variable ':
        Gdg.add_edge(fields[1], fields[3], kind=' static variable ')
        statics.add(fields[3])
f.close()

visited = set()
nodes = sorted(dataclasses)
for node in nodes:
    if node in visited:
        continue
    visited.add(node)
    stack = []
    if node in Hdg:
        stack = [(node, iter(Hdg[node]))]
    if node in Idg:
        Qdg = nx.dfs_preorder_nodes(Idg, node)
        for q in Qdg:
            print("class '"+q+"'")
            if q in Hdg:
                stack.append((q, iter(Hdg[q])))
    while stack:
        parent, children = stack[-1]
        print("class '"+parent+"'")
        try:
            child = next(children)
            if child not in visited:
                visited.add(child)
                if not stdcl.search(child):
                    print("class '"+child+"'")
                    stack.append((child, iter(Hdg[child])))
                    kind = Hdg[parent][child]['kind']
                    print(parent, kind, child)
                    if stdptr.search(kind):
                        if child in Idg:
                            Qdg = nx.dfs_preorder_nodes(Idg, child)
                            for q in Qdg:
                                print("class '"+q+"'")
                                if q in Hdg:
                                    stack.append((q, iter(Hdg[q])))
        except StopIteration:
            stack.pop()
