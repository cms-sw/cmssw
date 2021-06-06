#! /usr/bin/env python3
from __future__ import print_function
import fileinput
import networkx as nx
import re
datacl = re.compile("^class ")
bfunc = re.compile("^function ")
mbcl = re.compile("(base|data|flagged) class")
farg = re.compile(r"(.*)\(\w+\)")
nsep = re.compile(r"\:\:")
topfunc = re.compile(
    r"::(produce|analyze|filter|beginLuminosityBlock|beginRun|beginStream)\(")
baseclass = re.compile(
    r"edm::(one::|stream::|global::)?ED(Producer|Filter|Analyzer)(Base)?")
globalclass = re.compile("^edm::global::ED(Producer|Filter|Analyzer)$")
getfunc = re.compile(
    r"edm::eventsetup::EventSetupRecord::get\<.*\>\((.*)&\) const")
handle = re.compile("(.*),?class edm::ES(.*)Handle<(.*)>")
skip = re.compile(
    r"edm::serviceregistry::ServicesManager::MakerHolder::add() const")
rootclass = re.compile("T(H1|Tree|Enum|DataType|Class|Branch|Named|File)")
stdcl = re.compile("^std::(.*)[^>]$")
stdptr = re.compile("(.*)_ptr$")
statics = set()
toplevelfuncs = set()
onefuncs = set()
dataclassfuncs = set()
virtfuncs = set()
virtclasses = set()
badfuncs = set()
badclasses = set()
esdclasses = set()
dclasses = set()
dataclasses = set()
flaggedclasses = set()
globalclasses = set()
Gdg = nx.DiGraph()
Hdg = nx.DiGraph()
Idg = nx.DiGraph()

f = open('class-checker.txt')
for line in f:
    if mbcl.search(line):
        fields = line.split("'")
        classname = fields[1]
        funcname = fields[3]
        badclasses.add(classname)
        badfuncs.add(funcname)
f.close()

f = open('const-checker.txt')
for line in f:
    if mbcl.search(line):
        fields = line.split("'")
        classname = fields[1]
        badclasses.add(classname)
f.close()


f = open('classes.txt.dumperall')
for line in f:
    if mbcl.search(line):
        fields = line.split("'")
        if fields[2] == ' member data class ':
            Hdg.add_edge(fields[1], fields[3], kind=fields[2])
        if fields[2] == ' templated member data class ':
            Hdg.add_edge(fields[1], fields[3], kind=fields[2])
        if fields[2] == ' base class ':
            Hdg.add_edge(fields[1], fields[3], kind=fields[2])
            Idg.add_edge(fields[3], fields[1], kind=' derived class')
            if globalclass.match(fields[3]):
                globalclasses.add(fields[1])
                print("class '"+fields[1]+"' base class '"+fields[3]+"'")
f.close()

for line in fileinput.input(files=('function-statics-db.txt', 'function-calls-db.txt')):
    if not bfunc.search(line):
        continue
    fields = line.split("'")
    if skip.search(fields[1]) or skip.search(fields[3]):
        continue
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
fileinput.close()


for n, nbrdict in Gdg.adjacency():
    for nbr, eattr in nbrdict.items():
        if n in badfuncs or nbr in badfuncs:
            if 'kind' in eattr and eattr['kind'] == ' overrides function ':
                print("'"+n+"'"+eattr['kind']+"'"+nbr+"'")
                virtfuncs.add(nbr)
print()

print("-----------------------------------------------")
print("flagged functions found by checker")
print("-----------------------------------------------")
for dfunc in sorted(badfuncs):
    print(dfunc)
print()

print("-----------------------------------------------")
print("flagged classes found by checker ")
print("-----------------------------------------------")
for dclass in sorted(badclasses):
    print(dclass)
print()

nodes = sorted(badclasses)
for node in nodes:
    visited = set()
    if node in visited:
        continue
    visited.add(node)
    if node in Hdg:
        stack = [(node, iter(Hdg[node]))]
    if node in Idg:
        Qdg = nx.dfs_preorder_nodes(Idg, node)
        for q in Qdg:
            if q in Hdg:
                stack.append((q, iter(Hdg[q])))
    while stack:
        parent, children = stack[-1]
        try:
            child = next(children)
            if globalclass.search(child):
                visited.add(child)
            if rootclass.search(child):
                visited.add(child)
            if child not in visited:
                visited.add(child)
                stack.append((child, iter(Hdg[child])))
                kind = Hdg[parent][child]['kind']
                if stdptr.search(kind):
                    if child in Idg:
                        Qdg = nx.dfs_preorder_nodes(Idg, child)
                        for q in Qdg:
                            if q in Hdg:
                                stack.append((q, iter(Hdg[q])))
        except StopIteration:
            stack.pop()
    print("flagged class " + node + " contains or inherits from classes ", end=' ')
    for v in visited:
        print(v + ",", end=' ')
    print("\n\n")
    for v in sorted(visited):
        if v in globalclasses:
            print("EDM global class '" + v +
                  "' is flagged because it is connected to flagged class '" + node + "'")
