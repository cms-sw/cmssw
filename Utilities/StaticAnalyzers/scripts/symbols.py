#! /usr/bin/env python

from __future__ import print_function
import collections
import os
import re
import subprocess

addr_re = r"(?P<address>[0-9a-f]{1,16})?"
code_re = r"(?P<code>[a-zA-Z])"
symbol_re = r"(?P<symbol>[a-zA-Z0-9_.$@]+)"
symbol_demunged_re = r"(?P<symbol>[a-zA-Z0-9_.$@:&()<>{}\[\]|^!%,~*+-=# ]+)"
symbols_re_skip = re.compile("(@@)")
nm_line_re = re.compile(r"\s+".join([addr_re, code_re, symbol_re]) + "\s*$",
                        re.I)
ldd_line_re = re.compile(r"\s+(.*) => (.*) \(0x")

requires = collections.defaultdict(set)
provides = collections.defaultdict(set)
dependencies = collections.defaultdict(set)
libraries = collections.defaultdict(set)

def get_symbols(fname):
    lines = subprocess.check_output(["nm", "-g", fname])
    for l in lines.splitlines():
        m = nm_line_re.match(l)
        if not m : continue
        symbol = m.group('symbol')
        if m.group('code') == 'U':
            requires[os.path.basename(fname)].add(symbol)
        else:
            provides[symbol].add(os.path.basename(fname))

def get_libraries(fname):
    lines = subprocess.check_output(["ldd",fname])
    for l in lines.splitlines():
        m = ldd_line_re.match(l)
        if not m: continue
        library = m.group(2)
        libraries[os.path.basename(fname)].add(os.path.basename(library.rstrip('\r\n')))


paths=os.environ['LD_LIBRARY_PATH'].split(':')

for p in paths:
    for dirpath, dirnames, filenames in os.walk(p):
        for f in filenames:
            fpth=os.path.realpath(os.path.join(dirpath,f))
            filetype = subprocess.check_output(["file", fpth])
            if filetype.find("dynamically linked") >= 0 :
                get_symbols(fpth)
                get_libraries(fpth)

for fname, symbols in requires.items():
    deps=set()
    for library in libraries[fname]:
        for s in symbols : 
            if library in provides[s] : deps.add(library)
    dependencies[fname]=deps 
    print(fname + ' : primary dependencies : ' + ',  '.join(sorted(dependencies[fname]))+'\n')
    unmet = set()
    demangled = set()
    for s in symbols: 
        if s not in provides and not symbols_re_skip.search(s) : unmet.add(s) 
    for u in sorted(unmet):
        dm = subprocess.check_output(["c++filt",u])
        demangled.add(dm.rstrip('\r\n'))
    if demangled :  print(fname + ': undefined : ' + ',  '.join(sorted(demangled)))

import networkx as nx
G=nx.DiGraph()
for key,values in dependencies.items():
    G.add_node(key)
    for val in values: G.add_edge(key,val)

for node in nx.nodes_iter(G):
    s = nx.dfs_successors(G,node)
    deps=set()
    if s : 
        for key,vals in s.items() : 
            if key != node : deps.add(key)
            for v in vals :
                deps.add(v)
    print(node + ': primary and secondary dependencies :' + ', '.join(sorted(deps)))

import pydot

H=nx.DiGraph()
for key,values in dependencies.items():
    H.add_node(os.path.basename(key))
    for val in values: H.add_edge(os.path.basename(key),os.path.basename(val))
for node in nx.nodes_iter(H):
    T = nx.dfs_tree(H,node)
    name = node + ".dot"
    nx.write_dot(T,name)
