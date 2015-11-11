#! /usr/bin/env python

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

requires = collections.defaultdict(set)
provides = collections.defaultdict(set)
dependencies = collections.defaultdict(set)

def get_symbols(fname):
    lines = subprocess.check_output(["nm", "-g", fname])
    for l in lines.splitlines():
        m = nm_line_re.match(l)
	if not m : continue
        symbol = m.group('symbol')
        if m.group('code') == 'U':
            requires[fname].add(symbol)
        else:
            provides[symbol].add(fname)

paths=os.environ['LD_LIBRARY_PATH'].split(':')

for p in paths:
	for dirpath, dirnames, filenames in os.walk(p):
	    for f in filenames:
	        fpth=os.path.realpath(os.path.join(dirpath,f))
    		filetype = subprocess.check_output(["file", fpth])
    		if filetype.find("dynamically linked") >= 0 :
	            get_symbols(fpth)

def pick(symbols):
    # If several files provide a symbol, choose the one with the shortest name.
    best = None
    for s in symbols:
        if best is None or len(s) < len(best):
            best = s
#    if len(symbols) > 1:
#        best = "*" + best
    return best

for fname, symbols in requires.items():
    dependencies[fname] = set(pick(provides[s]) for s in symbols if s in provides)
#    print fname + ': ' + ' '.join(sorted(dependencies[fname]))
    unmet = set()
    demangled = set()
    for s in symbols: 
	if s not in provides and not symbols_re_skip.search(s) : unmet.add(s) 
    for u in sorted(unmet):
	dm = subprocess.check_output(["c++filt",u])
	demangled.add(dm.rstrip('\r\n'))
#    if demangled :  print fname + ': undefined : ' + ' '.join(sorted(demangled))

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
	print node + ': ' + ','.join(sorted(deps))
