#!/usr/bin/env python3

# find cycles in cmssw libs
import collections

class Graph(object):
    def __init__(self, edges):
        self.edges = edges

    @staticmethod
    def _build_adjacency_list(edges):
        adj = collections.defaultdict(list)
        for edge in edges:
            adj[edge[0]].append(edge[1])
        return adj

    def addEdge(self,edge):
        self.edges.append(edge)

    def build_adjacency_list(self):
        self.adj = Graph._build_adjacency_list(self.edges)
        

def dfs(G):
    discovered = set()
    finished = set()
    for u in G.adj:
        if u not in discovered and u not in finished:
            discovered, finished = dfs_visit(G, u, discovered, finished)

def dfs_visit(G, u, discovered, finished):
    if u not in G.adj:
        finished.add(u)
        return discovered, finished

    discovered.add(u)

    for v in G.adj[u]:
        # Detect cycles
        if v in discovered:
            print(f"Cycle detected: found a back edge from {u} to {v}. It involves")
            for i,d in enumerate(discovered):
                if i != len(discovered)-1:
                    print(d,end=', ')
                else:
                    print(d)

        # Recurse into DFS tree
        else:
            if v not in finished:
                dfs_visit(G, v, discovered, finished)

    discovered.remove(u)
    finished.add(u)

    return discovered, finished

import subprocess
def run_command(comm):
    encoding = 'utf-8'
    proc = subprocess.Popen(comm, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    stdout, stderr = proc.communicate()
    stdout = stdout.decode(encoding)
    if stderr is not None:
        stderr= stderr.decode(encoding)
    return proc.returncode, stdout, stderr


import os
def get_pack_list():
    src_base = os.environ.get('CMSSW_RELEASE_BASE')
    src_path = os.path.join(src_base,"src")
    comm = "ls -d "+src_path+"/*/*/interface"
    c_rt,c_out,c_err = run_command(comm)

    ret_val=[]
    for l in c_out.split():
        sp=l.split('/')
        ret_val.append('/'.join(sp[-3:-1]))

    return ret_val

def get_files(pack,subdir,file_types):
    src_base = os.environ.get('CMSSW_RELEASE_BASE')
    pack_path = os.path.join(src_base,"src",pack,subdir)
    if not os.path.exists(pack_path): return []
    ret_val=[]
    for root, dirs, files in os.walk(pack_path, topdown=False):
        for name in files:
            for t in file_types:
                if name.endswith(t):
                    ret_val.append(os.path.join(root,name))

    return ret_val

def get_lib_deps(lib_info):
    lib_path = lib_info[1]
    comm = "ldd "+lib_path+"| grep cms | grep -v \"/external\" | grep -v \"/lcg/\" | awk '{print $3}'"
    c_rt,c_out,c_err = run_command(comm)
    ret_val=[]
    for l in c_out.split():
        lib = l.strip()
        ret_val.append( (lib.split('/')[-1],lib))
    return ret_val

def get_include_packages(file_list,package=None,is_cuda=False):
    incs={}
    pack_incs={}
    if is_cuda:
        comm= "gcc -fpreprocessed -dD -E "
    else:
        comm= "nvcc --compiler-options  -fpreprocessed -dD -E "
    for f in file_list:
        comm=comm+f+" "
    comm=comm+" | grep \"#include\""
    c_rt,c_out,c_err = run_command(comm)
    for l in c_out.split():
        inc = l.strip().split()[-1][1:-1]
        if '/' in inc:
            incs['/'.join(inc.split('/')[0:2])]=1
            if package is not None and package in inc and "interface" in inc:
                pack_incs[os.path.join('/'.join(file_list[0].split('/')[0:-4]),inc)]=1
    if package is None:
        return list(incs.keys())
    else:
        return list(incs.keys()),list(pack_incs.keys())



import sys

if __name__ == "__main__":
    

    import argparse
    parser=argparse.ArgumentParser(description="CMSSW Cyclic dependency finder")
    parser.add_argument("--omit_header_only",dest="omit_header_only",
                        action="store_false", default=True,
                        help="Ignore cycles due to header only dependencies"
                    )
    parser.add_argument("--status_bar",dest="status_bar",
                        action="store_true", default=False,
                        help="Show progress bar when running"
                    )

    args = parser.parse_args()
    omit_header_only=args.omit_header_only
    show_status_bar=args.status_bar
    print(omit_header_only,show_status_bar)
    if 'CMSSW_RELEASE_BASE' not in os.environ:
        print("Execute within a cmssw environment")
        sys.exit(1)
    
    G = Graph([])

    lib_list = get_pack_list()

    if show_status_bar:
        import tqdm
        iter_lib = tqdm.tqdm(lib_list)
    else:
        iter_lib = lib_list
    for lib in iter_lib:
        header_list = get_files(lib,"interface",[".h",".hpp"])
        source_list = get_files(lib,"src",[".cc",".cpp",".cxx"])
        cuda_source_list = get_files(lib,"src",[".cu"])
        
        cpp_incs_packages, cpp_self_headers = get_include_packages(source_list,lib)
        cuda_incs_packages, cuda_self_headers = get_include_packages(cuda_source_list,lib,is_cuda=True)

        source_incs_packages = list(cpp_incs_packages)
        source_incs_packages.extend(x for x in cuda_incs_packages if x not in source_incs_packages)
        self_headers = list(cpp_self_headers)
        self_headers.extend(x for x in cuda_self_headers if x not in self_headers)

        if not omit_header_only:
            header_incs_packages = get_include_packages(header_list)
        else:
            header_incs_packages = get_include_packages(self_headers)
        
        for dep in header_incs_packages:
            if dep != lib:
                G.addEdge( (lib,dep) )
        for dep in source_incs_packages:
            if dep != lib and dep not in header_incs_packages:
                G.addEdge( (lib,dep) )

    print("Building adjacency graph")
    G.build_adjacency_list()
    print("Looking for cycles")
    dfs(G)

