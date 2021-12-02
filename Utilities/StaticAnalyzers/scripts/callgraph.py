#! /usr/bin/env python3
from __future__ import print_function
import networkx as nx
import re
import json
from yaml import load, dump, FullLoader
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

topfunc = r"::(dqmBeginRun|dqmEndRun|bookHistograms|accumulate|acquire|startingNewLoop|duringLoop|endOfLoop|beginOfJob|endOfJob|produce|analyze|filter|beginLuminosityBlock|beginRun|beginStream|streamBeginRun|streamBeginLuminosityBlock|streamEndRun|streamEndLuminosityBlock|globalBeginRun|globalEndRun|globalBeginLuminosityBlock|globalEndLuminosityBlock|endRun|endRunProduce|endLuminosityBlock)\("
topfuncre = re.compile(topfunc)

baseclass = r"\b(edm::)?(one::|stream::|global::)?((DQM)?(Global|One)?ED(Producer|Filter|Analyzer|(IterateNTimes|NavigateEvents)?Looper)(Base)?|impl::(ExternalWork|Accumulator|RunWatcher|RunCacheHolder)|FromFiles|ProducerSourceBase|OutputModuleBase|InputSource|ProducerSourceFromFiles|ProducerBase|PuttableSourceBase|OutputModule|RawOutput|RawInputSource|impl::RunWatcher<edm::one::EDProducerBase>|impl::EndRunProducer<edm::one::EDProducerBase>|DQMEDHarvester|AlignmentProducerBase|BMixingModule|TrackProducerBase|cms::CkfTrackCandidateMakerBase|CallEndRunProduceImpl|CallGlobalImpl|impl::makeGlobal|impl::makeStreamModule|impl::EndRunProducer)\b"
baseclassre = re.compile(baseclass)
assert(baseclassre.match('edm::one::impl::EndRunProducer'))
assert(baseclassre.match('DQMOneEDAnalyzer'))
assert(baseclassre.match('edm::EDAnalyzer'))
assert(baseclassre.match('edm::stream::EDProducerBase'))
assert(baseclassre.match('DQMEDHarvester'))
assert(baseclassre.match('edm::one::impl::RunWatcher<edm::one::EDProducerBase>'))
assert(baseclassre.match('edm::global::EDFilter::filter() virtual'))
assert(baseclassre.search('edm::stream::EDProducerBase::produce'))
assert(topfuncre.search('edm::global::EDFilterBase::filter(&) const virtual'))
assert(not baseclassre.match('edm::BaseFlatGunProducer'))
assert(not baseclassre.match('edm::FlatRandomEGunProducer'))
assert(baseclassre.match('edm::ProducerSourceBase'))
assert(baseclassre.match('edm::one::OutputModuleBase'))
farg = re.compile(r"\(.*?\)")
tmpl = re.compile(r'<(.*)>')
toplevelfuncs = set()

getfuncre = re.compile(r"edm::eventsetup::EventSetupRecord::get<")
epfuncre = re.compile(r"edm::eventsetup::EventSetupRecord::deprecated_get<")
f = 'edm::eventsetup::EventSetupRecord::get<class edm::ESHandle<class CaloTopology>>(const std::string &, class edm::ESHandle<class CaloTopology> &) const'
g = 'edm::eventsetup::EventSetupRecord::deprecated_get<class edm::ESHandle<class HcalDDDSimConstants>>(const char *, class edm::ESHandle<class HcalDDDSimConstants> &) const'
g2 = 'edm::eventsetup::EventSetupRecord::deprecated_get<class edm::ESHandle<class HcalDDDSimConstants>>(class edm::ESHandle<class HcalDDDSimConstants> &) const'
h = 'edm::eventsetup::EventSetupRecord::get<class edm::ESHandle<class CaloTPGTranscoder>>(class edm::ESHandle<class CaloTPGTranscoder> &) const'
assert(getfuncre.search(f))
assert(epfuncre.search(g))
assert(getfuncre.search(h))
assert(epfuncre.search(g2))
epf='edm::eventsetup::EventSetupRecord::get<class edm::ESHandle<class CaloTPGTranscoder>>(edm::ESHandle<CaloTPGTranscoder> &) const'
assert(getfuncre.search(epf))

skipfunc = re.compile(r"TGraph::IsA\(.*\)")
epfuncs = set()

function='(anonymous namespace in /src/RecoTracker/TkHitPairs/plugins/HitPairEDProducer.cc)::Impl<(anonymous namespace)::ImplSeedingHitSets, (anonymous namespace)::DoNothing, (anonymous namespace)::RegionsLayersSeparate>::produce(const _Bool, edm::Event &, const edm::EventSetup &);'
function2='HitPairEDProducer::produce(edm::Event &, const edm::EventSetup &);'
value='HitPairEDProducer'
valuere=r'\b%s\b' % value
vre=re.compile(valuere)
m=vre.search(function)
assert(m)
m2=vre.search(function2)
assert(m2)


module2package = dict()
with open('module_to_package.yaml') as m:
    module2package=load(m, Loader=FullLoader)

with open('packages.json', 'r') as j:
    p = json.load(j)

for k,v in p.items():
    module=k.split('|')[0]
    dirs=v.split('|')
    package=dirs[0]+'/'+dirs[1]
    module2package.setdefault(package, []).append(module)

for k in module2package.keys():
    module2package[k]=sorted(set(module2package[k]))

G = nx.DiGraph()
with open('function-calls-db.txt') as f:
    for line in f:
        fields = line.split("'")
        if len(fields) < 3:
            continue
        if fields[2] == ' calls function ':
            G.add_edge(fields[1], fields[3], kind=fields[2])
            if epfuncre.search(fields[3]) or getfuncre.search(fields[3]):
                epfuncs.add(fields[3])
        if fields[2] == ' overrides function ':
            if baseclassre.match(fields[3]):
                G.add_edge(fields[1], fields[3], kind=' overrides function ')
                if topfuncre.search(fields[1]) :
                    toplevelfuncs.add(fields[1])
            else:
                G.add_edge(fields[3], fields[1], kind=' overriden function calls function ')
            if epfuncre.search(fields[3]) or getfuncre.search(fields[3]):
                epfuncs.add(fields[3])

callstacks = set()
for tfunc in toplevelfuncs:
    for epfunc in epfuncs:
        if G.has_node(tfunc) and G.has_node(epfunc) and nx.has_path(G, tfunc, epfunc):
            for path in nx.all_shortest_paths(G, tfunc, epfunc):
                cs = str("")
                previous = str("")
                for p in path:
                    if getfuncre.search(p) or epfuncre.search(p):
                        break
                    #stripped=re.sub(farg, "()", p)
                    stripped=p
                    if previous != stripped:
                        cs += stripped + '; '
                        previous = stripped
                if cs not in callstacks:
                    callstacks.add(cs)

report = dict()
csset = set(callstacks)
for callstack in sorted(list(callstacks)):
    for key in module2package.keys():
        for value in module2package.get(key):
            regex_str = r'\b%s\b'%value
            vre=re.compile(regex_str)
            if vre.search(callstack):
                report.setdefault(str(key), {}).setdefault(str(value), []).append(str(callstack))
                csset.discard(callstack)

report['no-package']=dict()
for cs in sorted(list(csset)):
    key=tmpl.split(topfuncre.split(cs)[0])[0]
    report['no-package'].setdefault(key,[]).append(str(cs))

r = open('eventsetuprecord-get.yaml', 'w')
dump(report, r, width=float("inf"))
