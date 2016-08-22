#!/usr/bin/env python
import os
import PhysicsTools.HeppyCore.framework.config as cfg
cfg.Analyzer.nosubdir=True

import PSet
import sys
import re
print "ARGV:",sys.argv
JobNumber=sys.argv[1]
crabFiles=PSet.process.source.fileNames
print crabFiles
firstInput = crabFiles[0]
print "--------------- using edmFileUtil to convert PFN to LFN -------------------------"
for i in xrange(0,len(crabFiles)) :
     if os.getenv("GLIDECLIENT_Group","") != "overflow" :
       print "Data is local"
       pfn=os.popen("edmFileUtil -d %s"%(crabFiles[i])).read() 
       pfn=re.sub("\n","",pfn)
       print crabFiles[i],"->",pfn
       crabFiles[i]=pfn
     else:
       print "Data is not local, using AAA/xrootd"
       crabFiles[i]="root://cms-xrd-global.cern.ch/"+crabFiles[i]
print "Doing hadd:", "hadd tree.root %s" % (" ".join(crabFiles))
os.system("hadd tree.root %s" % (" ".join(crabFiles)))
#replace files with crab ones
os.system("ls -l")

import ROOT
f=ROOT.TFile.Open('tree.root')
entries=f.Get('tree').GetEntries()

fwkreport='''<FrameworkJobReport>
<ReadBranches>
</ReadBranches>
<PerformanceReport>
  <PerformanceSummary Metric="StorageStatistics">
    <Metric Name="Parameter-untracked-bool-enabled" Value="true"/>
    <Metric Name="Parameter-untracked-bool-stats" Value="true"/>
    <Metric Name="Parameter-untracked-string-cacheHint" Value="application-only"/>
    <Metric Name="Parameter-untracked-string-readHint" Value="auto-detect"/>
    <Metric Name="ROOT-tfile-read-totalMegabytes" Value="0"/>
    <Metric Name="ROOT-tfile-write-totalMegabytes" Value="0"/>
  </PerformanceSummary>
</PerformanceReport>

<GeneratorInfo>
</GeneratorInfo>

<InputFile>
<LFN>%s</LFN>
<PFN></PFN>
<Catalog></Catalog>
<InputType>primaryFiles</InputType>
<ModuleLabel>source</ModuleLabel>
<GUID></GUID>
<InputSourceClass>PoolSource</InputSourceClass>
<EventsRead>1</EventsRead>

</InputFile>

<File>
<LFN></LFN>
<PFN>tree.root</PFN>
<Catalog></Catalog>
<ModuleLabel>HEPPY</ModuleLabel>
<GUID></GUID>
<OutputModuleClass>PoolOutputModule</OutputModuleClass>
<TotalEvents>%s</TotalEvents>
<BranchHash>dc90308e392b2fa1e0eff46acbfa24bc</BranchHash>
</File>

</FrameworkJobReport>''' % (firstInput,entries)

f1=open('./FrameworkJobReport.xml', 'w+')
f1.write(fwkreport)
