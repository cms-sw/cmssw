import ROOT as r
import sys

inpu = file("./dec22List.txt","r")
#inpu = file("./incompList.txt","r")
flist = inpu.readlines()
print len(flist)
idx = sys.argv[1]
Analysis = sys.argv[2]

f =  flist[int(idx)]
print Analysis
outDir = sys.argv[3]
r.gROOT.ProcessLine(".x ~/cms03/CMSSW_4_1_3_patch3/src/UserCode/L1TriggerDPG/macros/initL1Analysis.C")
fi = f.strip()
r.gROOT.ProcessLine(".L ./"+str(Analysis) +".C+")
if Analysis == "L1EnergySumAnalysis": m = r.L1EnergySumAnalysis()
if Analysis == "L1JetAnalysis" : m = r.L1JetAnalysis()
if Analysis == "L1JetAnalysis_2011" : m = r.L1JetAnalysis_2011()
m.Open(fi)
m.run(-1,str(outDir)+"/"+str(fi).rpartition("/")[2])
m.Delete()
