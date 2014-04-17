import os
import sys
import ROOT

if len(sys.argv) < 4:
    print "Error. Expected at least 3 arguments\n\nUsage: python GetRecoTauVFromDQM_MC_cff.py InputFile OutputFile EventType"
    sys.exit()


Input  = sys.argv[1]
Output = sys.argv[2]
ET     = sys.argv[3]

InputFileName  = "%s" % Input
OutputFileName = "%s" % Output
EventType      = "%s" % ET


#opening output file 
outFile = ROOT.TFile(OutputFileName,"recreate")

#opening input file
fullFile = ROOT.TFile(InputFileName)

    #retrieving interesting Directory position
source = ROOT.gDirectory.ls()

    #retrieving interesting Directory position
source = ROOT.gDirectory.ls()
next=ROOT.TIter(fullFile.GetListOfKeys())
dirFound0 = None
dirFound1 = None
dirFound2 = None
dirFound3 = None

for key in next:
    cl = ROOT.gROOT.GetClass(key.GetClassName())
    if(cl.InheritsFrom("TDirectory")):
       dir=key.ReadObj()
       dirFound0=dir.GetName()
       next2=ROOT.TIter(dir.GetListOfKeys())
       for key in next2:
           cl2 = ROOT.gROOT.GetClass(key.GetClassName())
           if(cl2.InheritsFrom("TDirectory")):
              dir2=key.ReadObj()
              dirFound1 = dir2.GetName()
              next3=ROOT.TIter(dir2.GetListOfKeys())
              for key in next3:
                  cl3 = ROOT.gROOT.GetClass(key.GetClassName())
                  if(cl3.InheritsFrom("TDirectory")):
                     dir3=key.ReadObj()
                     if 'RecoTauV' in dir3.GetName():
                        dirFound2 = dir3.GetName()
                        next4=ROOT.TIter(dir3.GetListOfKeys())
                        for key in next4:
                            cl4 = ROOT.gROOT.GetClass(key.GetClassName())
                            if(cl4.InheritsFrom("TDirectory")):
                                dir4=key.ReadObj()
                                dirFound3 = dir4.GetName()

InputDir = dirFound0+"/"+dirFound1+"/"+dirFound2+"/"+dirFound3
ROOT.gDirectory.cd(InputDir)

    #Listing subdirectories of TargetDir
SubDirs = []
next5=ROOT.TIter(ROOT.gDirectory.GetListOfKeys())
for key in next5:
    cl5 = ROOT.gROOT.GetClass(key.GetClassName())
    if(cl5.InheritsFrom("TDirectory")):
       dirFound4=key.ReadObj()
       dirFound4_name=dirFound4.GetName()
       if EventType in dirFound4_name:
          SubDirs.append(dirFound4_name)  

    #Writing objects to file
for sub in SubDirs:
    outFile.cd()
    ROOT.gDirectory.mkdir(sub)
    fullFile.cd()
    ROOT.gDirectory.cd(InputDir+"/"+sub)
    HList = ROOT.TIter(ROOT.gDirectory.GetListOfKeys())
    for k in HList:
       kk = ROOT.gROOT.GetClass(k.GetClassName())
       if(kk.InheritsFrom("TH1F")):
          outFile.cd()
          ROOT.gDirectory.cd(sub)
          obj = k.ReadObj()
          obj.Write()
          
outFile.Close()
