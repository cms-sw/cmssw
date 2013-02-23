import ROOT

##########################################################################################
### author: Lucia Perrini
### how to use the script ____
### Change the InputFileName
### Change the Condition (GlobalTag)
### For the moment do not change EventType. We are running over RealData only
##########################################################################################


InputFileName = "DQM_JetHT_GR_R_53_V16.root" 
EventType = "RealData"
Condition = "GR_R_53_V16"
OutFileName = "RecoTauV_"+EventType+"_"+Condition+".root"

#opening output file 
outFile = ROOT.TFile(OutFileName,"recreate")

#opening input file
fullFile = ROOT.TFile(InputFileName)

#retrieving interesting Directory position
source = ROOT.gDirectory.ls()
next=ROOT.TIter(fullFile.GetListOfKeys())
dirFound1 = None
dirFound2 = None
dirFound3 = None
for key in next:
   cl = ROOT.gROOT.GetClass(key.GetClassName())
   if(cl.InheritsFrom("TDirectory")):
      dir=key.ReadObj()
      dirFound1=dir.GetName()
      next2=ROOT.TIter(dir.GetListOfKeys())
      for key in next2:
         cl2 = ROOT.gROOT.GetClass(key.GetClassName())
         if(cl2.InheritsFrom("TDirectory")):
            dir2=key.ReadObj()
            dirFound2 = dir2.GetName()
            next3=ROOT.TIter(dir2.GetListOfKeys())
            for key in next3:
               cl3 = ROOT.gROOT.GetClass(key.GetClassName())
               if(cl3.InheritsFrom("TDirectory")):
                  dir3=key.ReadObj()
                  if "Run" in dir3.GetName():
                     dirFound3 = dir3.GetName()

   #entering inside target directory
InputDir = "/"+dirFound1+"/"+dirFound3
print InputDir+" InputDir"
ROOT.gDirectory.cd(InputDir)

   #Listing subdirectories of TargetDir
next4=ROOT.TIter(ROOT.gDirectory.GetListOfKeys())
for key in next4:
   cl4 = ROOT.gROOT.GetClass(key.GetClassName())
   if(cl4.InheritsFrom("TDirectory")):
      dir4=key.ReadObj()
      if "TauV" in dir4.GetName():
         dirFound4 = dir4.GetName()
         next5=ROOT.TIter(dir4.GetListOfKeys())
         for key in next5:
            cl5 = ROOT.gROOT.GetClass(key.GetClassName())
            if(cl5.InheritsFrom("TDirectory")):
               dir5=key.ReadObj()
               if "summary" in dir5.GetName():
                  dirFound5 = dir5.GetName()


InputDir_RunSummary = "/"+dirFound1+"/"+dirFound3+"/"+dirFound4+"/"+dirFound5
ROOT.gDirectory.cd(InputDir_RunSummary)

SubDirs = []
next6=ROOT.TIter(ROOT.gDirectory.GetListOfKeys())
for key in next6:
   cl6 = ROOT.gROOT.GetClass(key.GetClassName())
   if(cl6.InheritsFrom("TDirectory")):
      dir6=key.ReadObj()
      if "hpsPFTauProducer"+EventType in dir6.GetName():
         dirFound6 = dir6.GetName()
         SubDirs.append(dirFound6)  

#Writing objects to file
for sub in SubDirs:
   outFile.cd()
   ROOT.gDirectory.mkdir(sub)
   fullFile.cd()
   ROOT.gDirectory.cd(InputDir_RunSummary+"/"+sub)
   HList = ROOT.TIter(ROOT.gDirectory.GetListOfKeys())
   for k in HList:
      kk = ROOT.gROOT.GetClass(k.GetClassName())
      if(kk.InheritsFrom("TH1F")):
         outFile.cd()
         ROOT.gDirectory.cd(sub)
         obj = k.ReadObj()
         obj.Write()

outFile.Close()
