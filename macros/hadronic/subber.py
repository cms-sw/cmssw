#!/vols/sl5_exp_software/cms/slc5_amd64_gcc434/cms/cmssw-patch/CMSSW_4_1_3_patch3/external/slc5_amd64_gcc434/bin/python

import ROOT as r
import sys, os
inpu = file("./dec22List.txt","r")
#inpu = file("./incompList.txt","r")
flist = inpu.readlines()
for i in range(0,len(flist)):
  os.system("qsub -q hepmedium.q subScript.sh "+str(i)+" " + sys.argv[1] +" "+sys.argv[2])
