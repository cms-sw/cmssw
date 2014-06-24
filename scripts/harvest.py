#!/usr/bin/env python
#!/afs/cern.ch/cms/sw/slc4_ia32_gcc345/external/python/2.4.2-cms6/bin/python

# get output from a given job and hadd it

import sys
import os
import re
import tarfile
import commands

#import ROOT
from subprocess import *

# arguments
crabdir=sys.argv[1]

# directories
ntupdir=os.environ['CAF_TRIGGER']+"/l1analysis/ntuples"
input_root="/castor/cern.ch/cms/store/caf/user/"+os.environ['USER']+"/"+"L1PromptAnalysis_"+crabdir
output_root="/castor/cern.ch/cms/store/caf/user/L1AnalysisNtuples"

print "Open crab directory "+crabdir+" ..."

# retrieve data
retrieve = Popen("crab -getoutput -c "+crabdir, shell=True)
retrieve.wait()

# determine if the output are stored in CASTOR
result=commands.getoutput("grep T1_CH_CERN_Buffer "+crabdir+"/log/crab.log")
output_mode=""
if (result!=""):
    output_mode="ROOTMODE"
    print "Data must be stored on CASTOR ..."
else :
    print "Data must be stored on AFS ..."

if (output_mode!="ROOTMODE"):
    print "Getting files for "+crabdir+" and putting in "+ntupdir
else:
    print "Getting files for "+input_root+" and putting in "+ntupdir
    
if (output_mode!="ROOTMODE"):
    # untar if required
    files=os.listdir(crabdir+"/res")
    for file in files:
        if re.search("tgz", file):
            tarfile.open(file, "r:gz")
            tarfile.extractall(crabdir+"/res")

# hadd all the files together
if (output_mode!="ROOTMODE"):
    hadd=Popen("hadd -f "+ntupdir+"/"+crabdir+".root "+crabdir+"/res/*.root", shell=True)
    hadd.wait()
else:
    result=commands.getoutput("nsls "+input_root+"/ > "+crabdir+"/list.txt")
    result=commands.getoutput("awk '{print \"rfio:"+input_root+"/\" $1}' "+crabdir+"/list.txt > "+crabdir+"/list2.txt");
    hadd=Popen("hadd -f "+crabdir+".root @"+crabdir+"/list2.txt", shell=True)
    hadd.wait()
    print ""
    print "Move "+crabdir+".root to "+output_root+"/"
    result=commands.getoutput("rfcp "+crabdir+".root "+output_root)
    result=commands.getoutput("mv "+crabdir+".root "+ntupdir+"/") 
    print "------------------------------------------------------------"
    print "Please do not forget to remove the crab directory on CASTOR :"
    print input_root
    


