#!/usr/bin/env python
#!/afs/cern.ch/cms/sw/slc4_ia32_gcc345/external/python/2.4.2-cms6/bin/python

# script to create & submit CRAB jobs for l1 prompt analysis

import sys
import os
import getopt
import re
import shutil

from subprocess import *


# constants
express     = "/ExpressPhysics/Commissioning10-Express-v9/FEVT"
minbias     = "/MinimumBias/Commissioning10-PromptReco-v9/RECO"
minbiasraw  = "/MinimumBias/Commissioning10-v4/RAW"
zerobias    = "/ZeroBias/Commissioning10-PromptReco-v9/RECO"
zerobiasraw = "/ZeroBias/Commissioning10-v4/RAW"
testenables = "/TestEnables/Commissioning10-v4/RAW"
hltmon      = "/OfflineMonitor/Commissioning10-Express-v9/FEVTHLTALL"
goodcoll    = "/MinimumBias/Commissioning10-GOODCOLL-v9/RAW-RECO"
goodvertex  = "/ZeroBias/Commissioning10-GOODVERTEX-v9/RAW-RECO"

jobpath=os.environ['CAF_TRIGGER']+"/l1analysis/cmssw/"+os.environ['L1CMSSW']+"/src/UserCode/L1TriggerDPG/test/"

#
def usage():
    print "Usage : submit.py [ -e | -n ] [ synchronisation ] <run>"
    print

# arguments : file location and local dir for results
try:
    opts, args = getopt.getopt(sys.argv[1:], "hen")
except getopt.GetoptError:
    usage()
    sys.exit(2)

doNtuple=False
doEmulator=False
for opt, arg in opts:
    if opt=='-h':
        usage()
        exit.sys()
    if opt=='-n':
        doNtuple=True
    if opt=='-e':
        doEmulator=True

print args

dataset = " "
if (args[0]=="synchronisation"):
    dataset="MinBias"

run = args[1]

if (len(run)==6):
    run="000"+run


# run
# make directory for results if it doesn't already exist
# and add
wwwdir = os.environ['CAF_TRIGGER']+"/l1analysis/www/"+str(run)
if not os.path.exists(wwwdir):
    os.mkdir(wwwdir)
shutil.copy(os.environ['CAF_TRIGGER']+"/l1analysis/www/subdir_htaccess.tmp", wwwdir+"/.htaccess")


# dataset
# set use_parent=1 for RECO datasets
datasetpath = minbias
useparent   = 0

if (dataset=="MinBias"):
    datasetpath=minbias
    useparent=1
elif (dataset=="ZeroBias"):
    datasetpath=zerobias
    useparent=1
elif (dataset=="MinBiasRaw"):
    datasetpath=minbiasraw
elif (dataset=="ZeroBiasRaw"):
    datasetpath=zerobiasraw
elif (dataset=="TestEnables"):
    datasetpath=testenables
elif (dataset=="HLTMon"):
    datasetpath=hltmon
elif (dataset=="ExpressPhysics"):
    datasetpath=express
elif (dataset=="GOODCOLL"):
    datasetpath=goodcoll
elif (dataset=="GoodVertex"):
    datasetpath=goodvertex
else :
    print "Dataset name is not correct !"
    sys.exit(1)
    
# set job dependent variables
job =""
label = ""
outfile = ""

# check if the dataset is RAW
p = re.compile('.+Raw')
raw = p.match(datasetpath)

if (doEmulator):
    # emulator jobs will work on RECO (2 file solution) or RAW
    job="l1EmulatorJob.py"
    label=dataset+"_Emulator_"+str(run)
    outfile="L1EmulHistos.root"
elif (doNtuple):
    # ntuple job needs to be adjusted if run on RAW
    label=dataset+"_Ntuple_"+str(run)
    outfile = "L1Tree.root"
    if raw:
        job="l1NtupleFromRaw.py"
    else:
        job="l1NtupleFromReco.py"
else:
    print "Please specify a type of job to run!"
    sys.exit(1)

# CRAB file
crabfile="crab-"+dataset+"-"+run+".cfg"


# create job file
print "Making CRAB job for "+dataset+", run "+run+", using "+datasetpath
crab_returndata="""
[USER]
return_data = 1
copy_data = 0
ui_working_dir = """+label+"""
"""

crab_copydata="""
[USER]
return_data = 0
copy_data = 1
storage_element = T1_CH_CERN_Buffer
user_remote_dir = L1PromptAnalysis_"""+label+"""
ui_working_dir = """+label+"""
"""

crab_output=""
if (doEmulator):
     crab_output=crab_returndata
else:
     crab_output=crab_copydata
     

crab1="""
[CRAB]
jobtype = cmssw
scheduler = caf

[CMSSW]
dbs_url=http://cmsdbsprod.cern.ch/cms_dbs_prod_global/servlet/DBSServlet
datasetpath="""+datasetpath+"""
runselection="""+run+"""
pset="""+jobpath+job+"""
total_number_of_events=-1 
events_per_job=100000
#number_of_jobs=-1
output_file = """+outfile+"""
use_parent="""+str(useparent)+"""
check_user_remote_dir=0
"""

crab2="""
[GRID]
rb = CERN
proxy_server = myproxy.cern.ch
"""

f = open(crabfile, 'w')
f.write(crab1)
f.write(crab_output)
f.write(crab2)

print "crab -create -submit -cfg "+crabfile

submit = Popen("crab -create -submit -cfg "+crabfile, shell=True)
#submit.wait()
