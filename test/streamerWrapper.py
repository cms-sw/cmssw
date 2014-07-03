import FWCore.ParameterSet.Config as cms

# wrapper around a CMSSW config file to make it run on streamer files on castor
#
# usage :
# cmsRun streamerWrapper.py <run> <first LS> <last LS>
#

# this is the job to run
from l1NtupleFromRaw import *

# this stuff is the wrapper
import sys
import os
from subprocess import *

# arguments
run=sys.argv[2]

firstLumi=1
lastLumi=999
if (len(sys.argv) != 2):
    firstLumi=sys.argv[3]
    lastLumi=sys.argv[4]

print "Going to run over lumi sections "+firstLumi+" to "+lastLumi+" in run "+run

runsplit=str(run)

while (len(runsplit)<9):
    runsplit="0"+runsplit

# directory on castor
root="/castor/cern.ch/cms"
dir="/store/streamer/Data/A/"+runsplit[0]+runsplit[1]+runsplit[2]+"/"+runsplit[3]+runsplit[4]+runsplit[5]+"/"+runsplit[6]+runsplit[7]+runsplit[8]+"/"

print "Looking in "+dir

# get list of files
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 

rfdir=Popen("rfdir "+root+dir, shell=True, stdout=PIPE)
files=rfdir.communicate()[0]

for file in files.splitlines():
    if ( int((file.rsplit()[8]).split(".")[2]) >= int(firstLumi) and
         int((file.rsplit()[8]).split(".")[2]) < int(lastLumi) ):
        readFiles.append(dir+file.rsplit()[8])

process.source = cms.Source(
    "NewEventStreamFileReader",
    fileNames = readFiles
)
