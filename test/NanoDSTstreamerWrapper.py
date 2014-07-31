import FWCore.ParameterSet.Config as cms

# wrapper for L1Ntuple-NanoDST production from streamer files
#
# usage :
# cmsRun NanoDSTstreamerWrapper.py <jsonfile>
# or
# cmsRun NanoDSTstreamerWrapper.py <jsonfile> castor
#

# this is the job to run


# this stuff is the wrapper
import sys
import os, json
from subprocess import *


if len(sys.argv)<3:
    print "Wrong parameters!"
    print "Please do:"
    print "       GetData.py <config_file>"
    print "to produce the output root file in current directory,"
    print "or"
    print "       GetData.py <config_file> castor"
    print "to produce to output root file in castor CAF directory"
    print "rfio:///castor/cern.ch/cms/store/caf/user/L1AnalysisNtuples/Nano/"
    print
    sys.exit()



######################## input and output file names #########################
jsonconfigfile=sys.argv[2]
fileradix=''
if len(sys.argv)==4:
    if sys.argv[3]=='castor':
        fileradix='rfio:///castor/cern.ch/cms/store/caf/user/L1AnalysisNtuples/Nano/'
fname=fileradix+'L1TreeNano'
#print "--",jsonconfigfile,"--"
#print sys.argv[0],sys.argv[1],sys.argv[2]
############################ get data file list ###############################
a={}        
with open(jsonconfigfile) as f:
    a = json.load(f)
    f.close()
readFiles = cms.untracked.vstring()
#print jsonconfigfile
for run, lumis in a.iteritems():
    fname = fname+'_'+str(run)
    root="/castor/cern.ch/cms"
    while len(run)<9:
        run='0'+run
        #print run
        #        spath="/store/t0streamer/Data/NanoDST/"
    dir="/store/t0streamer/Data/NanoDST/"+run[0]+run[1]+run[2]+"/"+run[3]+run[4]+run[5]+"/"+run[6]+run[7]+run[8]+"/"
    #print dir
    rfdir=Popen("rfdir "+root+dir, shell=True, stdout=PIPE)
    files=rfdir.communicate()[0]
    #print 'pippo'
    #print files
    for file in files.splitlines():
        if  int((file.rsplit()[8]).split(".")[2]) in [j[0] for j in lumis] :
            readFiles.append(str(dir+file.rsplit()[8]))
fname = fname+'.root'

print
#print readFiles

######################## now running #########################################
print "Running on "+str(len(readFiles))+" input files"
print "Writing output file",fname			
# cms process    
process = cms.Process("L1NTUPLE")
process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration/StandardSequences/EndOfProcess_cff')
process.load('Configuration/EventContent/EventContent_cff')
# analysis
process.load("L1TriggerDPG.L1Ntuples.l1NtupleProducerNano_cfi")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.source = cms.Source(
    "NewEventStreamFileReader",
    fileNames = readFiles
)
process.p = cms.Path(
     process.l1NtupleProducer
)
process.TFileService = cms.Service("TFileService",
    fileName = cms.string(fname)                                  
)
