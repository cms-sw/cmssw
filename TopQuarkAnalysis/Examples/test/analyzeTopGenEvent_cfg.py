import FWCore.ParameterSet.Config as cms

#-------------------------------------------------
# test cfg file for tqaflayer1 & 2 production from
# fullsim for semi-leptonic ttbar events 
#-------------------------------------------------
process = cms.Process("TEST")

## configure message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.categories = cms.untracked.vstring('decayChain')
process.MessageLogger.cerr = cms.untracked.PSet(threshold = cms.untracked.string('INFO'),
                                                decayChain = cms.untracked.PSet(limit = cms.untracked.int32(10))
                                                )

#-------------------------------------------------
# process configuration
#-------------------------------------------------

## define input
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
   #PAT test sample
   #'file:/afs/cern.ch/cms/PRS/top/cmssw-data/relval200-for-pat-testing/FullSimTTBar-2_1_X_2008-07-08_STARTUP_V4-AODSIM.100.root'
   #219 RelVal sample
   'rfio:/castor/cern.ch/cms/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/16D75F0F-1186-DD11-80B9-000423D98C20.root',
   'rfio:/castor/cern.ch/cms/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/24CD41BB-1A86-DD11-9CDA-000423D98EC8.root',
   'rfio:/castor/cern.ch/cms/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/264D79AA-1786-DD11-9F3C-001617C3B6DC.root',
   'rfio:/castor/cern.ch/cms/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/327DE1B9-0E86-DD11-B7B1-000423D6C8E6.root',
   'rfio:/castor/cern.ch/cms/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/345AE083-1186-DD11-8D43-000423D99658.root',
   'rfio:/castor/cern.ch/cms/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/4A0ADB7D-1086-DD11-BD16-000423D98E6C.root',
   'rfio:/castor/cern.ch/cms/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/4E31E969-1886-DD11-8398-000423D9989E.root',
   'rfio:/castor/cern.ch/cms/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/4EEEA6AE-0886-DD11-90F9-000423D94990.root'
   #'file:/afs/desy.de/user/r/rwolf/cms13/samples/21X/16D75F0F-1186-DD11-80B9-000423D98C20.root'
   #'file:/afs/desy.de/user/r/rwolf/cms13/samples/21X/24CD41BB-1A86-DD11-9CDA-000423D98EC8.root',
   #'file:/afs/desy.de/user/r/rwolf/cms13/samples/21X/264D79AA-1786-DD11-9F3C-001617C3B6DC.root',
   #'file:/afs/desy.de/user/r/rwolf/cms13/samples/21X/327DE1B9-0E86-DD11-B7B1-000423D6C8E6.root',
   #'file:/afs/desy.de/user/r/rwolf/cms13/samples/21X/345AE083-1186-DD11-8D43-000423D99658.root',
   #'file:/afs/desy.de/user/r/rwolf/cms13/samples/21X/4A0ADB7D-1086-DD11-BD16-000423D98E6C.root',
   #'file:/afs/desy.de/user/r/rwolf/cms13/samples/21X/4E31E969-1886-DD11-8398-000423D9989E.root',
   #'file:/afs/desy.de/user/r/rwolf/cms13/samples/21X/4EEEA6AE-0886-DD11-90F9-000423D94990.root'
   ),
   #skipEvents = cms.untracked.uint32(300)
)

## define maximal number of events to loop over
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)


## configure process options
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(False)
)

## configure geometry
process.load("Configuration.StandardSequences.Geometry_cff")

## configure conditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('STARTUP_V7::All')

# Magnetic field now needs to be in the high-level py
process.load("Configuration.StandardSequences.MagneticField_cff")




#-------------------------------------------------
# tqaf configuration; if the TQAF Layer 1 is
# already in place yuo can comment the following
# two lines
#-------------------------------------------------

## std sequence for tqaf layer1
#process.load("TopQuarkAnalysis.TopObjectProducers.tqafLayer1_full_cff")
#process.p0 = cms.Path(process.tqafLayer1)

## std sequence for tqaf layer2 ttGenEvent
process.load("TopQuarkAnalysis.TopEventProducers.sequences.ttGenEvent_cff")
process.p1 = cms.Path(process.makeGenEvt)

#-------------------------------------------------
# analyze genEvent
#-------------------------------------------------
from TopQuarkAnalysis.Examples.TopGenEventAnalyzer_cfi import analyzeTopGenEvent
process.analyzeTopGenEvent = analyzeTopGenEvent

# register TFService
process.TFileService = cms.Service("TFileService",fileName = cms.string('analyzegenevent.root')
)

## end path   
process.p2 = cms.Path(process.analyzeTopGenEvent)

