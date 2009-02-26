import FWCore.ParameterSet.Config as cms

#-------------------------------------------------
# test cfg file for the production of a 
# ttSemiEvent
#-------------------------------------------------
process = cms.Process("TEST")

## add message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.categories.append('TtSemiLeptonicEvent')
process.MessageLogger.categories.append('TtSemiLepKinFitter')
#process.MessageLogger.categories.append('KinFitter')
process.MessageLogger.cerr.INFO = cms.untracked.PSet(
    default             = cms.untracked.PSet( limit = cms.untracked.int32( 0) ),
    TtSemiLeptonicEvent = cms.untracked.PSet( limit = cms.untracked.int32(-1) ),
    TtSemiLepKinFitter  = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
#    KinFitter           = cms.untracked.PSet( limit = cms.untracked.int32(-1) )
)

#-------------------------------------------------
# process configuration
#-------------------------------------------------

## define input
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
   #PAT test sample for 2.2.X
   #'file:/afs/cern.ch/cms/PRS/top/cmssw-data/relval200-for-pat-testing/FullSimTTBar-2_2_X_2008-11-03-STARTUP_V7-AODSIM.100.root'
   #PAT test sample for 2.1.X
    'file:/afs/cern.ch/cms/PRS/top/cmssw-data/relval200-for-pat-testing/FullSimTTBar-2_1_X_2008-07-08_STARTUP_V4-AODSIM.100.root'    
   #219 RelVal sample
   #'rfio:/castor/cern.ch/cms/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/16D75F0F-1186-DD11-80B9-000423D98C20.root',
   #'rfio:/castor/cern.ch/cms/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/24CD41BB-1A86-DD11-9CDA-000423D98EC8.root',
   #'rfio:/castor/cern.ch/cms/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/264D79AA-1786-DD11-9F3C-001617C3B6DC.root',
   #'rfio:/castor/cern.ch/cms/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/327DE1B9-0E86-DD11-B7B1-000423D6C8E6.root',
   #'rfio:/castor/cern.ch/cms/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/345AE083-1186-DD11-8D43-000423D99658.root',
   #'rfio:/castor/cern.ch/cms/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/4A0ADB7D-1086-DD11-BD16-000423D98E6C.root',
   #'rfio:/castor/cern.ch/cms/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/4E31E969-1886-DD11-8398-000423D9989E.root',
   #'rfio:/castor/cern.ch/cms/store/relval/CMSSW_2_1_9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/4EEEA6AE-0886-DD11-90F9-000423D94990.root'
    )
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

## load magnetic field
process.load("Configuration.StandardSequences.MagneticField_cff")


#-------------------------------------------------
# tqaf configuration
#-------------------------------------------------

## std sequence for tqaf layer1
process.load("TopQuarkAnalysis.TopObjectProducers.tqafLayer1_cff")

## std sequence to produce the ttGenEvt
process.load("TopQuarkAnalysis.TopEventProducers.sequences.ttGenEvent_cff")

## std sequence to produce the ttSemiLepEvent
process.load("TopQuarkAnalysis.TopEventProducers.sequences.ttSemiLepEvtBuilder_cff")
process.ttSemiLepEvent.verbosity = 1

## change maximum number of jets taken into account per event (default: 4)
#process.ttSemiLepEvent.maxNJets = 5
## change maximum number of jet combinations taken into account per event (default: 1)
#process.findTtSemiLepJetCombMVA       .maxNComb = -1
#process.kinFitTtSemiLepEventHypothesis.maxNComb = -1

## necessary fixes to run 2.2.X on 2.1.X data
## comment this when running on samples produced
## with 22X
from PhysicsTools.PatAlgos.tools.cmsswVersionTools import run22XonSummer08AODSIM
run22XonSummer08AODSIM(process)

## process path
process.p = cms.Path(process.tqafLayer1  *
                     process.makeGenEvt  *
                     process.makeTtSemiLepEvent
                     )

## define event selection
process.EventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('p')
    )
)

## define tqaf layer1 event content
from TopQuarkAnalysis.TopObjectProducers.tqafLayer1_EventContent_cff import *
makeTqafLayer1EventContent(process)
## define tqaf layer2 event content
from TopQuarkAnalysis.TopEventProducers.tqafLayer2_EventContent_cff import *
makeTqafLayer2EventContent(process)

## configure output module
process.out = cms.OutputModule("PoolOutputModule",
    process.EventSelection,                      
    process.tqafEventContent,
    fileName = cms.untracked.string('ttSemiLepEvtBuilder.root')
)

## output path
## in order not to write the persistent output to
## file, comment the following outpath
process.outpath = cms.EndPath(process.out)
