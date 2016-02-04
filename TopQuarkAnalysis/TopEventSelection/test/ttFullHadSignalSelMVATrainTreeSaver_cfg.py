import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

## add message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.cerr.FwkReport.reportEvery = 1

## define input
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_4_0_pre1/RelValTTbar/GEN-SIM-RECO/MC_31X_V9-v1/0008/2C8CD8FE-B6B5-DE11-ACB8-001D09F2905B.root'
     ),
     skipEvents = cms.untracked.uint32(0)
)
## define maximal number of events to loop over
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
## configure process options
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

## configure geometry & conditions
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('MC_38Y_V14::All')

## std sequence for pat
process.load("PhysicsTools.PatAlgos.patSequences_cff")

#from PhysicsTools.PatAlgos.tools.cmsswVersionTools import *

# run the 3.3.x software on Summer 09 MC from 3.1.x:
#   - change the name from "ak" (3.3.x) to "antikt) (3.1.x)
#   - run jet ID (not run in 3.1.x)
#run33xOn31xMC( process,
#               jetSrc = cms.InputTag("antikt5CaloJets"),
#               jetIdTag = "antikt5"
#               )

#restrictInputToAOD31X(process)

## std sequence for ttGenEvent
process.load("TopQuarkAnalysis.TopEventProducers.sequences.ttGenEvent_cff")

## filter for full-hadronic 
process.load("TopQuarkAnalysis.TopEventProducers.producers.TtDecaySelection_cfi")
process.ttFullHadronicFilter = process.ttDecaySelection.clone()

## configure mva trainer
process.load("TopQuarkAnalysis.TopEventSelection.TtFullHadSignalSelMVATrainTreeSaver_cff")

## make trainer looper known to the process
from TopQuarkAnalysis.TopEventSelection.TtFullHadSignalSelMVATrainTreeSaver_cff import looper
process.looper = looper

## to tell the MVA trainer that an event is background and not signal
#process.buildTraintree.whatData = 0

## to filter ttbar background events instead of fully hadronic signal events
#process.ttFullHadronicFilter.invert = True

## jet count filter
process.load("PhysicsTools.PatAlgos.selectionLayer1.jetCountFilter_cfi")

## setup jet collection, right now at least 6 jets needed for the MVA trainer/computer
process.leadingJetSelection = process.countPatJets.clone(src = 'selectedPatJets',
                                                         minNumber = 6
                                                         )

## produce pat objects and ttGenEvt and make mva training
process.p = cms.Path(process.patDefaultSequence *
                     process.leadingJetSelection *
                     process.makeGenEvt *
                     process.ttFullHadronicFilter *
                     process.saveTrainTree)
