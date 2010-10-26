import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

## add message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.categories.append('TtFullLeptonicEvent')
process.MessageLogger.cerr.TtFullLeptonicEvent = cms.untracked.PSet(
    limit = cms.untracked.int32(-1)
)

## define input
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_8_2/RelValTTbar/GEN-SIM-RECO/MC_38Y_V9-v1/0018/E8B5D618-96AF-DF11-835A-003048679070.root'
    )
)

## define maximal number of events to loop over
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(50)
)

## configure process options
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(False)
)

## configure geometry
process.load("Configuration.StandardSequences.Geometry_cff")

## configure conditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('START38_V7::All')

## load magnetic field
process.load("Configuration.StandardSequences.MagneticField_cff")

## std sequence for pat
process.load("PhysicsTools.PatAlgos.patSequences_cff")

## std sequence to produce the ttGenEvt
process.load("TopQuarkAnalysis.TopEventProducers.sequences.ttGenEvent_cff")

## std sequence to produce the ttFullLepEvent
process.load("TopQuarkAnalysis.TopEventProducers.sequences.ttFullLepEvtBuilder_cff")
process.ttFullLepEvent.verbosity = 1

## optional change of settings
#from TopQuarkAnalysis.TopEventProducers.sequences.ttFullLepEvtBuilder_cff import *		      
#removeTtFullLepHypGenMatch(process)

#setForAllTtFullLepHypotheses(process,"muons","myMuons")
#setForAllTtFullLepHypotheses(process,"jets","myJets")
#setForAllTtFullLepHypotheses(process,"maxNJets",4)
#setForAllTtFullLepHypotheses(process,"jetCorrectionLevel","part")

## process path
process.p = cms.Path(process.patDefaultSequence *
                     process.makeGenEvt *
                     process.makeTtFullLepEvent
                    )

## configure output module
process.out = cms.OutputModule("PoolOutputModule",
    fileName     = cms.untracked.string('ttFullLepEvtBuilder.root'),
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('p') ),
    outputCommands = cms.untracked.vstring('drop *'),                      
    dropMetaData = cms.untracked.string('DROPPED')
)
process.outpath = cms.EndPath(process.out)

## pat content
from PhysicsTools.PatAlgos.patEventContent_cff import *
process.out.outputCommands += patTriggerEventContent
process.out.outputCommands += patExtraAodEventContent
process.out.outputCommands += patEventContentNoCleaning
## tqaf content
from TopQuarkAnalysis.TopEventProducers.tqafEventContent_cff import *
process.out.outputCommands += tqafEventContent
