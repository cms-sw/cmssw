import FWCore.ParameterSet.Config as cms

process = cms.Process("TQAF")

## configure message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'

## define input
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/data/BeamCommissioning09/MinimumBias/RAW-RECO/SD_InterestingEvents-Dec19thSkim_341_v1/0005/B641315C-ACED-DE11-82E1-0030486792B6.root'
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

## configure geometry & conditions
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('GR_R_38X_V7::All')

#-------------------------------------------------
# PAT and TQAF configuration
#-------------------------------------------------

## std sequence for PAT
process.load("PhysicsTools.PatAlgos.patSequences_cff")

from PhysicsTools.PatAlgos.tools.cmsswVersionTools import run36xOn35xInput
run36xOn35xInput(process)

## remove MC specific stuff in PAT
from PhysicsTools.PatAlgos.tools.coreTools import *
removeMCMatching(process, ["All"])

## std sequence for TQAF
process.load("TopQuarkAnalysis.TopEventProducers.tqafSequences_cff")

## remove MC specific stuff in TQAF
process.tqafTtSemiLeptonic.remove(process.makeGenEvt)
from TopQuarkAnalysis.TopEventProducers.sequences.ttSemiLepEvtBuilder_cff import *
addTtSemiLepHypotheses(process, ["kGeom", "kWMassMaxSumPt", "kMaxSumPtWMass"])
removeTtSemiLepHypGenMatch(process)

## process path
process.p = cms.Path(process.patDefaultSequence *
                     process.tqafTtSemiLeptonic
                     )

## configure output module
process.out = cms.OutputModule("PoolOutputModule",
    fileName       = cms.untracked.string('tqafOutput.woGeneratorInfo.root'),
    SelectEvents   = cms.untracked.PSet(SelectEvents = cms.vstring('p') ),
    outputCommands = cms.untracked.vstring('drop *'),                      
    dropMetaData   = cms.untracked.string("DROPPED")  ## NONE    for none
                                                      ## DROPPED for drop for dropped data
)
process.outpath = cms.EndPath(process.out)

## PAT content
from PhysicsTools.PatAlgos.patEventContent_cff import *
process.out.outputCommands += patTriggerEventContent
process.out.outputCommands += patExtraAodEventContent
process.out.outputCommands += patEventContentNoCleaning

## TQAF content
from TopQuarkAnalysis.TopEventProducers.tqafEventContent_cff import *
process.out.outputCommands += tqafEventContent
