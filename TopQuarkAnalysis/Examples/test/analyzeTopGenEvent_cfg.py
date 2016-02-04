import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

## configure message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = cms.untracked.string('INFO')
## dump content of TopGenEvent
process.MessageLogger.categories.append('TopGenEvent')

## define input
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_8_2/RelValTTbar/GEN-SIM-RECO/MC_38Y_V9-v1/0018/E8B5D618-96AF-DF11-835A-003048679070.root'
     ),
     skipEvents = cms.untracked.uint32(0)                            
)

## define maximal number of events to loop over
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

## configure process options
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(False)
)

process.load("TopQuarkAnalysis.TopEventProducers.sequences.ttGenEvent_cff")

from TopQuarkAnalysis.Examples.TopGenEventAnalyzer_cfi import analyzeTopGenEvent
process.analyzeTopGenEvent = analyzeTopGenEvent

# register TFileService
process.TFileService = cms.Service("TFileService",
    fileName = cms.string('analyzeTopGenEvent.root')
)

## end path   
process.p1 = cms.Path(process.makeGenEvt *
                      process.analyzeTopGenEvent)

