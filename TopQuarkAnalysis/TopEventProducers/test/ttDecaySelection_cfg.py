import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

## add message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.categories.append('TtDecayChannelSelector')
process.MessageLogger.cerr.TtDecayChannelSelector = cms.untracked.PSet(
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
    input = cms.untracked.int32(10)
)
## configure process options
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(False)
)

## configure genEvent
process.load("TopQuarkAnalysis.TopEventProducers.sequences.ttGenEvent_cff")
process.decaySubset.fillMode = "kME"
process.decaySubset.addRadiation = True
       
## configure decay channel selection 
process.load("TopQuarkAnalysis.TopEventProducers.producers.TtDecaySelection_cfi")
process.ttDecaySelection.src = "genEvt"
process.ttDecaySelection.allowedTopDecays.decayBranchA.electron = True
process.ttDecaySelection.allowedTopDecays.decayBranchA.muon     = True
process.ttDecaySelection.allowedTopDecays.decayBranchB.electron = False
process.ttDecaySelection.allowedTopDecays.decayBranchB.muon     = False
#process.ttDecaySelection.restrictTauDecays = cms.PSet(leptonic = cms.bool(True))
#process.ttDecaySelection.restrictTauDecays = cms.PSet(oneProng = cms.bool(True))
#process.ttDecaySelection.restrictTauDecays = cms.PSet(threeProng = cms.bool(True))

## path
process.path = cms.Path(process.makeGenEvt *
                        process.ttDecaySelection)
