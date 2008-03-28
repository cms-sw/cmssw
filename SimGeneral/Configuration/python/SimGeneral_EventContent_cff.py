import FWCore.ParameterSet.Config as cms

#Full Event content
SimGeneralFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *_trackingtruthprod_*_*', 'drop *_electrontruth_*_*', 'keep *_mergedtruth_*_*')
)
#RECO content
SimGeneralRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *_trackingtruthprod_*_*', 'drop *_electrontruth_*_*', 'keep *_mergedtruth_*_*')
)
#AOD content
SimGeneralAOD = cms.PSet(
    outputCommands = cms.untracked.vstring()
)

