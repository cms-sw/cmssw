import FWCore.ParameterSet.Config as cms

#Full Event content
SimGeneralFEVTDEBUG = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *_trackingtruthprod_*_*', 
        'drop *_electrontruth_*_*', 
        'keep *_mergedtruth_*_*')
)
#RAW content
SimGeneralRAW = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
#RECO content
SimGeneralRECO = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
#AOD content
SimGeneralAOD = cms.PSet(
    outputCommands = cms.untracked.vstring()
)

