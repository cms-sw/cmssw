import FWCore.ParameterSet.Config as cms

#Full Event content
SimGeneralFEVTDEBUG = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *_trackingtruthprod_*_*', 
        'drop *_electrontruth_*_*', 
        'keep *_mix_MergedTrackTruth_*',
        'keep CrossingFramePlaybackInfoExtended_*_*_*')
)
#RAW content
SimGeneralRAW = cms.PSet(
    outputCommands = cms.untracked.vstring('keep CrossingFramePlaybackInfoExtended_*_*_*',
                                           'keep PileupSummaryInfos_*_*_*',
                                           'keep int_addPileupInfo_bunchSpacing_*')
)
#RECO content
SimGeneralRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep PileupSummaryInfos_*_*_*',
                                           'keep int_addPileupInfo_bunchSpacing_*')
)
#AOD content
SimGeneralAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep PileupSummaryInfos_*_*_*',
                                           'keep int_addPileupInfo_bunchSpacing_*')
)

