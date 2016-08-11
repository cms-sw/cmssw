import FWCore.ParameterSet.Config as cms

#Full Event content
SimGeneralFEVTDEBUG = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *_trackingtruthprod_*_*', 
        'drop *_electrontruth_*_*', 
        'keep *_mix_MergedTrackTruth_*',
        'keep CrossingFramePlaybackInfoNew_*_*_*')
)
#RAW content
SimGeneralRAW = cms.PSet(
    outputCommands = cms.untracked.vstring('keep CrossingFramePlaybackInfoNew_*_*_*',
                                           'keep PileupSummaryInfos_*_*_*',
                                           'keep int_*_bunchSpacing_*')
)
#RECO content
SimGeneralRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep PileupSummaryInfos_*_*_*',
                                           'keep int_*_bunchSpacing_*')
)
#AOD content
SimGeneralAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep PileupSummaryInfos_*_*_*',
                                           'keep int_*_bunchSpacing_*')
)

# mods for HGCAL
_phase2_hgc_extraCommands = [ 'keep *_mix_HGCDigisEE_*', 'keep *_mix_HGCDigisHEfront_*', 'keep *_mix_HGCDigisHEback_*', 
                              'keep *_mix_MergedCaloTruth_*' ]
from Configuration.StandardSequences.Eras import eras
eras.phase2_hgcal.toModify( SimGeneralRAW, outputCommands = SimGeneralRAW.outputCommands + _phase2_hgc_extraCommands )
eras.phase2_hgcal.toModify( SimGeneralFEVTDEBUG, outputCommands = SimGeneralFEVTDEBUG.outputCommands + _phase2_hgc_extraCommands )
eras.phase2_hgcal.toModify( SimGeneralRECO, outputCommands = SimGeneralRECO.outputCommands + _phase2_hgc_extraCommands )
