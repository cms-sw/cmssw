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

def _modifySimGeneralEventContentForHGCal( obj ):
    obj.outputCommands.append('keep *_mix_HGCDigisEE_*')
    obj.outputCommands.append('keep *_mix_HGCDigisHEfront_*')
    obj.outputCommands.append('keep *_mix_HGCDigisHEback_*')

# mods for HGCAL
from Configuration.StandardSequences.Eras import eras
eras.phase2_hgcal.toModify( SimGeneralRAW, func=_modifySimGeneralEventContentForHGCal )
eras.phase2_hgcal.toModify( SimGeneralFEVTDEBUG, func=_modifySimGeneralEventContentForHGCal )
eras.phase2_hgcal.toModify( SimGeneralRECO, func=_modifySimGeneralEventContentForHGCal )
