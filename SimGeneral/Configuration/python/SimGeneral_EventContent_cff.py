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

_phase2_SimGeneralRAW_outputCommands = SimGeneralRAW.outputCommands
_phase2_SimGeneralFEVTDEBUG_outputCommands = SimGeneralFEVTDEBUG.outputCommands
_phase2_SimGeneralRECO_outputCommands = SimGeneralRECO.outputCommands

for _phase2_output in [_phase2_SimGeneralRAW_outputCommands, _phase2_SimGeneralFEVTDEBUG_outputCommands, _phase2_SimGeneralRECO_outputCommands]:
    _phase2_output.append('keep *_mix_HGCDigisEE_*')
    _phase2_output.append('keep *_mix_HGCDigisHEfront_*')
    _phase2_output.append('keep *_mix_HGCDigisHEback_*')

# mods for HGCAL
from Configuration.StandardSequences.Eras import eras
eras.phase2_hgcal.toModify( SimGeneralRAW, outputCommands = _phase2_SimGeneralRAW_outputCommands )
eras.phase2_hgcal.toModify( SimGeneralFEVTDEBUG, outputCommands = _phase2_SimGeneralFEVTDEBUG_outputCommands )
eras.phase2_hgcal.toModify( SimGeneralRECO, outputCommands = _phase2_SimGeneralRECO_outputCommands )
