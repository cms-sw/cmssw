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
                                           'keep int6stdbitsetstdpairs_*_AffectedAPVList_*',
                                           'keep int_*_bunchSpacing_*',
                                           'keep *_genPUProtons_*_*',
                                           'keep *_mix_MergedTrackTruth_*') 
)
#RECO content
SimGeneralRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep PileupSummaryInfos_*_*_*',
                                           'keep int_*_bunchSpacing_*',
                                           'keep *_genPUProtons_*_*') 
)
#AOD content
SimGeneralAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep PileupSummaryInfos_*_*_*',
                                           'keep int_*_bunchSpacing_*',
                                           'keep *_genPUProtons_*_*') 
)

# mods for HGCAL
_phase2_hgc_extraCommands = [ 'keep *_mix_HGCDigisEE_*', 'keep *_mix_HGCDigisHEfront_*', 'keep *_mix_HGCDigisHEback_*', 
                              'keep *_mix_MergedCaloTruth_*' ]
from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toModify( SimGeneralRAW, outputCommands = SimGeneralRAW.outputCommands + _phase2_hgc_extraCommands )
phase2_hgcal.toModify( SimGeneralFEVTDEBUG, outputCommands = SimGeneralFEVTDEBUG.outputCommands + _phase2_hgc_extraCommands )
phase2_hgcal.toModify( SimGeneralRECO, outputCommands = SimGeneralRECO.outputCommands + _phase2_hgc_extraCommands )

_phase2_timing_extraCommands = [ 'keep *_mix_FTLBarrel_*','keep *_mix_FTLEndcap_*','keep *_mix_InitialVertices_*' ]
from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing
phase2_timing.toModify( SimGeneralRAW, outputCommands = SimGeneralRAW.outputCommands + _phase2_timing_extraCommands )
phase2_timing.toModify( SimGeneralFEVTDEBUG, outputCommands = SimGeneralFEVTDEBUG.outputCommands + _phase2_timing_extraCommands )
phase2_timing.toModify( SimGeneralRECO, outputCommands = SimGeneralRECO.outputCommands + _phase2_timing_extraCommands )

