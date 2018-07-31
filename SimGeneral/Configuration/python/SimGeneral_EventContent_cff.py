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
# Event content for premixing library
SimGeneralPREMIX = cms.PSet(
    outputCommands = cms.untracked.vstring()
)

# mods for HGCAL; should these be moved under SimCalorimetry?
_phase2_hgc_extraCommands = cms.PSet( # using PSet in order to customize with Modifier
    v = cms.vstring('keep *_simHGCalUnsuppressedDigis_EE_*', 'keep *_simHGCalUnsuppressedDigis_HEfront_*', 'keep *_simHGCalUnsuppressedDigis_HEback_*', 'keep *_mix_MergedCaloTruth_*'),
)
# For phase2 premixing switch the sim digi collections to the ones including pileup
from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(_phase2_hgc_extraCommands,
    v = ['keep *_mixData_HGCDigisEE_*', 'keep *_mixData_HGCDigisHEfront_*', 'keep *_mixData_HGCDigisHEback_*', 'keep *_mixData_MergedCaloTruth_*']
)
from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toModify( SimGeneralRAW, outputCommands = SimGeneralRAW.outputCommands + _phase2_hgc_extraCommands.v )
phase2_hgcal.toModify( SimGeneralFEVTDEBUG, outputCommands = SimGeneralFEVTDEBUG.outputCommands + _phase2_hgc_extraCommands.v )
phase2_hgcal.toModify( SimGeneralRECO, outputCommands = SimGeneralRECO.outputCommands + _phase2_hgc_extraCommands.v )
phase2_hgcal.toModify( SimGeneralPREMIX, outputCommands = SimGeneralPREMIX.outputCommands + _phase2_hgc_extraCommands.v )

_phase2_timing_extraCommands = [ 'keep *_mix_FTLBarrel_*','keep *_mix_FTLEndcap_*','keep *_mix_InitialVertices_*' ]
from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing
phase2_timing.toModify( SimGeneralRAW, outputCommands = SimGeneralRAW.outputCommands + _phase2_timing_extraCommands )
phase2_timing.toModify( SimGeneralFEVTDEBUG, outputCommands = SimGeneralFEVTDEBUG.outputCommands + _phase2_timing_extraCommands )
phase2_timing.toModify( SimGeneralRECO, outputCommands = SimGeneralRECO.outputCommands + _phase2_timing_extraCommands )

_pp_on_AA_extraCommands = ['keep CrossingFramePlaybackInfoNew_mix_*_*','keep *_heavyIon_*_*']
from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
for e in [pp_on_XeXe_2017, pp_on_AA_2018]:
    e.toModify( SimGeneralRAW, outputCommands = SimGeneralRAW.outputCommands + _pp_on_AA_extraCommands )
    e.toModify( SimGeneralFEVTDEBUG, outputCommands = SimGeneralFEVTDEBUG.outputCommands + _pp_on_AA_extraCommands )
    e.toModify( SimGeneralRECO, outputCommands = SimGeneralRECO.outputCommands + _pp_on_AA_extraCommands )
    e.toModify( SimGeneralAOD, outputCommands = SimGeneralAOD.outputCommands + _pp_on_AA_extraCommands )
