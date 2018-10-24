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

_pp_on_AA_extraCommands = ['keep CrossingFramePlaybackInfoNew_mix_*_*','keep *_heavyIon_*_*']
from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
for e in [pp_on_XeXe_2017, pp_on_AA_2018]:
    e.toModify( SimGeneralRAW, outputCommands = SimGeneralRAW.outputCommands + _pp_on_AA_extraCommands )
    e.toModify( SimGeneralFEVTDEBUG, outputCommands = SimGeneralFEVTDEBUG.outputCommands + _pp_on_AA_extraCommands )
    e.toModify( SimGeneralRECO, outputCommands = SimGeneralRECO.outputCommands + _pp_on_AA_extraCommands )
    e.toModify( SimGeneralAOD, outputCommands = SimGeneralAOD.outputCommands + _pp_on_AA_extraCommands )
