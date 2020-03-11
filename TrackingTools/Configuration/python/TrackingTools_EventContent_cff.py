import FWCore.ParameterSet.Config as cms

#AOD content
TrackingToolsAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoTracks_GsfGlobalElectronTest_*_*',
        'keep recoGsfTracks_electronGsfTracks_*_*'
    )
)

from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toModify( TrackingToolsAOD,
                       outputCommands = TrackingToolsAOD.outputCommands + ['keep recoGsfTracks_electronGsfTracksFromMultiCl_*_*'])

#RECO content
TrackingToolsRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_CkfElectronCandidates_*_*', 
        'keep *_GsfGlobalElectronTest_*_*',
        'keep *_electronMergedSeeds_*_*',
        'keep recoGsfTrackExtras_electronGsfTracks_*_*', 
        'keep recoTrackExtras_electronGsfTracks_*_*', 
        'keep TrackingRecHitsOwned_electronGsfTracks_*_*'
    )
)
TrackingToolsRECO.outputCommands.extend(TrackingToolsAOD.outputCommands)
_phase2_hgcal_TrackingRECO_tokeep = [
        'keep recoGsfTracks_electronGsfTracksFromMultiCl_*_*',
        'keep recoGsfTrackExtras_electronGsfTracksFromMultiCl_*_*',
        'keep recoTrackExtras_electronGsfTracksFromMultiCl_*_*',
        'keep TrackingRecHitsOwned_electronGsfTracksFromMultiCl_*_*',
        'keep *_electronMergedSeedsFromMultiCl_*_*'
]
phase2_hgcal.toModify( TrackingToolsRECO,
                       outputCommands = TrackingToolsRECO.outputCommands + _phase2_hgcal_TrackingRECO_tokeep)

#FEVT content
TrackingToolsFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_electronGsfTracks_*_*'
    )
)
TrackingToolsFEVT.outputCommands.extend(TrackingToolsRECO.outputCommands)
