import FWCore.ParameterSet.Config as cms

TrackingToolsFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_CkfElectronCandidates_*_*', 
        'keep *_GsfGlobalElectronTest_*_*',
        'keep *_electronMergedSeeds_*_*',
        'keep *_electronGsfTracks_*_*'
        )
)
#RECO content
TrackingToolsRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_CkfElectronCandidates_*_*', 
        'keep *_GsfGlobalElectronTest_*_*',
        'keep *_electronMergedSeeds_*_*',
        'keep recoGsfTracks_electronGsfTracks_*_*', 
        'keep recoGsfTrackExtras_electronGsfTracks_*_*', 
        'keep recoTrackExtras_electronGsfTracks_*_*', 
        'keep TrackingRecHitsOwned_electronGsfTracks_*_*'                                            )
)

_phase2_hgcal_TrackingRECO_tokeep = [
        'keep recoGsfTracks_electronGsfTracksFromMultiCl_*_*',
        'keep recoGsfTrackExtras_electronGsfTracksFromMultiCl_*_*',
        'keep recoTrackExtras_electronGsfTracksFromMultiCl_*_*',
        'keep TrackingRecHitsOwned_electronGsfTracksFromMultiCl_*_*',
        'keep *_electronMergedSeedsFromMultiCl_*_*'
]

#AOD content
TrackingToolsAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoTracks_GsfGlobalElectronTest_*_*',
        'keep recoGsfTracks_electronGsfTracks_*_*'
                                         
    )
)

from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toModify( TrackingToolsRECO,
                       outputCommands = TrackingToolsRECO.outputCommands + _phase2_hgcal_TrackingRECO_tokeep)
phase2_hgcal.toModify( TrackingToolsAOD,
                       outputCommands = TrackingToolsAOD.outputCommands + ['keep recoGsfTracks_electronGsfTracksFromMultiCl_*_*'])

