import FWCore.ParameterSet.Config as cms

#AOD content
TrackingToolsAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep recoTracks_GsfGlobalElectronTest_*_*',
        'keep recoGsfTracks_electronGsfTracks_*_*'
    )
)

from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal

TICLrereco = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *Association*_generalTracks_*_*')
)

phase2_hgcal.toModify(TrackingToolsAOD, outputCommands = TrackingToolsAOD.outputCommands + TICLrereco.outputCommands)

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

#FEVT content
TrackingToolsFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_electronGsfTracks_*_*'
    )
)
TrackingToolsFEVT.outputCommands.extend(TrackingToolsRECO.outputCommands)
