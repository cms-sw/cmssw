import FWCore.ParameterSet.Config as cms

#Full Event content with DIGI
SimTrackerFEVTDEBUG = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_simSiPixelDigis_*_*', 
        'keep *_simSiStripDigis_*_*', 
        'drop *_mix_simSiPixelDigis*_*', 
        'drop *_mix_simSiStripDigis*_*', 
        'keep *_allTrackMCMatch_*_*', 
        'keep *_trackingParticleRecoTrackAsssociation_*_*', 
        'keep *_assoc2secStepTk_*_*', 
        'keep *_assoc2thStepTk_*_*', 
        'keep *_assoc2GsfTracks_*_*', 
        'keep *_assocOutInConversionTracks_*_*', 
        'keep *_assocInOutConversionTracks_*_*')
)

SimTrackerDEBUG = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep PixelDigiSimLinkedmDetSetVector_simSiPixelDigis_*_*', 
        'keep StripDigiSimLinkedmDetSetVector_simSiStripDigis_*_*', 
        'drop *_mix_simSiPixelDigis*_*', 
        'drop *_mix_simSiStripDigis*_*', 
        'keep *_allTrackMCMatch_*_*')
)
#RAW content 
SimTrackerRAW = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_allTrackMCMatch_*_*')
)
#RECO content
SimTrackerRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_allTrackMCMatch_*_*')
)
#AOD content
SimTrackerAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_allTrackMCMatch_*_*')
)


