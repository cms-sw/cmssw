# The following comments couldn't be translated into the new config version:

#save digis

#save match with GenParticles 

#save match with TrackingParticles 

#save match with GenParticles 

#save match with GenParticles 

#save match with GenParticles 

import FWCore.ParameterSet.Config as cms

#Full Event content with DIGI
SimTrackerFEVTDEBUG = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_simSiPixelDigis_*_*', 
        'keep *_simSiStripDigis_*_*', 
        'keep *_allTrackMCMatch_*_*', 
        'keep *_trackingParticleRecoTrackAsssociation_*_*', 
        'keep *_assoc2secStepTk_*_*', 
        'keep *_assoc2thStepTk_*_*', 
        'keep *_assoc2GsfTracks_*_*', 
        'keep *_assocOutInConversionTracks_*_*', 
        'keep *_assocInOutConversionTracks_*_*')
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


