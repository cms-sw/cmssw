# The following comments couldn't be translated into the new config version:

#save DigiSimLink

#save match with GenParticles 

#save match with TrackingParticles

#save digis

#save match with GenParticles 

#save match with TrackingParticles 

#save match with GenParticles 

#save match with TrackingParticles 

#save match with GenParticles 

import FWCore.ParameterSet.Config as cms

#Full Event content 
SimTrackerFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep PixelDigiSimLinkedmDetSetVector_siPixelDigis_*_*', 
        'keep StripDigiSimLinkedmDetSetVector_siStripDigis_*_*', 
        'keep *_trackMCMatch_*_*', 
        'keep *_trackingParticleRecoTrackAsssociation_*_*', 
        'keep *_assoc2secStepTk_*_*', 
        'keep *_assoc2thStepTk_*_*')
)
#Full Event content with DIGI
SimTrackerFEVTDIGI = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_siPixelDigis_*_*', 
        'keep *_siStripDigis_*_*', 
        'keep *_trackMCMatch_*_*', 
        'keep *_trackingParticleRecoTrackAsssociation_*_*', 
        'keep *_assoc2secStepTk_*_*', 
        'keep *_assoc2thStepTk_*_*')
)
#RECO content
SimTrackerRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_trackMCMatch_*_*', 
        'keep *_trackingParticleRecoTrackAsssociation_*_*', 
        'keep *_assoc2secStepTk_*_*', 
        'keep *_assoc2thStepTk_*_*')
)
#AOD content
SimTrackerAOD = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_trackMCMatch_*_*')
)

