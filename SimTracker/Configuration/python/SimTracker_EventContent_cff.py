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
        'keep *_assocInOutConversionTracks_*_*',
        'keep *_TTClusterAssociatorFromPixelDigis_*_*',
        'keep *_TTStubAssociatorFromPixelDigis_*_*')

)
# For phase2 premixing switch the sim digi collections to the ones including pileup
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
(premix_stage2 & phase2_tracker).toModify(SimTrackerFEVTDEBUG, outputCommands = SimTrackerFEVTDEBUG.outputCommands + [
    'drop *_simSiPixelDigis_*_*',
    'keep *_mixData_Pixel_*',
    'keep *_mixData_Tracker_*',
])

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

# Event content for premixing library
SimTrackerPREMIX = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_simSiPixelDigis_*_*', # covers digis and digiSimLinks
        'keep *_simSiStripDigis_ZeroSuppressed_*',
        'keep StripDigiSimLinkedmDetSetVector_simSiStripDigis_*_*',
        'keep *_mix_AffectedAPVList_*',
    )
)
phase2_tracker.toModify(SimTrackerPREMIX, outputCommands = [
        'keep Phase2TrackerDigiedmDetSetVector_mix_*_*',
        'keep *_*_Phase2OTDigiSimLink_*',
        'keep *_simSiPixelDigis_*_*', # covers digis and digiSimLinks
])
