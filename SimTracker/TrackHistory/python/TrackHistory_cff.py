import FWCore.ParameterSet.Config as cms

recoTrackModule = cms.string('ctfWithMaterialTracks')
trackingParticleModule = cms.string('trackingtruthprod')
trackingParticleProduct = cms.string('')
associationModule = cms.string('TrackAssociatorByHits')
bestMatchByMaxValue = cms.bool(True)

