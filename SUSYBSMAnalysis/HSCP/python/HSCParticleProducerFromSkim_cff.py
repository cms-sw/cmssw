import FWCore.ParameterSet.Config as cms

from SUSYBSMAnalysis.HSCP.HSCParticleProducer_cff import *

TrackRefitter.src                      = "generalTracksSkim"
muontiming.MuonCollection              = cms.InputTag("muonsSkim")
HSCParticleProducer.tracksIsolation    = cms.InputTag("generalTracksSkim")
HSCParticleProducer.muons              = cms.InputTag("muonsSkim")
HSCParticleProducer.MTmuons            = cms.InputTag("muonsSkim")
HSCParticleProducer.EBRecHitCollection = cms.InputTag("reducedHSCPEcalRecHitsEB")
HSCParticleProducer.EERecHitCollection = cms.InputTag("reducedHSCPEcalRecHitsEE")

HSCParticleProducer.TrackAssociatorParameters.EBRecHitCollectionLabel   = 'reducedHSCPEcalRecHitsEB'
HSCParticleProducer.TrackAssociatorParameters.EERecHitCollectionLabel   = 'reducedHSCPEcalRecHitsEE'
HSCParticleProducer.TrackAssociatorParameters.HBHERecHitCollectionLabel = 'reducedHSCPhbhereco'
HSCParticleProducer.TrackAssociatorParameters.HBHERecHitCollectionLabel = 'reducedHSCPhbhereco'
HSCParticleProducer.TrackAssociatorParameters.useHO                     = False
