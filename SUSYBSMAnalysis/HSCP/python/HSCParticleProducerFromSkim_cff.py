import FWCore.ParameterSet.Config as cms

from SUSYBSMAnalysis.HSCP.HSCParticleProducer_cff import *

TrackRefitter.src         = "generalTracksSkim"
muontiming.MuonCollection = cms.InputTag("muonsSkim")
HSCParticleProducer.muons = cms.InputTag("muonsSkim")


