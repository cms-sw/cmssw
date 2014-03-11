import FWCore.ParameterSet.Config as cms

from SimTracker.TrackAssociation.trackingParticleRecoTrackAsssociation_cff import *

from Validation.RecoVertex.v0validator_cfi import *

vertexValidation = cms.Sequence(trackingParticleRecoTrackAsssociation*v0Validator)
