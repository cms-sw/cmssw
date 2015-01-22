import FWCore.ParameterSet.Config as cms

from SimTracker.TrackAssociation.trackingParticleRecoTrackAsssociation_cfi import *
from SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi import *
from Validation.RecoVertex.v0validator_cfi import *
from Validation.RecoVertex.PrimaryVertexAnalyzer4PUSlimmed_cfi import *

vertexValidation = cms.Sequence(quickTrackAssociatorByHits
                                * trackingParticleRecoTrackAsssociation
                                * v0Validator
                                * vertexAnalysisSequence)
