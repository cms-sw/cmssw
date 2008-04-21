import FWCore.ParameterSet.Config as cms

from SimTracker.TrackAssociation.TrackAssociatorByChi2_cfi import *
from SimTracker.TrackAssociation.TrackAssociatorByHits_cfi import *
from Validation.RecoTrack.cuts_cff import *
from Validation.RecoTrack.MultiTrackValidator_cfi import *
tracksValidation = cms.Sequence(cutsTPEffic*cutsTPFake*multiTrackValidator)

