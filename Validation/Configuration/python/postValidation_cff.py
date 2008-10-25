import FWCore.ParameterSet.Config as cms

from Validation.RecoTrack.TrackValidation_cff import *
multiTrackValidator.UseAssociators = True

postValidation = cms.Sequence(tracksValidation)


