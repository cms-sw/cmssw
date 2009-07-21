import FWCore.ParameterSet.Config as cms

import SimTracker.TrackAssociation.TrackAssociatorByHits_cfi
#from Validation.RecoVertex.v0validator_cff import *
from SimTracker.TrackAssociation.trackingParticleRecoTrackAsssociation_cff import *

from Configuration.StandardSequences.MagneticField_cff import *
from DQMServices.Core.DQM_cfg import *
DQMStore.collateHistograms = cms.untracked.bool(True)
from Validation.RecoVertex.v0validator_cfi import *

#v0Validator_ = Validation.RecoVertex.v0validator_cfi.v0Validator.clone()

vertexValidation = cms.Sequence(trackingParticleRecoTrackAsssociation*v0Validator)
