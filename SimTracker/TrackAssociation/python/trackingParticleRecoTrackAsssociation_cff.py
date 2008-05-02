import FWCore.ParameterSet.Config as cms

from SimTracker.TrackAssociation.trackingParticleRecoTrackAsssociation_cfi import *
import copy
from SimTracker.TrackAssociation.trackingParticleRecoTrackAsssociation_cfi import *
assoc2secStepTk = copy.deepcopy(trackingParticleRecoTrackAsssociation)
import copy
from SimTracker.TrackAssociation.trackingParticleRecoTrackAsssociation_cfi import *
assoc2thStepTk = copy.deepcopy(trackingParticleRecoTrackAsssociation)
assoc2secStepTk.label_tr = 'secStep'
assoc2thStepTk.label_tr = 'thStep'

