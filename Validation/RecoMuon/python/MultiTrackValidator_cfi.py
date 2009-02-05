import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.selectors_cff import *
from Validation.RecoMuon.associators_cff import *

import Validation.RecoTrack.MultiTrackValidator_cfi

multiTrackValidator = Validation.RecoTrack.MultiTrackValidator_cfi.multiTrackValidator.clone()
multiTrackValidator.dirName = 'RecoMuonV/MultiTrack/'
multiTrackValidator.pdgIdTP = (13,-13)

multiTrackValidator.useFabsEta = True
multiTrackValidator.nint = cms.int32(25)
multiTrackValidator.min = cms.double(0)
multiTrackValidator.max = cms.double(2.5)

multiTrackValidator.nintPhi = cms.int32(36)
multiTrackValidator.minPhi = cms.double(-3.15)
multiTrackValidator.maxPhi = cms.double(3.15)

multiTrackValidator.nintpT = cms.int32(25)
multiTrackValidator.minpT = cms.double(0.0)
multiTrackValidator.maxpT = cms.double(3100.0)

multiTrackValidator.nintHit = cms.int32(75)
multiTrackValidator.minHit = cms.double(-0.5)
multiTrackValidator.maxHit = cms.double(74.5)

multiTrackValidator.UseAssociators = False





