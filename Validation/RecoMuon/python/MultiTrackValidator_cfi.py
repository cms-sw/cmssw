import FWCore.ParameterSet.Config as cms

from Validation.RecoMuon.selectors_cff import *
from Validation.RecoMuon.associators_cff import *

import Validation.RecoTrack.MultiTrackValidator_cfi

RMmultiTrackValidator = Validation.RecoTrack.MultiTrackValidator_cfi.multiTrackValidator.clone()
RMmultiTrackValidator.dirName = 'Muons/RecoMuonV/MultiTrack/'
RMmultiTrackValidator.pdgIdTP = (13,-13)

RMmultiTrackValidator.useFabsEta = False
RMmultiTrackValidator.nint = cms.int32(50)
RMmultiTrackValidator.min = cms.double(-2.5)
RMmultiTrackValidator.max = cms.double(2.5)

RMmultiTrackValidator.nintPhi = cms.int32(36)
RMmultiTrackValidator.minPhi = cms.double(-3.1416)
RMmultiTrackValidator.maxPhi = cms.double(3.1416)

RMmultiTrackValidator.nintpT = cms.int32(40)
RMmultiTrackValidator.minpT = cms.double(0.0)
RMmultiTrackValidator.maxpT = cms.double(1500.0)

RMmultiTrackValidator.nintHit = cms.int32(75)
RMmultiTrackValidator.minHit = cms.double(-0.5)
RMmultiTrackValidator.maxHit = cms.double(74.5)

RMmultiTrackValidator.phiRes_rangeMin = cms.double(-0.05)
RMmultiTrackValidator.phiRes_rangeMax = cms.double(0.05)
#RMmultiTrackValidator.etaRes_rangeMin = cms.double(-0.05)
#RMmultiTrackValidator.etaRes_rangeMax = cms.double(0.05)
RMmultiTrackValidator.ptRes_rangeMin = cms.double(-0.3)
RMmultiTrackValidator.ptRes_rangeMax = cms.double(0.3)
RMmultiTrackValidator.dxyRes_rangeMin = cms.double(-0.02)
RMmultiTrackValidator.dxyRes_rangeMax = cms.double(0.02)
RMmultiTrackValidator.dzRes_rangeMin = cms.double(-0.05)
RMmultiTrackValidator.dzRes_rangeMax = cms.double(0.05)

RMmultiTrackValidator.UseAssociators = False





