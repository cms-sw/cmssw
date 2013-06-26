import FWCore.ParameterSet.Config as cms

from SimGeneral.MixingModule.mixObjects_cfi import *
cfWriter = cms.EDProducer("CFWriter",
    maxBunch = cms.int32(3),
    minBunch = cms.int32(-5),
    
    mixObjects = cms.PSet(
    mixCH = cms.PSet(
      mixCaloHits
    ),
    mixTracks = cms.PSet(
      mixSimTracks
    ),
    mixVertices = cms.PSet(
      mixSimVertices
    ),
    mixSH = cms.PSet(
      mixSimHits
    ),
    mixHepMC = cms.PSet(
      mixHepMCProducts
    )
    )
)

