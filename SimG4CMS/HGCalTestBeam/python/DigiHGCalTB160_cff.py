import FWCore.ParameterSet.Config as cms
from SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi import *

mixSimHits = cms.PSet(
    input = cms.VInputTag(),
    type = cms.string('PSimHit'),
    crossingFrames = cms.untracked.vstring(),
    subdets = cms.vstring()
)
mixCaloHits = cms.PSet(
    input = cms.VInputTag(
        cms.InputTag("g4SimHits","HGCHitsEE"),
        cms.InputTag("g4SimHits","HGCHitsHEfront")
    ),
    type = cms.string('PCaloHit'),
    crossingFrames = cms.untracked.vstring(),
    subdets = cms.vstring(
        'HGCHitsEE',
        'HGCHitsHEfront'
    )
)

mixSimTracks = cms.PSet(
    makeCrossingFrame = cms.untracked.bool(False),
    input = cms.VInputTag(cms.InputTag("g4SimHits")),
    type = cms.string('SimTrack')
)
mixSimVertices = cms.PSet(
    makeCrossingFrame = cms.untracked.bool(False),
    input = cms.VInputTag(cms.InputTag("g4SimHits")),
    type = cms.string('SimVertex')
)
mixHepMCProducts = cms.PSet(
    makeCrossingFrame = cms.untracked.bool(False),
    input = cms.VInputTag(cms.InputTag("generatorSmeared"),cms.InputTag("generator")),
    type = cms.string('HepMCProduct')
)

theMixObjects = cms.PSet(
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

theDigitizers = cms.PSet(
    hgcalEE = cms.PSet(
        hgceeDigitizer     
    )
)

mix = cms.EDProducer("MixingModule",
    digitizers = cms.PSet(theDigitizers),
    LabelPlayback = cms.string(''),
    maxBunch = cms.int32(3),
    minBunch = cms.int32(-5), ## in terms of 25 ns

    bunchspace = cms.int32(450),
    mixProdStep1 = cms.bool(False),
    mixProdStep2 = cms.bool(False),

    playback = cms.untracked.bool(False),
    useCurrentProcessOnly = cms.bool(False),
    mixObjects = cms.PSet(theMixObjects)
)
