import FWCore.ParameterSet.Config as cms

# configuration to model pileup for initial physics phase
from SimGeneral.MixingModule.aliases_cfi import *
from SimGeneral.MixingModule.mixObjects_cfi import *
from SimGeneral.MixingModule.mixPoolSource_cfi import *
from SimGeneral.MixingModule.pixelDigitizer_cfi import *
from SimGeneral.MixingModule.stripDigitizer_cfi import *
from SimGeneral.MixingModule.ecalDigitizer_cfi import *
from SimGeneral.MixingModule.hcalDigitizer_cfi import *
from SimGeneral.MixingModule.castorDigitizer_cfi import *
from SimGeneral.MixingModule.trackingTruthProducer_cfi import *

mix = cms.EDProducer("MixingModule",
    digitizers = cms.PSet(
      pixel = cms.PSet(
        pixelDigitizer
      ),
      strip = cms.PSet(
        stripDigitizer
      ),
      ecal = cms.PSet(
        ecalDigitizer
      ),
      hcal = cms.PSet(
        hcalDigitizer
      ),
      castor  = cms.PSet(
        castorDigitizer
      ),
      mergedtruth = cms.PSet(
        trackingParticles
      )
    ),
    LabelPlayback = cms.string(''),
    maxBunch = cms.int32(3),
    minBunch = cms.int32(-5), ## in terms of 25 nsec

    bunchspace = cms.int32(450), ##ns
    mixProdStep1 = cms.bool(False),
    mixProdStep2 = cms.bool(False),

    playback = cms.untracked.bool(False),
    useCurrentProcessOnly = cms.bool(False),

    input = cms.SecSource("PoolSource",
        nbPileupEvents = cms.PSet(
            averageNumber = cms.double(3.0)
        ),
        type = cms.string('fixed'),
	sequential = cms.untracked.bool(False),
        fileNames = FileNames
    ),
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


