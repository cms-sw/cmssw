import FWCore.ParameterSet.Config as cms

# this is a minimum configuration of the Mixing module,
# to run it in the zero-pileup mode
#

from SimGeneral.MixingModule.mixObjects_cfi import * 
from SimGeneral.MixingModule.pixelDigitizer_cfi import *
from SimGeneral.MixingModule.stripDigitizer_cfi import *
from SimGeneral.MixingModule.ecalDigitizer_cfi import *
from SimGeneral.MixingModule.hcalDigitizer_cfi import *
from SimGeneral.MixingModule.castorDigitizer_cfi import *
from SimGeneral.MixingModule.trackingTruthProducer_cfi import *
from SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi import hgceeDigitizer, hgchebackDigitizer, hgchefrontDigitizer, HGCAL_noise_fC, HGCAL_noise_heback, HGCAL_chargeCollectionEfficiencies, HGCAL_noises

mix = cms.EDProducer("MixingModule",
    digitizers = cms.PSet(),
    LabelPlayback = cms.string(''),
    maxBunch = cms.int32(3),
    minBunch = cms.int32(-5), ## in terms of 25 ns

    bunchspace = cms.int32(25),
    mixProdStep1 = cms.bool(False),
    mixProdStep2 = cms.bool(False),

    playback = cms.untracked.bool(False),
    useCurrentProcessOnly = cms.bool(False),
    mixObjects = cms.PSet(
        mixCH = cms.PSet(
            mixCaloHits
        ),
        mixTracks = cms.PSet(
            mixSimTracks
        )
                          ,
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
mix.mixObjects.mixSH.crossingFrames = cms.untracked.vstring('MuonCSCHits',
'MuonDTHits',
'MuonRPCHits',
'TotemHitsRP',
'CTPPSPixelHits')

mix.mixObjects.mixSH.input = mix.mixObjects.mixSH.input + [ cms.InputTag("g4SimHits","TotemHitsRP"),cms.InputTag("g4SimHits","CTPPSPixelHits") ]
mix.mixObjects.mixSH.subdets= mix.mixObjects.mixSH.subdets + [ 'TotemHitsRP' , 'CTPPSPixelHits']


