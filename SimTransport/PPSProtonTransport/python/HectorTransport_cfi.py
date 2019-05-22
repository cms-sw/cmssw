import FWCore.ParameterSet.Config as cms

from SimG4Core.Application.hectorParameter_cfi import *

from SimTransport.PPSProtonTransport.HectorOpticsParameters_cfi import *

LHCTransport = cms.EDProducer("PPSSimTrackProducer",
    HepMCProductLabel = cms.InputTag('generatorSmeared'),  ## HepMC source to be processed
    Verbosity = cms.bool(False),
    TransportMethod = cms.string('Hector'),
    #VtxMeanX        = cms.double(0.),
    #VtxMeanY        = cms.double(0.),
    #VtxMeanZ        = cms.double(0.),
    PPSHector = cms.PSet(
        HectorEtaCut,
        Validated_PreTS2_2016,
        BeamLineLengthPPS = cms.double(250.0),
        PPSf = cms.double(212.55),    ##in meters
        PPSb = cms.double(212.55),    ##in meters
        ApplyZShift = cms.bool(True),
        MomentumMin = cms.double(3.000)
    )
)
