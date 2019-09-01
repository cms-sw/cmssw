import FWCore.ParameterSet.Config as cms

from SimTransport.PPSProtonTransport.TotemBeamConditions_cff import BeamConditionsGlobal
LHCTransport = cms.EDProducer('PPSSimTrackProducer',
    TransportMethod = cms.string('Totem'),
    HepMCProductLabel = cms.InputTag('generatorSmeared'),
    Verbosity = cms.bool(False),
    sqrtS = cms.double(13.0e3),

    # crossing angle
    checkApertures = cms.bool(True),

    BeamProtTransportSetup=BeamConditionsGlobal
)
