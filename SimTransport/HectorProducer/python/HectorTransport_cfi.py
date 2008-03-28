import FWCore.ParameterSet.Config as cms

from SimG4Core.Application.hectorParameter_cfi import *
LHCTransport = cms.EDProducer("HectorProducer",
    ZDCTransport = cms.bool(True), ## main flag to set transport for ZDC

    Hector = cms.PSet(
        HectorEtaCut,
        BeamLineLengthZDC = cms.double(140.0), ## length of beam line for ZDC: important for aperture checks

        sigmaEnergy = cms.double(0.0), ## beam energy dispersion (GeV)

        smearEnergy = cms.bool(True),
        sigmaSTX = cms.double(0.0), ## x angle dispersion at IP (m)

        sigmaSTY = cms.double(0.0), ## y angle dispersion at IP (m)

        BeamLineLengthFP420 = cms.double(430.0), ## length of beam line for FP420: important for aperture checks

        Beam2 = cms.string('SimTransport/HectorProducer/data/LHCB2IR5_v6.500.tfs'),
        BeamLineLengthD1 = cms.double(105.0), ## distance of transport for ZDC case, length of beam line 

        Beam1 = cms.string('SimTransport/HectorProducer/data/LHCB1IR5_v6.500.tfs'),
        smearAng = cms.bool(True),
        RP420f = cms.double(419.0), ## distance of transport in clockwise dir. for FP420

        RP420b = cms.double(419.0) ## distance of transport in anti-clockwise dir. for FP420

    ),
    Verbosity = cms.bool(False),
    HepMCProductLabel = cms.string('source'),
    FP420Transport = cms.bool(True) ## main flag to set transport for FP420

)


