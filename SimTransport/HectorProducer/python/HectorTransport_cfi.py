import FWCore.ParameterSet.Config as cms

from SimG4Core.Application.hectorParameter_cfi import *
LHCTransport = cms.EDProducer("HectorProducer",
    HepMCProductLabel = cms.string('generator'),  ## HepMC source to be processed
    FP420Transport = cms.bool(True),              ## main flag to set transport for FP420
    HPS240Transport = cms.bool(True),              ## main flag to set transport for HPS240
    ZDCTransport = cms.bool(True),                ## main flag to set transport for ZDC
    Verbosity = cms.bool(False),
    Hector = cms.PSet(
        HectorEtaCut,
        Beam1 = cms.string('SimTransport/HectorProducer/data/LHCB1IR5_5TeV.tfs'),
        Beam2 = cms.string('SimTransport/HectorProducer/data/LHCB2IR5_5TeV.tfs'),
        BeamLineLengthZDC = cms.double(139.0),    ## length of beam line for ZDC: important for aperture checks
        BeamLineLengthFP420 = cms.double(430.0),  ## length of beam line for FP420: important for aperture checks
        RP420f = cms.double(419.0),               ## distance of transport in clockwise dir. for FP420
        RP420b = cms.double(419.0),               ## distance of transport in anti-clockwise dir. for FP420
        BeamLineLengthHPS240 = cms.double(250.0),  ## length of beam line for HPS240: important for aperture checks
        RP240f = cms.double(239.0),               ## distance of transport in clockwise dir. for HPS240
        RP240b = cms.double(239.0),               ## distance of transport in anti-clockwise dir. for HPS240
        smearEnergy = cms.bool(True),       ## if False: no Energy smearing(i.e. sigmaEnergy =0.0)
        sigmaEnergy = cms.double(0.0),     ## beam energy dispersion (GeV); if =0.0 the default(=0.79) is used
        smearAng = cms.bool(True),       ## if False: no Angle smearing(i.e. sigmaSTX(Y) =0.0)
        sigmaSTX = cms.double(0.0),     ## x angle dispersion at IP (urad); if =0.0 the default(=30.23) is used
        sigmaSTY = cms.double(0.0)      ## y angle dispersion at IP (urad); if =0.0 the default(=30.23) is used
    )
)


