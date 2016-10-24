import FWCore.ParameterSet.Config as cms

from SimG4Core.Application.hectorParameter_cfi import *
LHCTransport = cms.EDProducer("HectorProducerForCTPPS",
    HepMCProductLabel = cms.string('generatorSmeared'),  ## HepMC source to be processed
    CTPPSTransport = cms.bool(True), 
    Verbosity = cms.bool(False),
    HectorForCTPPS = cms.PSet(
        HectorEtaCut,
        Beam1 = cms.string('SimTransport/HectorProducerForCTPPS/data/LHCB1_Beta0.40_6.5TeV_CR205_v6.503.tfs'),
        Beam2 = cms.string('SimTransport/HectorProducerForCTPPS/data/LHCB2_Beta0.40_6.5TeV_CR205_v6.503.tfs'),
        CrossingAngle  = cms.double(205.), #in mrad
        BeamLineLengthCTPPS = cms.double(250.0),
	    CTPPSf = cms.double(203.827),
        CTPPSb = cms.double(203.827),
        smearEnergy = cms.bool(True),       ## if False: no Energy smearing(i.e. sigmaEnergy =0.0)
        sigmaEnergy = cms.double(1.11e-4),     ## beam energy dispersion (GeV); if =0.0 the default(=0.79) is used
        smearAng = cms.bool(True),       ## if False: no Angle smearing(i.e. sigmaSTX(Y) =0.0)
        sigmaSTX = cms.double(30.03),     ## x angle dispersion at IP (urad); if =0.0 the default(=30.23) is used
        sigmaSTY = cms.double(30.03),      ## y angle dispersion at IP (urad); if =0.0 the default(=30.23) is used
        CrossAngleCorr = cms.bool(True),
        BeamEnergy = cms.double(6500.0),
        VtxMeanX       = cms.double(0.10482),
        VtxMeanY       = cms.double(0.16867),
        VtxMeanZ       = cms.double(-1.0985),
        MomentumMin = cms.double(3.000)		
    )
)



