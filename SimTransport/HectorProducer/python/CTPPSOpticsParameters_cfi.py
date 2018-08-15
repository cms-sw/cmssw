import FWCore.ParameterSet.Config as cms

Totem_PreTS2_2016 = cms.PSet(
        #TotemBeamLine = cms.bool(True),
        Beam1 = cms.string('SimTransport/HectorProducer/data/LHCB1_Beta0.40_6.5TeV_CR191.541_PreTS2_TOTEM.tfs'),
        Beam2 = cms.string('SimTransport/HectorProducer/data/LHCB2_Beta0.40_6.5TeV_CR179.394_PreTS2_TOTEM.tfs'),
        CrossingAngleBeam1  = cms.double(191.541), #in mrad
        CrossingAngleBeam2  = cms.double(179.394), #in mrad
        VtxMeanX       = cms.double( 0.7500),
        VtxMeanY       = cms.double(-0.1866),
        VtxMeanZ       = cms.double(0.),
        sigmaEnergy = cms.double(1.11e-4),     ## beam energy dispersion (GeV); if =0.0 the default(=0.79) is used
        sigmaSTX = cms.double(135.071),     ## x angle dispersion at IP (urad); if =0.0 the default(=30.23) is used
        sigmaSTY = cms.double(135.071),      ## y angle dispersion at IP (urad); if =0.0 the default(=30.23) is used
        sigmaSX  = cms.double(54.03),
        sigmaSY  = cms.double(54.03),
        BeamEnergy = cms.double(6500.0),
        BeamXatIP      = cms.double(0.),
        BeamYatIP      = cms.double(0.)
)
Validated_PreTS2_2016 = cms.PSet(
        #TotemBeamLine = cms.bool(False),
        Beam1 = cms.string('SimTransport/HectorProducer/data/LHCB1_Beta0.40_6.5TeV_CR191.541_PreTS2.tfs'),
        Beam2 = cms.string('SimTransport/HectorProducer/data/LHCB2_Beta0.40_6.5TeV_CR179.394_PreTS2.tfs'),
        CrossingAngleBeam1  = cms.double(191.541), #in mrad
        CrossingAngleBeam2  = cms.double(179.394), #in mrad
        #VtxMeanX       = cms.double(0.),
        #VtxMeanY       = cms.double(0.),
        #VtxMeanZ       = cms.double(0.),
        sigmaEnergy = cms.double(1.11e-4),     ## beam energy dispersion (GeV); if =0.0 the default(=0.79) is used
        sigmaSTX = cms.double(135.071),     ## x angle dispersion at IP (urad); if =0.0 the default(=30.23) is used
        sigmaSTY = cms.double(135.071),      ## y angle dispersion at IP (urad); if =0.0 the default(=30.23) is used
        sigmaSX  = cms.double(54.03),
        sigmaSY  = cms.double(54.03),
        BeamEnergy = cms.double(6500.0),
        BeamXatIP      = cms.double(0.),
        BeamYatIP      = cms.double(0.)
)

# Beam parameter for Nominal 2017 optics
Nominal_2017_beta40cm = cms.PSet(
        #TotemBeamLine = cms.bool(False),
        Beam1 = cms.string('SimTransport/HectorProducer/data/LHCB1_Beta0.40_6.5TeV_CR150_Nominal_2017.tfs'),
        Beam2 = cms.string('SimTransport/HectorProducer/data/LHCB2_Beta0.40_6.5TeV_CR150_Nominal_2017.tfs'),
        CrossingAngleBeam1  = cms.double(150.), #in mrad
        CrossingAngleBeam2  = cms.double(150.), #in mrad
        VtxMeanX       = cms.double(0.),
        VtxMeanY       = cms.double(-0.15),
        VtxMeanZ       = cms.double(0.),
        sigmaEnergy = cms.double(1.11e-4),     ## beam energy dispersion (GeV); if =0.0 the default(=0.79) is used
        sigmaSTX = cms.double(30.04),     ## x angle dispersion at IP (urad); if =0.0 the default(=30.23) is used
        sigmaSTY = cms.double(30.04),      ## y angle dispersion at IP (urad); if =0.0 the default(=30.23) is used
        sigmaSX  = cms.double(12.01),
        sigmaSY  = cms.double(12.01),
        BeamEnergy = cms.double(6500.0),
        BeamXatIP      = cms.double(0.),
        BeamYatIP      = cms.double(0.)
)
#
Nominal_2017_beta30cm = cms.PSet(
        #TotemBeamLine = cms.bool(False),
        Beam1 = cms.string('SimTransport/HectorProducer/data/LHCB1_Beta0.30_6.5TeV_CR175_Nominal_2017.tfs'),
        Beam2 = cms.string('SimTransport/HectorProducer/data/LHCB2_Beta0.30_6.5TeV_CR175_Nominal_2017.tfs'),
        CrossingAngleBeam1  = cms.double(175.), #in mrad
        CrossingAngleBeam2  = cms.double(175.), #in mrad
        VtxMeanX       = cms.double(0.),
        VtxMeanY       = cms.double(-0.95),
        VtxMeanZ       = cms.double(0.),
        sigmaEnergy = cms.double(1.11e-4),     ## beam energy dispersion (GeV); if =0.0 the default(=0.79) is used
        sigmaSTX = cms.double(34.68),     ## x angle dispersion at IP (urad); if =0.0 the default(=30.23) is used
        sigmaSTY = cms.double(34.68),      ## y angle dispersion at IP (urad); if =0.0 the default(=30.23) is used
        sigmaSX  = cms.double(10.40),
        sigmaSY  = cms.double(10.40),
        BeamEnergy = cms.double(6500.0),
        BeamXatIP      = cms.double(0.),
        BeamYatIP      = cms.double(0.)
)
# Beam parametes for Nominal 2017
Nominal_2016 = cms.PSet(
        #TotemBeamLine = cms.bool(False),
        Beam1 = cms.string('SimTransport/HectorProducer/data/LHCB1_Beta0.40_6.5TeV_CR185_Nominal_2016.tfs'),
        Beam2 = cms.string('SimTransport/HectorProducer/data/LHCB2_Beta0.40_6.5TeV_CR185_Nominal_2016.tfs'),
        CrossingAngleBeam1  = cms.double(185.), #in mrad
        CrossingAngleBeam2  = cms.double(185.), #in mrad
        VtxMeanX       = cms.double(0.),
        VtxMeanY       = cms.double(0.),
        VtxMeanZ       = cms.double(0.),
        sigmaEnergy = cms.double(1.11e-4),     ## beam energy dispersion (GeV); if =0.0 the default(=0.79) is used
        sigmaSTX = cms.double(35.54),     ## x angle dispersion at IP (urad); if =0.0 the default(=30.23) is used
        sigmaSTY = cms.double(35.54),      ## y angle dispersion at IP (urad); if =0.0 the default(=30.23) is used
        sigmaSX  = cms.double(14.22),
        sigmaSY  = cms.double(14.22),
        BeamEnergy = cms.double(6500.0),
        BeamXatIP      = cms.double(0.),
        BeamYatIP      = cms.double(0.)
)
