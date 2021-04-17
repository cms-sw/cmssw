import FWCore.ParameterSet.Config as cms

from SimG4Core.Application.hectorParameter_cfi import *

baseHectorParameters = cms.PSet(
                TransportMethod = cms.string('Hector'),
                produceHitsRelativeToBeam = cms.bool(True),
                ApplyZShift = cms.bool(True)
)

Totem_PreTS2_2016 = cms.PSet(
        #TotemBeamLine = cms.bool(True),
        Beam1Filename = cms.string('SimTransport/PPSProtonTransport/data/LHCB1_Beta0.40_6.5TeV_CR191.541_PreTS2_TOTEM.tfs'),
        Beam2Filename = cms.string('SimTransport/PPSProtonTransport/data/LHCB2_Beta0.40_6.5TeV_CR179.394_PreTS2_TOTEM.tfs'),
        halfCrossingAngleXSector45  = cms.double(191.541), #in mrad / Beam 1
        halfCrossingAngleYSector45  = cms.double(0.), #in mrad / Beam 1
        halfCrossingAngleXSector56  = cms.double(179.394), #in mrad / Beam 2
        halfCrossingAngleYSector56  = cms.double(0.), #in mrad / Beam 2
        BeamEnergyDispersion = cms.double(1.11e-4),     ## beam energy dispersion (GeV); if =0.0 the default(=0.79) is used
        BeamDivergenceX = cms.double(135.071),     ## x angle dispersion at IP (urad); if =0.0 the default(=30.23) is used
        BeamDivergenceY = cms.double(135.071),      ## y angle dispersion at IP (urad); if =0.0 the default(=30.23) is used
        BeamSigmaX  = cms.double(54.03),
        BeamSigmaY  = cms.double(54.03),
        BeamEnergy = cms.double(6500.0),
        BeamXatIP      = cms.untracked.double(0.),
        BeamYatIP      = cms.untracked.double(0.),
)
Validated_PreTS2_2016 = cms.PSet(
        #TotemBeamLine = cms.bool(False),
        Beam1Filename = cms.string('SimTransport/PPSProtonTransport/data/LHCB1_Beta0.40_6.5TeV_CR191.541_PreTS2.tfs'),
        Beam2Filename = cms.string('SimTransport/PPSProtonTransport/data/LHCB2_Beta0.40_6.5TeV_CR179.394_PreTS2.tfs'),
        halfCrossingAngleXSector45  = cms.double(191.541), #in mrad / Beam 1
        halfCrossingAngleYSector45  = cms.double(0.), #in mrad / Beam 1
        halfCrossingAngleXSector56  = cms.double(179.394), #in mrad / Beam 2
        halfCrossingAngleYSector56  = cms.double(0.), #in mrad / Beam 2
        BeamEnergyDispersion = cms.double(1.11e-4),     ## beam energy dispersion (GeV); if =0.0 the default(=0.79) is used
        BeamDivergenceX = cms.double(135.071),     ## x angle dispersion at IP (urad); if =0.0 the default(=30.23) is used
        BeamDivergenceY = cms.double(135.071),      ## y angle dispersion at IP (urad); if =0.0 the default(=30.23) is used
        BeamSigmaX  = cms.double(54.03),
        BeamSigmaY  = cms.double(54.03),
        BeamEnergy = cms.double(6500.0),
        #BeamXatIP = cms.untracked.double(0.499), # if not given, will take the CMS average vertex position
        #BeamYatIP = cms.untracked.double(-0.190), # if not given, will take the CMS average vertex position
)
# Beam parametes for Nominal 2016
Nominal_2016 = cms.PSet(
        Beam1Filename = cms.string('SimTransport/PPSProtonTransport/data/LHCB1_Beta0.40_6.5TeV_CR185_Nominal_2016.tfs'),
        Beam2Filename = cms.string('SimTransport/PPSProtonTransport/data/LHCB2_Beta0.40_6.5TeV_CR185_Nominal_2016.tfs'),
        halfCrossingAngleXSector45  = cms.double(185.), #in mrad / Beam 1
        halfCrossingAngleYSector45  = cms.double(0.), #in mrad / Beam 1
        halfCrossingAngleXSector56  = cms.double(185.), #in mrad / Beam 2
        halfCrossingAngleYSector56  = cms.double(0.), #in mrad / Beam 2
        BeamEnergyDispersion = cms.double(1.11e-4),     ## beam energy dispersion (GeV); if =0.0 the default(=0.79) is used
        BeamDivergenceX = cms.double(35.54),     ## x angle dispersion at IP (urad); if =0.0 the default(=30.23) is used
        BeamDivergenceY = cms.double(35.54),      ## y angle dispersion at IP (urad); if =0.0 the default(=30.23) is used
        BeamSigmaX  = cms.double(14.22),
        BeamSigmaY  = cms.double(14.22),
        BeamEnergy = cms.double(6500.0),
        BeamXatIP      = cms.untracked.double(0.),
        BeamYatIP      = cms.untracked.double(0.),
)

# Beam parameter for Nominal 2017 optics
Nominal_2017_beta40cm = cms.PSet(
        #TotemBeamLine = cms.bool(False),
        Beam1Filename = cms.string('SimTransport/PPSProtonTransport/data/LHCB1_Beta0.40_6.5TeV_CR150_Nominal_2017.tfs'),
        Beam2Filename = cms.string('SimTransport/PPSProtonTransport/data/LHCB2_Beta0.40_6.5TeV_CR150_Nominal_2017.tfs'),
        halfCrossingAngleXSector45  = cms.double(150.), #in mrad / Beam 1
        halfCrossingAngleYSector45  = cms.double(0.), #in mrad / Beam 1
        halfCrossingAngleXSector56  = cms.double(150.), #in mrad / Beam 2
        halfCrossingAngleYSector56  = cms.double(0.), #in mrad / Beam 2
        BeamEnergyDispersion = cms.double(1.11e-4),     ## beam energy dispersion (GeV); if =0.0 the default(=0.79) is used
        BeamDivergenceX = cms.double(30.04),     ## x angle dispersion at IP (urad); if =0.0 the default(=30.23) is used
        BeamDivergenceY = cms.double(30.04),      ## y angle dispersion at IP (urad); if =0.0 the default(=30.23) is used
        BeamSigmaX  = cms.double(12.01),
        BeamSigmaY  = cms.double(12.01),
        BeamEnergy = cms.double(6500.0),
        BeamXatIP      = cms.untracked.double(0.),
        BeamYatIP      = cms.untracked.double(0.),
)
#
Nominal_2017_beta30cm = cms.PSet(
        #TotemBeamLine = cms.bool(False),
        Beam1Filename = cms.string('SimTransport/PPSProtonTransport/data/LHCB1_Beta0.30_6.5TeV_CR175_Nominal_2017.tfs'),
        Beam2Filename = cms.string('SimTransport/PPSProtonTransport/data/LHCB2_Beta0.30_6.5TeV_CR175_Nominal_2017.tfs'),
        halfCrossingAngleXSector45  = cms.double(175.), #in mrad / Beam 1
        halfCrossingAngleYSector45  = cms.double(0.), #in mrad / Beam 1
        halfCrossingAngleXSector56  = cms.double(175.), #in mrad / Beam 2
        halfCrossingAngleYSector56  = cms.double(0.), #in mrad / Beam 2
        BeamEnergyDispersion = cms.double(1.11e-4),     ## beam energy dispersion (GeV); if =0.0 the default(=0.79) is used
        BeamDivergenceX = cms.double(34.68),     ## x angle dispersion at IP (urad); if =0.0 the default(=30.23) is used
        BeamDivergenceY = cms.double(34.68),      ## y angle dispersion at IP (urad); if =0.0 the default(=30.23) is used
        BeamSigmaX  = cms.double(10.40),
        BeamSigmaY  = cms.double(10.40),
        BeamEnergy = cms.double(6500.0),
        BeamXatIP      = cms.untracked.double(0.),
        BeamYatIP      = cms.untracked.double(0.),
)

Nominal_2018_beta30cm = cms.PSet(
        Beam1Filename = cms.string('SimTransport/PPSProtonTransport/data/LHCB1_Beta0.30_6.5TeV_CR129.8_Nominal_2018.tfs'),
        Beam2Filename = cms.string('SimTransport/PPSProtonTransport/data/LHCB2_Beta0.30_6.5TeV_CR129.8_Nominal_2018.tfs'),
        halfCrossingAngleXSector45  = cms.double(129.8), #in mrad / Beam 1
        halfCrossingAngleYSector45  = cms.double(0.290), #in mrad / Beam 1
        halfCrossingAngleXSector56  = cms.double(129.8), #in mrad / Beam 2
        halfCrossingAngleYSector56  = cms.double(0.210), #in mrad / Beam 2
        BeamEnergyDispersion = cms.double(1.11e-4),     ## beam energy dispersion (GeV); if =0.0 the default(=0.79) is used
        BeamDivergenceX = cms.double(34.71),     ## x angle dispersion at IP (urad); if =0.0 the default(=30.23) is used
        BeamDivergenceY = cms.double(34.67),      ## y angle dispersion at IP (urad); if =0.0 the default(=30.23) is used
        BeamSigmaX  = cms.double(10.40),
        BeamSigmaY  = cms.double(10.41),
        BeamEnergy = cms.double(6500.0),
        BeamXatIP      = cms.untracked.double(0.),  # mm
        BeamYatIP      = cms.untracked.double(-1.8)
)

Nominal_2018_beta27cm = cms.PSet(
        Beam1Filename = cms.string('SimTransport/PPSProtonTransport/data/LHCB1_Beta0.27_6.5TeV_CR130_Nominal_2018.tfs'),
        Beam2Filename = cms.string('SimTransport/PPSProtonTransport/data/LHCB2_Beta0.27_6.5TeV_CR130_Nominal_2018.tfs'),
        halfCrossingAngleXSector45  = cms.double(130.0), #in mrad / Beam 1
        halfCrossingAngleYSector45  = cms.double(0.), #in mrad / Beam 1
        halfCrossingAngleXSector56  = cms.double(130.0), #in mrad / Beam 2
        halfCrossingAngleYSector56  = cms.double(0.), #in mrad / Beam 2
        BeamEnergyDispersion = cms.double(1.11e-4),     ## beam energy dispersion (GeV); if =0.0 the default(=0.79) is used
        BeamDivergenceX = cms.double(36.56),     ## x angle dispersion at IP (urad); if =0.0 the default(=30.23) is used
        BeamDivergenceY = cms.double(36.56),      ## y angle dispersion at IP (urad); if =0.0 the default(=30.23) is used
        BeamSigmaX  = cms.double(9.87),
        BeamSigmaY  = cms.double(9.87),
        BeamEnergy = cms.double(6500.0),
        BeamXatIP      = cms.untracked.double(0.),  # mm
        BeamYatIP      = cms.untracked.double(-1.8)
)

Nominal_2018_beta25cm = cms.PSet(
        Beam1Filename = cms.string('SimTransport/PPSProtonTransport/data/LHCB1_Beta0.25_6.5TeV_CR130_Nominal_2018.tfs'),
        Beam2Filename = cms.string('SimTransport/PPSProtonTransport/data/LHCB2_Beta0.25_6.5TeV_CR130_Nominal_2018.tfs'),
        halfCrossingAngleXSector45  = cms.double(130.0), #in mrad / Beam 1
        halfCrossingAngleYSector45  = cms.double(0.), #in mrad / Beam 1
        halfCrossingAngleXSector56  = cms.double(130.0), #in mrad / Beam 2
        halfCrossingAngleYSector56  = cms.double(0.), #in mrad / Beam 2
        BeamEnergyDispersion = cms.double(1.11e-4),     ## beam energy dispersion (GeV); if =0.0 the default(=0.79) is used
        BeamDivergenceX = cms.double(37.99),     ## x angle dispersion at IP (urad); if =0.0 the default(=30.23) is used
        BeamDivergenceY = cms.double(37.98),      ## y angle dispersion at IP (urad); if =0.0 the default(=30.23) is used
        BeamSigmaX  = cms.double(9.50),
        BeamSigmaY  = cms.double(9.50),
        BeamEnergy = cms.double(6500.0),
        #BeamXatIP      = cms.untracked.double(0.),  # mm
        #BeamYatIP      = cms.untracked.double(-1.8)
)

Nominal_RunIII =  cms.PSet(
        #TotemBeamLine = cms.bool(False),
        Beam1Filename = cms.string('SimTransport/PPSProtonTransport/data/LHCB1_Beta0.15_7TeV_CR250_HLLHCv1.4.tfs'),
        Beam2Filename = cms.string('SimTransport/PPSProtonTransport/data/LHCB2_Beta0.15_7TeV_CR250_HLLHCv1.4.tfs'),
        halfCrossingAngleXSector45  = cms.double(0.04), #in mrad / Beam 1
        halfCrossingAngleYSector45  = cms.double(250.1), #in mrad / Beam 1
        halfCrossingAngleXSector56  = cms.double(0.150), #in mrad / Beam 2
        halfCrossingAngleYSector56  = cms.double(-250.), #in mrad / Beam 2
        BeamEnergyDispersion = cms.double(1.e-3),     ## beam energy dispersion (GeV); if =0.0 the default(=0.79) is used
        BeamDivergenceX = cms.double(47.32),     ## x angle dispersion at IP (urad); if =0.0 the default(=30.23) is used
        BeamDivergenceY = cms.double(47.25),      ## y angle dispersion at IP (urad); if =0.0 the default(=30.23) is used
        BeamSigmaX  = cms.double(7.08),
        BeamSigmaY  = cms.double(7.09),
        BeamEnergy = cms.double(7000.0),
        BeamXatIP      = cms.untracked.double(-0.750),
        BeamYatIP      = cms.untracked.double(0.),
)

# choose default optics for each year

hector_2016 = cms.PSet(
              baseHectorParameters,
              Nominal_2016,
)

hector_2017 = cms.PSet(
              baseHectorParameters,
              Nominal_2017_beta40cm
)

hector_2018 = cms.PSet(
              baseHectorParameters,
              Nominal_2018_beta25cm  # CHANGE THIS WHEN THE PROPER ONE GET READY
)

hector_2021 = cms.PSet(
              baseHectorParameters,
              Nominal_RunIII
)
