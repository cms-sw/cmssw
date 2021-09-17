import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedParameters_cfi import Realistic25ns13TeV2016CollisionVtxSmearingParameters

baseTotemParameters = cms.PSet(
                TransportMethod = cms.string('Totem'),
                checkApertures = cms.bool(True),
                ApplyZShift = cms.bool(True)
)

BeamConditions2016 = cms.PSet(
    Beam1Filename = cms.string('SimTransport/TotemRPProtonTransportParametrization/data/parametrization_6500GeV_0p4_185_reco_beam1.root'),
    Beam2Filename = cms.string('SimTransport/TotemRPProtonTransportParametrization/data/parametrization_6500GeV_0p4_185_reco_beam2.root'),
    Model_IP_150_R_Name = cms.string('ip5_to_beg_150_station_lhcb1'),
    Model_IP_150_L_Name = cms.string('ip5_to_beg_150_station_lhcb2'),
    BeamDivergenceX = cms.double(20.), # in urad
    BeamDivergenceY = cms.double(20.), # in urad
    BeamEnergyDispersion = cms.double(1.11e-4),
    halfCrossingAngleSector45 = cms.double(179.394), # in urad
    halfCrossingAngleSector56 = cms.double(191.541), # in urad
    BeamEnergy = cms.double(6500.), # in GeV
    #BeamXatIP = cms.untracked.double(0.499), # if not given, will take the CMS average vertex position
    #BeamYatIP = cms.untracked.double(-0.190), # if not given, will take the CMS average vertex position
    # in m, should be consistent with geometry xml definitions
    BeampipeApertureRadius = cms.double(0.04), # in meter
    BeamSigmaX = cms.double(20.),
    BeamSigmaY = cms.double(20.)
)

totemTransportSetup_2016 = cms.PSet(
             baseTotemParameters,
             BeamConditions2016
)
