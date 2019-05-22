import FWCore.ParameterSet.Config as cms

lhcBeamConditions_2016PreTS2 = cms.PSet(
    sqrtS = cms.double(13.e3), # in GeV
    vertexSize = cms.double(10.e-6), # in m
    beamDivergence = cms.double(20.e-6), # in rad

    # vertex vertical offset in both sectors
    yOffsetSector45 = cms.double(300.e-6), # in m
    yOffsetSector56 = cms.double(200.e-6), # in m

    # crossing angle
    halfCrossingAngleSector45 = cms.double(179.394e-6), # in rad
    halfCrossingAngleSector56 = cms.double(191.541e-6), # in rad
)

BeamConditionsGlobal = cms.PSet(
    ModelRootFile_R = cms.string('SimTransport/TotemRPProtonTransportParametrization/data/parametrization_6500GeV_0p4_185_reco_beam1.root'),
    ModelRootFile_L = cms.string('SimTransport/TotemRPProtonTransportParametrization/data/parametrization_6500GeV_0p4_185_reco_beam2.root'),
    # ModelRootFile = cms.string('Geometry/VeryForwardProtonTransport/data/parametrization_6500GeV_90_transp_75.root'),
    Model_IP_150_R_Name = cms.string('ip5_to_beg_150_station_lhcb1'),
    Model_IP_150_L_Name = cms.string('ip5_to_beg_150_station_lhcb2'),
    #beamDivergenceX = cms.double(135.071), # in urad
    #beamDivergenceY = cms.double(135.071), # in urad
    beamDivergenceX = cms.double(20.), # in urad
    beamDivergenceY = cms.double(20.), # in urad
    beamEnergyDispersion = cms.double(1.11e-4),
    halfCrossingAngleSector45 = cms.double(179.394), # in urad
    halfCrossingAngleSector56 = cms.double(191.541), # in urad
    sqrtS = cms.double(13.e3), # in GeV
    #BeamXatIP = cms.untracked.double(0.499), # if not given, will take the CMS average vertex position
    #BeamYatIP = cms.untracked.double(-0.190), # if not given, will take the CMS average vertex position
    # in m, should be consistent with geometry xml definitions
    Model_IP_150_R_Zmin = cms.double(0.0),
    #Model_IP_150_R_Zmax = cms.double(202.769),
    #Model_IP_150_L_Zmax = cms.double(-202.769),
    Model_IP_150_R_Zmax = cms.double(-212.46),
    Model_IP_150_L_Zmax = cms.double( 212.46),
    Model_IP_150_L_Zmin = cms.double(0.0),
    ApplyZShift = cms.bool(True),
    BeampipeApertureRadius = cms.double(0.04) # in meter
)
