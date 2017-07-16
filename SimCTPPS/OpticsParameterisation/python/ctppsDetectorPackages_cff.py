import FWCore.ParameterSet.Config as cms

genericStripsPackage = cms.PSet(
    scatteringAngle = cms.double(25.e-6), # physics scattering angle, rad
    resolution = cms.double(12.e-6), # RP resolution, m
    minXi = cms.double(0.03),
    maxXi = cms.double(0.17),
)

# list of detector packages to simulate
detectorPackages_2016PreTS2 = cms.VPSet(
    #----- sector 45
    genericStripsPackage.clone(
        potId = cms.uint32(0x76100000), # 002
        interpolatorName = cms.string('ip5_to_station_150_h_1_lhcb2'),
        zPosition = cms.double(-215.077), # z coordinate, m
    ),
    genericStripsPackage.clone(
        potId = cms.uint32(0x76180000), # 003
        interpolatorName = cms.string('ip5_to_station_150_h_2_lhcb2'),
        zPosition = cms.double(-215.077), # z coordinate, m
    ),
    #----- sector 56
    genericStripsPackage.clone(
        potId = cms.uint32(0x77100000), # 102
        interpolatorName = cms.string('ip5_to_station_150_h_1_lhcb1'),
        zPosition = cms.double(+215.077), # z coordinate, m
    ),
    genericStripsPackage.clone(
        potId = cms.uint32(0x77180000), # 103
        interpolatorName = cms.string('ip5_to_station_150_h_2_lhcb1'),
        zPosition = cms.double(+215.077), # z coordinate, m
    ),
)
