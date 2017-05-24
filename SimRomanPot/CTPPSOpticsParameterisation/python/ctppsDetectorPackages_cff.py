import FWCore.ParameterSet.Config as cms

# list of detector packages to simulate
detectorPackages_2017PreTS2 = cms.VPSet(
    cms.PSet(
        potId = cms.uint32(2),
        scatteringAngle = cms.double(25.e-6), # physics scattering angle, rad
        resolution = cms.double(12.e-6), # RP resolution, m
        minXi = cms.double(0.03),
        maxXi = cms.double(0.17),
    ),
    cms.PSet(
        potId = cms.uint32(3),
        scatteringAngle = cms.double(25.e-6), # physics scattering angle, rad
        resolution = cms.double(12.e-6), # RP resolution, m
        minXi = cms.double(0.03),
        maxXi = cms.double(0.17),
    ),
    cms.PSet(
        potId = cms.uint32(102),
        scatteringAngle = cms.double(25.e-6), # physics scattering angle, rad
        resolution = cms.double(12.e-6), # RP resolution, m
        minXi = cms.double(0.03),
        maxXi = cms.double(0.17),
    ),
    cms.PSet(
        potId = cms.uint32(103),
        scatteringAngle = cms.double(25.e-6), # physics scattering angle, rad
        resolution = cms.double(12.e-6), # RP resolution, m
        minXi = cms.double(0.03),
        maxXi = cms.double(0.17),
    ),
)
