import FWCore.ParameterSet.Config as cms

lhcBeamConditions_2016PreTS2 = cms.PSet(
    vertexSize = cms.double(10.e-6), # in m
    beamDivergence = cms.double(20.e-6), # in rad

    # vertex vertical offset in both sectors
    yOffsetSector45 = cms.double(300.e-6), # in m
    yOffsetSector56 = cms.double(200.e-6),

    # crossing angles
    halfCrossingAngleSector45 = cms.double(179.394e-6), # in rad
    halfCrossingAngleSector56 = cms.double(191.541e-6), # in rad
)
