import FWCore.ParameterSet.Config as cms

parameters = cms.PSet(
    discriminatorStart = cms.double(-0.2),
    nBinEffPur = cms.int32(100),
    # the constant b-efficiency for the differential plots versus pt and eta
    effBConst = cms.double(0.5),
    endEffPur = cms.double(1.005),
    discriminatorEnd = cms.double(1.2),
    startEffPur = cms.double(0.005)
)

