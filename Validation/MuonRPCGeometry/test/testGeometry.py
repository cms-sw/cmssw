import FWCore.ParameterSet.Config as cms

process = cms.Process("genDigi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")
process.load("Geometry.RPCGeometry.rpcGeometry_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.source = cms.Source("EmptySource")

process.test = cms.EDAnalyzer("RPCGeometryServTest")

process.a = cms.Path(process.test)

