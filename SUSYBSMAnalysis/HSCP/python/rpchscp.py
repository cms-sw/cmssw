import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")
process.load("Geometry.CMSCommonData.cmsSimIdealGeometryXML_cfi")

process.load("Geometry.RPCGeometry.rpcGeometry_cfi")

process.load("Geometry.CommonDetUnit.globalTrackingGeometry_cfi")

process.load("RecoMuon.DetLayers.muonDetLayerGeometry_cfi")

process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:/tmp/carrillo/kktau300GeVx1000WithRPC.root')
)

process.MessageLogger = cms.Service("MessageLogger")

process.rpcHSCP = cms.EDFilter("RPCHSCP",
    rootFileName = cms.untracked.string('hscp.root'),
    fileMatrixname = cms.untracked.string('matrix.txt'),
    partLabel = cms.untracked.string('genParticles')
)

process.p = cms.Path(process.rpcHSCP)

