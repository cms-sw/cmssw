import FWCore.ParameterSet.Config as cms

process = cms.Process("PROPAGATORTEST")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("MagneticField.Engine.volumeBasedMagneticField_cfi")

process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")

process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")

process.load("Geometry.DTGeometry.dtGeometry_cfi")

process.load("Geometry.RPCGeometry.rpcGeometry_cfi")

process.load("Geometry.CSCGeometry.cscGeometry_cfi")

process.load("Geometry.CommonDetUnit.bareGlobalTrackingGeometry_cfi")

process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi")

process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi")

process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:///somewhere/simevent.root') ##/somewhere/simevent.root" }

)

process.propAna = cms.EDAnalyzer("SteppingHelixPropagatorAnalyzer",
    ntupleTkHits = cms.bool(False),
    startFromPrevHit = cms.bool(False),
    radX0CorrectionMode = cms.bool(False),
    trkIndOffset = cms.int32(0),
    NtFile = cms.string('PropagatorDump.root'),
    testPCAPropagation = cms.bool(False),
    debug = cms.bool(False),
    g4SimName = cms.string('g4SimHits')
)

process.p = cms.Path(process.propAna)
process.PoolSource.fileNames = ['/store/relval/2008/4/28/RelVal-RelValSingleMuPt10-1209247429-IDEAL_V1-2nd/0001/04660E79-0115-DD11-A59B-001D09F290D8.root', '/store/relval/2008/4/28/RelVal-RelValSingleMuPt10-1209247429-IDEAL_V1-2nd/0001/5ECCCA2E-0615-DD11-8A55-000423D98800.root', '/store/relval/2008/4/28/RelVal-RelValSingleMuPt10-1209247429-IDEAL_V1-2nd/0001/D0E48A2F-0615-DD11-89AE-000423D94C68.root']

