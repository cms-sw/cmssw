import FWCore.ParameterSet.Config as cms

process = cms.Process("G4PrintGeometry")

process.load('Configuration.ProcessModifiers.dd4hep_cff')
process.load('Geometry.CMSCommonData.cmsExtendedGeometry2021XML_cfi')
process.load('Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi')
process.load('Geometry.EcalCommonData.ecalSimulationParameters_cff')
process.load('Geometry.HcalCommonData.hcalDDDSimConstants_cff')
process.load('Geometry.HGCalCommonData.hgcalParametersInitialization_cfi')
process.load('Geometry.HGCalCommonData.hgcalNumberingInitialization_cfi')
process.load('Geometry.MTDNumberingBuilder.mtdNumberingGeometry_cfi')
process.load("Geometry.MuonNumbering.muonGeometryConstants_cff")
process.load('FWCore.MessageService.MessageLogger_cfi')


from SimG4Core.PrintGeomInfo.g4PrintGeomInfo_cfi import *

process = printGeomInfo(process)

if hasattr(process,'MessageLogger'):
    process.MessageLogger.G4cerr=dict()
    process.MessageLogger.G4cout=dict()

process.DDDetectorESProducer = cms.ESSource("DDDetectorESProducer",
                                            confGeomXMLFiles = cms.FileInPath('Geometry/CMSCommonData/data/dd4hep/cmsExtendedGeometry2021.xml'),
                                            appendToDataLabel = cms.string('')
)

process.DDCompactViewESProducer = cms.ESProducer("DDCompactViewESProducer",
                                                appendToDataLabel = cms.string('')
)

process.g4SimHits.g4GeometryDD4hepSource = cms.bool(True)
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    DumpSummary    = cms.untracked.bool(True),
    DumpLVTree     = cms.untracked.bool(True),
    DumpMaterial   = cms.untracked.bool(False),
    DumpLVList     = cms.untracked.bool(True),
    DumpLV         = cms.untracked.bool(False),
    DumpSolid      = cms.untracked.bool(True),
    DumpAttributes = cms.untracked.bool(False),
    DumpPV         = cms.untracked.bool(False),
    DumpRotation   = cms.untracked.bool(False),
    DumpReplica    = cms.untracked.bool(False),
    DumpTouch      = cms.untracked.bool(True),
    DumpSense      = cms.untracked.bool(True),
    DD4Hep         = cms.untracked.bool(True),
    Name           = cms.untracked.string('csc:ME11*'),
    Names          = cms.untracked.vstring('EcalHitsEB'),
    type           = cms.string('PrintGeomInfoAction')
))

process.hcalParameters.fromDD4Hep = cms.bool(True)
process.hcalSimulationParameters.fromDD4Hep = cms.bool(True)
process.caloSimulationParameters.fromDD4Hep = cms.bool(True)
process.ecalSimulationParametersEB.fromDD4Hep = cms.bool(True)
process.ecalSimulationParametersEE.fromDD4Hep = cms.bool(True)
process.ecalSimulationParametersES.fromDD4Hep = cms.bool(True)
process.hgcalEEParametersInitialize.fromDD4Hep = cms.bool(True)
process.hgcalHESiParametersInitialize.fromDD4Hep = cms.bool(True)
process.hgcalHEScParametersInitialize.fromDD4Hep = cms.bool(True)
