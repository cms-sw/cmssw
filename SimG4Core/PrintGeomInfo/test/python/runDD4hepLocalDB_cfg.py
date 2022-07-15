import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_dd4hep_cff import Run3_dd4hep
process = cms.Process('G4PrintGeometry',Run3_dd4hep)
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.GeometrySimDB_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('CondCore.CondDB.CondDB_cfi')
from Configuration.AlCa.GlobalTag import GlobalTag

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.CondDB.timetype = cms.untracked.string('runnumber')
process.CondDB.connect = cms.string('sqlite_file:myfile.db')
process.PoolDBESSourceGeometry = cms.ESSource("PoolDBESSource",
                                      process.CondDB,
                                      toGet = cms.VPSet(cms.PSet(record = cms.string('GeometryFileRcd'),
                                                        tag = cms.string('XMLFILE_Geometry_TagXX_Extended2021_mc'),
                                                        label = cms.untracked.string("Extended")),
                                                        cms.PSet(record = cms.string('IdealGeometryRecord'),tag = cms.string('TKRECO_Geometry_TagXX')),
                                                        cms.PSet(record = cms.string('PTrackerParametersRcd'),tag = cms.string('TKParameters_Geometry_TagXX')),
                                                        cms.PSet(record = cms.string('PEcalBarrelRcd'),   tag = cms.string('EBRECO_Geometry_TagXX')),
                                                        cms.PSet(record = cms.string('PEcalEndcapRcd'),   tag = cms.string('EERECO_Geometry_TagXX')),
                                                        cms.PSet(record = cms.string('PEcalPreshowerRcd'),tag = cms.string('EPRECO_Geometry_TagXX')),
                                                        cms.PSet(record = cms.string('PHcalRcd'),         tag = cms.string('HCALRECO_Geometry_TagXX')),
                                                        cms.PSet(record = cms.string('HcalParametersRcd'), tag = cms.string('HCALParameters_Geometry_TagXX')),
                                                        cms.PSet(record = cms.string('PCaloTowerRcd'),    tag = cms.string('CTRECO_Geometry_TagXX')),
                                                        cms.PSet(record = cms.string('PZdcRcd'),          tag = cms.string('ZDCRECO_Geometry_TagXX')),
                                                        cms.PSet(record = cms.string('PCastorRcd'),       tag = cms.string('CASTORRECO_Geometry_TagXX')),
                                                        cms.PSet(record = cms.string('CSCRecoGeometryRcd'),tag = cms.string('CSCRECO_Geometry_TagXX')),
                                                        cms.PSet(record = cms.string('CSCRecoDigiParametersRcd'),tag = cms.string('CSCRECODIGI_Geometry_TagXX')),
                                                        cms.PSet(record = cms.string('DTRecoGeometryRcd'),tag = cms.string('DTRECO_Geometry_TagXX')),
                                                        cms.PSet(record = cms.string('RPCRecoGeometryRcd'),tag = cms.string('RPCRECO_Geometry_TagXX')),
                                                        cms.PSet(record = cms.string('GEMRecoGeometryRcd'),tag = cms.string('GEMRECO_Geometry_TagXX'))
                                                    )
                                )

process.es_prefer_geometry = cms.ESPrefer( "PoolDBESSource", "PoolDBESSourceGeometry" )
from SimG4Core.PrintGeomInfo.g4PrintGeomInfo_cfi import *

process = printGeomInfo(process)

if hasattr(process,'MessageLogger'):
    process.MessageLogger.G4cout=dict()
    process.MessageLogger.G4cerr=dict()

process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    DumpSummary      = cms.untracked.bool(False),
    DumpLVTree       = cms.untracked.bool(False),
    DumpMaterial     = cms.untracked.bool(False),
    DumpLVList       = cms.untracked.bool(False),
    DumpLV           = cms.untracked.bool(False),
    DumpSolid        = cms.untracked.bool(False),
    DumpAttributes   = cms.untracked.bool(False),
    DumpPV           = cms.untracked.bool(False),
    DumpRotation     = cms.untracked.bool(False),
    DumpReplica      = cms.untracked.bool(False),
    DumpTouch        = cms.untracked.bool(False),
    DumpSense        = cms.untracked.bool(True),
    DumpParams       = cms.untracked.bool(True),
    DumpRegion       = cms.untracked.bool(True),
    DD4hep           = cms.untracked.bool(True),
    Name             = cms.untracked.string(''),
    Names            = cms.untracked.vstring(''),
    FileDetail       = cms.untracked.bool(True),
    type             = cms.string('PrintGeomInfoAction')
))
