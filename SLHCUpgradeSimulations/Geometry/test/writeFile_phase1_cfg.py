import FWCore.ParameterSet.Config as cms

process = cms.Process("ICALIB")

process.load("Configuration.StandardSequences.Services_cff")
process.load('Configuration.Geometry.GeometryExtended2017_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')


#process.TrackerDigiGeometryESModule = cms.ESProducer( "TrackerDigiGeometryESModule",
#  appendToDataLabel = cms.string( "" ),
#  fromDDD = cms.bool( False ),
#  applyAlignment = cms.bool( False ),
#  alignmentsLabel = cms.string( "" )
#)
#rocess.TrackerGeometricDetESModule = cms.ESProducer( "TrackerGeometricDetESModule",
#  fromDDD = cms.bool( False )
#)
process.load('Geometry.TrackerGeometryBuilder.trackerGeometry_cfi')
process.trackerGeometry.applyAlignment = cms.bool(False)


from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgrade2017', '')

#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = 'DESIGN61_V10::All'
#print process.TrackerGeometricDetESModule.fromDDD
#print process.TrackerDigiGeometryESModule.fromDDD,process.TrackerDigiGeometryESModule.applyAlignment

process.source = cms.Source("EmptyIOVSource",
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    timetype = cms.string('runnumber'),
    interval = cms.uint64(1)
)

process.load("FWCore/MessageService/MessageLogger_cfi")
process.MessageLogger.destinations = cms.untracked.vstring("logfile")
process.MessageLogger.logfile = cms.untracked.PSet(threshold = cms.untracked.string('INFO'))

process.Timing = cms.Service("Timing")

process.prodstrip = cms.EDAnalyzer("SiStripDetInfoFileWriter",
    FilePath = cms.untracked.string('SiStripDetInfo_phase1.dat'),
)

process.prodpixel = cms.EDAnalyzer("SiPixelDetInfoFileWriter",
    FilePath = cms.untracked.string('PixelSkimmedGeometry_phase1.txt'),
    WriteROCInfo = cms.untracked.bool(True)
)

process.asciiPrint = cms.OutputModule("AsciiOutputModule")

process.p1 = cms.Path(process.prodstrip)
process.p2 = cms.Path(process.prodpixel)
process.ep = cms.EndPath(process.asciiPrint)


