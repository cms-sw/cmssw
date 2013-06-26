import FWCore.ParameterSet.Config as cms

process = cms.Process("ICALIB")
#process.load("Configuration.StandardSequences.FakeConditions_cff")
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("SLHCUpgradeSimulations.Geometry.Longbarrel_cmsSimIdealGeometryXML_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'DESIGN42_V17::All'
process.TrackerDigiGeometryESModule.applyAlignment = False

print process.TrackerGeometricDetESModule.fromDDD
print process.TrackerDigiGeometryESModule.fromDDD,process.TrackerDigiGeometryESModule.applyAlignment

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

#process.prodstrip = cms.EDAnalyzer("SiStripDetInfoFileWriter",
#    FilePath = cms.untracked.string('SiStripDetInfo_longbarrel.dat'),
#)

process.prodpixel = cms.EDAnalyzer("SiPixelDetInfoFileWriter",
    FilePath = cms.untracked.string('PixelSkimmedGeometry_longbarrel.txt'),
    WriteROCInfo = cms.untracked.bool(True)
)

process.asciiPrint = cms.OutputModule("AsciiOutputModule")

#process.p1 = cms.Path(process.prodstrip)
process.p2 = cms.Path(process.prodpixel)
process.ep = cms.EndPath(process.asciiPrint)


