import FWCore.ParameterSet.Config as cms

process = cms.Process("ICALIB")
process.load("Configuration.StandardSequences.FakeConditions_cff")
process.load("Configuration.StandardSequences.Geometry_cff")

process.load("SLHCUpgradeSimulations.Geometry.hybrid_cmsIdealGeometryXML_cff")

process.source = cms.Source("EmptyIOVSource",
    firstValue = cms.uint64(1),
    lastValue = cms.uint64(1),
    timetype = cms.string('runnumber'),
    interval = cms.uint64(1)
)

process.MessageLogger = cms.Service("MessageLogger",
    insert_logfile = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO')
    ),
    destinations = cms.untracked.vstring('./logfile.txt')
)

process.prodstrip = cms.EDFilter("SiStripDetInfoFileWriter",
    FilePath = cms.untracked.string('SiStripDetInfo_hybrid.dat'),
)

process.prodpixel = cms.EDFilter("SiPixelDetInfoFileWriter",
    FilePath = cms.untracked.string('PixelSkimmedGeometry_hybrid.txt'),
)

process.asciiPrint = cms.OutputModule("AsciiOutputModule")

process.p1 = cms.Path(process.prodstrip)
process.p2 = cms.Path(process.prodpixel)
process.ep = cms.EndPath(process.asciiPrint)


