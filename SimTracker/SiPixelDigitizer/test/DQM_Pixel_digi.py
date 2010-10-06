import FWCore.ParameterSet.Config as cms

process = cms.Process("SiPixelMonitorDigiProcess")
process.load("Geometry.TrackerSimData.trackerSimGeometryXML_cfi")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("DQM.SiPixelMonitorDigi.SiPixelMonitorDigi_cfi")
process.load("DQMServices.Core.DQM_cfg")
process.load("Alignment.CommonAlignmentProducer.FakeAlignmentSource_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/user/v/vesna/testDigis.root')
)

process.LockService = cms.Service("LockService",
    labels = cms.untracked.vstring('source')
)
process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('siPixelDigis'),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    ),
    destinations = cms.untracked.vstring('cout')
)

process.p1 = cms.Path(process.SiPixelDigiSource)
process.SiPixelDigiSource.saveFile = True
process.SiPixelDigiSource.isPIB = False
process.SiPixelDigiSource.slowDown = False
process.SiPixelDigiSource.modOn = True
process.SiPixelDigiSource.ladOn = True
process.SiPixelDigiSource.layOn = True
process.SiPixelDigiSource.phiOn = False
process.SiPixelDigiSource.ringOn = True
process.SiPixelDigiSource.bladeOn = True
process.SiPixelDigiSource.diskOn = True
process.DQM.collectorHost = ''


process.SiPixelDigiSource.twoDimOn = True
process.SiPixelDigiSource.hiRes = True
process.SiPixelDigiSource.reducedSet = False	


