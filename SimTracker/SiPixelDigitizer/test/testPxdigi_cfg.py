#
import FWCore.ParameterSet.Config as cms

process = cms.Process("digiTest")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.MessageLogger = cms.Service("MessageLogger",
    debugModules = cms.untracked.vstring('PixelDigisTest'),
    destinations = cms.untracked.vstring('cout'),
#    destinations = cms.untracked.vstring("log","cout"),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('ERROR')
    )
#    log = cms.untracked.PSet(
#        threshold = cms.untracked.string('DEBUG')
#    )
)

process.source = cms.Source("PoolSource",
    fileNames =  cms.untracked.vstring(
    'file:/afs/cern.ch/user/d/dkotlins/scratch0/data/digis11.root'
#    'file:/afs/cern.ch/user/d/dkotlins/scratch0/data/digis21.root'
    )
)


process.TFileService = cms.Service("TFileService",
    fileName = cms.string('histo.root')
)

#process.load("Geometry.TrackerSimData.trackerSimGeometryXML_cfi")
#process.load("Geometry.CMSCommonData.cmsSimIdealGeometryXML_cfi")

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
# what is this?
# process.load("Configuration.StandardSequences.Services_cff")

# what is this?
#process.load("SimTracker.Configuration.SimTracker_cff")

# needed for global transformation
# this crashes
# process.load("Configuration.StandardSequences.FakeConditions_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")# Choose the global tag here:
#process.GlobalTag.globaltag = 'MC_31X_V9::All'
#process.GlobalTag.globaltag = 'CRAFT09_R_V4::All'
process.GlobalTag.globaltag = 'GR10_P_V4::All'

#process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
#process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
#process.load("Configuration.StandardSequences.MagneticField_cff")
# include "MagneticField/Engine/data/volumeBasedMagneticField.cfi"
  
process.analysis = cms.EDAnalyzer("PixelDigisTest",
    Verbosity = cms.untracked.bool(False),
    src = cms.InputTag("siPixelDigis"),
)

process.p = cms.Path(process.analysis)

