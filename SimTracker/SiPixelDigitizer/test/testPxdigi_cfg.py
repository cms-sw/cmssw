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
    'file:/afs/cern.ch/work/d/dkotlins/public/MC/mu/pt100/digis/digis1.root'
#    'file:dummy_100.root'
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
# 2012
# process.GlobalTag.globaltag = 'GR_P_V40::All'
# 2013 MC
process.GlobalTag.globaltag = 'MC_70_V1::All'

#process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
#process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
#process.load("Configuration.StandardSequences.MagneticField_cff")
# include "MagneticField/Engine/data/volumeBasedMagneticField.cfi"
  
process.analysis = cms.EDAnalyzer("PixelDigisTest",
    Verbosity = cms.untracked.bool(False),
# sim in V7
    src = cms.InputTag("mix"),
# old default
#    src = cms.InputTag("siPixelDigis"),
)

process.p = cms.Path(process.analysis)

