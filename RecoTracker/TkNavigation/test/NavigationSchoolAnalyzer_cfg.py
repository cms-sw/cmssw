import FWCore.ParameterSet.Config as cms

process = cms.Process("NavigationSchoolAnalyze")

#process.load("Configuration.StandardSequences.Geometry_cff")
process.load('Configuration.Geometry.GeometryExtendedPhase2TkBE5DPixel10DReco_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = 'STARTUP_V4::All'
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("RecoTracker.TkNavigation.NavigationSchoolESProducer_cff")

process.Tracer = cms.Service("Tracer",
    indention = cms.untracked.string('$$')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.navigationSchoolAnalyzer = cms.EDAnalyzer("NavigationSchoolAnalyzer",
    #navigationSchoolName = cms.string('BeamHaloNavigationSchool')
    navigationSchoolName = cms.string('SimpleNavigationSchool')
)

process.p = cms.Path(process.navigationSchoolAnalyzer)


