import FWCore.ParameterSet.Config as cms

# set the geometry and the GlobalTag
import Configuration.Geometry.defaultPhase2ConditionsEra_cff as _settings
_PH2_GLOBAL_TAG, _PH2_ERA = _settings.get_era_and_conditions(_settings.DEFAULT_VERSION)

process = cms.Process("NavigationSchoolAnalyzer",_PH2_ERA)

# the following lines are kept for when the phase-2 geometry wil be moved to DB
#process.load("Configuration.StandardSequences.GeometryDB_cff")
#process.load('Configuration.StandardSequences.GeometryRecoDB_cff')

process.load('Configuration.Geometry.GeometryExtendedRun4DefaultReco_cff')
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, _PH2_GLOBAL_TAG, '')
process.load("RecoTracker.TkNavigation.NavigationSchoolESProducer_cff")

#process.MessageLogger = cms.Service("MessageLogger",
#    destinations = cms.untracked.vstring('detailedInfo')
#)

#process.Tracer = cms.Service("Tracer",
#    indentation = cms.untracked.string('$$')
#)


#This has to be modified in order to read the tracker + MTD structure
process.TrackerRecoGeometryESProducer = cms.ESProducer("TrackerMTDRecoGeometryESProducer",
    usePhase2Stacks = cms.bool(False)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.navigationSchoolAnalyzer = cms.EDAnalyzer("NavigationSchoolAnalyzer",
#    navigationSchoolName = cms.string('BeamHaloNavigationSchool')
#    navigationSchoolName = cms.string('CosmicNavigationSchool')
    navigationSchoolName = cms.string('SimpleNavigationSchool')
)

process.muonNavigationTest = cms.EDAnalyzer("MuonNavigationTest")

process.p = cms.Path(process.navigationSchoolAnalyzer+process.muonNavigationTest)

