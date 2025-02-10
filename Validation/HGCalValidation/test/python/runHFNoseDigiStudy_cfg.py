import FWCore.ParameterSet.Config as cms

geomName = "Run4D115"
geomFile = "Configuration.Geometry.GeometryExtended" + geomName + "Reco_cff"
import Configuration.Geometry.defaultPhase2ConditionsEra_cff as _settings
GLOBAL_TAG, ERA = _settings.get_era_and_conditions(geomName)
print("Geometry Name:  ", geomName)
print("Geom file Name: ", geomFile)
print("Global Tag Name: ", GLOBAL_TAG)
print("Era Name:        ", ERA)

process = cms.Process('HFNoseDigiStudy',ERA)

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load(geomFile)
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Validation.HGCalValidation.hfnoseDigiStudy_cfi')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, GLOBAL_TAG, '')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.HGCalValidation=dict()
    process.MessageLogger.HGCalGeom=dict()

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('file:step2D94.root')
                            )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string('hfnDigiD94tt.root'),
                                   closeFileFast = cms.untracked.bool(True)
                                   )

process.raw2digi_step = cms.Path(process.RawToDigi)
process.analysis_step = cms.Path(process.hfnoseDigiStudy)
process.hfnoseDigiStudy.verbosity = 1

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.analysis_step)
