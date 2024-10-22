import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run3_cff import Run3

process = cms.Process("PROPAGATORTEST",Run3)

####################################################
# Message Logger 
####################################################  
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load("Configuration.EventContent.EventContent_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2022_realistic', '')

####################################################
# MessageLogger customizations
####################################################
process.MessageLogger.cerr.enable = False
process.MessageLogger.cout.enable = True
labels = ["propTest", "geopro"] # Python module's label
messageLogger = dict()
for category in labels:
      main_key = '%sMessageLogger'%(category)
      category_key = 'Geant4e' # C++ EDProducer type
      messageLogger[main_key] = dict(
              filename = '%s_%s.log' % ("debugG4e", category),
              threshold = 'DEBUG',
              default = dict(limit=0)
              )
      messageLogger[main_key][category_key] = dict(limit=-1)
      # First create defaults
      setattr(process.MessageLogger.files, category, dict())
      # Then modify them
      setattr(process.MessageLogger.files, category, messageLogger[main_key])

process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

####################################################
## Set up geometry
####################################################
from SimG4Core.Application.g4SimHits_cfi import g4SimHits as _g4SimHits
process.geopro = cms.EDProducer("GeometryProducer",
     GeoFromDD4hep = cms.bool(False),
     UseMagneticField = cms.bool(True),
     UseSensitiveDetectors = cms.bool(False),
     MagneticField =  _g4SimHits.MagneticField.clone()
)

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep
dd4hep.toModify(process.geopro, GeoFromDD4hep = True )

####################################################
# Extrapolator
####################################################
process.propTest = cms.EDAnalyzer("SimpleGeant4ePropagatorTest",
)

process.g4TestPath = cms.Path( process.geopro*process.propTest )
process.schedule = cms.Schedule( process.g4TestPath )
