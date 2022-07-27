import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3

#process = cms.Process("PROPAGATORTEST")
process = cms.Process("PROPAGATORTEST",Run3)



  #####################################################################
  # Message Logger ####################################################
  #
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load("Configuration.EventContent.EventContent_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run1_mc', '')
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2022_realistic', '')

process.load("FWCore.MessageService.MessageLogger_cfi")

# MessageLogger customizations
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

## Set up geometry
process.geopro = cms.EDProducer("GeometryProducer",
     GeoFromDD4hep= cms.bool(True),
#     GeoFromDD4hep= cms.bool(False),
     UseMagneticField = cms.bool(True),
     UseSensitiveDetectors = cms.bool(False),
     MagneticField = cms.PSet(
        UseLocalMagFieldManager = cms.bool(False),
        Verbosity = cms.bool(False),
        ConfGlobalMFM = cms.PSet(
            Volume = cms.string('OCMS'),
            OCMS = cms.PSet(
                Stepper = cms.string('G4TDormandPrince45'),
                Type = cms.string('CMSIMField'),
                StepperParam = cms.PSet(
                    VacRegions = cms.vstring(),
#                   VacRegions = cms.vstring('DefaultRegionForTheWorld','BeamPipeVacuum','BeamPipeOutside'),
                    EnergyThTracker = cms.double(0.2),     ## in GeV
                    RmaxTracker = cms.double(8000),        ## in mm
                    ZmaxTracker = cms.double(11000),       ## in mm
                    MaximumEpsilonStep = cms.untracked.double(0.01),
                    DeltaOneStep = cms.double(0.001),      ## in mm
                    DeltaOneStepTracker = cms.double(1e-4),## in mm
                    MaximumLoopCounts = cms.untracked.double(1000.0),
                    DeltaChord = cms.double(0.002),        ## in mm
                    DeltaChordTracker = cms.double(0.001), ## in mm
                    MinStep = cms.double(0.1),             ## in mm
                    DeltaIntersectionAndOneStep = cms.untracked.double(-1.0),
                    DeltaIntersection = cms.double(0.0001),     ## in mm
                    DeltaIntersectionTracker = cms.double(1e-6),## in mm
                    MaxStep = cms.double(150.),            ## in cm
                    MinimumEpsilonStep = cms.untracked.double(1e-05),
                    EnergyThSimple = cms.double(0.015),    ## in GeV
                    DeltaChordSimple = cms.double(0.1),    ## in mm
                    DeltaOneStepSimple = cms.double(0.1),  ## in mm
                    DeltaIntersectionSimple = cms.double(0.01), ## in mm
                    MaxStepSimple = cms.double(50.),       ## in cm
                )
            )
        ),
        delta = cms.double(1.0)
    ),
   )




  #####################################################################
  # Extrapolator ######################################################
  #
process.propTest = cms.EDAnalyzer("SimpleGeant4ePropagatorTest",
)


process.g4TestPath = cms.Path( process.geopro*process.propTest )
process.schedule = cms.Schedule( process.g4TestPath )
