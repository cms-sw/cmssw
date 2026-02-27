import sys
FLOAT_MAX = 3.402823466E+38

import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
# Options
####################################################################
options = VarParsing.VarParsing()
options.register('dbFromEvent',
                 False, # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.bool, # string, int, or float
                 "use the Event for reading the DB?")
options.register('warningThreshold',
                 1., # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.float, # string, int, or float
                 "threshold to emit a warning")
options.register('errorThreshold',
                 3., # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.float, # string, int, or float
                 "threshold to emit a warning")
options.parseArguments()

import FWCore.ParameterSet.Config as cms
process = cms.Process("testBeamSpotCompatibility")

####################################################################
# Load source
####################################################################
readFiles = cms.untracked.vstring()
readFiles.extend(['/store/data/Run2023D/HLTPhysics/ALCARECO/TkAlMinBias-PromptReco-v1/000/370/580/00000/acdddb09-046c-4375-82f3-678138106ac7.root'])
process.source = cms.Source("PoolSource",
                            fileNames = readFiles ,
                            duplicateCheckMode = cms.untracked.string('checkAllFilesOpened')
                            )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.options = cms.untracked.PSet(
   wantSummary = cms.untracked.bool(False),
   Rethrow = cms.untracked.vstring("ProductNotFound"), # make this exception fatal
   fileMode  =  cms.untracked.string('NOMERGE') # no ordering needed, but calls endRun/beginRun etc. at file boundaries
)

####################################################################
# Load and configure Message Logger
####################################################################
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

process.load("RecoVertex.BeamSpotProducer.BeamSpot_cff")
#from RecoVertex.BeamSpotProducer.BeamSpot_cfi import offlineBeamSpot
#process.myOfflineBeamSpot = offlineBeamSpot.clone()

####################################################################
# Load and configure refitting includes
####################################################################
process.load("Configuration.Geometry.GeometryDB_cff")
process.load('Configuration.StandardSequences.Services_cff')
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")

####################################################################
# Load and Configure TrackRefitter
####################################################################
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
import RecoTracker.TrackProducer.TrackRefitters_cff
process.FinalTrackRefitter = RecoTracker.TrackProducer.TrackRefitter_cfi.TrackRefitter.clone()
process.FinalTrackRefitter.src = "ALCARECOTkAlMinBias"
process.FinalTrackRefitter.TrajectoryInEvent = True
process.FinalTrackRefitter.NavigationSchool = ''
process.FinalTrackRefitter.TTRHBuilder = "WithAngleAndTemplate"

####################################################################
#Global tag
####################################################################
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, "140X_dataRun3_v4")  ## DO NOT CHANGE (it will change the behaviour of the test)!

process.BeamSpotChecker = cms.EDAnalyzer("BeamSpotCompatibilityChecker",
                                         bsFromFile = cms.InputTag("offlineBeamSpot::RECO"),  # source of the event beamspot (in the ALCARECO files)
                                         #bsFromFile = cms.InputTag("offlineBeamSpot"),       # source of the event beamspot (in the ALCARECO files)
                                         dbFromEvent = cms.bool(options.dbFromEvent),         # take the DB beamspot from the event
                                         warningThr = cms.double(options.warningThreshold),   # significance threshold to emit a warning message
                                         errorThr = cms.double(options.errorThreshold),       # significance threshold to abort the job
                                         verbose = cms.untracked.bool(True)                   # verbose mode
                                         )
if(options.dbFromEvent):
    process.BeamSpotChecker.bsFromDB = cms.InputTag("offlineBeamSpot::@currentProcess"), # source of the DB beamspot (from Global Tag) NOTE: only if dbFromEvent is True!

process.p = cms.Path(
    #process.myOfflineBeamSpot*
    process.offlineBeamSpot*
    process.FinalTrackRefitter*
    process.BeamSpotChecker)

print("Done")
