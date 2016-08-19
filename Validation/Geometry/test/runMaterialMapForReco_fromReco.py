# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step3 --conditions auto:run2_mc -s RAW2DIGI,RECO:reconstruction_trackingOnly,VALIDATION:@trackingOnlyValidation,DQM:@trackingOnlyDQM --datatier GEN-SIM-RECO,DQMIO -n 10 --era Run2_2016 --eventcontent RECOSIM,DQM --filein file:step2.root --fileout file:step3.root
import FWCore.ParameterSet.Config as cms

from FWCore.ParameterSet.VarParsing import VarParsing

from Configuration.StandardSequences.Eras import eras

options = VarParsing ('analysis')
options.register('inputFile',
                 None,
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.string,
                 "Input file to use.")
options.register('fromLocalXML',                   # name
                False,                             # default value
                VarParsing.multiplicity.singleton, # kind of option, if single or multiple
                VarParsing.varType.bool,           # type of the option
                "Read Material description for XML files in release or in local installation" # Info message
                )
options.register('sample',
                'SingleMuPt100',
                VarParsing.multiplicity.singleton,
                VarParsing.varType.string,
                "Input sample to use. Will also modify the output filename accordingly")
options.parseArguments()

# DEFAULT VALUES
output_file_inRECO = 'file:matbdgForReco_FromReco_%s.root' % options.sample
output_file_inDQM  = 'file:matbdgForReco_FromReco_%s_inDQM.root'% options.sample
input_file = '/store/relval/CMSSW_8_1_0_pre6/RelValSingleMuPt100_UP15/GEN-SIM-DIGI-RAW-HLTDEBUG/80X_mcRun2_asymptotic_v14-v1/00000/58AD9241-0A2A-E611-AB04-0025905B856E.root'

if options.fromLocalXML == True:
  output_file_inRECO = 'file:matbdgForReco_FromReco_%s_FromLocalXML.root' % options.sample
  output_file_inDQM  = 'file:matbdgForReco_FromReco_%s_FromLocalXML_inDQM.root' % options.sample

if options.sample == 'SingleMuPt10':
  input_file = '/store/relval/CMSSW_8_1_0_pre6/RelValSingleMuPt10_UP15/GEN-SIM-DIGI-RAW-HLTDEBUG/80X_mcRun2_asymptotic_v14-v1/00000/0263098F-092A-E611-AC04-0CC47A745250.root'

if options.inputFile is not None:
  input_file = options.inputFile

process = cms.Process('MATERIAL',eras.Run2_2016)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
if options.fromLocalXML:
  print "Loading material from local XMLs"
  process.load('Configuration.Geometry.GeometryExtended2016Reco_cff')
else:
  process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.Validation_cff')
process.load('DQMOffline.Configuration.DQMOfflineMC_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(input_file),
#    fileNames = cms.untracked.vstring('file:/data/rovere/RelVal/CMSSW_8_1_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/80X_mcRun2_asymptotic_v14-v1/4A127630-092A-E611-AC30-0025905B858A.root'),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step3 nevts:10'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.RECOSIMoutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-RECO'),
        filterName = cms.untracked.string('')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    fileName = cms.untracked.string(output_file_inRECO),
    outputCommands = process.RECOSIMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('DQMIO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string(output_file_inDQM),
    outputCommands = process.DQMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)


process.materialdumper = cms.EDAnalyzer("TrackingRecoMaterialAnalyser",
        tracks = cms.InputTag("generalTracks"),
        vertices = cms.InputTag("offlinePrimaryVertices"),
        DoPredictionsOnly = cms.bool(False),
        Fitter = cms.string('KFFitterForRefitInsideOut'),
        TrackerRecHitBuilder = cms.string('WithAngleAndTemplate'),
        Smoother = cms.string('KFSmootherForRefitInsideOut'),
        MuonRecHitBuilder = cms.string('MuonRecHitBuilder'),
        RefitDirection = cms.string('alongMomentum'),
        RefitRPCHits = cms.bool(True),
        Propagator = cms.string('SmartPropagatorAnyRKOpposite'),
        #Propagators
        PropagatorAlong = cms.string("RungeKuttaTrackerPropagator"),
        PropagatorOpposite = cms.string("RungeKuttaTrackerPropagatorOpposite")
)
# Additional output definition

# Other statements
process.mix.playback = True
process.mix.digitizers = cms.PSet()
for a in process.aliases: delattr(process, a)
process.RandomNumberGeneratorService.restoreStateLabel=cms.untracked.string("randomEngineStateProducer")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
process.reconstruction_step = cms.Path(process.reconstruction_trackingOnly)
process.prevalidation_step = cms.Path(process.globalPrevalidationTrackingOnly)
process.dqmoffline_step = cms.Path(process.DQMOfflineTracking)
process.dqmofflineOnPAT_step = cms.Path(process.PostDQMOffline)
process.validation_step = cms.EndPath(process.globalValidationTrackingOnly)
process.RECOSIMoutput_step = cms.EndPath(process.RECOSIMoutput)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)
process.materialdumper_step = cms.Path(process.materialdumper)

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.reconstruction_step,process.materialdumper_step,process.prevalidation_step,process.validation_step,process.dqmoffline_step,process.dqmofflineOnPAT_step,process.RECOSIMoutput_step,process.DQMoutput_step)

# customisation of the process.

# Automatic addition of the customisation function from SimGeneral.MixingModule.fullMixCustomize_cff
from SimGeneral.MixingModule.fullMixCustomize_cff import setCrossingFrameOn 

#call to customisation function setCrossingFrameOn imported from SimGeneral.MixingModule.fullMixCustomize_cff
process = setCrossingFrameOn(process)

# End of customisation functions

#Setup FWK for multithreaded
process.options.numberOfThreads=cms.untracked.uint32(6)
process.options.numberOfStreams=cms.untracked.uint32(0)

