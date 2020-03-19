# Auto generated configuration file
# using:
# Revision: 1.19
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v
# with command line options: step3 --conditions auto:run2_mc -s RAW2DIGI,RECO:reconstruction_trackingOnly,VALIDATION:@trackingOnlyValidation,DQM:@trackingOnlyDQM --datatier GEN-SIM-RECO,DQMIO -n 10 --era Run2_2016 --eventcontent RECOSIM,DQM --filein file:step2.root --fileout file:step3.root
import FWCore.ParameterSet.Config as cms

from FWCore.ParameterSet.VarParsing import VarParsing


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
                """Input sample to use. Will also modify the output filename accordingly.
                       Currently supported samples are:
                       SingleMuPt10,
                       [SingleMuPt100],
                       SingleElectronPt35,
                       SingleElectronPt10,
                       TTbar_PU25""")
options.register('ntuple',
                False,
                VarParsing.multiplicity.singleton,
                VarParsing.varType.bool,
                "Flag to trigger the production of MTV ntuple",
                )
options.parseArguments()

# DEFAULT VALUES
output_file_inRECO = 'file:matbdgForReco_FromReco_%s.root' % options.sample
output_file_inDQM  = 'file:matbdgForReco_FromReco_%s_inDQM.root'% options.sample
input_file = '/store/relval/CMSSW_8_1_0_pre6/RelValSingleMuPt100_UP15/GEN-SIM-DIGI-RAW-HLTDEBUG/80X_mcRun2_asymptotic_v14-v1/00000/58AD9241-0A2A-E611-AB04-0025905B856E.root'
input_files = []

if options.fromLocalXML == True:
  output_file_inRECO = 'file:matbdgForReco_FromReco_%s_FromLocalXML.root' % options.sample
  output_file_inDQM  = 'file:matbdgForReco_FromReco_%s_FromLocalXML_inDQM.root' % options.sample

if options.sample == 'SingleMuPt10':
  input_file = ['/store/relval/CMSSW_8_1_0_pre6/RelValSingleMuPt10_UP15/GEN-SIM-DIGI-RAW-HLTDEBUG/80X_mcRun2_asymptotic_v14-v1/00000/0263098F-092A-E611-AC04-0CC47A745250.root',
                '/store/relval/CMSSW_8_1_0_pre6/RelValSingleMuPt10_UP15/GEN-SIM-DIGI-RAW-HLTDEBUG/80X_mcRun2_asymptotic_v14-v1/00000/2081A255-092A-E611-9EA6-0CC47A4C8E2E.root',
                '/store/relval/CMSSW_8_1_0_pre6/RelValSingleMuPt10_UP15/GEN-SIM-DIGI-RAW-HLTDEBUG/80X_mcRun2_asymptotic_v14-v1/00000/80727153-092A-E611-948F-0CC47A745294.root',
                '/store/relval/CMSSW_8_1_0_pre6/RelValSingleMuPt10_UP15/GEN-SIM-DIGI-RAW-HLTDEBUG/80X_mcRun2_asymptotic_v14-v1/00000/E676F884-092A-E611-8121-0CC47A4D7602.root',
                '/store/relval/CMSSW_8_1_0_pre6/RelValSingleMuPt10_UP15/GEN-SIM-DIGI-RAW-HLTDEBUG/80X_mcRun2_asymptotic_v14-v1/00000/F06FBE84-092A-E611-ADA9-0CC47A4D769C.root']

if options.sample == 'SingleElectronPt35':
  input_file = '/store/relval/CMSSW_8_1_0_pre6/RelValSingleElectronPt35_UP15/GEN-SIM-DIGI-RAW-HLTDEBUG/80X_mcRun2_asymptotic_v14-v1/00000/64866125-092A-E611-ACC3-0025905B8612.root'

if options.sample == 'SingleElectronPt10':
  input_file = '/store/relval/CMSSW_8_1_0_pre6/RelValSingleElectronPt10_UP15/GEN-SIM-DIGI-RAW-HLTDEBUG/80X_mcRun2_asymptotic_v14-v1/00000/E05284C8-092A-E611-AED9-0CC47A4D75EC.root'

if options.sample == 'TTbar_PU25':
  input_files = ['/store/relval/CMSSW_8_1_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_80X_mcRun2_asymptotic_v14-v1/00000/C262684A-B92C-E611-A9CD-0CC47A4C8F08.root',
                 '/store/relval/CMSSW_8_1_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_80X_mcRun2_asymptotic_v14-v1/00000/C2EC8F1F-BA2C-E611-9EF5-0CC47A4C8E56.root',
                 '/store/relval/CMSSW_8_1_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_80X_mcRun2_asymptotic_v14-v1/00000/C4989350-B92C-E611-998D-0CC47A4D76AC.root',
                 '/store/relval/CMSSW_8_1_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_80X_mcRun2_asymptotic_v14-v1/00000/C68B5B52-B92C-E611-9AA3-0025905A613C.root',
                 '/store/relval/CMSSW_8_1_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_80X_mcRun2_asymptotic_v14-v1/00000/CADAE916-B92C-E611-AC71-0CC47A78A414.root',
                 '/store/relval/CMSSW_8_1_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_80X_mcRun2_asymptotic_v14-v1/00000/CC36B308-B92C-E611-B34B-003048FFD7AA.root',
                 '/store/relval/CMSSW_8_1_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_80X_mcRun2_asymptotic_v14-v1/00000/CC92BC5C-B92C-E611-B68B-0025905B857A.root',
                 '/store/relval/CMSSW_8_1_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_80X_mcRun2_asymptotic_v14-v1/00000/DE9B7443-B92C-E611-BD58-0CC47A78A2EC.root',
                 '/store/relval/CMSSW_8_1_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_80X_mcRun2_asymptotic_v14-v1/00000/E6DCC67A-B92C-E611-A83A-0CC47A4D76B2.root',
                 '/store/relval/CMSSW_8_1_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_80X_mcRun2_asymptotic_v14-v1/00000/EAFB710C-B92C-E611-A31D-0025905B85AA.root',
                 '/store/relval/CMSSW_8_1_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_80X_mcRun2_asymptotic_v14-v1/00000/F628B94F-B92C-E611-836C-0CC47A4D7646.root',
                 '/store/relval/CMSSW_8_1_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_80X_mcRun2_asymptotic_v14-v1/00000/FAA55833-BA2C-E611-BAFA-0025905A6092.root',
                 '/store/relval/CMSSW_8_1_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_80X_mcRun2_asymptotic_v14-v1/00000/FC4C281A-BA2C-E611-B1C3-0CC47A4C8F06.root',
                 '/store/relval/CMSSW_8_1_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_80X_mcRun2_asymptotic_v14-v1/00000/FC5E3132-B92C-E611-8C42-0025905B85FC.root',
                 '/store/relval/CMSSW_8_1_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_80X_mcRun2_asymptotic_v14-v1/00000/FC929909-B92C-E611-851F-0025905B85D2.root',
                 '/store/relval/CMSSW_8_1_0_pre6/RelValTTbar_13/GEN-SIM-DIGI-RAW-HLTDEBUG/PU25ns_80X_mcRun2_asymptotic_v14-v1/00000/FEBE1221-B92C-E611-B8CE-0025905A608E.root']

if options.inputFile is not None:
  input_file = options.inputFile

from Configuration.Eras.Era_Run2_2016_cff import Run2_2016
process = cms.Process('MATERIAL',Run2_2016)

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
if len(input_files):
  process.source.fileNames = cms.untracked.vstring()
  process.source.fileNames.extend(input_files)
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


from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
process.materialdumper = DQMEDAnalyzer('TrackingRecoMaterialAnalyser',
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

if options.ntuple:
  from Validation.RecoTrack.customiseTrackingNtuple import customiseTrackingNtuple
  process = customiseTrackingNtuple(process)
  process.trackingNtuple.includeAllHits = False
  process.trackingNtuple.includeSeeds = False
