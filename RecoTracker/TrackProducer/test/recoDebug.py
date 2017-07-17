# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: RECO --conditions 76X_dataRun2_v1 -s RAW2DIGI,L1Reco,RECO,EI,PAT,DQM --runUnscheduled --data --eventcontent RECO,AOD,MINIAOD,DQM --scenario pp --datatier RECO,AOD,MINIAOD,DQMIO --customise Configuration/DataProcessing/RecoTLR.customiseDataRun2Common_25ns --filein /store/data/Run2015C/SingleElectron/RAW/v1/000/254/879/00000/8E51CA98-7349-E511-B9AE-02163E01427B.root -n 100 --no_exec --python_filename=RECO_Run2015C.py
import FWCore.ParameterSet.Config as cms

process = cms.Process('RECO')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_Data_cff')
process.load('CommonTools.ParticleFlow.EITopPAG_cff')
process.load('DQMOffline.Configuration.DQMOffline_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')


#Adding SimpleMemoryCheck service:
#process.SimpleMemoryCheck=cms.Service("SimpleMemoryCheck",
#                                   ignoreTotal=cms.untracked.int32(1),
#                                   oncePerEventMode=cms.untracked.bool(True))


process.Timing = cms.Service("Timing"
#    ,summaryOnly = cms.untracked.bool(True)
)

# process.add_(cms.Service("Tracer"))


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:jetHT256630_RAW.root'),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('RECO nevts:100'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.RECOoutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('RECO'),
        filterName = cms.untracked.string('')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    fileName = cms.untracked.string('RECO_RAW2DIGI_L1Reco_RECO_EI_PAT_DQM.root'),
    outputCommands = process.RECOEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 
  'auto:run2_data_GRun', '')  # if mc change in mc....

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
process.L1Reco_step = cms.Path(process.L1Reco)
process.reconstruction_step = cms.Path(process.reconstruction)
process.eventinterpretaion_step = cms.Path(process.EIsequence)
process.RECOoutput_step = cms.EndPath(process.RECOoutput)

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.L1Reco_step,process.reconstruction_step,process.eventinterpretaion_step,process.RECOoutput_step)

# customisation of the process.

# Automatic addition of the customisation function from Configuration.DataProcessing.RecoTLR
from Configuration.DataProcessing.RecoTLR import customiseDataRun2Common_25ns 

#call to customisation function customiseDataRun2Common_25ns imported from Configuration.DataProcessing.RecoTLR
process = customiseDataRun2Common_25ns(process)

# End of customisation functions

process.MessageLogger = cms.Service("MessageLogger",
                                   destinations = cms.untracked.vstring("cout"), #1
                                   debugModules = cms.untracked.vstring("electronGsfTracks"),#initialStepTrackCandidates), #2
                                   categories = cms.untracked.vstring(
                                                                     'TrackProducer',
                                                                     'GsfTrackFitters',
                                                                      # 'GsfMaterialEffectsUpdator',
                                                                      'AnalyticalPropagator','RKPropagatorInS',
                                                                   #   'CkfPattern'
                                                                      ), #3
                                   cout = cms.untracked.PSet(threshold = cms.untracked.string("DEBUG"), #4
                                                                      DEBUG = cms.untracked.PSet(limit = cms.untracked.int32(0)), #5
                                                                      default = cms.untracked.PSet(limit = cms.untracked.int32(0)), #6
                                                                      TrackProducer = cms.untracked.PSet(limit = cms.untracked.int32(-1)), #7
                                                                      GsfTrackFitters = cms.untracked.PSet(limit = cms.untracked.int32(-1)), #7
                                                                      # GsfMaterialEffectsUpdator = cms.untracked.PSet(limit = cms.untracked.int32(-1)), #7
                                                                      AnalyticalPropagator = cms.untracked.PSet(limit = cms.untracked.int32(-1)), #7
                                                                      RKPropagatorInS = cms.untracked.PSet(limit = cms.untracked.int32(-1)), #7
                                                                  #    CkfPattern = cms.untracked.PSet(limit = cms.untracked.int32(-1)) #7
                                                                      )
                                   )
                              

