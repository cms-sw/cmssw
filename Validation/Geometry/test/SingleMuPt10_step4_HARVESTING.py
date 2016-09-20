# Auto generated configuration file
# using:
# Revision: 1.19
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v
# with command line options: step4 --filetype DQM --conditions auto:run2_mc --mc -s HARVESTING:@trackingOnlyValidation+@trackingOnlyDQM --era Run2_2016 -n -1 --no_exec --filein file:step3_RAW2DIGI_L1Reco_RECO_VALIDATION_DQM_inDQM.root
import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras
from varParser import options

process = cms.Process('HARVESTING',eras.Run2_2016)
options.parseArguments()
input_filename_ = 'SingleMuPt10_step3_RAW2DIGI_L1Reco_RECO_VALIDATION_DQM'
output_workflow_ = '/SingleMuPt10/Run2016/DQMIO'
if options.geomDB:
  input_filename_ += '_fromDB'
  output_workflow_ = '/SingleMuPt10_fromDB/Run2016/DQMIO'
else:
  input_filename_ += '_fromLocalFiles'
  output_workflow_ = '/SingleMuPt10_fromLocalFiles/Run2016/DQMIO'


# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.DQMSaverAtRunEnd_cff')
process.load('Configuration.StandardSequences.Harvesting_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("DQMRootSource",
    fileNames = cms.untracked.vstring('file:%s_inDQM.root' % input_filename_)
)

process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound'),
    fileMode = cms.untracked.string('FULLMERGE')
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step4 nevts:-1'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

# Path and EndPath definitions
process.dqmHarvestingPOGMC = cms.Path(process.DQMOffline_SecondStep_PrePOGMC)
process.validationHarvesting = cms.Path(process.postValidation+process.hltpostvalidation+process.postValidation_gen)
process.validationprodHarvesting = cms.Path(process.hltpostvalidation_prod+process.postValidation_gen)
process.dqmHarvestingFakeHLT = cms.Path(process.DQMOffline_SecondStep_FakeHLT+process.DQMOffline_Certification)
process.validationpreprodHarvesting = cms.Path(process.postValidation_preprod+process.hltpostvalidation_preprod+process.postValidation_gen)
process.validationHarvestingMiniAOD = cms.Path(process.JetPostProcessor+process.METPostProcessorHarvesting+process.postValidationMiniAOD)
process.validationHarvestingHI = cms.Path(process.postValidationHI)
process.genHarvesting = cms.Path(process.postValidation_gen)
process.dqmHarvestingPOG = cms.Path(process.DQMOffline_SecondStep_PrePOG)
process.alcaHarvesting = cms.Path()
process.dqmHarvesting = cms.Path(process.DQMOffline_SecondStep+process.DQMOffline_Certification)
process.validationHarvestingFS = cms.Path(process.postValidation+process.hltpostvalidation+process.postValidation_gen)
process.postValidation_trackingOnly_step = cms.Path(process.postValidation_trackingOnly)
process.DQMHarvestTracking_step = cms.Path(process.DQMHarvestTracking)
process.dqmsave_step = cms.Path(process.DQMSaver)
process.dqmSaver.workflow = output_workflow_

# Schedule definition
process.schedule = cms.Schedule(process.postValidation_trackingOnly_step,process.DQMHarvestTracking_step,process.dqmsave_step)


# Customisation from command line
