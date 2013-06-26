# Auto generated configuration file
# using: 
# Revision: 1.155 
# Source: /cvs_server/repositories/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: harvest -s HARVESTING:dqmHarvesting -n -1 --conditions MC_3XY_V9A::All --filein file:RecoOutput.root --python_filename Harvesting_cfg.py --no_exec
import FWCore.ParameterSet.Config as cms
import FWCore.Utilities.FileUtils as FileUtils

process = cms.Process('HARVESTING')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryExtended_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.EDMtoMEAtRunEnd_cff')
process.load('Configuration.StandardSequences.Harvesting_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.EventContent.EventContent_cff')

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    annotation = cms.untracked.string('harvest nevts:100'),
    name = cms.untracked.string('PyReleaseValidation')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound'),
    fileMode = cms.untracked.string('FULLMERGE')
)
# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:valid_reco.root'),
    processingMode = cms.untracked.string('RunsAndLumis')
)

# Additional output definition

# Other statements
#process.GlobalTag.globaltag = 'DESIGN_42_V10::All'
process.GlobalTag.globaltag = 'MC_42_V10::All'

# Path and EndPath definitions
process.edmtome_step = cms.Path(process.EDMtoME)
process.validationpreprodHarvesting = cms.Path(process.postValidation*process.hltpostvalidation_preprod)
process.validationprodHarvesting = cms.Path(process.postValidation*process.hltpostvalidation_prod)
process.dqmHarvesting = cms.Path(process.DQMOffline_SecondStep*process.DQMOffline_Certification)
process.validationHarvesting = cms.Path(process.postValidation*process.hltpostvalidation)
process.validationHarvestingFS = cms.Path(process.HarvestingFastSim)
process.dqmHarvestingPOG = cms.Path(process.DQMOffline_SecondStep_PrePOG)
process.dqmsave_step = cms.Path(process.DQMSaver)

# Schedule definition
process.schedule = cms.Schedule(process.edmtome_step,process.dqmHarvesting,process.dqmsave_step)

#-----------------------------------------------------------------------------------
# Mark's changes start (everything above this point is the output from cmsDriver)
#

# For some reason a seed harvester isn't included in the standard sequences. If this next processor isn't
# run then things like efficiencies are just added together instead of recalculated.
process.postProcessorSeed = cms.EDAnalyzer("DQMGenericClient",
	profile = cms.vstring(),
	resolution = cms.vstring(),
	efficiency = cms.vstring("effic \'Efficiency vs #eta\' num_assoc(simToReco)_eta num_simul_eta", 
		"efficPt \'Efficiency vs p_{T}\' num_assoc(simToReco)_pT num_simul_pT", 
		"fakerate \'Fake rate vs #eta\' num_assoc(recoToSim)_eta num_reco_eta fake", 
		"fakeratePt \'Fake rate vs p_{T}\' num_assoc(recoToSim)_pT num_reco_pT fake" ),
	subDirs = cms.untracked.vstring('Tracking/Seed/*'),
	outputFileName = cms.untracked.string('')
)

process.dqmSaver.saveByRun = -1
process.dqmSaver.saveAtJobEnd = True
process.dqmSaver.forceRunNumber = 1

# Remove the HLT harvesting from the validation harvesting step
process.validationHarvesting = cms.Path(process.postValidation)
process.trackingOnlyHarvesting = cms.Path(process.postProcessorTrack)
process.seedingOnlyHarvesting = cms.Path(process.postProcessorSeed)
#process.schedule = cms.Schedule(process.edmtome_step,process.validationHarvesting,process.dqmsave_step)
#process.schedule = cms.Schedule(process.edmtome_step,process.trackingOnlyHarvesting,process.seedingOnlyHarvesting,process.dqmsave_step)
process.schedule = cms.Schedule(process.edmtome_step,process.trackingOnlyHarvesting,process.dqmsave_step)
#process.schedule = cms.Schedule(process.edmtome_step,process.seedingOnlyHarvesting,process.dqmsave_step)

process.source.fileNames = cms.untracked.vstring("file:valid_reco.root")
#process.source.fileNames = cms.untracked.vstring( FileUtils.loadListFromFile ('mtv_muPU50.txt') )
#process.source.fileNames = cms.untracked.vstring(
