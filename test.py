# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: test_11_a_1 -s RAW2DIGI,RECO,DQM -n 10 --eventcontent DQM --datatier DQMIO --conditions 80X_dataRun2_Prompt_v9 --era Run2_2016 --filein /store/data/Run2016A/ZeroBias1/RAW/v1/000/271/336/00000/00963A5A-BF0A-E611-A657-02163E0141FB.root --data --customise DQMServices/Components/test/customDQM.py --no_exec --python_filename=test_11_a_1.py --customise=L1Trigger/Configuration/customiseReEmul.L1TReEmulFromRAW
import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('RECO',eras.Run2_2016)

from HLTrigger.HLTfilters.hltHighLevel_cfi import hltHighLevel
process.hltFatEventFilters = hltHighLevel.clone()
process.hltFatEventFilters.throw = cms.bool(False)
process.hltFatEventFilters.HLTPaths = cms.vstring('HLT_L1FatEvents_v*')
# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
process.load('Configuration.StandardSequences.Reconstruction_Data_cff')
process.load('DQMOffline.Configuration.DQMOffline_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/data/Run2016A/ZeroBias1/RAW/v1/000/271/336/00000/00963A5A-BF0A-E611-A657-02163E0141FB.root'),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('test_11_a_1 nevts:10'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('DQMIO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('test_11_a_1_RAW2DIGI_RECO_DQM.root'),
    outputCommands = process.DQMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '80X_dataRun2_Prompt_v9', '')

# Path and EndPath definitions
#process.hltFatEventFilter=cms.Path(process.hltFatEventFilters)
#process.raw2digi_step = cms.Path(process.RawToDigi)
#process.reconstruction_step = cms.Path(process.reconstruction)
#process.dqmoffline_step = cms.Path(process.DQMOffline)
#process.dqmofflineOnPAT_step = cms.Path(process.DQMOffline)
#process.DQMoutput_step = cms.EndPath(process.DQMoutput)

# customisation of the process.
process.schedule=cms.Schedule()
# Automatic addition of the customisation function from DQMServices.Components.test.customDQM
from DQMServices.Components.test.customDQM import customise 

#call to customisation function customise imported from DQMServices.Components.test.customDQM
process = customise(process)

# Automatic addition of the customisation function from L1Trigger.Configuration.customiseReEmul
from L1Trigger.Configuration.customiseReEmul import L1TReEmulFromRAW 

#call to customisation function L1TReEmulFromRAW imported from L1Trigger.Configuration.customiseReEmul
process = L1TReEmulFromRAW(process)
process.p=cms.Path( process.hltFatEventFilters*
			process.RawToDigi*
			process.reconstruction*
			process.DQMOffline*			
			process.L1TReEmul
		)
process.e=cms.EndPath( process.DQMoutput )
process.schedule=cms.Schedule(process.p,process.e)

# End of customisation functions

# Schedule definition
#process.schedule = cms.Schedule(process.hltFatEventFilter,process.raw2digi_step,process.reconstruction_step,process.dqmoffline_step,process.dqmofflineOnPAT_step,process.DQMoutput_step)


