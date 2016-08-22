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
    fileNames = cms.untracked.vstring('/store/data/Run2016A/ZeroBias1/RAW/v1/000/271/336/00000/00963A5A-BF0A-E611-A657-02163E0141FB.root',
                                      '/store/data/Run2016A/ZeroBias1/RAW/v1/000/271/024/00000/1E1D9FE3-C908-E611-8D1C-02163E014238.root',
                                      '/store/data/Run2016A/ZeroBias1/RAW/v1/000/271/024/00000/2C2D75D0-D408-E611-AB10-02163E014251.root',
                                      '/store/data/Run2016A/ZeroBias1/RAW/v1/000/271/024/00000/70968FFB-CD08-E611-9878-02163E0134F4.root',
                                      '/store/data/Run2016A/ZeroBias1/RAW/v1/000/271/024/00000/7AFA99EA-D408-E611-9AB7-02163E0143D9.root',
                                      '/store/data/Run2016A/ZeroBias1/RAW/v1/000/271/024/00000/807129D4-D408-E611-A1B8-02163E0146D8.root',
                                      '/store/data/Run2016A/ZeroBias1/RAW/v1/000/271/024/00000/8A4789CB-D408-E611-9ACB-02163E014238.root',
                                      '/store/data/Run2016A/ZeroBias1/RAW/v1/000/271/024/00000/A28B0449-D508-E611-A1EF-02163E014115.root',
                                      '/store/data/Run2016A/ZeroBias1/RAW/v1/000/271/024/00000/AED1D93B-CF08-E611-AF37-02163E014126.root',
                                      '/store/data/Run2016A/ZeroBias1/RAW/v1/000/271/024/00000/B8D03CE3-D408-E611-88D2-02163E0139D8.root',
                                      '/store/data/Run2016A/ZeroBias1/RAW/v1/000/271/024/00000/EE4A5ED0-D408-E611-A7F5-02163E0141B5.root',
                                      '/store/data/Run2016A/ZeroBias1/RAW/v1/000/271/024/00000/F25432E9-D408-E611-8355-02163E011933.root',
                                      '/store/data/Run2016A/ZeroBias1/RAW/v1/000/271/025/00000/2E9CA495-CE08-E611-B006-02163E0135FA.root',
                                      '/store/data/Run2016A/ZeroBias1/RAW/v1/000/271/025/00000/5E654A69-CF08-E611-96BD-02163E013926.root',
                                      '/store/data/Run2016A/ZeroBias1/RAW/v1/000/271/025/00000/865B89F6-D008-E611-8A8F-02163E011E38.root',
                                      '/store/data/Run2016A/ZeroBias1/RAW/v1/000/271/025/00000/8E8D3CEB-D008-E611-96B0-02163E011F4E.root',
                                      '/store/data/Run2016A/ZeroBias1/RAW/v1/000/271/025/00000/A086D3A3-D008-E611-8311-02163E0137F8.root',
                                      '/store/data/Run2016A/ZeroBias1/RAW/v1/000/271/025/00000/C62D1FA6-CE08-E611-88FC-02163E011ECB.root',
                                      '/store/data/Run2016A/ZeroBias1/RAW/v1/000/271/025/00000/D42F63A5-D008-E611-8556-02163E014303.root',
                                      '/store/data/Run2016A/ZeroBias1/RAW/v1/000/271/025/00000/D6EFE5D4-CF08-E611-9C3D-02163E01382D.root',
                                      '/store/data/Run2016A/ZeroBias1/RAW/v1/000/271/026/00000/24C10B5D-D508-E611-AB37-02163E01203B.root'   ),
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


