import FWCore.ParameterSet.Config as cms

process = cms.Process("TopDQM")
## load TopDecayChannel validation
process.load("Validation.Generator.TopDecayChannelDQM_cfi")

process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.DQM.collectorHost = ''

process.dqmSaver.workflow = cms.untracked.string('/Test/TopDecayChannelDQM/DataSet')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source(
    "PoolSource"
    ,fileNames = cms.untracked.vstring(
     '/store/relval/CMSSW_3_5_0_pre1/RelValTTbar/GEN-SIM-RECO/STARTUP3X_V14-v1/0006/14920B0A-0DE8-DE11-B138-002618943926.root'
    ,'/store/relval/CMSSW_3_5_0_pre1/RelValTTbar/GEN-SIM-RECO/STARTUP3X_V14-v1/0006/1AD1F37E-0BE8-DE11-8D83-00261894396A.root'
    ,'/store/relval/CMSSW_3_5_0_pre1/RelValTTbar/GEN-SIM-RECO/STARTUP3X_V14-v1/0006/AC476888-0CE8-DE11-8EDC-0026189438D4.root'
    ,'/store/relval/CMSSW_3_5_0_pre1/RelValTTbar/GEN-SIM-RECO/STARTUP3X_V14-v1/0007/4ADBBCAE-37E8-DE11-AE89-00304867C1BA.root'
    ,'/store/relval/CMSSW_3_5_0_pre1/RelValTTbar/GEN-SIM-RECO/STARTUP3X_V14-v1/0007/6ABDD43B-13E8-DE11-8A47-001A92971BA0.root'
    ,'/store/relval/CMSSW_3_5_0_pre1/RelValTTbar/GEN-SIM-RECO/STARTUP3X_V14-v1/0007/744B08B2-12E8-DE11-A729-001A928116B8.root'
    ,'/store/relval/CMSSW_3_5_0_pre1/RelValTTbar/GEN-SIM-RECO/STARTUP3X_V14-v1/0007/A2CC4B57-11E8-DE11-B413-003048678D9A.root'
    ,'/store/relval/CMSSW_3_5_0_pre1/RelValTTbar/GEN-SIM-RECO/STARTUP3X_V14-v1/0007/B69516B8-12E8-DE11-982F-00304867BFAE.root'
    ,'/store/relval/CMSSW_3_5_0_pre1/RelValTTbar/GEN-SIM-RECO/STARTUP3X_V14-v1/0007/CEFA8143-12E8-DE11-A51F-0018F3D096E4.root' 
    )
)

#process.content = cms.EDAnalyzer("EventContentAnalyzer")

process.p = cms.Path(
   #process.content *
    process.topDecayChannelDQM    +
    process.dqmSaver
)
