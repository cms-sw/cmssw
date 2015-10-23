import FWCore.ParameterSet.Config as cms

process = cms.Process("simHitTest")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
#process.MessageLogger = cms.Service("MessageLogger",
#    debugModules = cms.untracked.vstring('siPixelRawData'),
#    destinations = cms.untracked.vstring("cout"),
#    cout = cms.untracked.PSet(
#        threshold = cms.untracked.string('ERROR')
#    )
#)
process.source = cms.Source("PoolSource",
    fileNames =  cms.untracked.vstring(
       'file:step1.root'
       )
)
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.Geometry.GeometryExtended2023MuonReco_cff')
process.load('Configuration.Geometry.GeometryExtended2023Muon_cff')

process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')

process.load("DQMServices.Components.DQMFileSaver_cfi")
process.load("DQMServices.Components.DQMEventInfo_cfi")
process.dqmEnv.subSystemFolder    = "TrackerSimHit"
process.dqmSaver.convention = cms.untracked.string('Online')
process.dqmSaver.producer = cms.untracked.string('DQM')
process.dqmSaver.saveAtJobEnd = cms.bool(True)


process.simHitMon = cms.EDAnalyzer("SimHitTest",
    TopFolderName = cms.string("TrackerSimHit")
)

process.ana_seq = cms.Sequence(process.simHitMon*process.dqmEnv*process.dqmSaver)

#process.digi_step = cms.Sequence(process.siPixelRawData*process.siPixelDigis)
process.p = cms.Path(process.ana_seq)  

# customisation of the process.                                                                                                                              

# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.combinedCustoms                                                 
from SLHCUpgradeSimulations.Configuration.combinedCustoms import cust_2023Muondev

#call to customisation function cust_2023Muondev imported from SLHCUpgradeSimulations.Configuration.combinedCustoms                                          
process = cust_2023Muondev(process)
