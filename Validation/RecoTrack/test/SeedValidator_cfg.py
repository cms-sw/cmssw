import FWCore.ParameterSet.Config as cms

process = cms.Process("SEEDVALIDATOR")
process.load("Configuration/StandardSequences/GeometryPilot2_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'MC_31X_V2::All'
#process.MessageLogger.categories = ['TrackAssociator', 'TrackValidator']
#process.MessageLogger.debugModules = ['*']
#process.MessageLogger.cout = cms.untracked.PSet(
#    threshold = cms.untracked.string('DEBUG'),
#    default = cms.untracked.PSet(
#        limit = cms.untracked.int32(0)
#    ),
#    TrackAssociator = cms.untracked.PSet(
#        limit = cms.untracked.int32(0)
#    ),
#    TrackValidator = cms.untracked.PSet(
#        limit = cms.untracked.int32(-1)
#    )
#)
#process.MessageLogger.cerr = cms.untracked.PSet(
#    placeholder = cms.untracked.bool(True)
#)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring([
       '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-RECO/MC_31X_V2-v1/0002/DEFAFFEA-596B-DE11-9CC0-001D09F252F3.root',
       '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-RECO/MC_31X_V2-v1/0002/DC0DE66D-586B-DE11-A659-001D09F232B9.root',
       '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-RECO/MC_31X_V2-v1/0002/CEC0E4DD-5D6B-DE11-91DB-001D09F2545B.root',
       '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-RECO/MC_31X_V2-v1/0002/ACFC5D1B-5B6B-DE11-9E13-001D09F29169.root',
       '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-RECO/MC_31X_V2-v1/0002/4E62490C-D66B-DE11-A0E6-001D09F2A690.root',
       '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-RECO/MC_31X_V2-v1/0002/2C0FEA5F-5C6B-DE11-B5FF-001D09F24353.root',
       '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-RECO/MC_31X_V2-v1/0002/2ACFA83D-576B-DE11-A957-001D09F29169.root'
]                                     ),
   secondaryFileNames=cms.untracked.vstring([
       '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/EA8E5AF7-576B-DE11-BA98-001D09F24498.root',
       '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/E8627E8B-5A6B-DE11-A8F4-001D09F2438A.root',
       '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/D66DD273-5C6B-DE11-A8DB-001D09F290CE.root',
       '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/CC3232F2-596B-DE11-8C47-0019B9F704D6.root',
       '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/AAFDF230-5C6B-DE11-BF0A-001D09F24498.root',
       '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/9A28D939-576B-DE11-811D-000423D944F0.root',
       '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/94F72FC0-5B6B-DE11-8215-000423D6AF24.root',
       '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/94710927-5B6B-DE11-92EA-001D09F290CE.root',
       '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/88D87820-5B6B-DE11-B522-0019B9F704D6.root',
       '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/7E7FB3BC-E16B-DE11-9374-000423D8F63C.root',
       '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/7640E138-576B-DE11-B907-000423D99AAE.root',
       '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/6EC3A9F2-596B-DE11-B800-001D09F290CE.root',
       '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/6A4E7D34-596B-DE11-BC50-000423D986A8.root',
       '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/6482606F-586B-DE11-A34C-000423D9880C.root',
       '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/50B56266-5E6B-DE11-9275-001D09F24664.root',
       '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/44793DC2-656B-DE11-B11D-000423D6CA72.root',
       '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/404CFED1-5C6B-DE11-875E-000423D6CA02.root',
       '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/38212970-586B-DE11-AB61-000423D6CA72.root',
       '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/2CF0A195-566B-DE11-92A5-000423D6B358.root',
       '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/0CD87279-5C6B-DE11-8711-000423D98BC4.root',
       '/store/relval/CMSSW_3_1_1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V2-v1/0002/0279970F-606B-DE11-89A8-001D09F2438A.root'
]
))

process.load("SimTracker.TrackAssociation.TrackAssociatorByChi2_cfi")
process.load("SimTracker.TrackAssociation.quickTrackAssociatorByHits_cfi")

process.load("Validation.RecoTrack.cuts_cff")

process.load("Validation.RecoTrack.TrackerSeedValidator_cff")
#process.multiTrackValidator.associators = cms.vstring('quickTrackAssociatorByHits','TrackAssociatorByChi2')
#process.multiTrackValidator.UseAssociators = True
#process.multiTrackValidator.label = ['cutsRecoTracks']
#process.multiTrackValidator.label_tp_effic = cms.InputTag("cutsTPEffic")
#process.multiTrackValidator.label_tp_fake  = cms.InputTag("cutsTPFake")
#process.multiTrackValidator.associatormap = cms.InputTag(assoc2GsfTracks)
process.trackerSeedValidator.outputFile = 'file.root'

# Tracking Truth and mixing module, if needed
#process.load("SimGeneral.MixingModule.mixNoPU_cfi")
#process.load("SimGeneral.TrackingAnalysis.trackingParticles_cfi")

process.evtInfo = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.siPixelRecHits*process.siStripMatchedRecHits*process.ckftracks*process.cutsTPEffic*process.cutsTPFake*process.trackerSeedValidator)
#process.p = cms.Path(process.multiTrackValidator)
process.ep = cms.EndPath(process.evtInfo)


