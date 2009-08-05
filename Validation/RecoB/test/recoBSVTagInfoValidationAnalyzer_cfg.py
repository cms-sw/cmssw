import FWCore.ParameterSet.Config as cms

process = cms.Process("testvalidationanalyzer")

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.MixingNoPileUp_cff')
process.load('Configuration.StandardSequences.GeometryExtended_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('Configuration.StandardSequences.VtxSmearedEarly10TeVCollision_cff')
process.load('Configuration.StandardSequences.Sim_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.EventContent.EventContent_cff')

process.load("SimTracker.TrackHistory.Playback_cff")
process.load("SimTracker.TrackHistory.SecondaryVertexTagInfoProxy_cff")
process.load("SimTracker.TrackHistory.VertexClassifier_cff")

process.load("DQMServices.Components.DQMEnvironment_cfi")
process.load("DQMServices.Core.DQM_cfg")

process.dqmEnv.subSystemFolder = 'BTAG'
process.dqmSaver.producer = 'DQM'
process.dqmSaver.workflow = '/POG/BTAG/SV'
process.dqmSaver.convention = 'Offline'
process.dqmSaver.saveByRun = cms.untracked.int32(-1)
process.dqmSaver.saveAtJobEnd =cms.untracked.bool(True) 
process.dqmSaver.forceRunNumber = cms.untracked.int32(1)

process.svTagInfoValidationAnalyzer = cms.EDFilter("recoBSVTagInfoValidationAnalyzer",
    process.vertexClassifier,
    svTagInfoProducer = cms.untracked.InputTag('secondaryVertexTagInfos')
)

process.GlobalTag.globaltag = 'MC_31X_V1::All'

process.svTagInfoValidationAnalyzer.enableSimToReco = cms.untracked.bool(True)
process.svTagInfoValidationAnalyzer.vertexProducer = cms.untracked.InputTag('svTagInfoProxy')

process.p = cms.Path(process.playback * process.svTagInfoProxy * process.svTagInfoValidationAnalyzer * process.dqmSaver)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)

readFiles.extend( (
            '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-RECO/MC_31X_V1-v1/0001/DE34DE28-C266-DE11-880D-001D09F2924F.root',
        '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-RECO/MC_31X_V1-v1/0001/B08ED361-C766-DE11-BB11-000423D999CA.root',
        '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-RECO/MC_31X_V1-v1/0001/A2AA961C-C366-DE11-ACBB-001D09F24F1F.root',
        '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-RECO/MC_31X_V1-v1/0001/8459B164-DE66-DE11-9A70-001D09F25393.root',
        '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-RECO/MC_31X_V1-v1/0001/60CDDB87-C866-DE11-BD66-001D09F2915A.root',
        '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-RECO/MC_31X_V1-v1/0001/1ADB3A72-C666-DE11-9D34-001D09F2525D.root',
        '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-RECO/MC_31X_V1-v1/0001/101B03DF-C466-DE11-ABE1-001D09F24934.root'

            ) );

secFiles.extend( (

            '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/F81AA535-C666-DE11-942A-001D09F24600.root',
        '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/F45A3761-C766-DE11-8274-001D09F24FBA.root',
        '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/ECDD6402-C466-DE11-AD8D-000423D99A8E.root',
        '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/D0B6652D-C266-DE11-A7A6-001D09F24600.root',
        '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/CA895E96-DE66-DE11-8768-001D09F248FD.root',
        '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/B4FF6350-C466-DE11-BB33-001D09F24DA8.root',
        '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/A80A52B0-C266-DE11-8A5A-001D09F25041.root',
        '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/A6C8A82A-C266-DE11-8704-001D09F23A6B.root',
        '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/A6350A56-C866-DE11-B573-001D09F24FBA.root',
        '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/A4C58176-C566-DE11-ADC0-001D09F28D4A.root',
        '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/A2C1AC27-C266-DE11-9667-001D09F2983F.root',
        '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/8AE98AF2-C766-DE11-B315-001D09F26C5C.root',
        '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/88F81419-C966-DE11-8481-001D09F24024.root',
        '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/5EC0F22B-C266-DE11-A2DA-001D09F23A61.root',
        '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/4A872C1B-C366-DE11-A844-001D09F25041.root',
        '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/32CED660-C766-DE11-B873-001D09F28F11.root',
        '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/308CF886-C866-DE11-95C7-001D09F28755.root',
        '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/2A4FA6DE-C466-DE11-A598-000423D99E46.root',
        '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/28327168-C666-DE11-9486-000423D99EEE.root',
        '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/24FF0D62-CB66-DE11-8F1F-001D09F24DA8.root',
        '/store/relval/CMSSW_3_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V1-v1/0001/14AEBDFE-C666-DE11-AA23-001D09F28755.root'
            ) )

