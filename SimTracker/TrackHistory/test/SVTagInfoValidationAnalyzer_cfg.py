import FWCore.ParameterSet.Config as cms

process = cms.Process("SVTagInfoValidationAnalyzer")

# import of standard configurations
process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration/StandardSequences/MixingNoPileUp_cff')
process.load('Configuration/StandardSequences/GeometryExtended_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/Generator_cff')
process.load('Configuration/StandardSequences/VtxSmearedEarly10TeVCollision_cff')
process.load('Configuration/StandardSequences/Sim_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load('Configuration/EventContent/EventContent_cff')

process.load("SimTracker.TrackHistory.Playback_cff")
process.load("SimTracker.TrackHistory.SecondaryVertexTagInfoProxy_cff")
process.load("SimTracker.TrackHistory.VertexClassifier_cff")

process.add_(
  cms.Service("TFileService",
      fileName = cms.string("SVTagInfoValidation.root")
  )
)

process.svTagInfoValidationAnalyzer = cms.EDAnalyzer("SVTagInfoValidationAnalyzer",
    process.vertexClassifier,
    svTagInfoProducer = cms.untracked.InputTag('secondaryVertexTagInfos')
)

process.GlobalTag.globaltag = 'MC_31X_V9::All'

process.svTagInfoValidationAnalyzer.enableSimToReco = cms.untracked.bool(True)
process.svTagInfoValidationAnalyzer.vertexProducer = cms.untracked.InputTag('svTagInfoProxy')

process.p = cms.Path(process.playback * process.svTagInfoProxy * process.svTagInfoValidationAnalyzer)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)

readFiles.extend( ( 
       '/store/relval/CMSSW_3_3_2/RelValTTbar/GEN-SIM-RECO/MC_31X_V9-v2/0000/CC9F5336-C6C7-DE11-9A0D-003048678F8E.root',
       '/store/relval/CMSSW_3_3_2/RelValTTbar/GEN-SIM-RECO/MC_31X_V9-v2/0000/C6F9DB29-5AC8-DE11-9660-00304867BEDE.root',
       '/store/relval/CMSSW_3_3_2/RelValTTbar/GEN-SIM-RECO/MC_31X_V9-v2/0000/C03DFC68-CCC7-DE11-ADE2-003048678FE0.root',
       '/store/relval/CMSSW_3_3_2/RelValTTbar/GEN-SIM-RECO/MC_31X_V9-v2/0000/B6E856D4-C4C7-DE11-A99C-00261894388A.root',
       '/store/relval/CMSSW_3_3_2/RelValTTbar/GEN-SIM-RECO/MC_31X_V9-v2/0000/9C3A0A7A-C5C7-DE11-B26A-003048D42DC8.root',
       '/store/relval/CMSSW_3_3_2/RelValTTbar/GEN-SIM-RECO/MC_31X_V9-v2/0000/5C3F5451-C6C7-DE11-A478-0026189438E9.root'
) );

secFiles.extend( (
       '/store/relval/CMSSW_3_3_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v2/0000/EE030734-C6C7-DE11-B1A0-002618943869.root',
       '/store/relval/CMSSW_3_3_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v2/0000/E4CF40CB-C4C7-DE11-9849-002618943910.root',
       '/store/relval/CMSSW_3_3_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v2/0000/B22CD7C8-C4C7-DE11-8E29-00261894386A.root',
       '/store/relval/CMSSW_3_3_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v2/0000/AE43C873-C5C7-DE11-B0E0-00304867C0EA.root',
       '/store/relval/CMSSW_3_3_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v2/0000/ACD18D3F-C6C7-DE11-9131-001A92810AEA.root',
       '/store/relval/CMSSW_3_3_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v2/0000/AC9CD402-C4C7-DE11-901E-00261894390B.root',
       '/store/relval/CMSSW_3_3_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v2/0000/A0369C37-C6C7-DE11-B5AC-003048D15DDA.root',
       '/store/relval/CMSSW_3_3_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v2/0000/9C66702A-5AC8-DE11-8100-002618943876.root',
       '/store/relval/CMSSW_3_3_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v2/0000/8C68B667-CCC7-DE11-A26F-0018F3D09644.root',
       '/store/relval/CMSSW_3_3_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v2/0000/86539F78-C5C7-DE11-A427-002618943896.root',
       '/store/relval/CMSSW_3_3_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v2/0000/5C0DE4D4-C6C7-DE11-BEBB-00304867C0EA.root',
       '/store/relval/CMSSW_3_3_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v2/0000/5A8660D2-C6C7-DE11-BC6B-002618943896.root',
       '/store/relval/CMSSW_3_3_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v2/0000/349E1C76-C5C7-DE11-AD7E-003048679010.root',
       '/store/relval/CMSSW_3_3_2/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v2/0000/32499275-C5C7-DE11-B681-003048679076.root'
) )

