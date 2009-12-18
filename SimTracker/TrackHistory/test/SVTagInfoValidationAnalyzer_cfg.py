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

process.svTagInfoValidationAnalyzer = cms.EDFilter("SVTagInfoValidationAnalyzer",
    process.vertexClassifier,
    svTagInfoProducer = cms.untracked.InputTag('secondaryVertexTagInfos')
)

process.GlobalTag.globaltag = 'IDEAL_31X::All'

process.svTagInfoValidationAnalyzer.enableSimToReco = cms.untracked.bool(True)
process.svTagInfoValidationAnalyzer.vertexProducer = cms.untracked.InputTag('svTagInfoProxy')

process.p = cms.Path(process.playback * process.svTagInfoProxy * process.svTagInfoValidationAnalyzer)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)

readFiles.extend( ( 
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0007/DE732988-5E4F-DE11-82ED-001D09F25208.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0007/82FD1C7A-6E4F-DE11-9198-0019B9F72CC2.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0007/82B99E83-5E4F-DE11-96FA-001D09F28EA3.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0007/345DA7B5-F64E-DE11-8C23-001617DBD556.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0007/22620711-524F-DE11-9D47-001617C3B65A.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0007/1C645B87-5E4F-DE11-A7BB-000423D985E4.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-RECO/IDEAL_31X_v1/0007/14981583-5E4F-DE11-8A22-001D09F253C0.root'
) );

secFiles.extend( (
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/EA6C6D8A-5E4F-DE11-BD34-0030487C6062.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/E8BC81A9-F64E-DE11-8727-000423D99AAA.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/E2AA7EE2-F64E-DE11-A06D-001617C3B6CE.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/C4CB7C79-5E4F-DE11-9EA2-001D09F28F0C.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/B691AE7B-5E4F-DE11-ABF5-001D09F24637.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/A4118E73-F64E-DE11-AE57-000423D6B358.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/8C610683-5E4F-DE11-817D-0030487A1FEC.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/8A108B79-5E4F-DE11-8CD7-001D09F253D4.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/7AE06281-6E4F-DE11-8C85-000423D99896.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/6A2FE089-5E4F-DE11-9BA6-000423D9863C.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/56845B82-5E4F-DE11-B375-000423D6CAF2.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/4824EE85-5E4F-DE11-8F3A-000423D98804.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/44A7AB85-5E4F-DE11-B80A-001D09F24600.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/40302A87-5E4F-DE11-A997-001D09F2AD7F.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/3CADAA7E-5E4F-DE11-B5C7-0019B9F730D2.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/3C7EC0B0-5E4F-DE11-AFB3-001D09F253D4.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/36A89B8D-F64E-DE11-91FF-001617C3B778.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/369D3C85-5E4F-DE11-A2F7-001D09F25456.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/2E42BD9B-524F-DE11-A1CF-000423D6C8EE.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/2ABD6A7A-5E4F-DE11-8F5F-001D09F242EA.root',
       '/store/relval/CMSSW_3_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_31X_v1/0007/24C60B86-5E4F-DE11-8C87-000423D6CA42.root'
) )

