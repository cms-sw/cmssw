import FWCore.ParameterSet.Config as cms

process = cms.Process('TrackingTruthPlayback')

# Global conditions
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')

# Playback
process.load("SimGeneral.TrackingAnalysis.Playback_cfi")
# TrackingTruth
process.load("SimGeneral.TrackingAnalysis.trackingParticles_cfi")

process.output = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(
        'drop *_*_*_*',
        'keep *_mix_*_*',
        'keep *_generator_*_*',
        'keep *_randomEngineStateProducer_*_*',
        'keep *_mix__*',
        'keep *_mix_*_*'
    ),
    fileName = cms.untracked.string('file:TrackingTruth.root')
)

process.GlobalTag.globaltag = 'STARTUP31X_V2::All'

process.path = cms.Path(process.mix*process.trackingParticles)
process.outpath = cms.EndPath(process.output)

# Input definition
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)

readFiles.extend( [
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0008/9A0ED789-DA7C-DE11-B6D3-001D09F29619.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0008/38598951-967C-DE11-8578-001D09F24D8A.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/FCAA7A27-557C-DE11-8064-000423D99658.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/F09E1E99-4D7C-DE11-B7F0-001D09F29619.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/D6E18955-527C-DE11-A90C-001D09F24691.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/C851451B-4C7C-DE11-916F-000423D944F0.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/BADEB400-5F7C-DE11-8CFB-000423D98EA8.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/B4E20C35-4D7C-DE11-96C6-000423D95030.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/AC51B347-507C-DE11-BA3E-001D09F29533.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/8E540997-4D7C-DE11-9DE7-000423D6CA42.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/8CAFCA68-4D7C-DE11-B74D-000423D987E0.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/8C80A0DF-4D7C-DE11-B927-000423D98BC4.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/82F85C38-5C7C-DE11-9560-000423D6B444.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/800E8389-717C-DE11-BB29-001D09F251FE.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/748AEF76-4D7C-DE11-8476-000423D60FF6.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/6A74F5A3-6C7C-DE11-8818-001D09F29597.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/6240240D-4D7C-DE11-A4B2-001D09F29619.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/620E14B8-4D7C-DE11-84D7-000423D9517C.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/60CFC2A5-4C7C-DE11-B10F-001D09F2A465.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/5E3D53A2-4E7C-DE11-93B5-001D09F29533.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/52C1668C-4C7C-DE11-9ED5-000423D996C8.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/525343E9-4E7C-DE11-BB53-000423D992A4.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/4E79CA36-4E7C-DE11-B9F4-000423D98B6C.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/4896D74B-4C7C-DE11-8C89-000423D95030.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/46915BAA-4D7C-DE11-982C-000423D98B6C.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/4266283B-517C-DE11-A330-001D09F29597.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/2EE85113-5A7C-DE11-B311-000423D9853C.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/2AD3566E-4F7C-DE11-919E-000423D98F98.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/20C67138-777C-DE11-88DB-000423D6B5C4.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/10E83CB6-587C-DE11-918C-000423D987FC.root',
       '/store/relval/CMSSW_3_2_2/RelValTTbar_Tauola/GEN-SIM-DIGI-RAW-HLTDEBUG/STARTUP31X_V2_156BxLumiPileUp-v1/0007/0AD718DF-517C-DE11-A59E-000423DD2F34.root'
] )


