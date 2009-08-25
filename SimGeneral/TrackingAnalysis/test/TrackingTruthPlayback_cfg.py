import FWCore.ParameterSet.Config as cms

process = cms.Process('TrackingTruthPlayback')

# Global conditions
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')

# Playback
process.load("SimGeneral.TrackingAnalysis.Playback_cfi")
process.load("SimGeneral.MixingModule.mixLowLumPU_mixProdStep1_cfi")
# TrackingTruth
process.load("SimGeneral.TrackingAnalysis.trackingParticles_cfi")

process.output = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(
        'keep *_*_*_*'
    ),
    fileName = cms.untracked.string('file:TrackingTruth.root')
)

process.GlobalTag.globaltag = 'MC_31X_V3::All'

process.path = cms.Path(process.mix*process.trackingParticles)
process.outpath = cms.EndPath(process.output)

# Input definition
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)

readFiles.extend( [
       '/store/relval/CMSSW_3_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V3-v1/0005/A4062B97-D875-DE11-84D3-001D09F28F1B.root',
       '/store/relval/CMSSW_3_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V3-v1/0004/F6216870-DD74-DE11-99B1-001D09F24259.root',
       '/store/relval/CMSSW_3_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V3-v1/0004/DEB45105-D074-DE11-B344-001D09F23A3E.root',
       '/store/relval/CMSSW_3_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V3-v1/0004/D6CB62FE-CF74-DE11-8A59-001D09F2514F.root',
       '/store/relval/CMSSW_3_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V3-v1/0004/D4F69810-DC74-DE11-AA2B-000423D6CA6E.root',
       '/store/relval/CMSSW_3_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V3-v1/0004/CA27533E-D274-DE11-A288-001D09F24DDA.root',
       '/store/relval/CMSSW_3_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V3-v1/0004/B8AC6EC3-C374-DE11-9332-0019B9F72BFF.root',
       '/store/relval/CMSSW_3_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V3-v1/0004/B6EB8449-DB74-DE11-B004-001D09F241F0.root',
       '/store/relval/CMSSW_3_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V3-v1/0004/AE4D54A0-CE74-DE11-8494-001D09F2462D.root',
       '/store/relval/CMSSW_3_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V3-v1/0004/AC007342-7A75-DE11-9DFD-0019B9F581C9.root',
       '/store/relval/CMSSW_3_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V3-v1/0004/9E3A47B6-CF74-DE11-AC45-001D09F29597.root',
       '/store/relval/CMSSW_3_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V3-v1/0004/986766D9-D474-DE11-82A3-0019B9F72D71.root',
       '/store/relval/CMSSW_3_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V3-v1/0004/68A2FE18-D974-DE11-91A9-0019B9F704D6.root',
       '/store/relval/CMSSW_3_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V3-v1/0004/687C7EC5-DC74-DE11-854A-001D09F23A34.root',
       '/store/relval/CMSSW_3_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V3-v1/0004/5C6FC4AF-CF74-DE11-A3F3-0019B9F72BFF.root',
       '/store/relval/CMSSW_3_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V3-v1/0004/56520C4E-DA74-DE11-AB19-000423D944F8.root',
       '/store/relval/CMSSW_3_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V3-v1/0004/4CC14EB1-CF74-DE11-B9BD-001D09F276CF.root',
       '/store/relval/CMSSW_3_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V3-v1/0004/4426FB60-D574-DE11-BDDC-001D09F252DA.root',
       '/store/relval/CMSSW_3_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V3-v1/0004/3216CD7F-CF74-DE11-9FF3-001D09F2514F.root',
       '/store/relval/CMSSW_3_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V3-v1/0004/184E8BBF-DB74-DE11-9718-000423D99F1E.root',
       '/store/relval/CMSSW_3_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V3-v1/0004/0879244F-CF74-DE11-A426-0019B9F72BFF.root',
       '/store/relval/CMSSW_3_2_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V3-v1/0004/00AC40FD-CF74-DE11-9CCF-001D09F2AD7F.root'
] );


