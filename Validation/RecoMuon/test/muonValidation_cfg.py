import FWCore.ParameterSet.Config as cms

processName = "MuonSuite"
process = cms.Process(processName)

readFiles = cms.untracked.vstring()
process.source = cms.Source ("PoolSource",fileNames = readFiles)
readFiles.extend( (
       '/store/relval/CMSSW_2_1_4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V5_v1/0004/2E5CADBF-7B6C-DD11-9888-0019DB29C614.root',
       '/store/relval/CMSSW_2_1_4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V5_v1/0004/32A2CC00-7C6C-DD11-90F6-000423D991F0.root',
       '/store/relval/CMSSW_2_1_4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V5_v1/0004/46C337CA-7B6C-DD11-A32C-000423D94A20.root',
       '/store/relval/CMSSW_2_1_4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V5_v1/0004/508B02CE-7B6C-DD11-B41E-001617E30CD4.root',
       '/store/relval/CMSSW_2_1_4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V5_v1/0004/56CBFDAA-7B6C-DD11-B77A-0016177CA7A0.root',
       '/store/relval/CMSSW_2_1_4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V5_v1/0004/72F743C7-7B6C-DD11-B502-000423D99BF2.root',
       '/store/relval/CMSSW_2_1_4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V5_v1/0004/80C31AE9-7B6C-DD11-A28D-001617C3B76A.root',
       '/store/relval/CMSSW_2_1_4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V5_v1/0004/8A9583F4-7B6C-DD11-81B0-000423D9989E.root',
       '/store/relval/CMSSW_2_1_4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V5_v1/0004/94E00D1F-7C6C-DD11-96CE-000423D98E6C.root',
       '/store/relval/CMSSW_2_1_4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V5_v1/0004/9C9C92DB-7B6C-DD11-AD8D-000423D98844.root',
       '/store/relval/CMSSW_2_1_4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V5_v1/0004/BE9C2BFD-7B6C-DD11-AB18-0019DB29C5FC.root',
       '/store/relval/CMSSW_2_1_4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V5_v1/0004/CA647D04-7C6C-DD11-B80B-000423D98FBC.root',
       '/store/relval/CMSSW_2_1_4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V5_v1/0004/DCCC75D8-7B6C-DD11-9937-000423D991F0.root',
       '/store/relval/CMSSW_2_1_4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V5_v1/0004/E0D32149-7C6C-DD11-BBA3-0019DB29C614.root',
       '/store/relval/CMSSW_2_1_4/RelValZMM/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V5_v1/0004/E463A009-7C6C-DD11-8693-000423D987E0.root'
))
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *', "keep *_MEtoEDMConverter_*_"+processName),
    fileName = cms.untracked.string('validationEDM.root')
)
process.outpath = cms.EndPath(process.out)

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")
process.MEtoEDMConverter_step = cms.Path(process.MEtoEDMConverter)

process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load("Geometry.CommonDetUnit.globalTrackingGeometry_cfi")
process.load("RecoMuon.DetLayers.muonDetLayerGeometry_cfi")
process.GlobalTag.globaltag = "STARTUP_V5::All"

#---- Validation stuffs ----#
## Default validation modules
process.load("Configuration.StandardSequences.Validation_cff")
process.validation_step = cms.Path(process.validation)
## Load muon validation modules
process.load("Validation.RecoMuon.muonValidation_cff")

#process.recoMuonVMuAssoc.outputFileName = 'validationME.root'

process.postMuon_step = cms.Path(process.muonSelector_seq*process.muonAssociation_seq)
process.muonValidation_step = cms.Path(process.muonValidation_seq)

process.schedule = cms.Schedule(process.postMuon_step,
                                process.validation_step,process.muonValidation_step,
                                process.MEtoEDMConverter_step,process.outpath)

