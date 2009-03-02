import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("RecoMET.Configuration.CaloTowersOptForMET_cff")

process.load("RecoMET.Configuration.RecoMET_cff")

process.load("RecoMET.Configuration.RecoHTMET_cff")

process.load("RecoMET.Configuration.RecoGenMET_cff")

process.load("RecoMET.Configuration.GenMETParticles_cff")

process.load("RecoMET.Configuration.RecoPFMET_cff")

process.load("RecoJets.Configuration.CaloTowersRec_cff")

process.load("Validation.RecoMET.METRelValForDQM_cff")

process.load("Configuration.StandardSequences.Geometry_cff")

process.load("Configuration.StandardSequences.MagneticField_cff")

process.DQMStore = cms.Service("DQMStore")

process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(10),
    fileNames = cms.untracked.vstring(

    '/store/relval/CMSSW_2_2_4/RelValLM1_sfts/GEN-SIM-RECO/IDEAL_V11_v1/0000/22BCD385-90F3-DD11-80E2-001D09F291D2.root',


    )


)


process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )


process.fileSaver = cms.EDFilter("METFileSaver",
    OutputFile = cms.untracked.string('METTester_data_LM1_sfts.root')
) 
process.p = cms.Path(process.fileSaver*
                     process.genMetTrue*
                     process.genMetCalo*
                     process.genMetCaloAndNonPrompt*
                     process.tcMet*
                     process.METRelValSequence
)
process.schedule = cms.Schedule(process.p)


