# Runs PFBenchmarkAnalyzer and PFJetBenchmark on PFJet sample to
# monitor performance of PFJets

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("DQMServices.Core.DQM_cfg")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'IDEAL_30X::All'
process.load("RecoTauTag.Configuration.RecoPFTauTag_cff")
process.load("Configuration.StandardSequences.GeometryPilot2_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")

  
process.source = cms.Source("PoolSource",
<<<<<<< benchmark_cfg.py
#                            fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/user/g/gennai/CMSSW_310_pre2/ZTT_fastsim.root'
#                            fileNames = cms.untracked.vstring('file:/tmp/ZTT_fastsim.root'
#                            fileNames = cms.untracked.vstring('file:aodNoMaterialEffect.root'
                                                                fileNames = cms.untracked.vstring(
#                            'file:aod.root'


    'file:aod_ZTT_Full.root'



    #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-RECO/IDEAL_30X_v2/0001/16D8CD2E-B309-DE11-B931-001617E30E28.root',
    #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-RECO/IDEAL_30X_v2/0001/28717F85-B409-DE11-8748-001617E30CD4.root',
    #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-RECO/IDEAL_30X_v2/0001/34CF5413-AB09-DE11-B705-000423D99996.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-RECO/IDEAL_30X_v2/0001/4C8F841A-AB09-DE11-BA62-000423D9880C.root',
    #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-RECO/IDEAL_30X_v2/0001/52EA62D2-B309-DE11-8DC2-001617E30F58.root',
    #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-RECO/IDEAL_30X_v2/0001/56EFC32B-B409-DE11-A4D5-001617E30D06.root',
    #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-RECO/IDEAL_30X_v2/0001/603B2F47-B409-DE11-BDED-000423DD2F34.root',
    #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-RECO/IDEAL_30X_v2/0001/6CCFD31A-B509-DE11-87FA-000423D98950.root',
    #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-RECO/IDEAL_30X_v2/0001/AC35C63B-C409-DE11-84E3-001617C3B66C.root',
    #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-RECO/IDEAL_30X_v2/0001/CC7EDEFC-B309-DE11-A718-001617C3B6E8.root'




#'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/0C00AAC1-4107-DE11-B3D9-0019DB29C5FC.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/161CC521-4107-DE11-B459-000423D98834.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/18959BC5-4107-DE11-BD2E-001617C3B78C.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/1CB3088C-4107-DE11-BC8D-000423D94E70.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/1E6245A3-4107-DE11-9A34-000423D986A8.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/26A6538A-4107-DE11-BC52-000423D98EC8.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/2E2FFED0-4107-DE11-A87E-000423D6BA18.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/347F8A8F-4107-DE11-A76C-000423D33970.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/36509CCD-4107-DE11-976E-000423D9997E.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/367284BE-4107-DE11-87BF-000423D98C20.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/3A27008A-4107-DE11-8CB7-000423D94AA8.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/3CE518C0-4107-DE11-B485-001617DBD472.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/427CD2C3-4107-DE11-90A7-001617C3B70E.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/46BBE4AF-4107-DE11-9DFD-000423D94E70.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/54CEA6D6-4107-DE11-B709-000423D9880C.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/58BD61CE-4107-DE11-A176-001617DBCF90.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/5E7CB5C9-4107-DE11-A34F-000423D9863C.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/66603CCA-4107-DE11-9445-000423D986A8.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/6A20E986-4107-DE11-9486-000423D987E0.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/6E27D2C9-4107-DE11-B4C6-001617C3B6C6.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/74F44BA5-4107-DE11-94CF-000423D987E0.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/789DD591-4107-DE11-8ADD-000423D98834.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/8842D1A4-4107-DE11-A106-000423D9880C.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/8A17AC15-4207-DE11-AA1E-000423D99614.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/8C4335C8-4107-DE11-BEF3-000423D33970.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/A04689D5-4107-DE11-84A2-000423D9853C.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/A4A40FCA-4107-DE11-9B7C-000423D6B358.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/A4F63EC6-4107-DE11-8D88-0019DB2F3F9A.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/A8238EC8-4107-DE11-8364-000423D98B28.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/B2E183A1-4107-DE11-A53E-0019DB29C5FC.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/B4B35DC6-4107-DE11-82DC-000423D9870C.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/B6BABD8A-4107-DE11-9B6C-000423D99EEE.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/BC9F7FA1-4107-DE11-9E4D-000423D6B358.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/C06CD9C1-4107-DE11-9D0A-000423D98DD4.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/C4843DC9-4107-DE11-8442-001617C3B76A.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/C4A333A0-4107-DE11-9868-000423D98844.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/CCBDCDCA-4107-DE11-B6E9-001617DBD332.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/CE5A5CCA-4107-DE11-96EF-000423D6CA72.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/CEF16EC1-4107-DE11-A0B9-001617E30D00.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/D29F5311-4207-DE11-B75D-000423D8FA38.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/D41BE48A-4107-DE11-8958-000423D98844.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/DAA761C1-4107-DE11-A559-000423D6B444.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/E4A005C0-4107-DE11-A9FC-000423D94494.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/E680C4C9-4107-DE11-9E35-001617C3B5F4.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/EAD64FC6-4107-DE11-882B-000423D60FF6.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/EEC2C5CE-4107-DE11-BD7B-000423D6A6F4.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/F2AD17AC-4107-DE11-894B-000423D94AA8.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/F8218588-4107-DE11-8073-000423D98C20.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/FC3AB2CA-4107-DE11-B431-001617DBD5AC.root',
        #'/store/relval/CMSSW_3_1_0_pre2/RelValZTT/GEN-SIM-DIGI-RECO/IDEAL_30X_FastSim_v1/0001/FE1B7EBC-4107-DE11-8F33-000423D99EEE.root'

                            ),
  noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck')
=======
                            fileNames = cms.untracked.vstring('rfio:/castor/cern.ch/user/g/gennai/CMSSW_310_pre2/ZTT_fastsim.root' )
                            )
>>>>>>> 1.8


)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(30000)
)


process.load("Validation.RecoParticleFlow.tauBenchmarkGeneric_cff")
<<<<<<< benchmark_cfg.py
process.pfRecoTauProducerHighEfficiency.TrackerSignalConeSize_max = cms.double(0.15)
=======
#process.pfRecoTauProducerHighEfficiency.TrackerSignalConeSize_max = cms.double(0.5)
>>>>>>> 1.8

process.p =cms.Path(
    process.pfRecoTauProducerHighEfficiency + 
    process.tauBenchmarkGeneric
    )


process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('keep *'),
    fileName = cms.untracked.string('tree.root')
)
#process.outpath = cms.EndPath(process.out)

process.load("FWCore.MessageLogger.MessageLogger_cfi")


process.MessageLogger.cerr.FwkReport.reportEvery = 100

