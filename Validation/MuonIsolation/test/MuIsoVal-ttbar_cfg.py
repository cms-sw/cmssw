# The following comments couldn't be translated into the new config version:

#  Obsolete service from 1_x_x
#	service = DaqMonitorROOTBackEnd{}
#  Replaced by

#	muIsoDepositCalByAssociatorTowers,

import FWCore.ParameterSet.Config as cms

process = cms.Process("A")
process.load("RecoMuon.MuonIsolationProducers.muIsoDeposits_cff")

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")

process.load("MagneticField.Engine.volumeBasedMagneticField_cfi")

process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")

process.load("Geometry.CommonDetUnit.globalTrackingGeometry_cfi")

process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2000)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('XXXX', 
        'XXXX', 
        'XXXX')
)

process.DQMStore = cms.Service("DQMStore")

process.analyzer_incMuon = cms.EDFilter("MuIsoValidation",
    Global_Muon_Label = cms.untracked.InputTag("muons"),
    requireCombinedMuon = cms.untracked.bool(False),
    ecalIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositCalByAssociatorTowers","ecal"),
    rootfilename = cms.untracked.string('ttbar-validation.root'),
    hcalIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositCalByAssociatorTowers","hcal"),
    tkIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositTk"),
    hoIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositCalByAssociatorTowers","ho")
)

process.analyzer_combinedMuon = cms.EDFilter("MuIsoValidation",
    Global_Muon_Label = cms.untracked.InputTag("muons"),
    requireCombinedMuon = cms.untracked.bool(True),
    ecalIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositCalByAssociatorTowers","ecal"),
    rootfilename = cms.untracked.string('ttbar-validation.root'),
    hcalIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositCalByAssociatorTowers","hcal"),
    tkIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositTk"),
    hoIsoDeposit_Label = cms.untracked.InputTag("muIsoDepositCalByAssociatorTowers","ho")
)

process.p = cms.Path(process.analyzer_incMuon+process.analyzer_combinedMuon)
process.PoolSource.fileNames = ['/store/relval/CMSSW_2_1_10_patch1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0000/06E18608-2A9E-DD11-BCB2-000423D98E54.root',
                                '/store/relval/CMSSW_2_1_10_patch1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0000/0CCBAF8A-2A9E-DD11-B657-000423D985E4.root',
                                '/store/relval/CMSSW_2_1_10_patch1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0000/101EF90B-2C9E-DD11-9567-001617C3B76E.root',
                                '/store/relval/CMSSW_2_1_10_patch1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0000/108E89E1-289E-DD11-AEA9-000423D6B5C4.root',
                                '/store/relval/CMSSW_2_1_10_patch1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0000/389B16CB-2C9E-DD11-8641-0019DB29C620.root',
                                '/store/relval/CMSSW_2_1_10_patch1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0000/408839D1-2C9E-DD11-84F5-000423D987E0.root',
                                '/store/relval/CMSSW_2_1_10_patch1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0000/4A41EF2C-2B9E-DD11-BFA1-000423D98BE8.root',
                                '/store/relval/CMSSW_2_1_10_patch1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0000/4A9BE29E-2A9E-DD11-A538-000423D9997E.root',
                                '/store/relval/CMSSW_2_1_10_patch1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0000/60C01A9D-2C9E-DD11-8F91-000423D9863C.root',
                                '/store/relval/CMSSW_2_1_10_patch1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0000/7A41E9C9-2C9E-DD11-B79E-001617C3B65A.root',
                                '/store/relval/CMSSW_2_1_10_patch1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0000/7C76340C-2C9E-DD11-847E-0019DB29C620.root',
                                '/store/relval/CMSSW_2_1_10_patch1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0000/80DCBF48-2C9E-DD11-AF1C-001617DBCF1E.root',
                                '/store/relval/CMSSW_2_1_10_patch1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0000/82DC6925-2B9E-DD11-9B0A-000423D6B5C4.root',
                                '/store/relval/CMSSW_2_1_10_patch1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0000/888CC87E-2B9E-DD11-A6A3-001617DBD224.root',
                                '/store/relval/CMSSW_2_1_10_patch1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0000/8A6E8061-2A9E-DD11-8751-000423D98EA8.root',
                                '/store/relval/CMSSW_2_1_10_patch1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0000/9285EBE5-2A9E-DD11-B7AC-001617DBD224.root',
                                '/store/relval/CMSSW_2_1_10_patch1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0000/98B61F88-279E-DD11-BBC3-001617E30CA4.root',
                                '/store/relval/CMSSW_2_1_10_patch1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0000/9ED73FD0-339E-DD11-84D7-000423D99F3E.root',
                                '/store/relval/CMSSW_2_1_10_patch1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0000/A817DAFA-299E-DD11-BF85-000423D6B5C4.root',
                                '/store/relval/CMSSW_2_1_10_patch1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0000/A89BFE61-299E-DD11-B9E6-000423D6B5C4.root',
                                '/store/relval/CMSSW_2_1_10_patch1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0000/B6CCCDE8-2A9E-DD11-B133-000423D6B358.root',
                                '/store/relval/CMSSW_2_1_10_patch1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0000/C2251582-2B9E-DD11-8106-001617E30CA4.root',
                                '/store/relval/CMSSW_2_1_10_patch1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0000/C2E21928-2D9E-DD11-8E14-001617DBCF1E.root',
                                '/store/relval/CMSSW_2_1_10_patch1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0000/CC74C82A-2B9E-DD11-8952-000423D6CA6E.root',
                                '/store/relval/CMSSW_2_1_10_patch1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0000/D23E01B3-2B9E-DD11-9661-001617C3B65A.root',
                                '/store/relval/CMSSW_2_1_10_patch1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0000/D650AF11-2C9E-DD11-9914-001617C3B654.root',
                                '/store/relval/CMSSW_2_1_10_patch1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0000/D68509C6-299E-DD11-98C2-001617E30CC8.root',
                                '/store/relval/CMSSW_2_1_10_patch1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0000/EE71F890-2A9E-DD11-870B-000423D98F98.root',
                                '/store/relval/CMSSW_2_1_10_patch1/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0000/F875CD2F-2B9E-DD11-AC1A-000423D99AA2.root']

