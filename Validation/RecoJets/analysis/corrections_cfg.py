import FWCore.ParameterSet.Config as cms

from JetMETCorrections.Configuration.JetCorrectionsRecord_cfi import *
from RecoJets.Configuration.RecoJetAssociations_cff import *

process = cms.Process("CORRECTION")

process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.FakeConditions_cff")
process.load("Configuration.StandardSequences.Simulation_cff")
process.load("Configuration.StandardSequences.MixingNoPileUp_cff")
process.load("Configuration.StandardSequences.VtxSmearedGauss_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("JetMETCorrections.Configuration.JetPlusTrackCorrections_cff")
process.load("JetMETCorrections.Configuration.ZSPJetCorrections152_cff")

#process.load("DQMServices.Core.DQM_cfg")

process.maxEvents = cms.untracked.PSet(
       input = cms.untracked.int32(400)
)

process.source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(0),

    fileNames = cms.untracked.vstring(

        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/00FBAA98-DE99-DD11-A673-000423D9989E.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/027ED180-DC99-DD11-BB34-000423D98B5C.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/06D6826F-DB99-DD11-85BB-000423D99B3E.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/06F11A6E-DE99-DD11-866F-000423D99AA2.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/088F0B55-DA99-DD11-AEBC-000423D94A20.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/1088DCA4-D699-DD11-B055-001617E30F4C.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/108D512F-D299-DD11-9D43-000423D94E70.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/10CD53E8-DB99-DD11-A1B7-000423D94AA8.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/146BF925-CA99-DD11-BEFB-000423D6CA6E.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/14F8BD5C-D399-DD11-B41A-000423D98750.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/18636BB6-C699-DD11-B83F-001617C3B710.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/1A2DAE11-DF99-DD11-8393-001617DBD230.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/1AC0E0C0-C999-DD11-9A22-001617E30F48.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/1E9DC0CE-CD99-DD11-9639-001617C3B77C.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/1EA3245C-D199-DD11-B9AE-001617E30F56.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/20B007A2-D599-DD11-9853-0016177CA7A0.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/26B979BD-CA99-DD11-B319-001617E30D40.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/2AE4168A-DC99-DD11-9C35-000423D98BC4.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/2CD5C6EE-D999-DD11-A0CD-000423D95220.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/2E8F685E-DA99-DD11-AEA7-000423D98834.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/30632163-D399-DD11-947E-001617C3B76E.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/3CEC7E31-DB99-DD11-B8EC-001617E30CD4.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/427AFC6F-DB99-DD11-B300-000423D944F0.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/42C8B8CE-D699-DD11-8B80-000423D9890C.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/48AEF2A9-D599-DD11-BFEF-001617C3B70E.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/4ACAD56D-DE99-DD11-B245-000423D94700.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/4E521965-DB99-DD11-BA0C-0019DB29C620.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/523BD650-D799-DD11-94BC-000423D6A6F4.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/52F47187-D399-DD11-89DF-0019DB29C620.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/5E3867F8-E099-DD11-A019-001617E30D4A.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/725BCA20-D399-DD11-89A1-001617E30CE8.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/74E6EE1D-D999-DD11-8072-001617E30D38.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/76697838-D899-DD11-8975-000423D6C8EE.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/7A9C767C-DC99-DD11-A5FD-000423D987E0.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/7E0D98EE-E299-DD11-9424-000423D8F63C.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/8418EA0A-D799-DD11-B0E2-001617E30F4C.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/8E958696-D299-DD11-847B-001617DBD332.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/9A398A8B-CA99-DD11-B252-000423D98B28.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/9C4411DA-D399-DD11-B829-001617C3B78C.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/A00880E9-D399-DD11-875E-000423D9880C.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/A0433B9D-DE99-DD11-BC11-000423D985E4.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/A2B2D820-D499-DD11-B1F2-000423D98E54.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/A6C33B8E-DF99-DD11-9682-001617C3B78C.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/A6D3FAC3-D099-DD11-89CE-000423D98950.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/AC06E9C0-D499-DD11-A1CF-001617E30D38.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/ACEFD2C1-D899-DD11-8DFD-001617C3B70E.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/AE472DA7-DA99-DD11-A753-000423D991D4.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/AE7789B7-D499-DD11-929D-001617C3B6C6.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/B0E13D99-D299-DD11-B04B-000423D986A8.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/B24D51A3-C999-DD11-A01F-000423D99996.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/B6966090-DA99-DD11-92EB-000423D6A6F4.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/B8502922-C999-DD11-9BB7-000423D6B5C4.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/B8ECE720-DE99-DD11-8157-000423D98B08.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/C00D9D7C-DB99-DD11-9A77-000423D987E0.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/C44A4E1E-CA99-DD11-B3BC-001617E30F4C.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/C44BDA20-DB99-DD11-A1FF-000423D98C20.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/C6D30954-D599-DD11-B2CB-000423D9853C.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/CAD46A62-DE99-DD11-AD9B-001617C3B79A.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/CC73E83D-DA99-DD11-ACB9-001617C3B6FE.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/CCAEF392-D099-DD11-9E29-000423D99BF2.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/D0B11263-DA99-DD11-98E2-000423D94AA8.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/D2226245-E099-DD11-A010-000423D98EA8.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/D617126F-CC99-DD11-8847-000423D98E30.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/D83E0AFA-D999-DD11-BD0E-000423D986C4.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/DA23BBE7-E099-DD11-988C-001617C3B5D8.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/DC1C7A8F-CA99-DD11-8468-000423D992DC.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/E47ACE30-D399-DD11-92D0-000423D6B48C.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/E488C61E-D599-DD11-99B6-001617C3B76E.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/EAA114A1-D199-DD11-899C-000423D98B28.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/F4A1DE37-FD99-DD11-B3DF-00161757BF42.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/F4BE5A83-C899-DD11-9E33-001617C3B6E2.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/F6F4BFC0-D499-DD11-92EE-000423D94E70.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/FCFBB4F7-D799-DD11-9649-000423D99B3E.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/FE550CF1-DA99-DD11-A074-000423D99E46.root',
        '/store/relval/CMSSW_2_1_10/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v2/0000/FEF84636-DB99-DD11-8441-000423D99896.root'
        
    )
)


process.out = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('p1')
    ),                               
        outputCommands = cms.untracked.vstring('drop *',
                                               'keep recoCaloMETs_*_*_*',
                                               'keep *_CaloMETCollection_*_*',
                                               'keep CaloTowersSorted_*_*_*',
                                               'keep *_iterativeCone5CaloJets_*_*',
                                               'keep *_JetPlusTrackZSPCorJetIcone5_*_*',
                                               'keep *_iterativeCone5GenJets_*_*',
                                               'keep *_iterativeCone5PFJets_*_*',  
                                               'keep *_kt4CaloJets_*_*',
                                               'keep *_kt4GenJets_*_*',
                                               'keep *_kt6CaloJets_*_*',
                                               'keep *_kt6GenJets_*_*',
                                               'keep *_sisCone5CaloJets_*_*',
                                               'keep *_sisCone5GenJets_*_*',
                                               'keep *_sisCone7CaloJets_*_*',
                                               'keep *_sisCone7GenJets_*_*',
                                               'keep *_L2L3CorJet_*_*'
                                               #      'keep *__*_*',
),                               
    fileName = cms.untracked.string('Corr_QCD_80_120_STARTUP_V7_v2.root')
)

process.L2JetCorrector = cms.ESSource("L2RelativeCorrectionService", 
    tagName = cms.string('iCSA08_S156_L2Relative_Icone5'),
    label = cms.string('L2RelativeJetCorrector')
)

process.L3JetCorrector = cms.ESSource("L3AbsoluteCorrectionService", 
    tagName = cms.string('iCSA08_S156_L3Absolute_Icone5'),
    label = cms.string('L3AbsoluteJetCorrector')
)

#process.L5JetCorrector = cms.ESSource("L5FlavorCorrectionService",
#    section = cms.string('uds'),  
#    tagName = cms.string('L5Flavor_fromQCD_iterativeCone5'),
#    label = cms.string('L5FlavorJetCorrector')
#)

#process.L7JetCorrector = cms.ESSource("L7PartonCorrectionService", 
#    section = cms.string('qJ'),
#    tagName = cms.string('L7parton_IC5_080301'),
#    label = cms.string('L7PartonJetCorrector')
#)


#process.L2CorJetIcone5 = cms.EDProducer("CaloJetCorrectionProducer",
#    src = cms.InputTag("iterativeCone5CaloJets"),
##    correctors = cms.vstring('L2JetCorrectorIcone5')
#    correctors = cms.vstring('L2RelativeJetCorrector')
#)

process.L2L3JetCorrector = cms.ESSource("JetCorrectionServiceChain",  
    correctors = cms.vstring('L2RelativeJetCorrector','L3AbsoluteJetCorrector'),
    label = cms.string('L2L3JetCorrector')
)

process.prefer("L2L3JetCorrector")

process.L2L3CorJet = cms.EDProducer("CaloJetCorrectionProducer",
    src = cms.InputTag("iterativeCone5CaloJets"),
    correctors = cms.vstring('L2L3JetCorrector')
)

process.p1 = cms.Path(process.L2L3CorJet*process.ZSPJetCorrections*process.JetPlusTrackCorrections)

process.p = cms.EndPath(process.out)

