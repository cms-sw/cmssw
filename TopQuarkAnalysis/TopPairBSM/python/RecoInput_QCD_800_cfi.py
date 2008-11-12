# Dataset path /QCDDiJetPt800to1000/Summer08_IDEAL_V9_v1/GEN-SIM-RECO

import FWCore.ParameterSet.Config as cms

def RecoInput() :

    maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
    readFiles = cms.untracked.vstring()
    secFiles = cms.untracked.vstring() 
    source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
    readFiles.extend( (
        '/store/mc/Summer08/QCDDiJetPt800to1000/GEN-SIM-RECO/IDEAL_V9_v1/0006/08043F8E-24A0-DD11-8F7B-001EC9ED88D8.root',
        '/store/mc/Summer08/QCDDiJetPt800to1000/GEN-SIM-RECO/IDEAL_V9_v1/0006/1849B37F-23A0-DD11-8597-00145ED6E7C8.root',
        '/store/mc/Summer08/QCDDiJetPt800to1000/GEN-SIM-RECO/IDEAL_V9_v1/0006/3215635E-22A0-DD11-9996-0030487C1154.root',
        '/store/mc/Summer08/QCDDiJetPt800to1000/GEN-SIM-RECO/IDEAL_V9_v1/0006/3E0B8639-27A0-DD11-990D-001EC9ED7E46.root',
        '/store/mc/Summer08/QCDDiJetPt800to1000/GEN-SIM-RECO/IDEAL_V9_v1/0006/3ED29688-E6A0-DD11-A2A5-001EC9ED88D8.root',
        '/store/mc/Summer08/QCDDiJetPt800to1000/GEN-SIM-RECO/IDEAL_V9_v1/0006/700A2B80-23A0-DD11-A73E-001EC9ED8F2B.root',
        '/store/mc/Summer08/QCDDiJetPt800to1000/GEN-SIM-RECO/IDEAL_V9_v1/0006/ACA05E81-26A0-DD11-97F2-003048C26CB6.root',
        '/store/mc/Summer08/QCDDiJetPt800to1000/GEN-SIM-RECO/IDEAL_V9_v1/0006/AE8F64DA-30A0-DD11-B95F-0015C5E5B335.root',
        '/store/mc/Summer08/QCDDiJetPt800to1000/GEN-SIM-RECO/IDEAL_V9_v1/0006/D017F01E-2FA0-DD11-8F66-00192165CCB4.root',
        '/store/mc/Summer08/QCDDiJetPt800to1000/GEN-SIM-RECO/IDEAL_V9_v1/0006/F2BF165E-22A0-DD11-B63E-0030487C1154.root',
        '/store/mc/Summer08/QCDDiJetPt800to1000/GEN-SIM-RECO/IDEAL_V9_v1/0008/B4566DA1-05A2-DD11-976A-001D09645B69.root',
        '/store/mc/Summer08/QCDDiJetPt800to1000/GEN-SIM-RECO/IDEAL_V9_v1/0009/8EC9BC10-8BA2-DD11-8CB5-001D09645A9D.root'
        ) );
    
    secFiles.extend( (
        ) )

    return source
