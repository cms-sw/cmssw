# Dataset path /QCDDiJetPt600to800/Summer08_IDEAL_V9_v1/GEN-SIM-RECO

import FWCore.ParameterSet.Config as cms

def RecoInput() :

    maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
    readFiles = cms.untracked.vstring()
    secFiles = cms.untracked.vstring() 
    source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
    readFiles.extend( ( 
        '/store/mc/Summer08/QCDDiJetPt600to800/GEN-SIM-RECO/IDEAL_V9_v1/0000/181EF424-3196-DD11-8120-0019B9E494F3.root',
        '/store/mc/Summer08/QCDDiJetPt600to800/GEN-SIM-RECO/IDEAL_V9_v1/0000/2ECFCD0B-3196-DD11-989E-0019B9E4963E.root',
        '/store/mc/Summer08/QCDDiJetPt600to800/GEN-SIM-RECO/IDEAL_V9_v1/0000/40E116F5-FD95-DD11-A956-001125C49152.root',
        '/store/mc/Summer08/QCDDiJetPt600to800/GEN-SIM-RECO/IDEAL_V9_v1/0000/5CA3BB61-F595-DD11-859C-0019B9E495A4.root',
        '/store/mc/Summer08/QCDDiJetPt600to800/GEN-SIM-RECO/IDEAL_V9_v1/0000/6A0853CB-EF95-DD11-830F-001125C46418.root',
        '/store/mc/Summer08/QCDDiJetPt600to800/GEN-SIM-RECO/IDEAL_V9_v1/0000/7CE76841-F595-DD11-B927-0019B9E48991.root',
        '/store/mc/Summer08/QCDDiJetPt600to800/GEN-SIM-RECO/IDEAL_V9_v1/0000/88C56302-3196-DD11-9E86-0019B9E4FFFF.root',
        '/store/mc/Summer08/QCDDiJetPt600to800/GEN-SIM-RECO/IDEAL_V9_v1/0000/96CF8C67-EF95-DD11-8A51-001125C4910A.root',
        '/store/mc/Summer08/QCDDiJetPt600to800/GEN-SIM-RECO/IDEAL_V9_v1/0000/9E988C1E-F995-DD11-80BB-001D0966E1E9.root',
        '/store/mc/Summer08/QCDDiJetPt600to800/GEN-SIM-RECO/IDEAL_V9_v1/0000/A061971A-3196-DD11-833A-001125C48EE4.root',
        '/store/mc/Summer08/QCDDiJetPt600to800/GEN-SIM-RECO/IDEAL_V9_v1/0000/A414B974-EF95-DD11-9E7E-0019B9E50117.root',
        '/store/mc/Summer08/QCDDiJetPt600to800/GEN-SIM-RECO/IDEAL_V9_v1/0000/AC8F2845-F595-DD11-A709-0019B9E4896E.root',
        '/store/mc/Summer08/QCDDiJetPt600to800/GEN-SIM-RECO/IDEAL_V9_v1/0000/B69CB9F7-3096-DD11-98F7-001D0967D896.root',
        '/store/mc/Summer08/QCDDiJetPt600to800/GEN-SIM-RECO/IDEAL_V9_v1/0000/D0683153-F595-DD11-AC75-0019B9E49600.root',
        '/store/mc/Summer08/QCDDiJetPt600to800/GEN-SIM-RECO/IDEAL_V9_v1/0000/D8EF28E1-3096-DD11-8B24-0019B9E4FE56.root',
        '/store/mc/Summer08/QCDDiJetPt600to800/GEN-SIM-RECO/IDEAL_V9_v1/0000/EA4BD642-F595-DD11-9676-0019B9E48FC0.root',
        '/store/mc/Summer08/QCDDiJetPt600to800/GEN-SIM-RECO/IDEAL_V9_v1/0000/FE455340-EF95-DD11-A054-0019B9E4B150.root') );

    secFiles.extend( (
        ) )

    return source
