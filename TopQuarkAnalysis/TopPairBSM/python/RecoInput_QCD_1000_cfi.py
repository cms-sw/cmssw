# Dataset path /QCDDiJetPt1000to14000/Summer08_IDEAL_V9_v1/GEN-SIM-RECO

import FWCore.ParameterSet.Config as cms

def RecoInput() :

    maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
    readFiles = cms.untracked.vstring()
    secFiles = cms.untracked.vstring() 
    source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
    readFiles.extend( (
        '/store/mc/Summer08/QCDDiJetPt1000to14000/GEN-SIM-RECO/IDEAL_V9_v1/0005/948F4772-C29E-DD11-A149-001EC9ED4FAA.root',
        '/store/mc/Summer08/QCDDiJetPt1000to14000/GEN-SIM-RECO/IDEAL_V9_v1/0006/96E4C3A6-379F-DD11-AC24-0019B9E4FCA3.root',
        '/store/mc/Summer08/QCDDiJetPt1000to14000/GEN-SIM-RECO/IDEAL_V9_v1/0006/D0B352A5-379F-DD11-B0FC-0019B9E4FC5D.root',
        '/store/mc/Summer08/QCDDiJetPt1000to14000/GEN-SIM-RECO/IDEAL_V9_v1/0006/F4E391CC-369F-DD11-B8CF-0019B9E7C4D2.root',
        '/store/mc/Summer08/QCDDiJetPt1000to14000/GEN-SIM-RECO/IDEAL_V9_v1/0009/94581C84-D2A2-DD11-AB01-00215AA62C2A.root',
        '/store/mc/Summer08/QCDDiJetPt1000to14000/GEN-SIM-RECO/IDEAL_V9_v1/0009/9A39A082-D2A2-DD11-A755-00215A45F86A.root',
        '/store/mc/Summer08/QCDDiJetPt1000to14000/GEN-SIM-RECO/IDEAL_V9_v1/0009/C20F7A87-D2A2-DD11-A989-0022640631AE.root',
        '/store/mc/Summer08/QCDDiJetPt1000to14000/GEN-SIM-RECO/IDEAL_V9_v1/0009/C28AB37D-B9A2-DD11-98F9-00221981B410.root',
        '/store/mc/Summer08/QCDDiJetPt1000to14000/GEN-SIM-RECO/IDEAL_V9_v1/0009/E2BE2986-D2A2-DD11-BE4A-00215A4909F6.root',
        '/store/mc/Summer08/QCDDiJetPt1000to14000/GEN-SIM-RECO/IDEAL_V9_v1/0010/54204960-E7A2-DD11-A964-0015C5EC47A2.root',
        '/store/mc/Summer08/QCDDiJetPt1000to14000/GEN-SIM-RECO/IDEAL_V9_v1/0010/8C2E8B5F-E7A2-DD11-81D6-0015C5E5B288.root',
        '/store/mc/Summer08/QCDDiJetPt1000to14000/GEN-SIM-RECO/IDEAL_V9_v1/0010/A64BF360-E7A2-DD11-B221-0015C5EC47A2.root',
        '/store/mc/Summer08/QCDDiJetPt1000to14000/GEN-SIM-RECO/IDEAL_V9_v1/0010/A8950B61-E7A2-DD11-AB5F-0015C5E5B335.root',
        '/store/mc/Summer08/QCDDiJetPt1000to14000/GEN-SIM-RECO/IDEAL_V9_v1/0010/B8271362-E7A2-DD11-B1B4-0015C5E5B335.root',
        '/store/mc/Summer08/QCDDiJetPt1000to14000/GEN-SIM-RECO/IDEAL_V9_v1/0010/EA535161-E7A2-DD11-8C3E-0015C5E5B335.root',
        '/store/mc/Summer08/QCDDiJetPt1000to14000/GEN-SIM-RECO/IDEAL_V9_v1/0010/FE1F5584-D6A2-DD11-A5EB-001EC9ED840F.root'
        ) );
    
    secFiles.extend( (
        ) )

    return source
