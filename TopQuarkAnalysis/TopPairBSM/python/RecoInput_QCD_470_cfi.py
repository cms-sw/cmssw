# Dataset path /QCDDiJetPt470to600/Summer08_IDEAL_V9_v1/GEN-SIM-RECO

import FWCore.ParameterSet.Config as cms

def RecoInput() :
    
    maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
    readFiles = cms.untracked.vstring()
    secFiles = cms.untracked.vstring()
    source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
    readFiles.extend( (
       '/store/mc/Summer08/QCDDiJetPt470to600/GEN-SIM-RECO/IDEAL_V9_v1/0000/1A3CF22E-F595-DD11-8A28-0019B9E4893C.root',
       '/store/mc/Summer08/QCDDiJetPt470to600/GEN-SIM-RECO/IDEAL_V9_v1/0000/1A73AD50-F595-DD11-9F2E-0019B9E4527A.root',
       '/store/mc/Summer08/QCDDiJetPt470to600/GEN-SIM-RECO/IDEAL_V9_v1/0000/22BE613A-F595-DD11-ACD8-0019B9E4ACE1.root',
       '/store/mc/Summer08/QCDDiJetPt470to600/GEN-SIM-RECO/IDEAL_V9_v1/0000/3E517714-F295-DD11-A3AF-001125C472E4.root',
       '/store/mc/Summer08/QCDDiJetPt470to600/GEN-SIM-RECO/IDEAL_V9_v1/0000/465F18C5-F895-DD11-9E6D-0019B9E7C51D.root',
       '/store/mc/Summer08/QCDDiJetPt470to600/GEN-SIM-RECO/IDEAL_V9_v1/0000/4EF8AB34-F595-DD11-893C-0019B9E7C79F.root',
       '/store/mc/Summer08/QCDDiJetPt470to600/GEN-SIM-RECO/IDEAL_V9_v1/0000/52EA201C-FD95-DD11-9C4C-0019B9E4FD57.root',
       '/store/mc/Summer08/QCDDiJetPt470to600/GEN-SIM-RECO/IDEAL_V9_v1/0000/5EEA9C37-F595-DD11-AF61-0019B9E48B8C.root',
       '/store/mc/Summer08/QCDDiJetPt470to600/GEN-SIM-RECO/IDEAL_V9_v1/0000/600BCE3A-FD95-DD11-A279-00145EDD7971.root',
       '/store/mc/Summer08/QCDDiJetPt470to600/GEN-SIM-RECO/IDEAL_V9_v1/0000/64EED024-F295-DD11-A867-001125C49180.root',
       '/store/mc/Summer08/QCDDiJetPt470to600/GEN-SIM-RECO/IDEAL_V9_v1/0000/6EDE172F-F295-DD11-9334-0019B9E48FFC.root',
       '/store/mc/Summer08/QCDDiJetPt470to600/GEN-SIM-RECO/IDEAL_V9_v1/0000/7EEA911F-FD95-DD11-856A-0019B9E7E112.root',
       '/store/mc/Summer08/QCDDiJetPt470to600/GEN-SIM-RECO/IDEAL_V9_v1/0000/949C2459-F595-DD11-BE17-0019B9E7CD78.root',
       '/store/mc/Summer08/QCDDiJetPt470to600/GEN-SIM-RECO/IDEAL_V9_v1/0000/AED94115-F295-DD11-B586-001125C4664A.root',
       '/store/mc/Summer08/QCDDiJetPt470to600/GEN-SIM-RECO/IDEAL_V9_v1/0000/C63D4F42-F595-DD11-8547-001125C464F6.root',
       '/store/mc/Summer08/QCDDiJetPt470to600/GEN-SIM-RECO/IDEAL_V9_v1/0000/E065C04F-FD95-DD11-948D-00145EDD7381.root'
       ) );


    secFiles.extend( (
        ) )

    return source
