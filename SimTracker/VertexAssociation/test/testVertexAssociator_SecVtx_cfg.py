import FWCore.ParameterSet.Config as cms

process = cms.Process("ana")

process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("SimGeneral.TrackingAnalysis.trackingParticles_cfi")
process.load("SimGeneral.TrackingAnalysis.Playback_cfi")
process.load("SimTracker.TrackAssociatorProducers.trackAssociatorByChi2_cfi")
process.load("SimTracker.TrackAssociatorProducers.trackAssociatorByHits_cfi")
process.load("SimTracker.VertexAssociation.VertexAssociatorByTracks_cfi")
process.load("RecoTracker.Configuration.RecoTracker_cff")

process.load("SimTracker.TrackHistory.SecondaryVertexTagInfoProxy_cff")
process.load("SimTracker.TrackHistory.VertexClassifier_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
process.source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [
       '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-RECO/IDEAL_V12_v1/0003/68248864-043E-DE11-86C8-001D09F290BF.root',
       '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-RECO/IDEAL_V12_v1/0003/5EF66575-9E3D-DE11-B71D-001D09F290BF.root',
       '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-RECO/IDEAL_V12_v1/0003/180778D9-9E3D-DE11-8FB0-001D09F24D8A.root',
       '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-RECO/IDEAL_V12_v1/0002/64CE5CCA-9C3D-DE11-8106-001D09F231C9.root',
       '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-RECO/IDEAL_V12_v1/0002/5A518D10-9E3D-DE11-BE13-000423D6CA72.root',
       '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-RECO/IDEAL_V12_v1/0002/00797956-993D-DE11-AAC0-001D09F2A690.root' ] );


secFiles.extend( [
       '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/FC0E794B-9F3D-DE11-8969-000423D6C8E6.root',
       '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/DE8F4018-9E3D-DE11-993A-001D09F2423B.root',
       '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/D88A54EB-9E3D-DE11-95AC-001617DBD230.root',
       '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/D0D136FB-033E-DE11-A44E-001D09F28D4A.root',
       '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/8E1C5431-9E3D-DE11-AF4B-001D09F28F11.root',
       '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/86FC99FF-9D3D-DE11-92AF-001D09F290BF.root',
       '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/7EAEEC8E-9E3D-DE11-8BC3-001D09F231C9.root',
       '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/664A1CAD-9F3D-DE11-95D0-001D09F241B9.root',
       '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/648417C1-9E3D-DE11-A52F-001D09F24682.root',
       '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/58F61F49-9E3D-DE11-9B27-001D09F2523A.root',
       '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/58117FD8-9E3D-DE11-8EEC-001617C3B778.root',
       '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/3A53E076-9E3D-DE11-B98A-001D09F23A84.root',
       '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/389C28A6-9E3D-DE11-843E-001D09F2447F.root',
       '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0003/305A2B75-9E3D-DE11-BFAB-001D09F2423B.root',
       '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0002/EE5B4533-933D-DE11-AD30-001D09F24DA8.root',
       '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0002/EC88F7D0-9A3D-DE11-9836-001617E30E28.root',
       '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0002/E2A462B0-9D3D-DE11-A2B6-001D09F244BB.root',
       '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0002/CC9E91FC-933D-DE11-972F-001D09F25109.root',
       '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0002/B845EA9A-9B3D-DE11-A9F9-001617C3B6FE.root',
       '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0002/B67E5CE0-9D3D-DE11-83F1-001D09F291D2.root',
       '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0002/92912A15-9D3D-DE11-B3C4-001D09F24448.root',
       '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0002/749492B7-993D-DE11-9FBF-001617E30F50.root',
       '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0002/706DA2E3-923D-DE11-97DA-001D09F241B4.root',
       '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0002/6CDD71F8-973D-DE11-A993-001D09F297EF.root',
       '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0002/6694F56B-9A3D-DE11-95EA-001D09F291D7.root',
       '/store/relval/CMSSW_2_2_10/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG/IDEAL_V12_v1/0002/2255EDC6-9D3D-DE11-A02F-001D09F24D8A.root'] );



process.TFileService = cms.Service("TFileService",
    fileName = cms.string("testVertexAssociator_SecVtx_2210_2210.root")
)


process.testanalyzer = cms.EDAnalyzer("testVertexAssociator",
    process.vertexClassifier,
    vertexCollection = cms.untracked.InputTag('svTagInfoProxy'),
)

process.p = cms.Path( process.mix * process.trackingParticles * process.svTagInfoProxy * process.trackAssociatorByChi2 * process.trackAssociatorByHits * process.vertexAssociatorSequence * process.testanalyzer )





