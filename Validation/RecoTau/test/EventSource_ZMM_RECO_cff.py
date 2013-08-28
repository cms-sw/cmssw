import FWCore.ParameterSet.Config as cms

maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [
   '/store/relval/CMSSW_3_1_0_pre10/RelValZMM/GEN-SIM-RECO/STARTUP_31X_v1/0008/6ABB6AD6-E357-DE11-8EBC-001D09F2437B.root',
   '/store/relval/CMSSW_3_1_0_pre10/RelValZMM/GEN-SIM-RECO/STARTUP_31X_v1/0008/4C1E17B0-0458-DE11-A15D-001D09F26509.root',
   '/store/relval/CMSSW_3_1_0_pre10/RelValZMM/GEN-SIM-RECO/STARTUP_31X_v1/0008/36A83DC8-E357-DE11-B8DC-000423D996B4.root' ] );


secFiles.extend( [
   ] )


