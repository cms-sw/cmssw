import FWCore.ParameterSet.Config as cms

maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [
    '/store/relval/CMSSW_3_6_0_pre2/RelValZTT/GEN-SIM-RECO/START3X_V24-v1/0001/C27387A2-6F27-DF11-A4D5-002618943880.root',
    '/store/relval/CMSSW_3_6_0_pre2/RelValZTT/GEN-SIM-RECO/START3X_V24-v1/0000/EE23E60D-1A27-DF11-AD51-002618943920.root',
    '/store/relval/CMSSW_3_6_0_pre2/RelValZTT/GEN-SIM-RECO/START3X_V24-v1/0000/2AB69378-1A27-DF11-BBB4-002354EF3BDA.root',
    '/store/relval/CMSSW_3_6_0_pre2/RelValZTT/GEN-SIM-RECO/START3X_V24-v1/0000/22EC2405-1A27-DF11-8F83-002618943943.root',
    '/store/relval/CMSSW_3_6_0_pre2/RelValZTT/GEN-SIM-RECO/START3X_V24-v1/0000/1C3D776D-1927-DF11-A6FC-002618943894.root' 
] )



secFiles.extend( [
   ] )

