import FWCore.ParameterSet.Config as cms

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [
   '/store/relval/CMSSW_4_2_0_pre6/RelValZTT/GEN-SIM-RECO/START42_V4-v1/0023/267CF021-A645-E011-A05B-00261894391D.root',
   '/store/relval/CMSSW_4_2_0_pre6/RelValZTT/GEN-SIM-RECO/START42_V4-v1/0020/52137829-5945-E011-B764-00261894383F.root',
   '/store/relval/CMSSW_4_2_0_pre6/RelValZTT/GEN-SIM-RECO/START42_V4-v1/0020/24E484D7-5A45-E011-8B00-00261894394B.root'
] )


secFiles.extend( [
   ] )
