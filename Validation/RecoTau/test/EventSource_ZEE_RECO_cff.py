import FWCore.ParameterSet.Config as cms

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [
        'file:./Samples/RelValZEE/12F886E5-5A1B-DF11-8625-001A928116E0.root',
        'file:./Samples/RelValZEE/644D54F3-551B-DF11-892A-0018F3D09708.root',
        'file:./Samples/RelValZEE/6A421747-4E1B-DF11-B902-0018F34D0D62.root',
        'file:./Samples/RelValZEE/701D60CC-531B-DF11-955A-001731AF6B85.root',
        'file:./Samples/RelValZEE/D8886291-541B-DF11-8BC0-003048678ADA.root'
      ] )


secFiles.extend( [
   ] )

