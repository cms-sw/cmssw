import FWCore.ParameterSet.Config as cms

readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring() 
source = cms.Source ("PoolSource",fileNames = readFiles, secondaryFileNames = secFiles)
readFiles.extend( [
] )


secFiles.extend( [
   ] )

# foo bar baz
# JTSJJcOe5dNYn
# Zm36Ntlp6etYg
