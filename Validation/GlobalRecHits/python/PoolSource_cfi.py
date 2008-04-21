import FWCore.ParameterSet.Config as cms

# actual location is /pnfs/cms/WAX/11/
source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:MinBias.root')
)


