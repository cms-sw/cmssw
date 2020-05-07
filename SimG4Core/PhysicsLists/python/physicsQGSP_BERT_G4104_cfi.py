import FWCore.ParameterSet.Config as cms
from SimG4Core.Configuration.SimG4Core_cff import *

g4SimHits.Physics.EmaxBERT   = cms.double(8.)  # in GeV
g4SimHits.Physics.EmaxBERTpi = cms.double(8.)  # in GeV
g4SimHits.Physics.EminFTFP   = cms.double(6.)  # in GeV
g4SimHits.Physics.EmaxFTFP   = cms.double(25.) # in GeV
g4SimHits.Physics.EminQGSP   = cms.double(12.) # in GeV
