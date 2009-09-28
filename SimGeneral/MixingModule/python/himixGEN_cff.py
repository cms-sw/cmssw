import FWCore.ParameterSet.Config as cms

##################################################################################
# Match vertex of the hiSignal event to the background event
from SimGeneral.MixingModule.MatchVtx_cfi import *

##################################################################################
# Produce GenParticles of the two HepMCProducts
from PhysicsTools.HepMCCandAlgos.HiGenParticles_cfi import *
hiGenParticles.srcVector = ["hiSignal","generator"]

hiSignalGenSequence = cms.Sequence(matchVtx*hiGenParticles)
