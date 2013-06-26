import FWCore.ParameterSet.Config as cms

# Produce GenParticles of the two HepMCProducts
from PhysicsTools.HepMCCandAlgos.HiGenParticles_cfi import *
hiGenParticles.srcVector = ["hiSignal","generator"]

