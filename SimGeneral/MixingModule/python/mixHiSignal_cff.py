import FWCore.ParameterSet.Config as cms

##################################################################################
# Run SIM on Pyquen hiSignal
from Configuration.StandardSequences.Sim_cff import *
hiSignalG4SimHits = g4SimHits.clone()
hiSignalG4SimHits.Generator.HepMCProductLabel = 'hiSignal' # By default it's "generator" in 3_x_y

##################################################################################
# Match vertex of the hiSignal event to the background event
from SimGeneral.MixingModule.MatchVtx_cfi import *

##################################################################################
# Embed Pyquen hiSignal into Background source at SIM level
from SimGeneral.MixingModule.HiEventMixing_cff import *

##################################################################################
# Produce GenParticles of the two HepMCProducts
from PhysicsTools.HepMCCandAlgos.HiGenParticles_cfi import *
hiGenParticles.srcVector = ["hiSignal","generator"]

hiSignalSequence = cms.Sequence(cms.SequencePlaceholder("hiSignal")*matchVtx*hiGenParticles*hiSignalG4SimHits)

