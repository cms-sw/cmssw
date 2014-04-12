import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.SimIdeal_cff import *
hiSignalG4SimHits = g4SimHits.clone()
hiSignalG4SimHits.Generator.HepMCProductLabel = 'hiSignal' # By default it's "generator" in 3_x_y

psim.replace(g4SimHits,hiSignalG4SimHits)
