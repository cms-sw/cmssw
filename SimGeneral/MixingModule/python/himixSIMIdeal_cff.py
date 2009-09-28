import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.SimIdeal_cff import *

g4SimHits.Generator.HepMCProductLabel = 'hiSignal' # By default it's "generator" in 3_x_y
