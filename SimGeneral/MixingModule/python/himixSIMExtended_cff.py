import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.SimExtended_cff import *

LHCTransport.HepMCProductLabel = 'hiSignal' # By default it's "generator" in 3_x_y

hiSignalG4SimHits = g4SimHits.clone()

psim.replace(g4SimHits,hiSignalG4SimHits)
