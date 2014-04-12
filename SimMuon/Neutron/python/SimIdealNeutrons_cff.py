# Modified version of
# Configuration/StandardSequences/python/SimIdeal_cff.py

import FWCore.ParameterSet.Config as cms

# CMSSW/Geant4 interface
from SimG4Core.Configuration.SimG4Core_cff import *

# rename g4SimHits so we can reuse the module label for neutron simhits collection
g4SimHitsNeutrons = g4SimHits.clone()
g4SimHitsNeutrons.Generator.HepMCProductLabel = cms.string('generatorNeutrons')
del g4SimHits

# Configure G4 for neutron hits:
g4SimHitsNeutrons.Physics.type = 'SimG4Core/Physics/QGSP_BERT_HP'
g4SimHitsNeutrons.Physics.FlagBERT = True
g4SimHitsNeutrons.StackingAction.NeutronThreshold = 0.
g4SimHitsNeutrons.StackingAction.MaxTrackTime = 1e9
g4SimHitsNeutrons.SteppingAction.MaxTrackTime = 1e9
# the following two enable simulation in the Quad region
# (commenting them out would make debug runs faster)
g4SimHitsNeutrons.StackingAction.MaxTrackTimes[2] = 1e9
g4SimHitsNeutrons.SteppingAction.MaxTrackTimes[2] = 1e9
#  cuts on generator-level particles
g4SimHitsNeutrons.Generator.ApplyPCuts = False
g4SimHitsNeutrons.Generator.ApplyEtaCuts = False
#only affects weighting of energy deposit, so unneeded
#g4SimHitsNeutrons.CaloSD.NeutronThreshold = 0.

# special psim sequence:
psim_neutrons = cms.Sequence(cms.SequencePlaceholder("randomEngineStateProducer")*g4SimHitsNeutrons)
