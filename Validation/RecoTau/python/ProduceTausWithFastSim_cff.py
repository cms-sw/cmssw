import FWCore.ParameterSet.Config as cms

from FastSimulation.Configuration.FamosSequences_cff import *
from FastSimulation.Configuration.CommonInputs_cff import *
from FastSimulation.Configuration.RandomServiceInitialization_cff import *
from Configuration.StandardSequences.MagneticField_cff import *
from PhysicsTools.HepMCCandAlgos.genParticles_cfi import *

makeTausWithFastSim = cms.Sequence(cms.SequencePlaceholder("ProductionFilterSequence")*genParticles*famosWithElectrons*famosWithPFTauTagging*famosWithTauTagging)
