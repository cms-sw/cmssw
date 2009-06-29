import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Services_cff import *
from Configuration.StandardSequences.Geometry_cff import *
from Configuration.StandardSequences.Generator_cff import *
from Configuration.StandardSequences.MagneticField_cff import *
from Configuration.StandardSequences.FrontierConditions_GlobalTag_cff import *
from Configuration.StandardSequences.Simulation_cff import *
from Configuration.StandardSequences.MixingNoPileUp_cff import *
from Configuration.StandardSequences.DigiToRaw_cff import *
from Configuration.StandardSequences.VtxSmearedEarly10TeVCollision_cff import *
from PhysicsTools.HepMCCandAlgos.genParticles_cfi import *

simAndDigitizeForTaus = cms.Sequence(cms.SequencePlaceholder("ProductionFilterSequence")*pgen*psim*pdigi*genParticles)
