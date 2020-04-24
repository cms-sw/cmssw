import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Generator_cff import *
from GeneratorInterface.Core.generatorSmearingProducer_cfi import *

GenSmeared = cms.Sequence("generatorSmeared")
pgen_neutrons = cms.Sequence(cms.SequencePlaceholder("randomEngineStateProducer")+VertexSmearing+GenSmeared)

