import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Generator_cff import *

pgen_neutrons = cms.Sequence(cms.SequencePlaceholder("randomEngineStateProducer")+VertexSmearing)

