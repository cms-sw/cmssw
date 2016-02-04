import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedRealistic7TeVCollision_cfi import *

# update source label for vertex smearing:
VtxSmeared.src = cms.InputTag("generatorNeutrons")

