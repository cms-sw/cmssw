import FWCore.ParameterSet.Config as cms

from IOMC.EventVertexGenerators.VtxSmearedGauss_cfi import *

# update source label for vertex smearing:
VtxSmeared.src = cms.InputTag("generatorNeutrons")

# foo bar baz
# OWj5UMW7QBkzC
