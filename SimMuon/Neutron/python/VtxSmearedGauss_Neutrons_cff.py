import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.VtxSmearedGauss_cff import *

# update source label for vertex smearing:
VtxSmeared.src = cms.InputTag("generatorNeutrons")

