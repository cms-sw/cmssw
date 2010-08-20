import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.VtxSmearedRealistic7TeVCollision_cff import *

# update source label for vertex smearing:
VtxSmeared.src = cms.InputTag("generatorNeutrons")

