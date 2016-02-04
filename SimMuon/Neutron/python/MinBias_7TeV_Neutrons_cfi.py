import FWCore.ParameterSet.Config as cms

from Configuration.Generator.MinBias_cfi import *

generatorNeutrons = generator.clone()
generatorNeutrons.comEnergy = cms.double(7000.0)
del generator

