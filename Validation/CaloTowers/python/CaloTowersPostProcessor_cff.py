import FWCore.ParameterSet.Config as cms

from Validation.CaloTowers.CaloTowersClient_cfi import *

calotowersPostProcessor = cms.Sequence(calotowersClient)
