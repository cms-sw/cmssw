import FWCore.ParameterSet.Config as cms

from Validation.HcalRecHits.HcalRecHitsClient_cfi import *
from Validation.HcalRecHits.NoiseRatesClient_cfi import *

hcalrechitsPostProcessor = cms.Sequence(noiseratesClient*hcalrechitsClient)
