import FWCore.ParameterSet.Config as cms

from Validation.HcalRecHits.HcalRecHitsClient_cfi import *
from Validation.HcalRecHits.NoiseRatesClient_cfi import *

hcalrechitsPostProcessor = cms.Sequence(noiseratesClient*hcalrechitsClient)
# foo bar baz
# gRY1YSx7i3Rkf
# LcnKGlk3uQoMe
