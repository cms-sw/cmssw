import FWCore.ParameterSet.Config as cms

from Validation.TrackerRecHits.SiPixelRecHitsValid_cfi import *
from Validation.TrackerRecHits.SiStripRecHitsValid_cfi import *
trackerRecHitsValidation = cms.Sequence(pixRecHitsValid*stripRecHitsValid)

