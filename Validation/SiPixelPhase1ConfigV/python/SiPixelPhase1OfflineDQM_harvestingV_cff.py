import FWCore.ParameterSet.Config as cms

from Validation.SiPixelPhase1ConfigV.SiPixelPhase1OfflineDQM_sourceV_cff import *

siPixelPhase1OfflineDQM_harvestingV = cms.Sequence(SiPixelPhase1DigisHarvesterV
                                                + SiPixelPhase1TrackClustersHarvesterV
                                                + SiPixelPhase1RecHitsHarvesterV
                                                )
