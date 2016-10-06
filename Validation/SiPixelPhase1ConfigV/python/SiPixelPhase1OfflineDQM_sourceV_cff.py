import FWCore.ParameterSet.Config as cms

# Pixel Digi Monitoring
from Validation.SiPixelPhase1DigisV.SiPixelPhase1DigisV_cfi import *
# Pixel Clusters
from Validation.SiPixelPhase1TrackClustersV.SiPixelPhase1TrackClustersV_cfi import *
# RecHit (clusters)
from Validation.SiPixelPhase1RecHitsV.SiPixelPhase1RecHitsV_cfi import *

PerModule.enabled = False

siPixelPhase1OfflineDQM_sourceV = cms.Sequence(SiPixelPhase1DigisAnalyzerV
                                            + SiPixelPhase1TrackClustersAnalyzerV
                                            + SiPixelPhase1RecHitsAnalyzerV
                                            )
