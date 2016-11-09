from RecoTracker.TkSeedGenerator.trackerClusterCheckDefault_cfi import trackerClusterCheckDefault as _trackerClusterCheckDefault
trackerClusterCheck = _trackerClusterCheckDefault.clone()

from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
# Disable too many clusters check until we have an updated cut string for phase1 and phase2
phase1Pixel.toModify(trackerClusterCheck, doClusterCheck=False) # FIXME
phase2_tracker.toModify(trackerClusterCheck, doClusterCheck=False) # FIXME
