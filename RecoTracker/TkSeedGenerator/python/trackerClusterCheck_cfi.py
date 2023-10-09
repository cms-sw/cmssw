from RecoTracker.TkSeedGenerator.trackerClusterCheckDefault_cfi import trackerClusterCheckDefault as _trackerClusterCheckDefault
trackerClusterCheck = _trackerClusterCheckDefault.clone()

from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
# Disable too many clusters check until we have an updated cut string for phase1 and phase2
phase1Pixel.toModify(trackerClusterCheck, doClusterCheck=False) # FIXME
phase2_tracker.toModify(trackerClusterCheck, doClusterCheck=False) # FIXME

from Configuration.Eras.Modifier_peripheralPbPb_cff import peripheralPbPb
peripheralPbPb.toModify(trackerClusterCheck,
                        doClusterCheck=True,  #FIXMETOO
                        cut = "strip < 400000 && pixel < 40000 && (strip < 60000 + 7.0*pixel) && (pixel < 8000 + 0.14*strip)"
                        )
from Configuration.Eras.Modifier_pp_on_XeXe_2017_cff import pp_on_XeXe_2017
pp_on_XeXe_2017.toModify(trackerClusterCheck,
               doClusterCheck=True, #FIXMETOO
               cut = "strip < 1000000 && pixel < 100000 && (strip < 50000 + 10*pixel) && (pixel < 5000 + strip/2.)",
               MaxNumberOfPixelClusters = 100000
               )

from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toModify(trackerClusterCheck,
               doClusterCheck=True, 
               cut = "strip < 1000000 && pixel < 150000 && (strip < 50000 + 10*pixel) && (pixel < 5000 + strip/2.)",
               MaxNumberOfPixelClusters = 150000,
               MaxNumberOfStripClusters = 500000
               )

from Configuration.ProcessModifiers.egamma_lowPt_exclusive_cff import egamma_lowPt_exclusive
egamma_lowPt_exclusive.toModify(trackerClusterCheck,
               doClusterCheck=True,
               cut = "strip < 1000 && pixel < 300 ",
               MaxNumberOfPixelClusters = 300,
               MaxNumberOfStripClusters = 1000
               )

