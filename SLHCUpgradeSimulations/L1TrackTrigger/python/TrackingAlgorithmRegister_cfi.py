import FWCore.ParameterSet.Config as cms

# First register all the clustering algorithms, then specify preferred
# ones at end.

# Tracking algorithm a
#TrackingAlgorithm_a_PSimHit_   = cms.ESProducer("TrackingAlgorithm_a_PSimHit_")
#TrackingAlgorithm_a_PixelDigi_ = cms.ESProducer("TrackingAlgorithm_a_PixelDigi_")

TrackingAlgorithm_exactLongBarrel_PixelDigi_ = cms.ESProducer("TrackingAlgorithm_exactLongBarrel_PixelDigi_",
   NumSectors = cms.int32(28),
   NumWedges = cms.int32(10),
   ProjectionWindows = cms.VPSet(
#     cms.PSet( RhoPhiWin = cms.vdouble( 0 ),
#               ZWin      = cms.vdouble( 0 ), ),  #Use 0 as dummy to have direct access to the SL without re-starting from 0
#     cms.PSet( RhoPhiWin = cms.vdouble( 0, 0.00, 0.06, 0.30 ),    # from SL 1 to SL ...
#               ZWin      = cms.vdouble( 0, 0.00, 0.84, 2.50 ), ),
#     cms.PSet( RhoPhiWin = cms.vdouble( 0, 0.04, 0.00, 0.15 ),    # from SL 2 to SL ...
#               ZWin      = cms.vdouble( 0, 0.84, 0.00, 1.74 ), ),
#     cms.PSet( RhoPhiWin = cms.vdouble( 0, 0.06, 0.06, 0.00 ),    # from SL 3 to SL ...
#               ZWin      = cms.vdouble( 0, 2.40, 1.65, 0.00 ), ),
     cms.PSet( RhoPhiWin = cms.vdouble( 0 ),
               ZWin      = cms.vdouble( 0 ), ),  #Use 0 as dummy to have direct access to the SL without re-starting from 0
     cms.PSet( RhoPhiWin = cms.vdouble( 0, 0.00, 0.09, 0.38 ),    # from SL 1 to SL ...
               ZWin      = cms.vdouble( 0, 0.00, 1.05, 3.00 ), ),
     cms.PSet( RhoPhiWin = cms.vdouble( 0, 0.06, 0.00, 0.19 ),    # from SL 2 to SL ...
               ZWin      = cms.vdouble( 0, 1.02, 0.00, 2.10 ), ),
     cms.PSet( RhoPhiWin = cms.vdouble( 0, 0.07, 0.07, 0.00 ),    # from SL 3 to SL ...
               ZWin      = cms.vdouble( 0, 2.76, 1.99, 0.00 ), ),
   )
)




TrackingAlgorithm_exactBarrelEndcap_PixelDigi_ = cms.ESProducer("TrackingAlgorithm_exactBarrelEndcap_PixelDigi_",
   NumSectors = cms.int32(28),
   NumWedges = cms.int32(10),
   ProjectionWindowsBarrelBarrel = cms.VPSet(
     cms.PSet( RhoPhiWin = cms.vdouble( 0 ),
               ZWin      = cms.vdouble( 0 ), ),  #Use 0 as dummy to have direct access to the SL without re-starting from 0
     cms.PSet( RhoPhiWin = cms.vdouble( 0, 0.00, 0.00, 0.08, 0.12, 0.15, 0.20 ),   # from L 1,2 to L ...
               ZWin      = cms.vdouble( 0, 0.00, 0.00, 0.50, 3.00, 3.00, 3.00 ), ),
     cms.PSet( RhoPhiWin = cms.vdouble( 0, 0.04, 0.00, 0.00, 0.08, 0.10, 0.15 ),
               ZWin      = cms.vdouble( 0, 0.50, 0.00, 0.00, 3.00, 3.00, 3.00 ), ),
     cms.PSet( RhoPhiWin = cms.vdouble( 0, 0.03, 0.03, 0.00, 0.00, 0.03, 0.05 ),
               ZWin      = cms.vdouble( 0, 5.00, 3.00, 0.00, 0.00, 8.00, 10.0 ), ),
     cms.PSet( RhoPhiWin = cms.vdouble( 0, 0.03, 0.03, 0.03, 0.00, 0.00, 0.05 ),
               ZWin      = cms.vdouble( 0, 5.00, 3.00, 8.00, 0.00, 0.00, 10.0 ), ),
     cms.PSet( RhoPhiWin = cms.vdouble( 0, 0.10, 0.10, 0.10, 0.10, 0.00, 0.00 ),   # from L 5,6 to L ...
               ZWin      = cms.vdouble( 0, 8.00, 8.00, 8.00, 8.00, 0.00, 0.00 ), ),
   ),
   ProjectionWindowsBarrelEndcap = cms.VPSet(
     cms.PSet( RhoPhiWin = cms.vdouble( 0 ),
               ZWin      = cms.vdouble( 0 ),
               RhoPhiWinPS = cms.vdouble( 0 ), 
               ZWinPS      = cms.vdouble( 0 ), ),  #Use 0 as dummy to have direct access to the SL without re-starting from 0
     cms.PSet( RhoPhiWin = cms.vdouble( 0, 0.30, 0.30, 0.40, 0.40, 0.40 ),      # from L 1,2 to D ...
               ZWin      = cms.vdouble( 0, 3.00, 3.00, 3.00, 3.00, 3.00 ),
               RhoPhiWinPS = cms.vdouble( 0, 0.10, 0.10, 0.20, 0.20, 0.20 ),
               ZWinPS      = cms.vdouble( 0, 0.20, 0.20, 0.40, 1.50, 1.50 ), ),
     cms.PSet( RhoPhiWin = cms.vdouble( 0, 0.20, 0.20, 0.20, 0.20, 0.00 ),
               ZWin      = cms.vdouble( 0, 3.00, 3.00, 3.00, 3.00, 0.00 ),
               RhoPhiWinPS = cms.vdouble( 0, 0.20, 0.20, 0.20, 0.20, 0.00 ),
               ZWinPS      = cms.vdouble( 0, 0.50, 0.50, 0.50, 0.50, 0.00 ), ),
     cms.PSet( RhoPhiWin = cms.vdouble( 0, 0.60, 0.60, 0.30, 0.00, 0.00 ),
               ZWin      = cms.vdouble( 0, 3.00, 3.00, 3.00, 0.00, 0.00 ),
               RhoPhiWinPS = cms.vdouble( 0, 0.00, 0.00, 0.00, 0.00, 0.00 ),
               ZWinPS      = cms.vdouble( 0, 0.00, 0.00, 0.00, 0.00, 0.00 ), ),
     cms.PSet( RhoPhiWin = cms.vdouble( 0, 0.20, 0.00, 0.00, 0.00, 0.00 ),
               ZWin      = cms.vdouble( 0, 4.00, 0.00, 0.00, 0.00, 0.00 ),
               RhoPhiWinPS = cms.vdouble( 0, 0.00, 0.00, 0.00, 0.00, 0.00 ),
               ZWinPS      = cms.vdouble( 0, 0.00, 0.00, 0.00, 0.00, 0.00 ), ),
     cms.PSet( RhoPhiWin = cms.vdouble( 0, 0.00, 0.00, 0.00, 0.00, 0.00 ),      # from L 5,6 to D ...
               ZWin      = cms.vdouble( 0, 0.00, 0.00, 0.00, 0.00, 0.00 ),
               RhoPhiWinPS = cms.vdouble( 0, 0.00, 0.00, 0.00, 0.00, 0.00 ),
               ZWinPS      = cms.vdouble( 0, 0.00, 0.00, 0.00, 0.00, 0.00 ), ),
   ),
   ProjectionWindowsEndcapBarrel = cms.VPSet(
     cms.PSet( RhoPhiWin = cms.vdouble( 0 ),
               ZWin      = cms.vdouble( 0 ),
               RhoPhiWinPS = cms.vdouble( 0 ),
               ZWinPS      = cms.vdouble( 0 ), ),  #Use 0 as dummy to have direct access to the SL without re-starting from 0
     cms.PSet( RhoPhiWin = cms.vdouble( 0, 0.40, 0.40, 0.40, 0.40, 0.00, 0.00 ),
               ZWin      = cms.vdouble( 0, 9.00, 9.00, 9.00, 9.00, 0.00, 0.00 ),
               RhoPhiWinPS = cms.vdouble( 0, 0.10, 0.10, 0.20, 0.20, 0.00, 0.00 ),
               ZWinPS      = cms.vdouble( 0, 1.50, 1.00, 8.00, 8.00, 0.00, 0.00 ), ),
     cms.PSet( RhoPhiWin = cms.vdouble( 0, 0.40, 0.40, 0.40, 0.40, 0.00, 0.00 ),
               ZWin      = cms.vdouble( 0, 8.00, 8.00, 8.00, 8.00, 0.00, 0.00 ),
               RhoPhiWinPS = cms.vdouble( 0, 0.10, 0.10, 0.10, 0.10, 0.00, 0,00 ),
               ZWinPS      = cms.vdouble( 0, 1.50, 1.50, 8.00, 8.00, 0.00, 0.00 ), ),
     cms.PSet( RhoPhiWin = cms.vdouble( 0, 0.30, 0.30, 0.30, 0.00, 0.00, 0.00 ),
               ZWin      = cms.vdouble( 0, 8.00, 8.00, 8.00, 0.00, 0.00, 0.00 ),
               RhoPhiWinPS = cms.vdouble( 0, 0.10, 0.10, 0.10, 0.00, 0.00, 0.00 ),
               ZWinPS      = cms.vdouble( 0, 2.00, 3.00, 8.00, 0.00, 0.00, 0.00 ), ),
     cms.PSet( RhoPhiWin = cms.vdouble( 0, 0.30, 0.30, 0.00, 0.00, 0.00, 0.00 ),
               ZWin      = cms.vdouble( 0, 8.00, 8.00, 0.00, 0.00, 0.00, 0.00 ),
               RhoPhiWinPS = cms.vdouble( 0, 0.10, 0.10, 0.00, 0.00, 0.00, 0.00 ),
               ZWinPS      = cms.vdouble( 0, 2.00, 3.00, 0.00, 0.00, 0.00, 0.00 ), ),
   ),
   ProjectionWindowsEndcapEndcap = cms.VPSet(
     cms.PSet( RhoPhiWin = cms.vdouble( 0 ),
               ZWin      = cms.vdouble( 0 ),
               RhoPhiWinPS = cms.vdouble( 0 ),
               ZWinPS      = cms.vdouble( 0 ), ),  #Use 0 as dummy to have direct access to the SL without re-starting from 0
     cms.PSet( RhoPhiWin = cms.vdouble( 0, 0.00, 0.00, 0.05, 0.10, 0.10 ),
               ZWin      = cms.vdouble( 0, 0.00, 0.00, 3.00, 3.00, 3.00 ),
               RhoPhiWinPS = cms.vdouble( 0, 0.00, 0.00, 0.05, 0.10, 0.15 ),
               ZWinPS      = cms.vdouble( 0, 0.00, 0.00, 0.5, 1.00, 1.00 ), ),
     cms.PSet( RhoPhiWin = cms.vdouble( 0, 0.40, 0.00, 0.00, 0.08, 0.10 ),
               ZWin      = cms.vdouble( 0, 3.00, 0.00, 0.00, 3.00, 3.00 ),
               RhoPhiWinPS = cms.vdouble( 0, 0.04, 0.00, 0.00, 0.08, 0.10 ),
               ZWinPS      = cms.vdouble( 0, 0.40, 0.00, 0.00, 0.50, 1.00 ), ),
     cms.PSet( RhoPhiWin = cms.vdouble( 0, 0.05, 0.05, 0.00, 0.00, 0.08 ),
               ZWin      = cms.vdouble( 0, 3.00, 3.00, 0.00, 0.00, 3.00 ),
               RhoPhiWinPS = cms.vdouble( 0, 0.05, 0.05, 0.00, 0.00, 0.08 ),
               ZWinPS      = cms.vdouble( 0, 0.50, 0.50, 0.00, 0.00, 1.00 ), ),
     cms.PSet( RhoPhiWin = cms.vdouble( 0, 0.05, 0.05, 0.05, 0.00, 0.00 ),
               ZWin      = cms.vdouble( 0, 3.00, 3.00, 3.00, 0.00, 0.00 ),
               RhoPhiWinPS = cms.vdouble( 0, 0.05, 0.05, 0.05, 0.00, 0.00 ),
               ZWinPS      = cms.vdouble( 0, 0.50, 0.50, 0.80, 0.00, 0.00 ), ),
   )
)


# Set the preferred hit matching algorithms.
# We prefer the a algorithm for now in order not to break anything.
# Override with process.TrackingAlgorithm_PSimHit_ = ..., etc. in your
# configuration.
#TrackingAlgorithm_PSimHit_   = cms.ESPrefer("TrackingAlgorithm_a_PSimHit_")
TrackingAlgorithm_PixelDigi_ = cms.ESPrefer("TrackingAlgorithm_exactLongBarrel_PixelDigi_")

