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
     cms.PSet( RhoPhiWin = cms.vdouble( 0, 0.00, 0.00, 0.07, 0.09, 0.16, 0.25 ),    # from L 1,2 to L ...
               ZWin      = cms.vdouble( 0, 0.00, 0.00, 4.20, 4.20, 3.80, 4.10 ), ),
     cms.PSet( RhoPhiWin = cms.vdouble( 0, 0.08, 0.00, 0.00, 0.05, 0.09, 0.14 ),    # from L 2,3 to L ...
               ZWin      = cms.vdouble( 0, 0.50, 0.00, 0.00, 4.20, 3.80, 4.00 ), ),
     cms.PSet( RhoPhiWin = cms.vdouble( 0, 0.05, 0.06, 0.00, 0.00, 0.06, 0.07 ),    # from L 3,4 to L ...
               ZWin      = cms.vdouble( 0, 5.10, 3.70, 0.00, 0.00, 5.20, 5.20 ), ),
     cms.PSet( RhoPhiWin = cms.vdouble( 0, 0.05, 0.06, 0.06, 0.00, 0.00, 0.06 ),    # from L 4,5 to L ...
               ZWin      = cms.vdouble( 0, 5.20, 5.20, 5.20, 0.00, 0.00, 4.20 ), ),
     cms.PSet( RhoPhiWin = cms.vdouble( 0, 0.08, 0.06, 0.06, 0.06, 0.00, 0.00 ),    # from L 5,6 to L ...
               ZWin      = cms.vdouble( 0, 5.10, 5.10, 5.10, 4.30, 0.00, 0.00 ), ),
   ),
   ProjectionWindowsBarrelEndcap = cms.VPSet(
     cms.PSet( RhoPhiWin = cms.vdouble( 0 ),
               ZWin      = cms.vdouble( 0 ), ),  #Use 0 as dummy to have direct access to the SL without re-starting from 0
     cms.PSet( RhoPhiWin = cms.vdouble( 0, 0.19, 0.20, 0.22, 0.24, 0.26, 0.25, 0.28 ),    # from L 1,2 to D ...
               ZWin      = cms.vdouble( 0, 3.40, 3.60, 3.70, 4.20, 4.30, 4.40, 4.30 ), ),
     cms.PSet( RhoPhiWin = cms.vdouble( 0, 0.17, 0.17, 0.17, 0.17, 0.19, 0.20, 0.00 ),    # from L 2,3 to D ...
               ZWin      = cms.vdouble( 0, 4.20, 4.30, 4.30, 4.30, 4.40, 4.40, 0.00 ), ),
     cms.PSet( RhoPhiWin = cms.vdouble( 0, 0.34, 0.34, 0.35, 0.38, 0.00, 0.00, 0.00 ),    # from L 3,4 to D ...
               ZWin      = cms.vdouble( 0, 5.00, 4.90, 4.90, 5.00, 0.00, 0.00, 0.00 ), ),
     cms.PSet( RhoPhiWin = cms.vdouble( 0, 0.32, 0.38, 0.00, 0.00, 0.00, 0.00, 0.00 ),    # from L 4,5 to D ...
               ZWin      = cms.vdouble( 0, 4.80, 2.20, 0.00, 0.00, 0.00, 0.00, 0.00 ), ),
     cms.PSet( RhoPhiWin = cms.vdouble( 0, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00 ),    # from L 5,6 to D ...
               ZWin      = cms.vdouble( 0, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00 ), ),
   ),
   ProjectionWindowsEndcapBarrel = cms.VPSet(
     cms.PSet( RhoPhiWin = cms.vdouble( 0 ),
               ZWin      = cms.vdouble( 0 ), ),  #Use 0 as dummy to have direct access to the SL without re-starting from 0
     cms.PSet( RhoPhiWin = cms.vdouble( 0, 0.20, 0.31, 0.42, 0.40, 0.00, 0.00 ),    # from D 1,2 to L ...
               ZWin      = cms.vdouble( 0, 2.80, 2.80, 5.20, 5.30, 0.00, 0.00 ), ),
     cms.PSet( RhoPhiWin = cms.vdouble( 0, 0.18, 0.32, 0.42, 0.40, 0.00, 0.00 ),    # from D 2,3 to L ...
               ZWin      = cms.vdouble( 0, 2.90, 3.20, 5.20, 5.00, 0.00, 0.00 ), ),
     cms.PSet( RhoPhiWin = cms.vdouble( 0, 0.17, 0.36, 0.38, 0.00, 0.00, 0.00 ),    # from D 3,4 to L ...
               ZWin      = cms.vdouble( 0, 3.20, 4.50, 5.40, 0.00, 0.00, 0.00 ), ),
     cms.PSet( RhoPhiWin = cms.vdouble( 0, 0.20, 0.45, 0.40, 0.00, 0.00, 0.00 ),    # from D 4,5 to L ...
               ZWin      = cms.vdouble( 0, 3.60, 5.40, 5.30, 0.00, 0.00, 0.00 ), ),
     cms.PSet( RhoPhiWin = cms.vdouble( 0, 0.24, 0.45, 0.40, 0.00, 0.00, 0.00 ),    # from D 5,6 to L ...
               ZWin      = cms.vdouble( 0, 4.00, 5.10, 4.80, 0.00, 0.00, 0.00 ), ),
     cms.PSet( RhoPhiWin = cms.vdouble( 0, 0.28, 0.40, 0.00, 0.00, 0.00, 0.00 ),    # from D 6,7 to L ...
               ZWin      = cms.vdouble( 0, 4.20, 5.30, 0.00, 0.00, 0.00, 0.00 ), ),
   ),
   ProjectionWindowsEndcapEndcap = cms.VPSet(
     cms.PSet( RhoPhiWin = cms.vdouble( 0 ),
               ZWin      = cms.vdouble( 0 ), ),  #Use 0 as dummy to have direct access to the SL without re-starting from 0
     cms.PSet( RhoPhiWin = cms.vdouble( 0, 0.00, 0.00, 0.25, 0.36, 0.37, 0.34, 0.34 ),    # from D 1,2 to D ...
               ZWin      = cms.vdouble( 0, 0.00, 0.00, 3.00, 3.50, 3.70, 3.80, 4.20 ), ),
     cms.PSet( RhoPhiWin = cms.vdouble( 0, 0.18, 0.00, 0.00, 0.25, 0.34, 0.35, 0.35 ),    # from D 2,3 to D ...
               ZWin      = cms.vdouble( 0, 2.60, 0.00, 0.00, 2.90, 3.30, 3.50, 3.80 ), ),
     cms.PSet( RhoPhiWin = cms.vdouble( 0, 0.23, 0.18, 0.00, 0.00, 0.24, 0.33, 0.36 ),    # from D 3,4 to D ...
               ZWin      = cms.vdouble( 0, 3.10, 2.60, 0.00, 0.00, 2.90, 3.20, 3.60 ), ),
     cms.PSet( RhoPhiWin = cms.vdouble( 0, 0.29, 0.24, 0.19, 0.00, 0.00, 0.24, 0.35 ),    # from D 4,5 to D ...
               ZWin      = cms.vdouble( 0, 3.60, 3.10, 2.60, 0.00, 0.00, 2.90, 3.40 ), ),
     cms.PSet( RhoPhiWin = cms.vdouble( 0, 0.32, 0.27, 0.22, 0.19, 0.00, 0.00, 0.26 ),    # from D 5,6 to D ...
               ZWin      = cms.vdouble( 0, 3.80, 3.40, 2.90, 2.60, 0.00, 0.00, 3.30 ), ),
     cms.PSet( RhoPhiWin = cms.vdouble( 0, 0.37, 0.31, 0.27, 0.23, 0.19, 0.00, 0.00 ),    # from D 6,7 to D ...
               ZWin      = cms.vdouble( 0, 4.40, 3.60, 3.20, 2.80, 2.70, 0.00, 0.00 ), ),
   )
)


# Set the preferred hit matching algorithms.
# We prefer the a algorithm for now in order not to break anything.
# Override with process.TrackingAlgorithm_PSimHit_ = ..., etc. in your
# configuration.
#TrackingAlgorithm_PSimHit_   = cms.ESPrefer("TrackingAlgorithm_a_PSimHit_")
TrackingAlgorithm_PixelDigi_ = cms.ESPrefer("TrackingAlgorithm_exactLongBarrel_PixelDigi_")

