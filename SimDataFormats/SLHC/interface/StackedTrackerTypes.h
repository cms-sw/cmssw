/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
/// Written by:                          ///
/// Andrew W. Rose                       ///
/// 2008                                 ///
///                                      ///
/// Changed by:                          ///
/// Nicola Pozzobon                      ///
/// UNIPD                                ///
/// 2010, Oct; 2011 July, Sep            ///
///                                      ///
/// Added feature:                       ///
/// Removed (NOT commented) TTHits       ///
/// (Maybe in the future they will be    ///
/// reintroduced in the framework...)    ///
/// Adapted to the new approach          ///
/// Completed with Tracks                ///
/// ////////////////////////////////////////

#ifndef STACKED_TRACKER_TYPES_H
#define STACKED_TRACKER_TYPES_H

/// Specific Data Formats for Tracking Trigger
#include "SimDataFormats/SLHC/interface/L1TkBeam.h"
#include "SimDataFormats/SLHC/interface/L1TkCluster.h"
#include "SimDataFormats/SLHC/interface/L1TkStub.h"
#include "SimDataFormats/SLHC/interface/L1TkTracklet.h"
#include "SimDataFormats/SLHC/interface/L1TkTrack.h"

/// Standard CMS Formats
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

namespace cmsUpgrades{

  typedef L1TkBeam             Beam_;
  typedef std::vector< Beam_ > Beam_Collection;

  /// The reference types
  typedef edm::Ref< edm::PSimHitContainer >                      Ref_PSimHit_;
  typedef edm::Ref< edm::DetSetVector<PixelDigi> , PixelDigi >   Ref_PixelDigi_;

  /// Cluster data types
  typedef L1TkCluster< Ref_PSimHit_ >            L1TkCluster_PSimHit_;
  typedef L1TkCluster< Ref_PixelDigi_ >          L1TkCluster_PixelDigi_;

  typedef std::vector< L1TkCluster_PSimHit_ >    L1TkCluster_PSimHit_Collection;
  typedef std::vector< L1TkCluster_PixelDigi_ >  L1TkCluster_PixelDigi_Collection;

  typedef std::map< std::pair<StackedTrackerDetId,int>, L1TkCluster_PSimHit_Collection >    L1TkCluster_PSimHit_Map;
  typedef std::map< std::pair<StackedTrackerDetId,int>, L1TkCluster_PixelDigi_Collection >  L1TkCluster_PixelDigi_Map;

  typedef edm::Ptr< L1TkCluster_PSimHit_ >             L1TkCluster_PSimHit_Pointer;
  typedef edm::Ptr< L1TkCluster_PixelDigi_ >           L1TkCluster_PixelDigi_Pointer;

  typedef std::vector< L1TkCluster_PSimHit_Pointer >   L1TkCluster_PSimHit_Pointer_Collection;
  typedef std::vector< L1TkCluster_PixelDigi_Pointer > L1TkCluster_PixelDigi_Pointer_Collection;

  /// Stub data types
  typedef L1TkStub< Ref_PSimHit_ >            L1TkStub_PSimHit_;
  typedef L1TkStub< Ref_PixelDigi_ >          L1TkStub_PixelDigi_;

  typedef std::vector< L1TkStub_PSimHit_ >    L1TkStub_PSimHit_Collection;
  typedef std::vector< L1TkStub_PixelDigi_ >  L1TkStub_PixelDigi_Collection;

  /// Tracklet data types
  typedef L1TkTracklet< Ref_PSimHit_ >            L1TkTracklet_PSimHit_;
  typedef L1TkTracklet< Ref_PixelDigi_ >          L1TkTracklet_PixelDigi_;

  typedef std::vector< L1TkTracklet_PSimHit_ >    L1TkTracklet_PSimHit_Collection;
  typedef std::vector< L1TkTracklet_PixelDigi_ >  L1TkTracklet_PixelDigi_Collection;

  /// Track data types
  typedef L1TkTrack< Ref_PSimHit_ >            L1TkTrack_PSimHit_;
  typedef L1TkTrack< Ref_PixelDigi_ >          L1TkTrack_PixelDigi_;
  
  typedef std::vector< L1TkTrack_PSimHit_ >    L1TkTrack_PSimHit_Collection;
  typedef std::vector< L1TkTrack_PixelDigi_ >  L1TkTrack_PixelDigi_Collection;
  

}
#endif


