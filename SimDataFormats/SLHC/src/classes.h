/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
/// Written by:                          ///
/// Unknown                              ///
///                                      ///
/// Changed by:                          ///
/// Emmanuele Salvati                    ///
/// Cornell                              ///
/// 2010, June                           ///
///                                      ///
/// Removed TrackTriggerHits             ///
/// Fixed some issues about Clusters     ///
/// ////////////////////////////////////////


/** Begin Tracking Trigger **/

#include "DataFormats/Common/interface/Wrapper.h"
#include "SimDataFormats/SLHC/interface/StackedTrackerTypes.h"

namespace {
  namespace {

    cmsUpgrades::Ref_PSimHit_    PSH_;
    cmsUpgrades::Ref_PixelDigi_  PD_;

    /// SimHit type
    cmsUpgrades::L1TkStub_PSimHit_                         S_PSH_;
    cmsUpgrades::L1TkStub_PSimHit_Collection               S_PSH_C;
    edm::Wrapper<cmsUpgrades::L1TkStub_PSimHit_Collection> S_PSH_CW;

    cmsUpgrades::L1TkTracklet_PSimHit_                         T_PSH_;
    cmsUpgrades::L1TkTracklet_PSimHit_Collection               T_PSH_C;
    edm::Wrapper<cmsUpgrades::L1TkTracklet_PSimHit_Collection> T_PSH_CW;

    cmsUpgrades::L1Track_PSimHit_                         L1T_PSH_;
    cmsUpgrades::L1Track_PSimHit_Collection               L1T_PSH_C;
    edm::Wrapper<cmsUpgrades::L1Track_PSimHit_Collection> L1T_PSH_CW;

    /// PixelDigi type
    cmsUpgrades::L1TkStub_PixelDigi_                         S_PD_;
    cmsUpgrades::L1TkStub_PixelDigi_Collection               S_PD_C;
    edm::Wrapper<cmsUpgrades::L1TkStub_PixelDigi_Collection> S_PD_CW;

    cmsUpgrades::L1TkTracklet_PixelDigi_                         T_PD_;
    cmsUpgrades::L1TkTracklet_PixelDigi_Collection               T_PD_C;
    edm::Wrapper<cmsUpgrades::L1TkTracklet_PixelDigi_Collection> T_PD_CW;

    cmsUpgrades::L1Track_PixelDigi_                         L1T_PD_;
    cmsUpgrades::L1Track_PixelDigi_Collection               L1T_PD_C;
    edm::Wrapper<cmsUpgrades::L1Track_PixelDigi_Collection> L1T_PD_CW;


/// WARNING NP** This has to be crosschecked after new class
/// of Clusters has been setup
/* ========================================================================== */      
//Cluster types
    std::vector< std::vector< cmsUpgrades::Ref_PixelDigi_ > > STV_PD;

    std::pair<cmsUpgrades::StackedTrackerDetId,int> STP_STDI_I; // why ???

    // Emmanuele's modification 
    cmsUpgrades::L1TkCluster_PSimHit_                         CL_PSH_;
    cmsUpgrades::L1TkCluster_PSimHit_Map                      CL_PSH_M;
    edm::Wrapper<cmsUpgrades::L1TkCluster_PSimHit_Map>        CL_PSH_MW;
    cmsUpgrades::L1TkCluster_PSimHit_Collection               CL_PSH_C;
    edm::Wrapper<cmsUpgrades::L1TkCluster_PSimHit_Collection> CL_PSH_CW;
    cmsUpgrades::L1TkCluster_PSimHit_Pointer                  CL_PSH_P;
    edm::Wrapper<cmsUpgrades::L1TkCluster_PSimHit_Pointer>    CL_PSH_PW;

    cmsUpgrades::L1TkCluster_PixelDigi_                         CL_PD_; 
    cmsUpgrades::L1TkCluster_PixelDigi_Map                      CL_PD_M;
    edm::Wrapper<cmsUpgrades::L1TkCluster_PixelDigi_Map>        CL_PD_MW;
    cmsUpgrades::L1TkCluster_PixelDigi_Collection               CL_PD_C;
    edm::Wrapper<cmsUpgrades::L1TkCluster_PixelDigi_Collection> CL_PD_CW;
    cmsUpgrades::L1TkCluster_PixelDigi_Pointer                  CL_PD_P;
    edm::Wrapper<cmsUpgrades::L1TkCluster_PixelDigi_Pointer>    CL_PD_PW;

    std::pair<unsigned int, cmsUpgrades::L1TkCluster_PSimHit_ >   P_INT_PSHC;
    std::pair<unsigned int, cmsUpgrades::L1TkCluster_PixelDigi_ > P_INT_PDC;

    std::pair<unsigned int , edm::Ptr< cmsUpgrades::L1TkStub_PSimHit_ > >   P_INT_PTRS_PSH; 
    std::pair<unsigned int , edm::Ptr< cmsUpgrades::L1TkStub_PixelDigi_ > > P_INT_PTRS_PD; 

  }
}

/** End Tracking Trigger **/


/** Begin Barrel DT **/

/// WARNING NP** This has to be crosschecked
#include "DataFormats/Common/interface/Wrapper.h"

#include <vector>
#include <set>

#include "L1Trigger/DTTrackFinder/interface/L1MuDTTrack.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTAddressArray.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTSecProcId.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTTrackSegLoc.h"

#include "SimDataFormats/SLHC/interface/DTBtiTrigger.h"
#include "SimDataFormats/SLHC/interface/DTTSPhiTrigger.h"
#include "SimDataFormats/SLHC/interface/DTTSThetaTrigger.h"
#include "SimDataFormats/SLHC/interface/DTStubMatch.h"
#include "SimDataFormats/SLHC/interface/DTStubMatchPt.h"
#include "SimDataFormats/SLHC/interface/DTStubMatchPtVariety.h"
#include "SimDataFormats/SLHC/interface/DTStubMatchPtAlgorithms.h"
#include "SimDataFormats/SLHC/interface/DTTrackerStub.h"
#include "SimDataFormats/SLHC/interface/DTStubMatchesCollection.h"
#include "SimDataFormats/SLHC/interface/DTSeededTracklet.h"

namespace {
  namespace {
    edm::Wrapper<DTBtiTrigger>                   Bti1;
    std::vector<DTBtiTrigger>                    Btv1;
    edm::Wrapper<std::vector<DTBtiTrigger> >     Btc1;

    edm::Wrapper<DTTSPhiTrigger>                 phi1;
    std::vector<DTTSPhiTrigger>                  phiv1;
    edm::Wrapper<std::vector<DTTSPhiTrigger> >   phic1;

    edm::Wrapper<DTStubMatchPt>                  DTPt1;
    std::vector<DTStubMatchPt>                   DTPtV1;
    edm::Wrapper<std::vector<DTStubMatchPt> >    DTPtW1;

    edm::Wrapper<DTStubMatch>                    DTM1;
    std::vector<DTStubMatch*>                    DTMv1;
    edm::Wrapper<std::vector<DTStubMatch*> >     DTSMwv1;

    std::vector<TrackerStub*>                    DTTSv1;
    edm::Wrapper<std::vector<TrackerStub*> >     DTTSwv1;
    edm::Wrapper<DTStubMatchesCollection>        DTSMc1;

    edm::Wrapper<TrackerStub>                      TS1;
    edm::Wrapper<lt_stub>                          LT1;
    std::set<TrackerStub*, lt_stub>                TSv1;
    edm::Wrapper<std::set<TrackerStub*, lt_stub> > TSwv1;

    DTSeededTracklet                               DTTST1;
    edm::Wrapper<DTSeededTracklet>                 DTTSTw1;
    std::vector<DTSeededTracklet*>                 DTTSTv1;
    edm::Wrapper<std::vector<DTSeededTracklet*> >  DTTSTa1;
    DTSeededTrackletsCollection                    DTTSTc1;
    edm::Wrapper<DTSeededTrackletsCollection>      DTTSTcw1;
  
  }
}

/** End Barrel DT **/



