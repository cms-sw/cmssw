// Modified by Emmanuele

#include "DataFormats/Common/interface/Wrapper.h"

/* ========================================================================================= */
/* ================================== TRACKING TRIGGER INCLUDES =================================== */
/* ========================================================================================= */

#include "SimDataFormats/SLHC/interface/StackedTrackerTypes.h"

namespace {
  namespace {

    cmsUpgrades::Ref_PSimHit_ PSH_;
    cmsUpgrades::Ref_PixelDigi_	PD_;

/* ========================================================================== */
//SimHit type
    cmsUpgrades::LocalStub_PSimHit_ LS_PSH_;
    cmsUpgrades::LocalStub_PSimHit_Collection LS_PSH_C;
    edm::Wrapper<cmsUpgrades::LocalStub_PSimHit_Collection> LS_PSH_CW;

    cmsUpgrades::GlobalStub_PSimHit_ GS_PSH_;
    cmsUpgrades::GlobalStub_PSimHit_Collection GS_PSH_C;
    edm::Wrapper<cmsUpgrades::GlobalStub_PSimHit_Collection> GS_PSH_CW;

    cmsUpgrades::Tracklet_PSimHit_ T_PSH_;
    cmsUpgrades::Tracklet_PSimHit_Collection T_PSH_C;
    edm::Wrapper<cmsUpgrades::Tracklet_PSimHit_Collection> T_PSH_CW;

    cmsUpgrades::L1Track_PSimHit_ L1T_PSH_;
    cmsUpgrades::L1Track_PSimHit_Collection L1T_PSH_C;
    edm::Wrapper<cmsUpgrades::L1Track_PSimHit_Collection> L1T_PSH_CW;


/* ========================================================================== */
//PixelDigi type
    cmsUpgrades::LocalStub_PixelDigi_ LS_PD_;
    cmsUpgrades::LocalStub_PixelDigi_Collection	LS_PD_C;
    edm::Wrapper<cmsUpgrades::LocalStub_PixelDigi_Collection> LS_PD_CW;

    cmsUpgrades::GlobalStub_PixelDigi_ GS_PD_;
    cmsUpgrades::GlobalStub_PixelDigi_Collection GS_PD_C;
    edm::Wrapper<cmsUpgrades::GlobalStub_PixelDigi_Collection> GS_PD_CW;

    cmsUpgrades::Tracklet_PixelDigi_ T_PD_;
    cmsUpgrades::Tracklet_PixelDigi_Collection T_PD_C;
    edm::Wrapper<cmsUpgrades::Tracklet_PixelDigi_Collection> T_PD_CW;

    cmsUpgrades::L1Track_PixelDigi_ L1T_PD_;
    cmsUpgrades::L1Track_PixelDigi_Collection L1T_PD_C;
    edm::Wrapper<cmsUpgrades::L1Track_PixelDigi_Collection> L1T_PD_CW;

/* ========================================================================== */      
//Cluster types
    std::vector< std::vector< cmsUpgrades::Ref_PixelDigi_ > > STV_PD;


    std::pair<cmsUpgrades::StackedTrackerDetId,int> STP_STDI_I; // why ???

    // Emmanuele's modification 
    cmsUpgrades::Cluster_PSimHit CL_PSH_; 
    cmsUpgrades::Cluster_PixelDigi CL_PD_; 
    //    
    cmsUpgrades::Cluster_PSimHit_Map CL_PSH_M;
    edm::Wrapper<cmsUpgrades::Cluster_PSimHit_Map> CL_PSH_MW;
    cmsUpgrades::Cluster_PixelDigi_Map CL_PD_M;
    edm::Wrapper<cmsUpgrades::Cluster_PixelDigi_Map> CL_PD_MW;
    
    std::pair<unsigned int, cmsUpgrades::Cluster_PSimHit > P_INT_PSHC;
    std::pair<unsigned int, cmsUpgrades::Cluster_PixelDigi > P_INT_PDC;

    //edm::Ptr< cmsUpgrades::GlobalStub_PSimHit_ > PTR_GS_PSH;
    //edm::Ptr< cmsUpgrades::GlobalStub_PixelDigi_ > PTR_GS_PD;
    //edm::Ptr< cmsUpgrades::GlobalStub_TTHit_ > PTR_GS_TTH;

    std::pair<unsigned int , edm::Ptr< cmsUpgrades::GlobalStub_PSimHit_ > > P_INT_PTRGS_PSH; 
    std::pair<unsigned int , edm::Ptr< cmsUpgrades::GlobalStub_PixelDigi_ > > P_INT_PTRGS_PD; 
    


    /* std::pair<unsigned int , edm::Ptr< cmsUpgrades::GlobalStub_TTHit_ > > P_INT_PTRGS_TTH; */
  }
}



/*****************************************************************************/
/*                                                                           */
/*                 Barrel DT                                                 */
/*                                                                           */
/*****************************************************************************/
/***************  Barrel DT includes  ****************************************/
 
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


/***************  End Barrel DT includes  *************************************/ 

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


/***************  End Barrel DT  **********************************************/ 

/* ========================================================================================= */
/* ========================================================================================= */
/* ========================================================================================= */

