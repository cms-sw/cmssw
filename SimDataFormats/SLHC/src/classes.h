#include "DataFormats/Common/interface/Wrapper.h"

/* ========================================================================================= */
/* ==================================== CALO TRIGGER INCLUDES ==================================== */
/* ========================================================================================= */

#include "SimDataFormats/SLHC/interface/L1CaloTriggerSetup.h"

#include "SimDataFormats/SLHC/interface/L1CaloTower.h"
#include "SimDataFormats/SLHC/interface/L1CaloCluster.h"
#include "SimDataFormats/SLHC/interface/L1CaloJet.h"

namespace {
  namespace {

    l1slhc::L1CaloTower l1tower;
    std::map<int,l1slhc::L1CaloTower>   maplcalo;
    std::vector<l1slhc::L1CaloTower>    l1caloto;
    edm::Wrapper< std::vector<l1slhc::L1CaloTower> >   wl1caloto;

    l1slhc::L1CaloCluster                 calocl;
    std::vector<l1slhc::L1CaloCluster>    l1calocl;
    edm::Wrapper< std::vector<l1slhc::L1CaloCluster> >   wl1calocl;

    l1slhc::L1CaloJet                     calojet;
    std::vector<l1slhc::L1CaloJet>       l1calojetcol;
    edm::Wrapper< std::vector<l1slhc::L1CaloJet> >   wl1calojetcol;

  }
}

/* ========================================================================================= */
/* ========================================================================================= */
/* ========================================================================================= */






/* ========================================================================================= */
/* ================================== TRACKING TRIGGER INCLUDES =================================== */
/* ========================================================================================= */

#include "SimDataFormats/SLHC/interface/StackedTrackerTypes.h"

namespace {
  namespace {

	cmsUpgrades::Ref_PSimHit_ 									PSH_;
	cmsUpgrades::Ref_PixelDigi_ 								PD_;

/* ========================================================================== */
//SimHit type LocalStub
	cmsUpgrades::LocalStub_PSimHit_ 							LS_PSH_;

//SimHit type LocalStub Collections
	cmsUpgrades::LocalStub_PSimHit_Collection					LS_PSH_C;
	edm::Wrapper<cmsUpgrades::LocalStub_PSimHit_Collection>		LS_PSH_CW;

//SimHit type GlobalStub
	cmsUpgrades::GlobalStub_PSimHit_							GS_PSH_;

//SimHit type GlobalStub Collections
	cmsUpgrades::GlobalStub_PSimHit_Collection					GS_PSH_C;
	edm::Wrapper<cmsUpgrades::GlobalStub_PSimHit_Collection>	GS_PSH_CW;

//SimHit type Tracklet
	cmsUpgrades::Tracklet_PSimHit_								T_PSH_;

//SimHit type Tracklet Collections
	cmsUpgrades::Tracklet_PSimHit_Collection					T_PSH_C;
	edm::Wrapper<cmsUpgrades::Tracklet_PSimHit_Collection>		T_PSH_CW;


/* ========================================================================== */
//PixelDigi type LocalStub
	cmsUpgrades::LocalStub_PixelDigi_							LS_PD_;

//PixelDigi type LocalStub Collections
	cmsUpgrades::LocalStub_PixelDigi_Collection					LS_PD_C;
	edm::Wrapper<cmsUpgrades::LocalStub_PixelDigi_Collection>	LS_PD_CW;

//PixelDigi type GlobalStub
	cmsUpgrades::GlobalStub_PixelDigi_							GS_PD_;

//PixelDigi type GlobalStub Collections
	cmsUpgrades::GlobalStub_PixelDigi_Collection				GS_PD_C;
	edm::Wrapper<cmsUpgrades::GlobalStub_PixelDigi_Collection>	GS_PD_CW;

//PixelDigi type Tracklet
	cmsUpgrades::Tracklet_PixelDigi_							T_PD_;

//PixelDigi type Tracklet Collections
	cmsUpgrades::Tracklet_PixelDigi_Collection					T_PD_C;
	edm::Wrapper<cmsUpgrades::Tracklet_PixelDigi_Collection>	T_PD_CW;

/* ========================================================================== */
//Finally the TTHit type
/* ========================================================================== */

	TrackTriggerHit												TTH;
	TrackTriggerHitCollection									TTHC;
	edm::Wrapper<TrackTriggerHitCollection>						TTHCW;
	cmsUpgrades::Ref_TTHit_ 									TTH_;

    edm::Wrapper<TrackTriggerHit> 								zs0;
    edm::Wrapper< std::vector<TrackTriggerHit>  > 				zs1;
    edm::Wrapper< edm::DetSet<TrackTriggerHit> > 				zs2;
    edm::Wrapper< std::vector<edm::DetSet<TrackTriggerHit> > > 	zs3;
    edm::Wrapper< edm::DetSetVector<TrackTriggerHit> > 			zs4;

//TTHit type LocalStub
	cmsUpgrades::LocalStub_TTHit_								LS_TTH_;

//PixelDigi type LocalStub Collections
	cmsUpgrades::LocalStub_TTHit_Collection						LS_TTH_C;
	edm::Wrapper<cmsUpgrades::LocalStub_TTHit_Collection>		LS_TTH_CW;

//TTHit type GlobalStub
	cmsUpgrades::GlobalStub_TTHit_								GS_TTH_;

//TTHit type GlobalStub Collections
	cmsUpgrades::GlobalStub_TTHit_Collection					GS_TTH_C;
	edm::Wrapper<cmsUpgrades::GlobalStub_TTHit_Collection>		GS_TTH_CW;

//TTHit type Tracklet
	cmsUpgrades::Tracklet_TTHit_								T_TTH_;

//TTHit type Tracklet Collections
	cmsUpgrades::Tracklet_TTHit_Collection						T_TTH_C;
	edm::Wrapper<cmsUpgrades::Tracklet_TTHit_Collection>		T_TTH_CW;

/* ========================================================================== */
      
//Cluster types
      cmsUpgrades::Cluster_PSimHit_Map                          CL_PSH_M;
      edm::Wrapper<cmsUpgrades::Cluster_PSimHit_Map>            CL_PSH_MW;
      cmsUpgrades::Cluster_PixelDigi_Map                        CL_PD_M;
      edm::Wrapper<cmsUpgrades::Cluster_PixelDigi_Map>          CL_PD_MW;
      cmsUpgrades::Cluster_TTHit_Map                            CL_TTH_M;
      edm::Wrapper<cmsUpgrades::Cluster_TTHit_Map>              CL_TTH_MW;

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

//#include "SLHCUpgradeSimulations/L1Trigger/interface/DTBtiTrigger.h"
//#include "SLHCUpgradeSimulations/L1Trigger/interface/DTTSPhiTrigger.h"
//#include "SLHCUpgradeSimulations/L1Trigger/interface/DTStubMatch.h"
//#include "SLHCUpgradeSimulations/L1Trigger/interface/DTStubMatchPt.h"
//#include "SLHCUpgradeSimulations/L1Trigger/interface/DTTrackerStub.h"
//#include "SLHCUpgradeSimulations/L1Trigger/interface/DTStubMatchesCollection.h"
//#include "SLHCUpgradeSimulations/L1Trigger/interface/DTSeededTracklet.h"


#include "SimDataFormats/SLHC/interface/DTBtiTrigger.h"
#include "SimDataFormats/SLHC/interface/DTTSPhiTrigger.h"
#include "SimDataFormats/SLHC/interface/DTStubMatch.h"
#include "SimDataFormats/SLHC/interface/DTStubMatchPt.h"
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
    DTStubMatchPtVariety                         DTPtc1;
    edm::Wrapper<DTStubMatchPtVariety>           DTPtcw1;

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
