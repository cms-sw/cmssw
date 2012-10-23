/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
/// Written by:                          ///
/// Unknown                              ///
///                                      ///
/// Changed by:                          ///
/// Emmanuele Salvati                    ///
/// Cornell                              ///
/// 2010, June                           ///
/// Pierluigi Zotto, Nicola Pozzobon     ///
/// Padova                               ///
/// 2012, Oct                            ///
///                                      ///
/// Removed TrackTriggerHits             ///
/// Fixed some issues about Clusters     ///
/// Added L1DT data formats              ///
/// Added L1Calo data formats (agreement ///
/// with Evan Friis)                     ///
/// ////////////////////////////////////////

#include "DataFormats/Common/interface/Wrapper.h"

/**********************/
/** L1 TRACK TRIGGER **/
/**********************/

#include "SimDataFormats/SLHC/interface/StackedTrackerTypes.h"

namespace {
  namespace {

    /// Beam storage class
    cmsUpgrades::Beam_                                BeamSC_;
    edm::Wrapper<cmsUpgrades::Beam_>                  BeamSC_W;
    cmsUpgrades::Beam_Collection                      BeamSC_C;
    edm::Wrapper<cmsUpgrades::Beam_Collection>        BeamSC_CW; 

    cmsUpgrades::Ref_PSimHit_    PSH_;
    cmsUpgrades::Ref_PixelDigi_  PD_;

    /// SimHit type
    cmsUpgrades::L1TkStub_PSimHit_                         S_PSH_;
    cmsUpgrades::L1TkStub_PSimHit_Collection               S_PSH_C;
    edm::Wrapper<cmsUpgrades::L1TkStub_PSimHit_Collection> S_PSH_CW;

    cmsUpgrades::L1TkTracklet_PSimHit_                         T_PSH_;
    cmsUpgrades::L1TkTracklet_PSimHit_Collection               T_PSH_C;
    edm::Wrapper<cmsUpgrades::L1TkTracklet_PSimHit_Collection> T_PSH_CW;

    cmsUpgrades::L1TkTrack_PSimHit_                         L1T_PSH_;
    cmsUpgrades::L1TkTrack_PSimHit_Collection               L1T_PSH_C;
    edm::Wrapper<cmsUpgrades::L1TkTrack_PSimHit_Collection> L1T_PSH_CW;

    /// PixelDigi type
    cmsUpgrades::L1TkStub_PixelDigi_                         S_PD_;
    cmsUpgrades::L1TkStub_PixelDigi_Collection               S_PD_C;
    edm::Wrapper<cmsUpgrades::L1TkStub_PixelDigi_Collection> S_PD_CW;

    cmsUpgrades::L1TkTracklet_PixelDigi_                         T_PD_;
    cmsUpgrades::L1TkTracklet_PixelDigi_Collection               T_PD_C;
    edm::Wrapper<cmsUpgrades::L1TkTracklet_PixelDigi_Collection> T_PD_CW;

    cmsUpgrades::L1TkTrack_PixelDigi_                         L1T_PD_;
    cmsUpgrades::L1TkTrack_PixelDigi_Collection               L1T_PD_C;
    edm::Wrapper<cmsUpgrades::L1TkTrack_PixelDigi_Collection> L1T_PD_CW;


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


/************************/
/** L1 DT MUON TRIGGER **/
/************************/

#include <vector>
#include <set>

#include "L1Trigger/DTTrackFinder/interface/L1MuDTTrack.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTAddressArray.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTSecProcId.h"
#include "L1Trigger/DTTrackFinder/src/L1MuDTTrackSegLoc.h"

#include "SimDataFormats/SLHC/interface/DTBtiTrigger.h"
#include "SimDataFormats/SLHC/interface/DTTSPhiTrigger.h"
#include "SimDataFormats/SLHC/interface/DTTSThetaTrigger.h"
#include "SimDataFormats/SLHC/interface/DTMatch.h"
#include "SimDataFormats/SLHC/interface/DTMatchPt.h"
#include "SimDataFormats/SLHC/interface/DTMatchPtVariety.h"
#include "SimDataFormats/SLHC/interface/DTMatchPtAlgorithms.h"
#include "SimDataFormats/SLHC/interface/DTTrackerStub.h"
#include "SimDataFormats/SLHC/interface/DTTrackerTracklet.h"
#include "SimDataFormats/SLHC/interface/DTTrackerTrack.h"
#include "SimDataFormats/SLHC/interface/DTMatchesCollection.h"
#include "SimDataFormats/SLHC/interface/DTSeededStubTrack.h"

namespace {
  namespace {
    edm::Wrapper<DTBtiTrigger>                   Bti1;
    std::vector<DTBtiTrigger>                    Btv1;
    edm::Wrapper<std::vector<DTBtiTrigger> >     Btc1;

    edm::Wrapper<DTTSPhiTrigger>                 phi1;
    std::vector<DTTSPhiTrigger>                  phiv1;
    edm::Wrapper<std::vector<DTTSPhiTrigger> >   phic1;

    edm::Wrapper<DTMatchPt>                  DTPt1;
    std::vector<DTMatchPt>                   DTPtV1;
    edm::Wrapper<std::vector<DTMatchPt> >    DTPtW1;

    edm::Wrapper<DTMatch>                    DTM1;
    std::vector<DTMatch*>                    DTMv1;
    edm::Wrapper<std::vector<DTMatch*> >     DTSMwv1;

    std::vector<DTMatchPt*>                    DTMm1;
    edm::Wrapper<std::vector<DTMatchPt*> >     DTSMwm1;

    std::vector<TrackerStub*>                    DTTSv1;
    edm::Wrapper<std::vector<TrackerStub*> >     DTTSwv1;
    edm::Wrapper<DTMatchesCollection>            DTSMc1;
    
    std::vector<TrackerTracklet*>                DTTTv1;
    edm::Wrapper<std::vector<TrackerTracklet*> > DTTTwv1;
    
    std::vector<TrackerTrack*>                DTTTTv1;
    edm::Wrapper<std::vector<TrackerTrack*> > DTTTTwv1;

    edm::Wrapper<TrackerStub>                      TS1;
    edm::Wrapper<TrackerTracklet>                  TT1;
    edm::Wrapper<TrackerTrack>                     XT1;
    edm::Wrapper<lt_stub>                          LT1;
    std::set<TrackerStub*, lt_stub>                TSv1;
    edm::Wrapper<std::set<TrackerStub*, lt_stub> > TSwv1;

    DTSeededStubTrack                               DTTST1;
    edm::Wrapper<DTSeededStubTrack>                 DTTSTw1;
    std::vector<DTSeededStubTrack*>                 DTTSTv1;
    edm::Wrapper<std::vector<DTSeededStubTrack*> >  DTTSTa1;
    DTSeededStubTracksCollection                    DTTSTc1;
    edm::Wrapper<DTSeededStubTracksCollection>      DTTSTcw1; 
  
  }
}


/*********************/
/** L1 CALO TRIGGER **/
/*********************/

#include "SimDataFormats/SLHC/interface/L1CaloTriggerSetup.h"
#include "SimDataFormats/SLHC/interface/L1CaloTower.h"
#include "SimDataFormats/SLHC/interface/L1CaloTowerFwd.h"
#include "SimDataFormats/SLHC/interface/L1CaloCluster.h"
#include "SimDataFormats/SLHC/interface/L1CaloClusterFwd.h"
#include "SimDataFormats/SLHC/interface/L1CaloJet.h"
#include "SimDataFormats/SLHC/interface/L1CaloJetFwd.h"
#include "SimDataFormats/SLHC/interface/L1CaloRegion.h"
#include "SimDataFormats/SLHC/interface/L1CaloRegionFwd.h"

#include "SimDataFormats/SLHC/interface/L1TowerJet.h"
#include "SimDataFormats/SLHC/interface/L1TowerJetFwd.h"

#include "SimDataFormats/SLHC/interface/EtaPhiContainer.h"

namespace {
  namespace {

    l1slhc::L1CaloTower                   tower;
    std::vector<l1slhc::L1CaloCluster>    l1calotl;
    l1slhc::L1CaloTowerRef                towerRef;

    l1slhc::L1CaloTowerCollection                towerColl;
    l1slhc::L1CaloTowerRefVector                 towerRefColl;
    edm::Wrapper<l1slhc::L1CaloTowerCollection>  wtowerColl;
    edm::Wrapper<l1slhc::L1CaloTowerRefVector>   wtowerRefColl;

    l1slhc::L1CaloCluster                             calocl;
    std::vector<l1slhc::L1CaloCluster>                l1calocl;
    l1slhc::L1CaloClusterCollection                   l1caloclcoll;
    edm::Wrapper< l1slhc::L1CaloClusterCollection >   wl1calocl;

    l1slhc::L1CaloJet                             calojet;
    std::vector<l1slhc::L1CaloJet>                l1calojetvec;
    l1slhc::L1CaloJetCollection                   l1calojetcoll;
    edm::Wrapper< l1slhc::L1CaloJetCollection >   wl1calojetcol;

    l1slhc::L1CaloRegion                                caloregion;
    std::vector<l1slhc::L1CaloRegion>                   l1caloregion;
    l1slhc::L1CaloRegionRef                             caloregionRef;
    l1slhc::L1CaloRegionCollection                      caloregionC;
    l1slhc::L1CaloRegionRefVector                       caloregionRefC;

    edm::Wrapper<l1slhc::L1CaloRegionCollection>        wcaloregionC;
    edm::Wrapper<l1slhc::L1CaloRegionRefVector>         qaloregionRefC;

    l1slhc::L1TowerJet                             towerjet;
    std::vector<l1slhc::L1TowerJet>                l1towerjetvec;
    l1slhc::L1TowerJetCollection                   l1towerjetcoll;
    edm::Wrapper< l1slhc::L1TowerJetCollection >   wl1towerjetcol;

  }
}


