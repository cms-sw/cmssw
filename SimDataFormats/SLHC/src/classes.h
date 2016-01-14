/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
/// ////////////////////////////////////////

#include "DataFormats/Common/interface/Wrapper.h"

/**********************/
/** L1 TRACK TRIGGER **/
/**********************/

#include "SimDataFormats/SLHC/interface/StackedTrackerTypes.h"
#include "SimDataFormats/Track/interface/SimTrack.h"

namespace
{
  namespace
  {
    std::vector<edm::Ptr<SimTrack> > STk_PC;

    Ref_PSimHit_    PSH_;
    Ref_PixelDigi_  PD_;

    std::vector<Ref_PixelDigi_> VRP;

  }
}

/************************/
/** L1 DT MUON TRIGGER **/
/************************/
/*
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
*/

/*********************/
/** L1 CALO TRIGGER **/
/*********************/

#include "SimDataFormats/SLHC/interface/L1CaloTriggerSetup.h"
#include "SimDataFormats/SLHC/interface/L1CaloTower.h"
#include "SimDataFormats/SLHC/interface/L1CaloTowerFwd.h"
#include "SimDataFormats/SLHC/interface/L1CaloCluster.h"
#include "SimDataFormats/SLHC/interface/L1CaloClusterFwd.h"
#include "SimDataFormats/SLHC/interface/L1CaloClusterWithSeed.h"
#include "SimDataFormats/SLHC/interface/L1CaloClusterWithSeedFwd.h"
#include "SimDataFormats/SLHC/interface/L1CaloJet.h"
#include "SimDataFormats/SLHC/interface/L1CaloJetFwd.h"
#include "SimDataFormats/SLHC/interface/L1CaloRegion.h"
#include "SimDataFormats/SLHC/interface/L1CaloRegionFwd.h"

#include "SimDataFormats/SLHC/interface/L1TowerJet.h"
#include "SimDataFormats/SLHC/interface/L1TowerJetFwd.h"

#include "SimDataFormats/SLHC/interface/EtaPhiContainer.h"

#include "SimDataFormats/SLHC/interface/L1EGCrystalCluster.h"


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

    l1slhc::L1CaloClusterWithSeed                 calocls;
    std::vector<l1slhc::L1CaloClusterWithSeed>    l1calocls;
	l1slhc::L1CaloClusterWithSeedCollection		  l1caloclscoll;
    edm::Wrapper< l1slhc::L1CaloClusterWithSeedCollection >   wl1calocls;

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


    l1slhc::L1EGCrystalCluster                       egcrystalcluster;
    std::vector<l1slhc::L1EGCrystalCluster>         l1egcrystalclustervec;
    l1slhc::L1EGCrystalClusterCollection            l1egcrystalclustercoll;
    edm::Wrapper<l1slhc::L1EGCrystalClusterCollection>   wl1egcrystalclustercoll;

  }
}

