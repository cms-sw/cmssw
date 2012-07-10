#include "DataFormats/Common/interface/Wrapper.h"

/* ========================================================================================= */
/* ==================================== CALO TRIGGER INCLUDES ==================================== */
/* ========================================================================================= */

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

    l1slhc::L1CaloTower      tower;
    std::vector<l1slhc::L1CaloCluster>    l1calotl;
    l1slhc::L1CaloTowerRef   towerRef;

    l1slhc::L1CaloTowerCollection towerColl;
    l1slhc::L1CaloTowerRefVector  towerRefColl;
    edm::Wrapper<l1slhc::L1CaloTowerCollection>   wtowerColl;
    edm::Wrapper<l1slhc::L1CaloTowerRefVector>   wtowerRefColl;

    l1slhc::L1CaloCluster                 calocl;
    std::vector<l1slhc::L1CaloCluster>    l1calocl;
	l1slhc::L1CaloClusterCollection		  l1caloclcoll;
    edm::Wrapper< l1slhc::L1CaloClusterCollection >   wl1calocl;


    l1slhc::L1CaloJet                     calojet;
    std::vector<l1slhc::L1CaloJet>       l1calojetvec;
 	l1slhc::L1CaloJetCollection		  l1calojetcoll;
    edm::Wrapper< l1slhc::L1CaloJetCollection >   wl1calojetcol;

    l1slhc::L1CaloRegion                                caloregion;
    std::vector<l1slhc::L1CaloRegion>   				l1caloregion;
    l1slhc::L1CaloRegionRef                             caloregionRef;
    l1slhc::L1CaloRegionCollection                      caloregionC;
    l1slhc::L1CaloRegionRefVector                       caloregionRefC;


    edm::Wrapper<l1slhc::L1CaloRegionCollection>        wcaloregionC;
    edm::Wrapper<l1slhc::L1CaloRegionRefVector>         qaloregionRefC;
  

    l1slhc::L1TowerJet                     towerjet;
    std::vector<l1slhc::L1TowerJet>       l1towerjetvec;
 	l1slhc::L1TowerJetCollection		  l1towerjetcoll;
    edm::Wrapper< l1slhc::L1TowerJetCollection >   wl1towerjetcol; 


  }
}

