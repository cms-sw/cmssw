#include "SimDataFormats/SLHC/interface/L1CaloTriggerSetup.h"

#include "SimDataFormats/SLHC/interface/L1CaloTower.h"
#include "DataFormats/Common/interface/Wrapper.h"
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
