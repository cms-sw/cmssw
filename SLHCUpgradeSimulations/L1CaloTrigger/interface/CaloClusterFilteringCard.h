/* SLHC Calo Trigger
Class that separates minimal isolated clusters from
Jet Fragments after removing Cluster overlap

M.Bachtis,S.Dasu
University of Wisconsin-Madison
*/

#ifndef CaloClusterFilteringCard_h
#define CaloClusterFilteringCard_h

#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/CaloCard.h"
#include "SimDataFormats/SLHC/interface/L1CaloCluster.h"
#include "SimDataFormats/SLHC/interface/L1CaloTriggerSetup.h"
#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/TriggerTowerGeometry.h"

class CaloClusterFilteringCard : public CaloCard<l1slhc::L1CaloCluster>
{
 public:
  CaloClusterFilteringCard();
  CaloClusterFilteringCard(const L1CaloTriggerSetup&);
  ~CaloClusterFilteringCard();


  void cleanClusters(const l1slhc::L1CaloClusterCollection&,l1slhc::L1CaloClusterCollection&);//Remove Overlap
  std::pair<int,int> calculateClusterPosition(l1slhc::L1CaloCluster&);

 private:
  L1CaloTriggerSetup s;
  int clusterCut_;
  void populateLattice(const l1slhc::L1CaloClusterCollection&); //Import Clusters into the lattice
};

#endif

