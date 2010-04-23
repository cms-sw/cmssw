/* SLHC Calo Trigger
Class that performs the Calorimeter Clustering
MBachtis,S.Dasu
Univeristy of Wisconsin-Madison
*/

#ifndef ClusteringCard_h
#define ClusteringCard_h

#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/CaloCard.h"
#include "SimDataFormats/SLHC/interface/L1CaloTriggerSetup.h"
#include "SimDataFormats/SLHC/interface/L1CaloTower.h"
#include "SimDataFormats/SLHC/interface/L1CaloCluster.h"


class CaloClusteringCard : public CaloCard<l1slhc::L1CaloTower>
{
 public:
  CaloClusteringCard();
  CaloClusteringCard(const L1CaloTriggerSetup&);
  ~CaloClusteringCard();


  //MainAlgorithm:Creates Clusters from the Lattice
  void clusterize(l1slhc::L1CaloClusterCollection&,const l1slhc::L1CaloTowerCollection&);
  void setThresholds(int,int,int); //Set Tower, Cluster thresholds

 private:
  void populateLattice(const l1slhc::L1CaloTowerCollection&); //Import Tower Information to the Lattice

  //Setup record
  L1CaloTriggerSetup s;

};

#endif

