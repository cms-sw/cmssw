/* SLHC Calo Trigger
Class that performs the Calorimeter Clustering
MBachtis,S.Dasu
Univeristy of Wisconsin-Madison
*/

#ifndef ClusteringModule_h
#define ClusteringModule_h

#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/ModuleBase.h"
#include "SimDataFormats/SLHC/interface/L1CaloTower.h"
#include "SimDataFormats/SLHC/interface/L1CaloCluster.h"
#include "SimDataFormats/SLHC/interface/L1CaloTowerFwd.h"
#include "SimDataFormats/SLHC/interface/L1CaloClusterFwd.h"


class ClusteringModule : public L1CaloTowerModule
{
 public:
  ClusteringModule();
  ClusteringModule(const L1CaloTriggerSetup&);
  ~ClusteringModule();

  //MainAlgorithm:Creates Clusters from the Lattice
  void clusterize(l1slhc::L1CaloClusterCollection&,const edm::Handle<l1slhc::L1CaloTowerCollection>&);

  //Set Tower, Cluster thresholds
  void setThresholds(int,int,int); 

 private:
  void populateLattice(const edm::Handle<l1slhc::L1CaloTowerCollection>&); //Import Tower Information to the Lattice



};

#endif
