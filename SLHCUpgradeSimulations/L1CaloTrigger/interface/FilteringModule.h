/* SLHC Calo Trigger
Class that separates minimal isolated clusters from
Jet Fragments after removing Cluster overlap

M.Bachtis,S.Dasu
University of Wisconsin-Madison
*/

#ifndef FilteringModule_h
#define FilteringModule_h

#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/ModuleBase.h"
#include "SimDataFormats/SLHC/interface/L1CaloCluster.h"
#include "SimDataFormats/SLHC/interface/L1CaloClusterFwd.h"
#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/TriggerTowerGeometry.h"

class FilteringModule : public L1CaloClusterModule
{
 public:
  FilteringModule();
  FilteringModule(const L1CaloTriggerSetup&);
  ~FilteringModule();


  void cleanClusters(const edm::Handle<l1slhc::L1CaloClusterCollection>&,l1slhc::L1CaloClusterCollection&);//Remove Overlap
  std::pair<int,int> calculateClusterPosition(l1slhc::L1CaloCluster&);

 private:
  void populateLattice(const edm::Handle<l1slhc::L1CaloClusterCollection>&); //Import Clusters into the lattice
};

#endif
