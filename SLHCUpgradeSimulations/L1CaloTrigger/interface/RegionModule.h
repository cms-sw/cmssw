/* SLHC Calo Trigger
Class that performs the Calorimeter Clustering
MBachtis,S.Dasu
Univeristy of Wisconsin-Madison
*/

#ifndef RegionModule_h
#define RegionModule_h

#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/ModuleBase.h"
#include "SimDataFormats/SLHC/interface/L1CaloTower.h"
#include "SimDataFormats/SLHC/interface/L1CaloRegionFwd.h"


class RegionModule : public ModuleBase<l1slhc::L1CaloTower>
{
 public:
  RegionModule();
  RegionModule(const L1CaloTriggerSetup&);
  ~RegionModule();

  //MainAlgorithm:Creates Clusters from the Lattice
  void clusterize(l1slhc::L1CaloRegionCollection&,const edm::Handle<l1slhc::L1CaloTowerCollection>&);
  //Set Tower, Cluster thresholds
  void setThresholds(int,int,int); 

 private:
  void populateLattice(const edm::Handle<l1slhc::L1CaloTowerCollection>&); //Import Tower Information to the Lattice

};

#endif
