/* SLHC Calo Trigger
Class that performs the Calorimeter Clustering
MBachtis,S.Dasu
Univeristy of Wisconsin-Madison
*/

#ifndef JetModule_h
#define JetModule_h

#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/ModuleBase.h"
#include "SimDataFormats/SLHC/interface/L1CaloTriggerSetup.h"
#include "SimDataFormats/SLHC/interface/L1CaloJet.h"
#include "SimDataFormats/SLHC/interface/L1CaloRegionFwd.h"


class JetModule : public L1CaloRegionModule
{
 public:
  JetModule();
  JetModule(const L1CaloTriggerSetup&);
  ~JetModule();

  //MainAlgorithm:Creates Clusters from the Lattice
  void clusterize(l1slhc::L1CaloJetCollection&,const edm::Handle<l1slhc::L1CaloRegionCollection>&);

 private:
  void populateLattice(const edm::Handle<l1slhc::L1CaloRegionCollection>&); //Import Tower Information to the Lattice
  void calculateJetPosition(l1slhc::L1CaloJet&);
};

#endif
