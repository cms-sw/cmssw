/* SLHC Calo Trigger
Class that separates minimal isolated clusters from
Jet Fragments after removing Cluster overlap

M.Bachtis,S.Dasu
University of Wisconsin-Madison
*/

#ifndef JetFilteringModule_h
#define JetFilteringModule_h

#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/ModuleBase.h"
#include "SimDataFormats/SLHC/interface/L1CaloJet.h"
#include "SimDataFormats/SLHC/interface/L1CaloJetFwd.h"
#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/TriggerTowerGeometry.h"

class JetFilteringModule : public L1CaloJetModule
{
 public:
  JetFilteringModule();
  JetFilteringModule(const L1CaloTriggerSetup&);
  ~JetFilteringModule();


  void cleanClusters(const edm::Handle<l1slhc::L1CaloJetCollection>&,l1slhc::L1CaloJetCollection&);//Remove Overlap

 private:
  void populateLattice(const edm::Handle<l1slhc::L1CaloJetCollection>&); //Import Clusters into the lattice
};

#endif
