/* SLHC Calo Trigger

M.Bachtis,S.Dasu
University of Wisconsin-Madison
*/

#ifndef JetExapndingModule_h
#define JetExpandingModule_h

#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/ModuleBase.h"
#include "SimDataFormats/SLHC/interface/L1CaloJet.h"
#include "SimDataFormats/SLHC/interface/L1CaloJetFwd.h"
#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/TriggerTowerGeometry.h"


class JetExpandingModule : public L1CaloJetModule
{
 public:
  JetExpandingModule();
  JetExpandingModule(const L1CaloTriggerSetup&);
  ~JetExpandingModule();

  typedef std::map<int,l1slhc::L1CaloRegionRefVector> AdderMap;


  void expandClusters(const edm::Handle<l1slhc::L1CaloJetCollection>&,l1slhc::L1CaloJetCollection&);//Remove Overlap
  
 private:
  void populateLattice(const edm::Handle<l1slhc::L1CaloJetCollection>&); //Import Clusters into the lattice


 AdderMap  addersInput;
};

#endif
