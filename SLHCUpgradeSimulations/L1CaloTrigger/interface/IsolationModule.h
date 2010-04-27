/* SLHC Calo Trigger
Class that calculates Isolation Deposits
MBachtis,S.Dasu
Univeristy of Wisconsin-Madison
*/

#ifndef IsolationModule_h
#define IsolationModule_h

#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/ModuleBase.h"
#include "SimDataFormats/SLHC/interface/L1CaloClusterFwd.h"


class IsolationModule : public L1CaloClusterModule
{
 public:
  IsolationModule();
  IsolationModule(const L1CaloTriggerSetup&);
  ~IsolationModule();


  //MainAlgorithm:Creates Clusters from the Lattice
  void isoDeposits(const edm::Handle<l1slhc::L1CaloClusterCollection>&,l1slhc::L1CaloClusterCollection&);

 private:
  void populateLattice(const edm::Handle<l1slhc::L1CaloClusterCollection>&); //Import Tower Information to the Lattice
  bool isoLookupTable(int clusters, int coeffA,int coeffB,int E); //takes as input the # Clusters the isolation coefficients
                                                       //and the energy of teh central cluster and gives the lookup result  

};

#endif
