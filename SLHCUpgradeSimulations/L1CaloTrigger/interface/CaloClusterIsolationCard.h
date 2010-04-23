/* SLHC Calo Trigger
Class that calculates Isolation Deposits
MBachtis,S.Dasu
Univeristy of Wisconsin-Madison
*/

#ifndef CaloClusterIsolationCard_h
#define CaloClusterIsolationCard_h

#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/CaloCard.h"
#include "SimDataFormats/SLHC/interface/L1CaloTriggerSetup.h"
#include "SimDataFormats/SLHC/interface/L1CaloCluster.h"


class CaloClusterIsolationCard : public CaloCard<l1slhc::L1CaloCluster>
{
 public:
  CaloClusterIsolationCard();
  CaloClusterIsolationCard(const L1CaloTriggerSetup&);
  ~CaloClusterIsolationCard();


  //MainAlgorithm:Creates Clusters from the Lattice
  void isoDeposits(const l1slhc::L1CaloClusterCollection&,l1slhc::L1CaloClusterCollection&);

 private:
  void populateLattice(const l1slhc::L1CaloClusterCollection&); //Import Tower Information to the Lattice

  //Setup record
  L1CaloTriggerSetup s;

};

#endif

