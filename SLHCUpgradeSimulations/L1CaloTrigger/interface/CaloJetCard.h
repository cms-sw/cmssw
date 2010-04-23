/* SLHC Calo Trigger
Class that Builds Jets from clusters
MBachtis,S.Dasu
University of Wisconsin-Madison
*/

#ifndef JetCard_h
#define JetCard_h

#include "SLHCUpgradeSimulations/L1CaloTrigger/interface/CaloCard.h"
#include "SimDataFormats/SLHC/interface/L1CaloTriggerSetup.h"
#include "SimDataFormats/SLHC/interface/L1CaloCluster.h"
#include "SimDataFormats/SLHC/interface/L1CaloJet.h"


class CaloJetCard : public CaloCard<l1slhc::L1CaloCluster>
{
 public:
  CaloJetCard();
  CaloJetCard(const L1CaloTriggerSetup&);
  ~CaloJetCard();


  //MainAlgorithm
  void makeJets(const l1slhc::L1CaloClusterCollection&,l1slhc::L1CaloJetCollection&);
  void filterJets(l1slhc::L1CaloJetCollection&);

 private:
  void populateLattice(const l1slhc::L1CaloClusterCollection&); //Import Tower Information to the Lattice

  //Setup record
  L1CaloTriggerSetup s;

};

#endif

