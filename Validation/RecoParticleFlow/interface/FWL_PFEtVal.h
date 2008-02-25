#ifndef FWL_PFEtVal_h
#define FWL_PFEtVal_h
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include <vector>

class TTree;
class TBranch;
class TFile;



class FWL_PFEtVal{

  
 public:
  FWL_PFEtVal();
  ~FWL_PFEtVal();
  void readData();
  TTree* tree_;
  int iEvent_;
  TFile* file_;
  TBranch* RECO_caloJetBranch_;
  TBranch* PF_JetBranch_;
  TBranch* GEN_caloJetBranch_;

  reco::CaloJetCollection *RECO_caloJet_;
  reco::PFJetCollection *PF_Jet_;
  reco::GenJetCollection *GEN_caloJet_;
  
};

#endif
