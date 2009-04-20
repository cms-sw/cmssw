#ifndef L1CaloJet_h
#define L1CaloJet_h

/*
This class describves the L1 Reconstructed jet
M.Bachtis,S.Dasu
University of Wisconsin - Madison
*/

#include "DataFormats/Math/interface/LorentzVector.h"
#include <vector>


namespace l1slhc {

class L1CaloJet 
{

public:

  L1CaloJet();
  L1CaloJet(int,int,int);
  ~L1CaloJet();
  //getters
  int iEta() const;
  int iPhi() const;
  int et() const;


  math::PtEtaPhiMLorentzVector p4() const;//returns LorentzVector in eta,phi space



  //Setters
  void setIEta(int);
  void setIPhi(int);
  void setEt(int);



private:
  int iEta_;
  int iPhi_;
  int et_;

};

 typedef std::vector<L1CaloJet> L1CaloJetCollection;


struct HigherJetEt
{
  bool operator()(L1CaloJet cl1,L1CaloJet cl2)
  {
    return cl1.et() >  cl2.et() ; 
  }

};


}
#endif
