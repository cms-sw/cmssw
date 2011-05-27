#ifndef L1CaloJet_h
#define L1CaloJet_h

/*
This class describves the L1 Reconstructed jet
M.Bachtis,S.Dasu
University of Wisconsin - Madison
*/

#include "DataFormats/Math/interface/LorentzVector.h"
#include <vector>
#include "SimDataFormats/SLHC/interface/L1CaloRegionFwd.h"
#include "SimDataFormats/SLHC/interface/L1CaloRegion.h"

namespace l1slhc {

class L1CaloJet 
{

public:

  L1CaloJet();
  L1CaloJet(int,int);
  ~L1CaloJet();

  //getters
  int iEta() const;
  int iPhi() const;
  int E() const;
  bool central() const;
  math::PtEtaPhiMLorentzVector p4() const;//returns LorentzVector in eta,phi space

  //Setters
  void setP4(const math::PtEtaPhiMLorentzVector& p4);
  void setCentral(bool);
  void setE(int);

  void  addConstituent(const L1CaloRegionRef&);
  int   hasConstituent(int,int);
  void  removeConstituent(int,int);
  L1CaloRegionRefVector  getConstituents() const;



private:
  int iEta_;
  int iPhi_;
  int E_;

  L1CaloRegionRefVector constituents_;
  math::PtEtaPhiMLorentzVector p4_;
  bool central_;

};

 typedef std::vector<L1CaloJet> L1CaloJetCollection;


struct HigherJetEt
{
  bool operator()(L1CaloJet cl1,L1CaloJet cl2)
  {
    return cl1.E() >  cl2.E() ; 
  }

};


}

std::ostream& operator<<(std::ostream& s, const l1slhc::L1CaloJet&);

#endif

