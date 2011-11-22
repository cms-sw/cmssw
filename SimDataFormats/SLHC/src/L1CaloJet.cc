#include "SimDataFormats/SLHC/interface/L1CaloJet.h"

using namespace l1slhc;

L1CaloJet::L1CaloJet()
{
  iEta_=0;
  iPhi_=0;
  E_=0;
}



L1CaloJet::L1CaloJet(int iEta,int iPhi)
{
  iEta_=iEta;
  iPhi_=iPhi;
  E_=0;
}

L1CaloJet::~L1CaloJet()
{


}

void L1CaloJet::setP4(const math::PtEtaPhiMLorentzVector& p4)
{
  p4_ = p4;
}

void L1CaloJet::setCentral(bool central)
{
  central_ = central;
}

int
L1CaloJet::iEta() const
{
  return iEta_;
} 


int
L1CaloJet::iPhi() const
{
  return iPhi_;
} 


int
L1CaloJet::E() const
{
  return E_;
} 

bool
L1CaloJet::central() const
{
  return central_;
} 


void
L1CaloJet::setE(int E)
{
  E_=E;
}

math::PtEtaPhiMLorentzVector 
L1CaloJet::p4() const
{
  return p4_;
}


void 
L1CaloJet::addConstituent(const L1CaloRegionRef& region)
{
  E_+=region->E();
  constituents_.push_back(region);
}

L1CaloRegionRefVector
L1CaloJet::getConstituents() const
{
  return constituents_;
}

int  
L1CaloJet::hasConstituent(int eta,int phi)
{
  int pos=-1;
  for(unsigned int i=0;i<constituents_.size();++i) {
    L1CaloRegionRef tower = constituents_.at(i);
    if(tower->iEta()==eta+iEta_&&tower->iPhi()==phi+iPhi_) {
      pos = i;
      break;
    }
  }

  return pos;
}

void  
L1CaloJet::removeConstituent(int eta, int phi)
{
  int pos = hasConstituent(eta,phi);
  if(pos!=-1) {
    E_=E_-constituents_.at(pos)->E(); 
    constituents_.erase(constituents_.begin()+pos);
  }
}


// pretty print
// pretty print
std::ostream& operator<<(std::ostream& s, const l1slhc::L1CaloJet& cand) {
  s << "L1CaloJet ";
  s << "iEta=" << cand.iEta() << "iPhi="<<cand.iPhi() << "\n";
  s << "Constituents"<< "\n";
  for(unsigned int i=0;i<cand.getConstituents().size();++i)
    s << "---------iEta=" << cand.getConstituents().at(i)->iEta() << "iPhi="<<cand.getConstituents().at(i)->iPhi() <<"ET="<<cand.getConstituents().at(i)->E()<<"\n";
  return s;
}
