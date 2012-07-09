#include "SimDataFormats/SLHC/interface/L1CaloCluster.h"

#include <math.h>


using namespace l1slhc;

L1CaloCluster::L1CaloCluster()
{
  iEta_ = 0;
  iPhi_ = 0;
  fg_=false;
  eGamma_=false;
  leadTowerTau_ =false;
  eGammaValue_ = 0;
  innerEta_=0;
  innerPhi_=0;
  isoClustersEG_=0;
  isoClustersTau_=0;
  isoEG_=false;
  isoTau_=false;
  central_=false;
  E_=0;
  p4_ = math::PtEtaPhiMLorentzVector(0.001,0,0,0.);

}

L1CaloCluster::L1CaloCluster(int iEta, int iPhi)
{
  iEta_ = iEta;
  iPhi_ = iPhi;
  E_=0;
  fg_=false;
  eGamma_=false;
  eGammaValue_ = 0;
  leadTowerTau_ =false;
  isoEG_=false;
  isoTau_=false;
  central_=false;
  innerEta_=0;
  innerPhi_=0;
  isoClustersEG_=0;
  isoClustersTau_=0;
  p4_ = math::PtEtaPhiMLorentzVector(0.001,0,0,0.);
}



L1CaloCluster::~L1CaloCluster()
{}


int
L1CaloCluster::iEta() const
{
  return iEta_;
}

int
L1CaloCluster::iPhi() const
{
  return iPhi_;
}

int
L1CaloCluster::innerEta() const
{
  return innerEta_;
}

int
L1CaloCluster::innerPhi() const
{
  return innerPhi_;
}


int
L1CaloCluster::E() const
{
  return E_;
}


bool
L1CaloCluster::fg() const
{
  return fg_;
}

bool
L1CaloCluster::eGamma() const
{
  return eGamma_;
}

bool
L1CaloCluster::hasLeadTower() const
{
  return leadTowerTau_;
}


int
L1CaloCluster::eGammaValue() const
{
  return eGammaValue_;
}

bool
L1CaloCluster::isoEG() const
{
  return isoEG_;
}

bool
L1CaloCluster::isoTau() const
{
  return isoTau_;
}

bool
L1CaloCluster::isCentral() const
{
  return central_;
}

int
L1CaloCluster::isoClustersEG() const
{
  return isoClustersEG_;
}

int
L1CaloCluster::isoClustersTau() const
{
  return isoClustersTau_;
}


bool
L1CaloCluster::isEGamma() const
{
  return (!fg() && eGamma() &&isCentral()); 
}

bool
L1CaloCluster::isIsoEGamma() const
{
  return (!fg() && eGamma() &&isoEG() && isCentral()); 
}

bool
L1CaloCluster::isIsoTau() const
{
  return hasLeadTower()&&isoTau()&& isCentral(); 
}

bool
L1CaloCluster::isTau() const
{
  return hasLeadTower() && isCentral(); 
}


void 
L1CaloCluster::setE(int E )
{
  E_ = E;
}


void 
L1CaloCluster::setConstituents(const L1CaloTowerRefVector& cons)
{
  constituents_ = cons;
}

L1CaloTowerRefVector
L1CaloCluster::getConstituents() const 
{
  return constituents_;
}  


void 
L1CaloCluster::setFg(bool fg)
{
  fg_ = fg;
}


void 
L1CaloCluster::setEGamma(bool eg)
{
  eGamma_ = eg;
}

void 
L1CaloCluster::setLeadTower(bool eg)
{
  leadTowerTau_ = eg;
}


void 
L1CaloCluster::setEGammaValue(int eg)
{
  eGammaValue_ = eg;
}

void 
L1CaloCluster::setIsoEG(bool eg)
{
  isoEG_ = eg;
}

void 
L1CaloCluster::setIsoTau(bool eg)
{
  isoTau_ = eg;
}

void 
L1CaloCluster::setIsoClusters(int eg,int tau)
{
  isoClustersEG_ = eg;
  isoClustersTau_ = tau;
}


void 
L1CaloCluster::setCentral(bool eg)
{
  central_ = eg;
}

void 
L1CaloCluster::setPosBits(int eta,int phi)
{
  innerEta_=eta;
  innerPhi_=phi;

}

void 
L1CaloCluster::setLorentzVector(const math::PtEtaPhiMLorentzVector& v)
{
  p4_=v;
}


void 
L1CaloCluster::addConstituent(const L1CaloTowerRef& tower)
{
  E_+=tower->E();
  E_+=tower->H();
  constituents_.push_back(tower);
}

int  
L1CaloCluster::hasConstituent(int eta,int phi)
{
  int pos=-1;
  for(unsigned int i=0;i<constituents_.size();++i) {
    L1CaloTowerRef tower = constituents_.at(i);
    if(tower->iEta()==iEta_+eta && tower->iPhi()==iPhi_+phi) {
      pos = i;
    }
  }
  return pos;
}


L1CaloTowerRef
L1CaloCluster::getConstituent(int pos)
{
  return constituents_.at(pos);
}



void  
L1CaloCluster::removeConstituent(int eta, int phi)
{

  int pos = hasConstituent(eta,phi);

  if(pos!=-1) {
    E_=E_-constituents_.at(pos)->E(); 
    E_=E_-constituents_.at(pos)->H(); 
    constituents_.erase(constituents_.begin()+pos);
  }
}



math::PtEtaPhiMLorentzVector 
L1CaloCluster::p4() const
{
  return p4_;
}

// pretty print
std::ostream& operator<<(std::ostream& s, const L1CaloCluster& cand) {
  s << "L1CaloCluster ";
  s << "iEta=" << cand.iEta() << "iPhi="<<cand.iPhi() <<" E"<<cand.E()<< "eta "<< cand.p4().eta() <<"phi "<< cand.p4().phi() <<"pt "<<cand.p4().pt() << "egamma " <<cand.eGammaValue() <<"central "<<cand.isCentral()<< "fg"<< cand.fg() << "\n";
  s << "Constituents"<< "\n";
  for(unsigned int i=0;i<cand.getConstituents().size();++i)
    s << "---------iEta=" << cand.getConstituents().at(i)->iEta() << "iPhi="<<cand.getConstituents().at(i)->iPhi() <<"ET="<<cand.getConstituents().at(i)->E()<<"\n";
  return s;
}
