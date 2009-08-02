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
  for(size_t i=0;i<4;++i)
    towerE_.push_back(0);

  p4_ = math::PtEtaPhiMLorentzVector(0.001,0,0,0.);

}

L1CaloCluster::L1CaloCluster(int iEta, int iPhi)
{
  iEta_ = iEta;
  iPhi_ = iPhi;

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


  for(size_t i=0;i<4;++i)
    towerE_.push_back(0);

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
  int E=0;

  for(int i=0;i<4;++i)
    E+=towerE_[i];

  return E;
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
L1CaloCluster::hasLeadTowerTau() const
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
L1CaloCluster::isTau() const
{
  return hasLeadTowerTau()&&isoTau()&& isCentral(); 
}


int 
L1CaloCluster::towerE(int i) const
{
  return towerE_[i];

}

int 
L1CaloCluster::seedTowerE() const
{
  int max = 0;
  for(unsigned int i=0;i<4;i++)
    if(towerE(i)>max)
      max=towerE(i);

 return max;

}


void 
L1CaloCluster::setTower(int i,int Et)
{
  towerE_[i] = Et;
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
L1CaloCluster::setLeadTowerTau(bool eg)
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



math::PtEtaPhiMLorentzVector 
L1CaloCluster::p4() const
{
  return p4_;
}
