#include "SimDataFormats/SLHC/interface/L1CaloTower.h"

using namespace l1slhc;

L1CaloTower::L1CaloTower()
{
  E_   = 0;
  H_   = 0;
  fineGrain_ = 0;
  iEta_ = 0;
  iPhi_ = 0;

}

L1CaloTower::~L1CaloTower()
{

}


int
L1CaloTower::E() const
{ return E_;}

int
L1CaloTower::H() const
{ return H_;}


int 
L1CaloTower::iEta() const
{ return iEta_;}

int 
L1CaloTower::iPhi() const
{ return iPhi_;}

void
L1CaloTower::setPos(int eta,int phi)
{
  iEta_ = eta;
  iPhi_ = phi;

}

void
L1CaloTower::setParams(int ecal,int hcal,bool fg)
{
  E_  = ecal;
  H_  = hcal;
  fineGrain_ = fg;
}


bool
L1CaloTower::fineGrain() const
{ return fineGrain_;}


