#include "SimDataFormats/SLHC/interface/L1CaloRegion.h"

#include <math.h>


using namespace l1slhc;

L1CaloRegion::L1CaloRegion()
{
  iEta_ = 0;
  iPhi_ = 0;
  E_=0;

}

L1CaloRegion::~L1CaloRegion()
{}

L1CaloRegion::L1CaloRegion(int iEta, int iPhi,int E)
{
  iEta_ = iEta;
  iPhi_ = iPhi;
  E_ = E;
}



int
L1CaloRegion::iEta() const
{
  return iEta_;
}

int
L1CaloRegion::iPhi() const
{
  return iPhi_;
}

int
L1CaloRegion::E() const
{
  return E_;
}
