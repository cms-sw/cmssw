#include "SimDataFormats/SLHC/interface/L1CaloTower.h"

using namespace l1slhc;

L1CaloTower::L1CaloTower():
  mEcal(0),
  mHcal(0),
  mIeta(0),
  mIphi(0),
  mEcalFG(false),
  mHcalFG(false)
{}

L1CaloTower::~L1CaloTower()
{}


void
L1CaloTower::setPos(const int& aIeta,const int& aIphi)
{
	mIeta = aIeta;
	mIphi = aIphi;
}


/*void
L1CaloTower::setParams(const int& aEcal,const int& aHcal,const bool& aFG)
{
  mEcal  = aEcal;
  mHcal  = aHcal;
  mEcalFG = aFG;
}*/


void
L1CaloTower::setEcal( const int& aEcal , const bool& aFG )
{
	mEcal  = aEcal;
	mEcalFG = aFG;
}

void
L1CaloTower::setHcal( const int& aHcal , const bool& aFG )
{
	mHcal  = aHcal;
	mHcalFG = aFG;
}




const int&
L1CaloTower::E() const
{
	return mEcal;
}

const int&
L1CaloTower::H() const
{
	return mHcal;
}


const int& 
L1CaloTower::iEta() const
{
	return mIeta;
}

const int& 
L1CaloTower::iPhi() const
{
	return mIphi;
}

const bool&
L1CaloTower::EcalFG() const
{
	return mEcalFG;
}

const bool&
L1CaloTower::HcalFG() const
{
	return mHcalFG;
}


