#include "SimDataFormats/SLHC/interface/L1CaloJet.h"

using namespace l1slhc;

L1CaloJet::L1CaloJet()
{
  iEta_=0;
  iPhi_=0;
  et_=0;
}



L1CaloJet::L1CaloJet(int iEta,int iPhi,int et)
{
  iEta_=iEta;
  iPhi_=iPhi;
  et_=et;

}

L1CaloJet::~L1CaloJet()
{


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
L1CaloJet::et() const
{
  return et_;
} 



void
L1CaloJet::setIEta(int eta)
{
   iEta_=eta;
} 


void
L1CaloJet::setIPhi(int phi)
{
  iPhi_=phi;
} 


void
L1CaloJet::setEt(int et)
{
  et_=et;
} 


math::PtEtaPhiMLorentzVector 
L1CaloJet::p4() const
{
  int et = et_;


  //Calculate float value of eta for barrel+endcap(L.Gray)
  double eta =-1982;//an important year...
  
  double etaOffset=0.087/2.0;

  int abs_eta = abs(iEta_);
  const double endcapEta[8] = {0.09,0.1,0.113,0.129,0.15,0.178,0.15,0.35};

  if(abs_eta <=20)
    eta =  (abs_eta*0.0870)-etaOffset;
  else
    {
      int offset = abs(iEta_) -21;
      eta = (20*0.0870);//-etaOffset;
      for(int i = 0;i<= offset;++i)
	{
	  eta+=endcapEta[i];
	}
      eta-=endcapEta[abs(iEta_)-21]/2.;
    }

  if(iEta_<0) eta  = -eta;
  
  double phi = (iPhi_*0.087)-etaOffset;
  double Et= (double)et/2.;

  return math::PtEtaPhiMLorentzVector(Et,eta,phi,0. ) ;
}


