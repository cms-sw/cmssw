#include <iostream.h>

class HistoCompare {

 public:

  HistoCompare() { std::cout << "Initializing HistoCompare... " << std::endl; } ;

  void PVCompute(TH1 * oldHisto , TH1 * newHisto , TText * te );
  void PVCompute(TH2 * oldHisto , TH2 * newHisto , TText * te );
  void PVCompute(TProfile * oldHisto , TProfile * newHisto , TText * te );

 private:
  
  Double_t mypv;

  TH1 * myoldHisto1;
  TH1 * mynewHisto1;

  TH2 * myoldHisto2;
  TH2 * mynewHisto2;

  TProfile * myoldProfile;
  TProfile * mynewProfile;

  TText * myte;

};

HistoCompare::PVCompute(TH1 * oldHisto , TH1 * newHisto , TText * te )
{

  myoldHisto1 = oldHisto;
  mynewHisto1 = newHisto;
  myte = te;
  Double_t * res;

  Double_t mypv = 0.;
  Double_t chi2;
  Int_t ndf, igood;
  if (myoldHisto1->GetEntries() == mynewHisto1->GetEntries()
       && myoldHisto1->GetSumOfWeights() ==0 && mynewHisto1->GetSumOfWeights() ==0) { 
    mypv = 1.;
  } else {
     mypv = myoldHisto1->Chi2TestX(mynewHisto1,chi2,ndf,igood,"UU NORM");
     if (ndf==0) mypv=1;
  }

  std::strstream buf;
  std::string value;
  buf<<"PV="<<mypv<<std::endl;
  buf>>value;
  
  myte->DrawTextNDC(0.2,0.7, value.c_str());

  std::cout << "[OVAL] " << myoldHisto1->GetName() << " PV= "<< mypv <<std::endl;
  return;

}

HistoCompare::PVCompute(TH2 * oldHisto , TH2 * newHisto , TText * te )
{

  myoldHisto2 = oldHisto;
  mynewHisto2 = newHisto;
  myte = te;

  Double_t mypv = myoldHisto2->Chi2Test(mynewHisto2,"UU");
  std::strstream buf;
  std::string value;
  buf<<"PV="<<mypv<<std::endl;
  buf>>value;
  
  myte->DrawTextNDC(0.2,0.7, value.c_str());

  std::cout << "[OVAL] " << myoldHisto2->GetName() << " PV= " << mypv << std::endl;
  return;

}


HistoCompare::PVCompute(TProfile * oldHisto , TProfile * newHisto , TText * te )
{

  myoldProfile = oldHisto;
  mynewProfile = newHisto;
  myte = te;

  Double_t mypv = myoldProfile->Chi2Test(mynewProfile,"UU");
  std::strstream buf;
  std::string value;
  buf<<"PV="<<mypv<<std::endl;
  buf>>value;
  
  myte->DrawTextNDC(0.2,0.7, value.c_str());

  std::cout << "[OVAL] " << myoldProfile->GetName() << " PV = " << mypv << std::endl;
  return;

}
