#include <iostream.h>

class HistoCompare {

 public:

  HistoCompare() { std::cout << "Initializing HistoCompare... " << std::endl; } ;

  void PVCompute(TH1 * oldHisto , TH1 * newHisto , TText * te , int method=1);
  void PVCompute(TH2 * oldHisto , TH2 * newHisto , TText * te, int method=1 );
  void PVCompute(TProfile * oldHisto , TProfile * newHisto , TText * te , int method=1);

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

HistoCompare::PVCompute(TH1 * oldHisto , TH1 * newHisto , TText * te , int method)
{

  myoldHisto1 = oldHisto;
  mynewHisto1 = newHisto;
  myte = te;

  Double_t *res;

  Double_t mypv;
  if (method == 1) {
     mypv = myoldHisto1->Chi2Test(mynewHisto1,"UU",res);
  }else {
     mypv = myoldHisto1->KolmogorovTest(mynewHisto1, "UO");
  }

  std::strstream buf;
  std::string value;
  buf<<"PV="<<mypv<<std::endl;
  buf>>value;
  
  myte->DrawTextNDC(0.2,0.7, value.c_str());

  std::cout << "[OVAL] " << myoldHisto1->GetName() << " PV = " << mypv << std::endl;
  return;

}

HistoCompare::PVCompute(TH2 * oldHisto , TH2 * newHisto , TText * te , int method )
{

  myoldHisto2 = oldHisto;
  mynewHisto2 = newHisto;
  myte = te;

  Double_t *res ;
  Double_t mypv ;
  if (method == 1) {
   mypv = myoldHisto2->Chi2Test(mynewHisto2,"UU",res);
  }else {
     mypv = myoldHisto2->KolmogorovTest(mynewHisto2, "UO");
  }


  std::strstream buf;
  std::string value;
  buf<<"PV="<<mypv<<std::endl;
  buf>>value;
  
  myte->DrawTextNDC(0.2,0.7, value.c_str());

  std::cout << "[OVAL] " << myoldHisto2->GetName() << " PV = " << mypv << std::endl;
  return;

}


HistoCompare::PVCompute(TProfile * oldHisto , TProfile * newHisto , TText * te , int method)
{

  myoldProfile = oldHisto;
  mynewProfile = newHisto;
  myte = te;

  Double_t *res ;

  Double_t mypv;
  if (method == 1) {
   mypv = myoldProfile->Chi2Test(mynewProfile,"UU",res);
  }else {
     mypv = myoldProfile->KolmogorovTest(mynewProfile, "UO");
  }


  std::strstream buf;
  std::string value;
  buf<<"PV="<<mypv<<std::endl;
  buf>>value;
  
  myte->DrawTextNDC(0.2,0.7, value.c_str());

  std::cout << "[OVAL] " << myoldProfile->GetName() << " PV = " << mypv << std::endl;
  return;

}
