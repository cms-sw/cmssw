#include <iostream>     // std::cout
#include <string>       // std::string
#include <sstream>      // std::stringstream
#include <fstream>      // std::filestream
#include "TH1F.h"
#include "TH2.h"
#include "TProfile.h"
#include "TText.h"

class HistoCompare {

 public:

  HistoCompare() { std::cout << "Initializing HistoCompare... " << std::endl; } ;

  void PVCompute(TH1 * oldHisto , TH1 * newHisto , TText * te, const double x=0.6, const double y=0.7);
  void PVCompute(TH2 * oldHisto , TH2 * newHisto , TText * te, const double x=0.2, const double y=0.7);
  void PVCompute(TProfile * oldHisto , TProfile * newHisto , TText * te, const double x=0.2, const double y=0.7);

 private:
  
  TH1 * myoldHisto1;
  TH1 * mynewHisto1;

  TH2 * myoldHisto2;
  TH2 * mynewHisto2;

  TProfile * myoldProfile;
  TProfile * mynewProfile;

  TText * myte;

};

void HistoCompare::PVCompute(TH1 * oldHisto , TH1 * newHisto , TText * te, const double x, const double y)
{

  myoldHisto1 = oldHisto;
  mynewHisto1 = newHisto;
  myte = te;

  Double_t mypv = myoldHisto1->Chi2Test(mynewHisto1,"UUNORM");
  std::stringstream buf;
  std::string value;
  buf<<"PV="<<mypv<<std::endl;
  buf>>value;
  
  myte->DrawTextNDC(x, y, value.c_str());

  std::cout << "[OVAL] " << myoldHisto1->GetName() << " PV = " << mypv << std::endl;
  return;

}

void HistoCompare::PVCompute(TH2 * oldHisto , TH2 * newHisto , TText * te, const double x, const double y)
{

  myoldHisto2 = oldHisto;
  mynewHisto2 = newHisto;
  myte = te;

  Double_t mypv = myoldHisto2->Chi2Test(mynewHisto2,"UUNORM");
  std::stringstream buf;
  std::string value;
  buf<<"PV="<<mypv<<std::endl;
  buf>>value;
  
  myte->DrawTextNDC(x, y, value.c_str());

  std::cout << "[OVAL] " << myoldHisto2->GetName() << " PV = " << mypv << std::endl;
  return;

}


void HistoCompare::PVCompute(TProfile * oldHisto , TProfile * newHisto , TText * te, const double x, const double y)
{

  myoldProfile = oldHisto;
  mynewProfile = newHisto;
  myte = te;

  Double_t mypv = myoldProfile->Chi2Test(mynewProfile,"UUNORM");
  std::stringstream buf;
  std::string value;
  buf<<"PV="<<mypv<<std::endl;
  buf>>value;
  
  myte->DrawTextNDC(x, y, value.c_str());

  std::cout << "[OVAL] " << myoldProfile->GetName() << " PV = " << mypv << std::endl;
  return;

}
