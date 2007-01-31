#include "TCanvas.h"
#include <strstream>
#include "TString.h"
#include "TH1.h"
#include "TH2.h"
#include "TProfile.h"
#include "TPostScript.h"
#include "TText.h"
#include "TCanvas.h"
#include <iostream>

class HistoCompare {
public:
  virtual void compare(TH1 * oldHisto , TH1 * newHisto)
  {
    Double_t *res;
    Double_t mypv = oldHisto->Chi2Test( newHisto,"UU", res);

    std::strstream buf;
    std::string value;
    buf<<"PV="<<mypv<<std::endl;
    buf>>value;

    //  myte->DrawTextNDC(0.2,0.7, value.c_str());

    std::cout << "[OVAL] " << oldHisto->GetName() << " PV = " << mypv << std::endl;
    return;
  }

  virtual void fits(TH1 * oldHisto , TH1 * newHisto)
  {
   oldHisto->Fit("gaus");
   newHisto->Fit("gaus");
   TF1 *f1 = oldHisto->GetFunction("gaus");
   TF2 *f2 = newHisto->GetFunction("gaus");
   cout << "OLD:  mean " << f1->GetParameter(1) << "  sigma " << f1->GetParameter(2) << endl;
   cout << "NEW:  mean " << f1->GetParameter(1) << "  sigma " << f1->GetParameter(2) << endl;
  }

  virtual void openPage(string name, int nx=1, int ny=1) {}
  virtual void closePage() {}

};


class HistoCompareDraw : public HistoCompare
{
public:
  HistoCompareDraw() : theCanvas(0), subpad(0) {}

  virtual void compare(TH1 * oldHisto , TH1 * newHisto) { 
    theCanvas->cd(++subpad);
    oldHisto->UseCurrentStyle();
    newHisto->UseCurrentStyle();
    oldHisto->SetLineColor(kRed);
    newHisto->SetLineColor(kBlue);
    oldHisto->Draw();
    newHisto->Draw("same");
    HistoCompare::compare(oldHisto, newHisto);
  }

protected:
  TCanvas * theCanvas;
  int subpad;
};


class HistoCompareGif : public HistoCompareDraw
{
public:
  HistoCompareGif(string title, string suffix)
  : theTitle(title),
    theName(""),
    theSuffix(suffix)
  {
  }

  virtual void openPage(string name, int nx, int ny) {
    theCanvas =  new TCanvas(theTitle.c_str(), theTitle.c_str(), 800, 600);
    theCanvas->Divide(nx, ny);
    theName = name + theSuffix + ".gif";
    subpad = 0;
  }
  
  virtual void closePage() {
    theCanvas->Print(theName.c_str());
  }

private:
  string theTitle;
  string theName;
  string theSuffix;
};


class HistoComparePS: public HistoCompareDraw
{
public:
  HistoComparePS(string filename)
  : thePostscript( new TPostScript(filename , -112) )
  {
    theCanvas = new TCanvas("ps", "ps", 800, 600);
    thePostscript->Range(29.7 , 21.0);
  }

  HistoComparePS::~HistoComparePS()
  {
    thePostscript->Close();
  }

  virtual void openPage(string name, int nx=1, int ny=1) {
    subpad = 0;
  }
    
  virtual void closePage() {
    theCanvas->Update();
    thePostscript->NewPage();
  }

private:
  TPostScript * thePostscript;
};

