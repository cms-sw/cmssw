// F. Cossutti 
// $Date: 2010/11/11 16:28:30 $
// $Revision: 1.1 $
//
// ROOT macro for graphical compariosn of Monitor Elements in a user
// supplied directory between two files with the same histogram content

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

  void printRes(TString theName, Double_t thePV, TText * te);

};

HistoCompare::printRes(TString myName, Double_t mypv, TText * myte)
{
  std::strstream buf;
  std::string value;
  buf<<"PV="<<mypv<<std::endl;
  buf>>value;
  
  myte->DrawTextNDC(0.7,0.92, value.c_str());

  std::cout << "[Compatibility test] " << myName << " PV = " << mypv << std::endl;

}


HistoCompare::PVCompute(TH1 * oldHisto , TH1 * newHisto , TText * te )
{

  myoldHisto1 = oldHisto;
  mynewHisto1 = newHisto;
  myte = te;

  Double_t *res;

  Double_t mypv = myoldHisto1->Chi2Test(mynewHisto1,"WW",res);
  TString title = myoldHisto1->GetName();
  printRes(title, mypv, myte);
  return;

}

HistoCompare::PVCompute(TH2 * oldHisto , TH2 * newHisto , TText * te )
{

  myoldHisto2 = oldHisto;
  mynewHisto2 = newHisto;
  myte = te;

  Double_t *res ;
  Double_t mypv = myoldHisto2->Chi2Test(mynewHisto2,"WW",res);
  TString title = myoldHisto2->GetName();
  printRes(title, mypv, myte);
  return;

}


HistoCompare::PVCompute(TProfile * oldHisto , TProfile * newHisto , TText * te )
{

  myoldProfile = oldHisto;
  mynewProfile = newHisto;
  myte = te;

  Double_t *res ;

  Double_t mypv = myoldProfile->Chi2Test(mynewProfile,"WW",res);
  TString title = myoldProfile->GetName();
  printRes(title, mypv, myte);
  return;

}

#include "TObject.h"
#include "TDirectory.h"
#include "TKey.h"
#include "TFile.h"
#include "TTree.h"
#include "TText.h"


void MECompare( TString currentfile = "new.root",
                TString referencefile = "ref.root",
                TString theDir = "DQMData/Run 1/Generator/Run summary/MBUEandQCD" )
{
  
  std::vector<TString> theList =  histoList(currentfile, theDir);
  
  gROOT ->Reset();
  char*  rfilename = referencefile ;
  char*  sfilename = currentfile ;
  
  delete gROOT->GetListOfFiles()->FindObject(rfilename);
  delete gROOT->GetListOfFiles()->FindObject(sfilename);
  
  TFile * rfile = new TFile(rfilename);
  TFile * sfile = new TFile(sfilename);
  
  char* baseDir=theDir;
  
  rfile->cd(baseDir);
  gDirectory->ls();
  
  sfile->cd(baseDir);
  gDirectory->ls();

  for ( unsigned int index = 0; index < theList.size() ; index++ ) {

    std::cout << index << std::endl;

    TString theName = theDir+"/"+theList[index];
    std::cout << theName << std::endl;

    TH1* href_;
    rfile->GetObject(theName,href_);
    href_;
    
    TH1* hnew_;
    sfile->GetObject(theName,hnew_);
    hnew_;
    
    MEComparePlot(href_, hnew_, currentfile, referencefile, theDir, theList[index]); 

  }
 
}

void MEComparePlot(TH1 * href_, TH1 * hnew_, TString currentfile, TString referencefile, TString theDir, TString theHisto )
{

 TString theName = theDir+"/"+theHisto;
 std::cout << "Histogram name = " << theName << std::endl;

 HistoCompare * myPV = new HistoCompare();

 int rcolor = 2;
 int scolor = 4;
 
 int rmarker = 21;
 int smarker = 20;

 Double_t markerSize = 0.75;
 
 href_->SetLineColor(rcolor);
 href_->SetMarkerStyle(rmarker);
 href_->SetMarkerSize(markerSize);
 href_->SetMarkerColor(rcolor);

 hnew_->SetLineColor(scolor);
 hnew_->SetMarkerStyle(smarker);
 hnew_->SetMarkerSize(markerSize);
 hnew_->SetMarkerColor(scolor);    

 if ( href_ && hnew_ ) {

 
   TCanvas *myPlot = new TCanvas("myPlot","Histogram comparison",200,10,700,900);
   TPad *pad1 = new TPad("pad1",
                         "The pad with the function",0.03,0.62,0.50,0.92);
   TPad *pad2 = new TPad("pad2",
                         "The pad with the histogram",0.51,0.62,0.98,0.92);
   TPad *pad3 = new TPad("pad3",
                         "The pad with the histogram",0.03,0.02,0.97,0.57);
   pad1->Draw();
   pad2->Draw();
   pad3->Draw();

   // Draw a global picture title
   TPaveLabel *title = new TPaveLabel(0.1,0.94,0.9,0.98,theName);
   title->SetFillColor(16);
   title->SetTextFont(52);
   title->Draw();

   TText * titte = new TText();

   // Draw reference
   pad1->cd();
   href_->DrawCopy("e1");
   titte->DrawTextNDC(0.,0.02, referencefile);

   // Draw new
   pad2->cd();
   hnew_->DrawCopy("e1");
   titte->DrawTextNDC(0.,0.02, currentfile);
   gStyle->SetOptStat("nemruoi");

   // Draw the two overlayed
   pad3->cd();
   pad3->SetGridx();
   pad3->SetGridy();
   href_->DrawCopy("e1");
   hnew_->DrawCopy("e1same");

   TText* te = new TText();
   te->SetTextSize(0.1);
   myPV->PVCompute( href_ , hnew_ , te );

   gStyle->SetOptStat(0000000);

 }
 TString plotFile = theHisto+".jpg";
 myPlot->Print(plotFile); 
 
 delete myPV;
 delete myPlot; 

 }

std::vector<TString> histoList( TString currentfile, TString theDir )
{

 gROOT ->Reset();
 char*  sfilename = currentfile ;

 delete gROOT->GetListOfFiles()->FindObject(sfilename);

 TFile * sfile = new TFile(sfilename);

 char* baseDir=theDir;

 sfile->cd(baseDir);

 TDirectory * d = gDirectory;

 std::vector<TString> theHistList;

 TIter i( d->GetListOfKeys() );
 TKey *k;
 while( (k = (TKey*)i())) {
   TClass * c1 = gROOT->GetClass(k->GetClassName());
   if ( !c1->InheritsFrom("TH1")) continue;
   theHistList.push_back(k->GetName());
 }
 
 std::cout << "Histograms considered: " << std::endl;
 for (unsigned int index = 0; index < theHistList.size() ; index++ ) {
   std::cout << index << " " << theHistList[index] << std::endl;
 }

 return theHistList;

}
