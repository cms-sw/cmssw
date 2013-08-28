// F. Cossutti 
// $Date: 2012/02/10 15:11:41 $
// $Revision: 1.1 $
//
// ROOT macro for graphical compariosn of Monitor Elements in a user
// supplied directory between two files with the same histogram content

#include <iostream.h>

class HistoCompare {

 public:

  HistoCompare() { std::cout << "Initializing HistoCompare... " << std::endl;printresy=0.96; } ;

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
  double printresy;
};

HistoCompare::printRes(TString theName, Double_t thePV, TText * te)
{
  myte->DrawTextNDC(0.1,printresy,theName );
  std::cout << "[Compatibility test] " << theName << std::endl;
  printresy+=-0.04;
}


HistoCompare::PVCompute(TH1 * oldHisto , TH1 * newHisto , TText * te )
{

  myoldHisto1 = oldHisto;
  mynewHisto1 = newHisto;
  myte = te;

  Double_t *res;

  Double_t mypvchi = myoldHisto1->Chi2Test(mynewHisto1,"WW",res);

  char str [128];
  sprintf(str,"chi^2 P Value: %f",mypvchi);
  TString title = str;
  printRes(title, mypvchi, myte);
  Double_t mypvKS = myoldHisto1->KolmogorovTest(mynewHisto1,"");
  sprintf(str,"KS (prob): %f",mypvKS);
  TString title = str;
  std::strstream buf;
  std::string value;

  printRes(title, mypvKS, myte);

  return;

}

HistoCompare::PVCompute(TH2 * oldHisto , TH2 * newHisto , TText * te )
{

  myoldHisto2 = oldHisto;
  mynewHisto2 = newHisto;
  myte = te;

  Double_t *res ;
  Double_t mypvchi = myoldHisto2->Chi2Test(mynewHisto2,"WW",res);
  char str [128];
  sprintf(str,"chi^2 P Value: %f",mypvchi);
  TString title = str;
  printRes(title, mypvchi, myte);

  return;

}


HistoCompare::PVCompute(TProfile * oldHisto , TProfile * newHisto , TText * te )
{

  myoldProfile = oldHisto;
  mynewProfile = newHisto;
  myte = te;

  Double_t *res ;

  Double_t mypv = myoldProfile->Chi2Test(mynewProfile,"WW",res);
  TString title =  "chi^2 P Value: ";
  printRes(title, mypv, myte);
  return;

}

#include "TObject.h"
#include "TDirectory.h"
#include "TKey.h"
#include "TFile.h"
#include "TTree.h"
#include "TText.h"


void DetailedCompare( TString currentfile = "new.root",
		      TString referencefile = "ref.root",
		      TString theDir = "DQMData/Run 1/Generator/Run summary/Tau")
{
  std::cout << "Note: This code correct the Histograms errors to sqrt(N) - this is not correct for samples with weights" << std::endl;
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

  gStyle->SetLabelFont(63,"X");
  gStyle->SetLabelSize(30,"X");
  gStyle->SetTitleOffset(1.25,"X");
  gStyle->SetTitleFont(63,"X");
  gStyle->SetTitleSize(35,"X");
  gStyle->SetLabelFont(63,"Y");
  gStyle->SetLabelSize(30,"Y");
  gStyle->SetTitleOffset(3.0,"Y");
  gStyle->SetTitleFont(63,"Y");
  gStyle->SetTitleSize(35,"Y");
  gStyle->SetLabelFont(63,"Z");
  gStyle->SetLabelSize(30,"Z");


  for ( unsigned int index = 0; index < theList.size() ; index++ ) {

    std::cout << index << std::endl;

    TString theName = theDir+"/"+theList[index];
    std::cout << theName << std::endl;

    TH1* href_;
    rfile->GetObject(theName,href_);
    href_;
    double nentries=href_->GetEntries();
    double integral=href_->Integral(0,href_->GetNbinsX()+1);
    if(integral!=0)href_->Scale(nentries/integral);
    href_->Sumw2();
   

    TH1* hnew_;
    sfile->GetObject(theName,hnew_);
    hnew_;
    // Set errors to sqrt(# entries)
    double nentries=hnew_->GetEntries();
    double integral=hnew_->Integral(0,hnew_->GetNbinsX()+1);
    if(integral!=0)hnew_->Scale(nentries/integral);
    hnew_->Sumw2();
    cout << referencefile << " " << nentries << " " << integral << endl;    
   

    DetailedComparePlot(href_, hnew_, currentfile, referencefile, theDir, theList[index]); 

  }
 
}

void DetailedComparePlot(TH1 * href_, TH1 * hnew_, TString currentfile, TString referencefile, TString theDir, TString theHisto )
{

  gStyle->SetOptTitle(0);

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
   TPad *pad0 = new TPad("pad0","The pad with the function ",0.0,0.9,0.0,1.0);
   TPad *pad1 = new TPad("pad1","The pad with the function ",0.0,0.6,0.5,0.9);
   TPad *pad2 = new TPad("pad2","The pad with the histogram",0.5,0.6,1.0,0.9);
   TPad *pad3 = new TPad("pad3","The pad with the histogram",0.0,0.3,0.5,0.6);
   TPad *pad4 = new TPad("pad4","The pad with the histogram",0.5,0.3,1.0,0.6);
   TPad *pad5 = new TPad("pad5","The pad with the histogram",0.0,0.0,0.5,0.3);
   TPad *pad6 = new TPad("pad6","The pad with the histogram",0.5,0.0,1.0,0.3);

   pad1->Draw();
   pad2->Draw();
   pad3->Draw();
   pad4->Draw();
   pad5->Draw();
   pad6->Draw();


   // Draw a global picture title
   TText titte;
   titte.SetTextSize(0.02);
   titte.DrawTextNDC(0.1,0.98,theName);
   titte.DrawTextNDC(0.1,0.96,"Reference File (A):");
   titte.DrawTextNDC(0.1,0.94,referencefile);
   titte.DrawTextNDC(0.1,0.92,"Current File (B):");
   titte.DrawTextNDC(0.1,0.90,currentfile);

   hnew_->SetXTitle(href_->GetTitle());
   href_->SetXTitle(href_->GetTitle());
   hnew_->SetTitleOffset(1.5,"Y");
   href_->SetTitleOffset(1.5,"Y");


   // Draw reference
   pad1->cd();
   href_->SetYTitle("Events (A)");
   href_->DrawCopy("e1");


   // Draw new
   pad2->cd();
   hnew_->SetYTitle("Events (B)");
   hnew_->DrawCopy("e1");

   gStyle->SetOptStat("nemruoi");

   // Draw the two overlayed Normalized display region to 1
   pad3->cd();

   pad3->SetGridx();
   pad3->SetGridy();
   TH1 *hnew_c=hnew_->Clone();
   TH1 *href_c=href_->Clone();
   double integral=href_c->Integral(1,href_c->GetNbinsX());
   if(integral!=0)href_c->Scale(1/integral);
   double integral=hnew_c->Integral(1,href_c->GetNbinsX());
   if(integral!=0)hnew_c->Scale(1/integral);
   href_c->SetYTitle("Comparison of A and B");
   href_c->DrawCopy("e1");
   hnew_c->DrawCopy("e1same");
   TText* te = new TText();
   te->SetTextSize(0.04);
   myPV->PVCompute( href_ , hnew_ , te );



   TH1 *ratio_=hnew_c->Clone();
   ratio_->Divide(href_c);
   ratio_->SetYTitle("Ratio: B/A");
   pad4->cd();
   pad4->SetGridx();
   pad4->SetGridy();
   ratio_->DrawCopy("e1");

   TH1 *diff_=hnew_c->Clone();
   diff_->Add(href_c,-1);
   diff_->SetYTitle("Difference: B-A");
   pad5->cd();
   pad5->SetGridx();
   pad5->SetGridy();
   diff_->DrawCopy("e1");


   TH1 *sigma_=diff_->Clone();
   for(unsigned int i=1; i<=sigma_->GetNbinsX(); i++ ){
     double v=sigma_->GetBinContent(i);
     double e=sigma_->GetBinError(i);
     //     cout <<  v << "+/-" << e << " " << v/fabs(e) << endl;
     if(e!=0)sigma_->SetBinContent(i,v/fabs(e));
     else sigma_->SetBinContent(i,0);
   }
   sigma_->SetYTitle("Sigma on Difference: (B-A)/sigma(B-A)");
   pad6->cd();
   pad6->SetGridx();
   pad6->SetGridy();
   sigma_->DrawCopy("phisto");



   pad0->cd();

   gStyle->SetOptStat(0000000);

 }
 TString plotFile = theHisto+".eps";
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
