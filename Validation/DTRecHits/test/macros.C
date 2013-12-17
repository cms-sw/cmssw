/** 
 * A collection of simple ROOT macros
 *
 * N. Amapane 2002-2004
 */

#include <sstream>
#include <iomanip>

#if !defined(__CINT__) || defined(__MAKECINT__)
#include "TProfile.h"
#include "TLegend.h"
#include "TROOT.h"
#include "TVirtualPad.h"
#include "TLine.h"
#include "TCanvas.h"
#include "TPostScript.h"
#include "TH1.h"
#include "TH2.h"
#include "TF1.h"
#include "TAxis.h"
#include "TMath.h"
#include "TROOT.h"
#include "TStyle.h"

#include <iostream>

using namespace std;

#endif



TString getPitchString(TH1 *histo, int prec = 5);


TLegend * getLegend(float x1=0.48, float y1=0.81, float x2=0.98, float y2=0.995);
void setStyle(TH1 *histo);
void setStyle(TH2 *histo);


void setStyle(TH1 *histo) {
  histo->GetXaxis()->SetTitleFont(gStyle->GetTitleFont());
  histo->GetXaxis()->SetTitleSize(gStyle->GetTitleFontSize());
  histo->GetXaxis()->SetLabelFont(gStyle->GetLabelFont());
  histo->GetXaxis()->SetLabelSize(gStyle->GetLabelSize());

  histo->GetYaxis()->SetTitleFont(gStyle->GetTitleFont());
  histo->GetYaxis()->SetTitleSize(gStyle->GetTitleFontSize());
  histo->GetYaxis()->SetLabelFont(gStyle->GetLabelFont());
  histo->GetYaxis()->SetLabelSize(gStyle->GetLabelSize());
}

void setStyle(TH2 *histo) {
  histo->GetXaxis()->SetTitleFont(gStyle->GetTitleFont());
  histo->GetXaxis()->SetTitleSize(gStyle->GetTitleFontSize());
  histo->GetXaxis()->SetLabelFont(gStyle->GetLabelFont());
  histo->GetXaxis()->SetLabelSize(gStyle->GetLabelSize());

  histo->GetYaxis()->SetTitleFont(gStyle->GetTitleFont());
  histo->GetYaxis()->SetTitleSize(gStyle->GetTitleFontSize());
  histo->GetYaxis()->SetLabelFont(gStyle->GetLabelFont());
  histo->GetYaxis()->SetLabelSize(gStyle->GetLabelSize());
}


bool addProfile=false;
bool addSlice=true;
bool addMedian  = false;
TString opt2Dplot = "col";

// Plot a TH2 + add profiles on top of it
// minY, maxY: Y range for plotting and for computing profile if addProfile==true.
//             Note that the simple profile is very sensitive to the Y range used!
TH1F* plotAndProfileX (TH2* theh, int rebinX, int rebinY, int rebinProfile, float minY, float maxY, float minX=0, float maxX=0) {
  TH2* h2=theh->Clone();
  
  //  setStyle(h2);
  if (h2==0) {
    cout << "plotAndProfileX: null histo ptr" << endl;
    return;
  }
  
  gPad->SetGrid(1,1);
  gStyle->SetGridColor(15);
  h2->Rebin2D(rebinX,rebinY);
  //  h2->GetYaxis()->SetRangeUser(minY,maxY);

  TLine * l = new TLine(h2->GetXaxis()->GetXmin(),0,h2->GetXaxis()->GetXmax(),0);
  if (maxX>minX) {
    h2->GetXaxis()->SetRangeUser(minX,maxX);  
    l->SetX1(minX);
    l->SetX2(maxX);
  }

  h2->SetMarkerStyle(1);
  h2->Draw(opt2Dplot);
  l->SetLineColor(3);
  l->SetLineWidth(2);
  l->Draw();
  if (addProfile) {
    TAxis* yaxis = h2->GetYaxis();
    //Add option "s" to draw RMS as error instead than RMS/sqrt(N)
    TProfile* prof = h2->ProfileX("_pfx", 
				  TMath::Max(1,yaxis->FindBin(minY)), 
				  TMath::Min(yaxis->GetNbins(),yaxis->FindBin(maxY)));
//     cout << yaxis->FindBin(minY) << " " << yaxis->FindBin(maxY) << endl;
//     cout << yaxis->GetNbins();
    //TProfile* prof = h2->ProfileX("_pfx");
    prof->SetMarkerColor(2);
    prof->SetMarkerStyle(20);
    prof->SetMarkerSize(0.4);
    prof->SetLineColor(2);
    prof->Rebin(rebinProfile);
    prof->Draw("same");
  }

  TH1F* ht=0;

  if (addSlice) {
    TObjArray aSlices;
    //    TF1 fff("a", "gaus", -0.1, 0.1);   
    h2->FitSlicesY(0, 0, -1, 0, "QNR", &aSlices); // add "G2" to merge 2 consecutive bins

    TH1F*  ht = (TH1F*) aSlices[1]->Clone();    
    // Remove bins with failed fits, based on fit errors
    float thr = (maxY-minY)/4.;
    for (int bin=1; bin<=ht->GetNbinsX();++bin){
      if (ht->GetBinError(bin)>thr) {
	ht->SetBinContent(bin,0);
	ht->SetBinError(bin,0);
      }
    }
    ht->SetMarkerColor(4);
    ht->Draw("same");    
  }

  if (addMedian) {
    double xq[1] = {0.5};
    double median[1];

    TAxis* axis =  h2->GetXaxis();
    TH1F* medprof = new TH1F(h2->GetName()+TString("medians"),"medians", axis->GetNbins(), axis->GetXmin(), axis->GetXmax());
    float bw =  h2->GetYaxis()->GetBinLowEdge(2)-h2->GetYaxis()->GetBinLowEdge(1);
    

    TString projname = h2->GetName()+TString("_pmedian");
    for (int bin=1; bin<=h2->GetNbinsX(); ++bin){
      TH1D * proj = h2->ProjectionY(projname, bin, bin);
      double integral = proj->Integral();
      if (integral==0) continue;
      // Take overflow and underflow into account
      int nbins = proj->GetNbinsX();
      proj->SetBinContent(1, proj->GetBinContent(0)+proj->GetBinContent(1));
      proj->SetBinContent(0,0);
      proj->SetBinContent(nbins, proj->GetBinContent(nbins)+proj->GetBinContent(nbins+1));
      proj->SetBinContent(nbins+1,0);
      proj->GetQuantiles(1,median,xq);
      medprof->SetBinContent(bin,median[0]);
      // Approximated uncertainty on median, probably underestimated.
      medprof->SetBinError(bin,bw*sqrt(integral/2.)/2./TMath::Max(1.,proj->GetBinContent(proj->FindBin(median[0]))));
    }
    medprof->SetMarkerColor(2);
    medprof->SetMarkerStyle(20);
    medprof->SetMarkerSize(0.4);
    medprof->Draw("Esame");
  }

  h2->GetYaxis()->SetRangeUser(minY,maxY);

  return ht;
}


// void plotAndProfileX (TH2* h2, float min, float max, bool profile=false) {
//   setStyle(h2);
//   gPad->SetGrid(1,1);
//   gStyle->SetGridColor(15);
//   h2->GetYaxis()->SetRangeUser(min,max);
//   h2->Draw();
//   if (profile) {
//     TProfile* prof = h2->ProfileX();
//     prof->SetMarkerColor(2);
//     prof->SetLineColor(2);
//     prof->Draw("same");
//   }
//   TLine * l = new TLine(h2->GetXaxis()->GetXmin(),0,h2->GetXaxis()->GetXmax(),0);
//   l->SetLineColor(3);
//   l->Draw();
// }


// Draw a 2-D plot within the specified Y range and superimpose its X profile,
// setting as sigmas that of the fit (and not the error of the mean)
void plotAndProfileXSpread (TH2* h2, float min, float max, bool profile=false, float ymin=-5., float ymax=5.) {
  setStyle(h2);
  gPad->SetGrid(1,1);
  gStyle->SetGridColor(15);
  gStyle->SetOptStat(0);
  // h2->RebinX(3);
  // h2->RebinY(2);
  // h2->SetXTitle("distance from anode (cm)");
  // h2->SetYTitle("(d_{reco}-d_{sim})/#sigma_{reco}");
  h2->SetMarkerColor(2);
  h2->SetLineColor(2);
  h2->GetYaxis()->SetTitleOffset(1.4);
  h2->GetXaxis()->SetRangeUser(min,max);
  h2->GetYaxis()->SetRangeUser(ymin,ymax);
  h2->DrawCopy("box");
  if (profile) {
    TProfile* prof = h2->ProfileX("profile",-1,-1,"s");
    prof->SetMarkerStyle(20);
    prof->SetMarkerSize(1.2);
    prof->SetMarkerColor(1);
    prof->SetLineColor(1);
    prof->SetLineWidth(2);
    prof->DrawCopy("same e1");
    delete prof;
  }
  TLine * l = new TLine(h2->GetXaxis()->GetXmin(),0,h2->GetXaxis()->GetXmax(),0);
  l->SetLineColor(3);
  l->Draw();
}



// Fit the gaussian core of an histogram with in  the range mean+-nsigmas.
TF1* drawGFit(TH1 * h1, float nsigmas, float min, float max){

  gPad->SetGrid(1,1);
  gStyle->SetGridColor(15);
  h1->GetXaxis()->SetRangeUser(min,max);
  float minfit = h1->GetMean() - h1->GetRMS();
  float maxfit = h1->GetMean() + h1->GetRMS();
 
  TLine * l = new TLine(0,0,0,h1->GetMaximum()*1.05);
  
  l->SetLineColor(3);
  l->SetLineWidth(2);
  
  static int i = 0;
  TString nameF1 = TString("g") + (Long_t)i;
  i++;
  TF1* g1 = new TF1(nameF1,"gaus",minfit,maxfit);

  g1->SetLineColor(2);
  g1->SetLineWidth(2);
  h1->Fit(g1,"RQ");
  
  minfit = g1->GetParameter("Mean") - nsigmas*g1->GetParameter("Sigma");
  maxfit = g1->GetParameter("Mean") + nsigmas*g1->GetParameter("Sigma");
  g1->SetRange(minfit,maxfit);

  h1->Fit(g1,"RQ");
  TF1* fh=h1->GetFunction(nameF1);
  if (fh) fh->FixParameter(0,g1->GetParameter(0)); // so that it is not shown in legend

  gPad->Draw(); 
  l->Draw();
  h1->Draw("same"); //redraw on top of the line
  return g1;
}


/*
 * Create a new TCanvas setting its properties
 *
 * 2003 NCA 
 */
// Specify name, title, x/y divisions, form or x,y sizes.
// If no name is specified, a new name is generated automatically
TCanvas * newCanvas(TString name="", TString title="",
                     Int_t xdiv=0, Int_t ydiv=0, Int_t form = 1, Int_t w=-1){
  static int i = 1;
  if (name == "") {
    name = TString("Canvas ") + i;
    i++;
  }
  if (title == "") title = name;
  if (w<0) {
    TCanvas * c = new TCanvas(name,title, form);
  } else {
    TCanvas * c = new TCanvas(name,title,form,w);
  }
  if (xdiv*ydiv!=0) c->Divide(xdiv,ydiv);
  c->cd(1);
  return c;
}
// Create a new canvas with an automatic generated name and the specified 
// divisions and form
TCanvas * newCanvas(Int_t xdiv, Int_t ydiv, Int_t form = 1) {
  return newCanvas("","",xdiv,ydiv,form);
}
// Create a new canvas with an automatic generated name and the specified 
// form
TCanvas * newCanvas(Int_t form = 1)
{
  return newCanvas(0,0,form);
}
// ...without specifying the title...
TCanvas * newCanvas(TString name, Int_t xdiv, Int_t ydiv, Int_t form,
                    Int_t w) {
  return newCanvas(name, name,xdiv,ydiv,form,w);
}
// ...without specifying title and divisions.
TCanvas * newCanvas(TString name, Int_t form, Int_t w=-1)
{
  return newCanvas(name, name, 0,0,form,w);
}
/*
 * Print all open canvases to PS or EPS files.
 *
 * 2003 NCA 
 */
// Print all canvases in a single PS file
void printCanvasesPS(TString name){
  TPostScript * ps = new TPostScript(name,112);
  TIter iter(gROOT->GetListOfCanvases());
  TCanvas *c;
  while( (c = (TCanvas *)iter()) )
    {
      c->cd();
      cout << "Printing " << c->GetName() << endl;
      ps->NewPage();
      c->Draw();
    }
  cout << " File " << name << " was created" << endl;
  ps->Close();
}
// Print all canvases in separate EPS files
void printCanvasesEps(){
  TIter iter(gROOT->GetListOfCanvases());
  TCanvas *c;
  while( (c = (TCanvas *)iter()) ) {
    c->Print(0,".eps");
  }
}
// Print all canvases in separate EPS files (another way)
void printCanvasesEps2() {
  gROOT->GetListOfCanvases()->Print(".eps");
}

// Print all canvases in separate files
void printCanvases(TString type="png", TString path="."){
  TIter iter(gROOT->GetListOfCanvases());
  TCanvas *c;
  while( (c = (TCanvas *)iter()) ) {
    TString name =  c->GetTitle();
    c->Print(path+"/"+name+"."+type,type);
  }
}

/*
 * Define different TStyles; use them with:
 * getStyle->cd();
 *
 * 2003 NCA
 */
TStyle * getStyle(TString name="myStyle")
{
  TStyle *theStyle;

  gROOT->ForceStyle();

  if ( name == "myStyle" ) {
    theStyle = new TStyle("myStyle", "myStyle");
    //    theStyle->SetOptStat(0);
    theStyle->SetPadBorderMode(0);
    theStyle->SetCanvasBorderMode(0);
    theStyle->SetPadColor(0);
    theStyle->SetCanvasColor(0);
    theStyle->SetMarkerStyle(8);
    theStyle->SetMarkerSize(0.7);
    theStyle->SetStatH(0.3);
    theStyle->SetStatW(0.15);
    //   theStyle->SetTextFont(132);
    //   theStyle->SetTitleFont(132);
    theStyle->SetTitleBorderSize(1);
    theStyle->SetPalette(1);

  } else if( name == "tdr" ) {
    theStyle = new TStyle("tdrStyle","Style for P-TDR");

    // For the canvas:
    theStyle->SetCanvasBorderMode(0);
    theStyle->SetCanvasColor(kWhite);
//      theStyle->SetCanvasDefH(600); //Height of canvas
//      theStyle->SetCanvasDefW(800); //Width of canvas
    theStyle->SetCanvasDefH(750); //Height of canvas
    theStyle->SetCanvasDefW(1000); //Width of canvas

    theStyle->SetCanvasDefX(0);   //POsition on screen
    theStyle->SetCanvasDefY(0);

    // For the Pad:
    theStyle->SetPadBorderMode(0);
    // theStyle->SetPadBorderSize(Width_t size = 1);
    theStyle->SetPadColor(kWhite);
    theStyle->SetPadGridX(true);
    theStyle->SetPadGridY(true);
    theStyle->SetGridColor(0);
    theStyle->SetGridStyle(3);
    theStyle->SetGridWidth(1);

    // For the frame:
    theStyle->SetFrameBorderMode(0);
    theStyle->SetFrameBorderSize(1);
    theStyle->SetFrameFillColor(0);
    theStyle->SetFrameFillStyle(0);
    theStyle->SetFrameLineColor(1);
    theStyle->SetFrameLineStyle(1);
    theStyle->SetFrameLineWidth(1);

    // For the histo:
    // theStyle->SetHistFillColor(1);
    // theStyle->SetHistFillStyle(0);
    theStyle->SetHistLineColor(kBlue);
    theStyle->SetMarkerColor(kBlue);
    //    theStyle->SetHistLineStyle(0);
    //    theStyle->SetHistLineWidth(1);
    // theStyle->SetLegoInnerR(Float_t rad = 0.5);
    // theStyle->SetNumberContours(Int_t number = 20);


     theStyle->SetEndErrorSize(2);
//     theStyle->SetErrorMarker(20);
//     theStyle->SetErrorX(0.);

    theStyle->SetMarkerStyle(20);
    theStyle->SetMarkerSize(0.5);


    //For the fit/function:
    theStyle->SetOptFit(1);
    theStyle->SetFitFormat("5.4g");
    theStyle->SetFuncColor(2);
    theStyle->SetFuncStyle(1);
    theStyle->SetFuncWidth(1);

    //For the date:
    theStyle->SetOptDate(0);
    // theStyle->SetDateX(Float_t x = 0.01);
    // theStyle->SetDateY(Float_t y = 0.01);

    // For the statistics box:
    theStyle->SetOptFile(0);
//     theStyle->SetOptStat(0); // To display the mean and RMS:   SetOptStat("mr");

    theStyle->SetOptStat("e");
    theStyle->SetStatColor(kWhite);
    //    theStyle->SetStatFont(42);
    //    theStyle->SetStatFontSize(0.05);
    theStyle->SetStatTextColor(1);
    theStyle->SetStatFormat("6.4g");
    theStyle->SetStatBorderSize(1);
//     theStyle->SetStatH(0.02);
//     theStyle->SetStatW(0.2);
    // theStyle->SetStatStyle(Style_t style = 1001);
    theStyle->SetStatX(0.94);
    theStyle->SetStatY(0.96);

    // Margins:
//      theStyle->SetPadTopMargin(0.1);
      theStyle->SetPadBottomMargin(0.11);
//      theStyle->SetPadLeftMargin(0.1);
//      theStyle->SetPadRightMargin(0.05);
    theStyle->SetPadLeftMargin(0.15);

    // For the Global title:
    
    //    theStyle->SetOptTitle(0); // Uncomment to remove title
//     theStyle->SetTitleFont(42);
//     theStyle->SetTitleColor(1);
//     theStyle->SetTitleTextColor(1);
    theStyle->SetTitleFillColor(0);
//     theStyle->SetTitleFontSize(0.05);
    // theStyle->SetTitleH(0); // Set the height of the title box
    // theStyle->SetTitleW(0); // Set the width of the title box
    // theStyle->SetTitleX(0); // Set the position of the title box
    theStyle->SetTitleY(0.96); // Set the position of the title box
    theStyle->SetTitleStyle(0);
    theStyle->SetTitleBorderSize(0);


    // For the axis titles:

//     theStyle->SetTitleColor(1, "XYZ");
//     theStyle->SetTitleFont(42, "XYZ");
    //    theStyle->SetTitleSize(0.05, "XYZ");
    // theStyle->SetTitleXSize(Float_t size = 0.02); // Another way to set the size?
    // theStyle->SetTitleYSize(Float_t size = 0.02);
//     theStyle->SetTitleXOffset(0.9);
//     theStyle->SetTitleYOffset(1.25);
    // theStyle->SetTitleOffset(1.1, "Y"); // Another way to set the Offset

    // For the axis labels:

//     theStyle->SetLabelColor(1, "XYZ");
//     theStyle->SetLabelFont(42, "XYZ");
//     theStyle->SetLabelOffset(0.007, "XYZ");
//     theStyle->SetLabelSize(0.045, "XYZ");

    // For the axis:

    theStyle->SetAxisColor(1, "XYZ");
    theStyle->SetStripDecimals(kTRUE);
    theStyle->SetTickLength(0.03, "XYZ");
    theStyle->SetNdivisions(510, "XYZ");
    theStyle->SetPadTickX(1);  // To get tick marks on the opposite side of the frame
    theStyle->SetPadTickY(1);

    // Change for log plots:
    theStyle->SetOptLogx(0);
    theStyle->SetOptLogy(0);
    theStyle->SetOptLogz(0);

    // Postscript options:
    theStyle->SetPaperSize(20.,20.);
    // theStyle->SetLineScalePS(Float_t scale = 3);
    // theStyle->SetLineStyleString(Int_t i, const char* text);
    // theStyle->SetHeaderPS(const char* header);
    // theStyle->SetTitlePS(const char* pstitle);

    // theStyle->SetBarOffset(Float_t baroff = 0.5);
    // theStyle->SetBarWidth(Float_t barwidth = 0.5);
    // theStyle->SetPaintTextFormat(const char* format = "g");
    // theStyle->SetPalette(Int_t ncolors = 0, Int_t* colors = 0);
    // theStyle->SetTimeOffset(Double_t toffset);
    // theStyle->SetHistMinimumZero(kTRUE);
    theStyle->SetTextSize(0.045);
    //    theStyle->SetTextFont(42);
    
    //   style->SetOptFit(101);
    //   style->SetOptStat(1111111);

  } else {
    // Avoid modifying the default style!
    theStyle = gStyle;
  }
  return theStyle;
}


setPalette()
{
  const Int_t NRGBs = 5;
  const Int_t NCont = 255;
 
//   { // Fine rainbow
//     Double_t stops[NRGBs] = { 0.00, 0.34, 0.61, 0.84, 1.00 };
//     Double_t red[NRGBs]   = { 0.00, 0.00, 0.87, 1.00, 0.51 };
//     Double_t green[NRGBs] = { 0.00, 0.81, 1.00, 0.20, 0.00 };
//     Double_t blue[NRGBs]  = { 0.51, 1.00, 0.12, 0.00, 0.00 };
//   }
 
//   { // blues
//     Double_t stops[NRGBs] = { 0.00, 0.34, 0.61, 0.84, 1.00 };
//     Double_t red[NRGBs]   = { 1.00, 0.84, 0.61, 0.34, 0.00 };
//     Double_t green[NRGBs] = { 1.00, 0.84, 0.61, 0.34, 0.00 };
//     Double_t blue[NRGBs]  = { 1.00, 1.00, 1.00, 1.00, 1.00 };
//   }


//   { // Gray (white->black)
//     Double_t stops[NRGBs] = { 0.00, 0.34, 0.61, 0.84, 1.00 };
//     Double_t red[NRGBs]   = { 1.00, 0.84, 0.61, 0.34, 0.00 };
//     Double_t green[NRGBs] = { 1.00, 0.84, 0.61, 0.34, 0.00 };
//     Double_t blue[NRGBs]  = { 1.00, 0.84, 0.61, 0.34, 0.00 };
//   }


  { // Gray (white->gray)
    //  similar to gStyle->SetPalette(5);
    float max = 0.3;
    float step=(1-max)/(NRGBs-1);
    Double_t stops[NRGBs] = { 0.00, 0.34, 0.61, 0.84, 1.00 };
    Double_t red[NRGBs]   = { 1.00, 1-step, 1-2*step, 1-3*step, 1-4*step };
    Double_t green[NRGBs] = { 1.00, 1-step, 1-2*step, 1-3*step, 1-4*step };
    Double_t blue[NRGBs]  = { 1.00, 1-step, 1-2*step, 1-3*step, 1-4*step };
  }


 TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
 gStyle->SetNumberContours(NCont);
}


// Overlay efficiency plots for S1, S2, S3
void plotEff(TH1* h1, TH1* h2=0, TH1* h3=0) {
  float minY=0.6;
  float maxY=1.05;
  h1->GetYaxis()->SetRangeUser(minY,maxY);
  h1->GetXaxis()->SetRangeUser(0.,2.1);
  h1->SetMarkerStyle(20);
  h1->SetMarkerSize(0.5);
  h1->SetStats(0);
  h1->Draw();
  
  if (h2) {
    h2->SetLineColor(kRed);
    h2->SetMarkerColor(kRed);
    h2->SetMarkerStyle(20);
    h2->SetMarkerSize(0.5);
    h2->SetStats(0);
    h2->Draw("same");
  }
  
  if (h3) {
    h3->SetLineColor(kBlack);
    h3->SetMarkerColor(kBlack);
    h3->SetMarkerStyle(20);
    h3->SetMarkerSize(0.5);
    h3->SetStats(0);
    h3->Draw("same");
  }
}

TH1F* getEffPlot(TH1* hnum, TH1* hden, int rebin=1) {
  hnum->Rebin(rebin);
  hden->Rebin(rebin);
  TH1F* h = hnum->Clone();
  h->Sumw2();
  h->Divide(hnum,hden,1.,1.,"B");
  h->SetStats(0);
  return h;
}
