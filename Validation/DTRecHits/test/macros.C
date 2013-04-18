/** 
 * A collection of simple ROOT macros
 *
 * N. Amapane 2002-2004
 */
/*
 * Draw 2-D plots superimposed to their profiles
 *
 * 2003 NCA
 */
// Draw a 2-D plot within the specified Y range and superimpose its X profile

#include <sstream>
#include <iomanip>

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

void plotAndProfileX (TH2* h2, float min, float max, bool profile=false) {
  setStyle(h2);
  gPad->SetGrid(1,1);
  gStyle->SetGridColor(15);
  h2->GetYaxis()->SetRangeUser(min,max);
  h2->Draw();
  if (profile) {
    TProfile* prof = h2->ProfileX();
    prof->SetMarkerColor(2);
    prof->SetLineColor(2);
    prof->Draw("same");
  }
  TLine * l = new TLine(h2->GetXaxis()->GetXmin(),0,h2->GetXaxis()->GetXmax(),0);
  l->SetLineColor(3);
  l->Draw();
}

// void plotAndProfileY (TH2* h2, float min, float max) {
//   h2->GetYaxis()->SetRangeUser(min,max);
//   h2->Draw();
//   TProfile* prof = h2->ProfileY();
//   prof->SetMarkerStyle(8);
//   prof->SetMarkerSize(0.7);
//   prof->SetMarkerColor(2);
//   prof->SetLineColor(2);
//   prof->Draw("same");
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
  }
  TLine * l = new TLine(h2->GetXaxis()->GetXmin(),0,h2->GetXaxis()->GetXmax(),0);
  l->SetLineColor(3);
  l->Draw();
}
/*
 * Draw and format a fitted histogram 
 *
 * 2003 NCA
 */
// Fit a histogram with a gaussian and draw it in the range 
// mean+-nsigmas*RMS.
void drawGFit(TH1 * h1, float nsigmas, float min, float max){
  float minfit = h1->GetMean() - h1->GetRMS();
  float maxfit = h1->GetMean() + h1->GetRMS();
  drawGFit(h1, min, max, minfit, maxfit);
  gPad->Draw();
}
// Fit a histogram with a gaussian and draw it in the specified range.
void drawGFit(TH1 * h1, float min, float max){
  drawGFit(h1, min, max, min, max);
  gPad->Draw();
}
// Fit a histogram in the range (minfit, maxfit) with a gaussian and
// draw it in the range (min, max)
void drawGFit(TH1 * h1, float min, float max, float minfit, float maxfit) {
  setStyle(h1);
  static int i = 0;
  i++;
  gPad->SetGrid(1,1);
  gStyle->SetGridColor(15);
  h1->GetXaxis()->SetRangeUser(min,max);
  TF1* g1 = new TF1(TString("g")+i,"gaus",minfit,maxfit);
  g1->SetLineColor(2);
  g1->SetLineWidth(2);
  h1->Fit(g1,"R");
  h1->Draw();
//   TPaveStats *st = (TPaveStats*)h1->GetListOfFunctions()->FindObject("stats");
//   st->SetX2NDC(0.905);
//   st->SetY2NDC(0.905);
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
// Print all canvases in separate EPS files
void printCanvases(TString type=".eps"){
  TIter iter(gROOT->GetListOfCanvases());
  TCanvas *c;
  while( (c = (TCanvas *)iter()) ) {
    c->cd();
    c->Print(0,type);
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
    theStyle->SetCanvasDefH(600); //Height of canvas
    theStyle->SetCanvasDefW(600); //Width of canvas
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
    theStyle->SetHistLineColor(1);
    theStyle->SetHistLineStyle(0);
    theStyle->SetHistLineWidth(1);
    // theStyle->SetLegoInnerR(Float_t rad = 0.5);
    // theStyle->SetNumberContours(Int_t number = 20);

    theStyle->SetEndErrorSize(2);
//     theStyle->SetErrorMarker(20);
    theStyle->SetErrorX(0.);
  
    theStyle->SetMarkerStyle(20);

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
    theStyle->SetOptStat(10);
    theStyle->SetStatColor(kWhite);
    theStyle->SetStatFont(42);
    theStyle->SetStatFontSize(0.07);
    theStyle->SetStatTextColor(1);
    theStyle->SetStatFormat("6.4g");
    theStyle->SetStatBorderSize(1);
    theStyle->SetStatH(0.3);
    theStyle->SetStatW(0.2);
    // theStyle->SetStatStyle(Style_t style = 1001);
    // theStyle->SetStatX(Float_t x = 0);
    // theStyle->SetStatY(Float_t y = 0);

    // Margins:
    theStyle->SetPadTopMargin(0.05);
    theStyle->SetPadBottomMargin(0.13);
    theStyle->SetPadLeftMargin(0.16);
    theStyle->SetPadRightMargin(0.02);

    // For the Global title:

    theStyle->SetOptTitle(0);
    theStyle->SetTitleFont(42);
    theStyle->SetTitleColor(1);
    theStyle->SetTitleTextColor(1);
    theStyle->SetTitleFillColor(10);
    theStyle->SetTitleFontSize(0.05);
    // theStyle->SetTitleH(0); // Set the height of the title box
    // theStyle->SetTitleW(0); // Set the width of the title box
    // theStyle->SetTitleX(0); // Set the position of the title box
    // theStyle->SetTitleY(0.985); // Set the position of the title box
    // theStyle->SetTitleStyle(Style_t style = 1001);
    // theStyle->SetTitleBorderSize(2);

    // For the axis titles:

    theStyle->SetTitleColor(1, "XYZ");
    theStyle->SetTitleFont(42, "XYZ");
    theStyle->SetTitleSize(0.06, "XYZ");
    // theStyle->SetTitleXSize(Float_t size = 0.02); // Another way to set the size?
    // theStyle->SetTitleYSize(Float_t size = 0.02);
    theStyle->SetTitleXOffset(0.9);
    theStyle->SetTitleYOffset(1.25);
    // theStyle->SetTitleOffset(1.1, "Y"); // Another way to set the Offset

    // For the axis labels:

    theStyle->SetLabelColor(1, "XYZ");
    theStyle->SetLabelFont(42, "XYZ");
    theStyle->SetLabelOffset(0.007, "XYZ");
    theStyle->SetLabelSize(0.045, "XYZ");

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


    //   style->SetOptFit(101);
    //   style->SetOptStat(1111111); 

  } else {
    // Avoid modifying the default style!
    theStyle = gStyle;
  }
  return theStyle;
}
