#include "Validation/DTRecHits/interface/utils.h"
#include "TF1.h"
#include "TProfile.h"
//#include "TLine.h"
void Tutils::drawGFit(TH1 * h1, float min, float max, float minfit, float maxfit) {
  setStyle(h1);
  static int i = 0;
  i++;
  //h1->SetGrid(1,1);
  //h1->SetGridColor(15);
  h1->GetXaxis()->SetRangeUser(min,max);
  TString  fitName = "g";
  fitName += i;
    TF1* g1 = new TF1(fitName.Data(),"gaus",minfit,maxfit);
  g1->SetLineColor(2);
  g1->SetLineWidth(2);
  h1->Fit(g1,"RQ");
  h1->Draw();
//   TPaveStats *st = (TPaveStats*)h1->GetListOfFunctions()->FindObject("stats");
//   st->SetX2NDC(0.905);
//   st->SetY2NDC(0.905);
}
void Tutils::setStyle(TH1 *histo) {
  mystyle = getStyle("tdr");
  histo->GetXaxis()->SetTitleFont(mystyle->GetTitleFont());
  histo->GetXaxis()->SetTitleSize(mystyle->GetTitleFontSize());
  histo->GetXaxis()->SetLabelFont(mystyle->GetLabelFont());
  histo->GetXaxis()->SetLabelSize(mystyle->GetLabelSize());

  histo->GetYaxis()->SetTitleFont(mystyle->GetTitleFont());
  histo->GetYaxis()->SetTitleSize(mystyle->GetTitleFontSize());
  histo->GetYaxis()->SetLabelFont(mystyle->GetLabelFont());
  histo->GetYaxis()->SetLabelSize(mystyle->GetLabelSize());
}

void Tutils::setStyle(TH2 *histo) {
  mystyle = getStyle("tdr");
  histo->GetXaxis()->SetTitleFont(mystyle->GetTitleFont());
  histo->GetXaxis()->SetTitleSize(mystyle->GetTitleFontSize());
  histo->GetXaxis()->SetLabelFont(mystyle->GetLabelFont());
  histo->GetXaxis()->SetLabelSize(mystyle->GetLabelSize());

  histo->GetYaxis()->SetTitleFont(mystyle->GetTitleFont());
  histo->GetYaxis()->SetTitleSize(mystyle->GetTitleFontSize());
  histo->GetYaxis()->SetLabelFont(mystyle->GetLabelFont());
  histo->GetYaxis()->SetLabelSize(mystyle->GetLabelSize());
}

TStyle * Tutils::getStyle(TString name)
{
  TStyle *theStyle;
  if ( name == "mstyle" ) {
    theStyle = new TStyle("mstyle", "mstyle");
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

void Tutils::plotAndProfileX (TH2* h2, float min, float max,bool profile) {
  setStyle(h2);
  //  gPad->SetGrid(1,1);
  //gStyle->SetGridColor(15);
  h2->GetYaxis()->SetRangeUser(min,max);
  h2->Draw();
  if (profile) {
    TProfile* prof = h2->ProfileX();
    prof->SetMarkerColor(2);
    prof->SetLineColor(2);
    prof->Draw("same");
  }
  //TLine * l = new TLine(h2->GetXaxis()->GetXmin(),0,h2->GetXaxis()->GetXmax(),0);
  //l->SetLineColor(3);
  //l->Draw();
}
