#ifndef STYLE__H
#define STYLE__H

#include "TStyle.h"

// tdrGrid: Turns the grid lines on (true) or off (false)
inline TStyle genStyle() {
  TStyle myStyle("myStyle", "Style similar to P-TDR");

  // For the canvas:
  myStyle.SetCanvasBorderMode(0);
  myStyle.SetCanvasColor(kWhite);
  myStyle.SetCanvasDefH(600);  // Height of canvas
  myStyle.SetCanvasDefW(600);  // Width of canvas
  myStyle.SetCanvasDefX(0);    // POsition on screen
  myStyle.SetCanvasDefY(0);

  // For the Pad:
  myStyle.SetPadBorderMode(0);
  // myStyle.SetPadBorderSize(Width_t size = 1);
  myStyle.SetPadColor(kWhite);
  myStyle.SetPadGridX(false);
  myStyle.SetPadGridY(false);
  myStyle.SetGridColor(0);
  myStyle.SetGridStyle(3);
  myStyle.SetGridWidth(1);

  // For the frame:
  myStyle.SetFrameBorderMode(0);
  myStyle.SetFrameBorderSize(1);
  myStyle.SetFrameFillColor(0);
  myStyle.SetFrameFillStyle(0);
  myStyle.SetFrameLineColor(1);
  myStyle.SetFrameLineStyle(1);
  myStyle.SetFrameLineWidth(1);

  // For the histo:
  // myStyle.SetHistFillColor(1);
  // myStyle.SetHistFillStyle(0);
  myStyle.SetHistLineColor(1);
  myStyle.SetHistLineStyle(0);
  myStyle.SetHistLineWidth(1);
  // myStyle.SetLegoInnerR(Float_t rad = 0.5);
  // myStyle.SetNumberContours(Int_t number = 20);

  myStyle.SetEndErrorSize(2);
  // myStyle.SetErrorMarker(20);
  myStyle.SetErrorX(0.);

  myStyle.SetMarkerStyle(20);

  // For the fit/function:
  myStyle.SetOptFit(1);
  myStyle.SetFitFormat("5.4g");
  myStyle.SetFuncColor(1);
  myStyle.SetFuncStyle(0);
  myStyle.SetFuncWidth(0);

  // For the date:
  myStyle.SetOptDate(0);
  // myStyle.SetDateX(Float_t x = 0.01);
  // myStyle.SetDateY(Float_t y = 0.01);

  // For the statistics box:
  myStyle.SetOptFile(1);
  myStyle.SetOptStat("mre");  // To display the mean and RMS:   SetOptStat("mr");
  myStyle.SetStatColor(kWhite);
  myStyle.SetStatFont(42);
  myStyle.SetStatFontSize(0.025);
  myStyle.SetStatTextColor(1);
  myStyle.SetStatFormat("6.4g");
  myStyle.SetStatBorderSize(1);
  myStyle.SetStatH(0.1);
  myStyle.SetStatW(0.15);
  // myStyle.SetStatStyle(Style_t style = 1001);
  // myStyle.SetStatX(Float_t x = 0);
  // myStyle.SetStatY(Float_t y = 0);

  // Margins:
  //  myStyle.SetPadTopMargin(0.05);
  //  myStyle.SetPadBottomMargin(0.13);
  //  myStyle.SetPadLeftMargin(0.16);
  //  myStyle.SetPadRightMargin(/*0.02*/0.1);

  // For the Global title:

  myStyle.SetOptTitle(0);
  myStyle.SetTitleFont(42);
  myStyle.SetTitleColor(1);
  myStyle.SetTitleTextColor(1);
  myStyle.SetTitleFillColor(10);
  myStyle.SetTitleFontSize(0.05);
  // myStyle.SetTitleH(0); // Set the height of the title box
  // myStyle.SetTitleW(0); // Set the width of the title box
  // myStyle.SetTitleX(0); // Set the position of the title box
  // myStyle.SetTitleY(0.985); // Set the position of the title box
  // myStyle.SetTitleStyle(Style_t style = 1001);
  // myStyle.SetTitleBorderSize(2);

  // For the axis titles:

  myStyle.SetTitleColor(1, "XYZ");
  myStyle.SetTitleFont(42, "XYZ");
  myStyle.SetTitleSize(0.06, "XYZ");
  // myStyle.SetTitleXSize(Float_t size = 0.02); // Another way to set the size?
  // myStyle.SetTitleYSize(Float_t size = 0.02);
  //  myStyle.SetTitleXOffset(0.9);
  //  myStyle.SetTitleYOffset(1.25);
  // myStyle.SetTitleOffset(1.1, "Y"); // Another way to set the Offset

  // For the axis labels:

  myStyle.SetLabelColor(1, "XYZ");
  myStyle.SetLabelFont(42, "XYZ");
  //  myStyle.SetLabelOffset(0.01, "XYZ");
  //  myStyle.SetLabelSize(0.03/*0.05*/, "XYZ");

  // For the axis:

  myStyle.SetAxisColor(1, "XYZ");
  myStyle.SetStripDecimals(kTRUE);
  myStyle.SetTickLength(0.03, "XYZ");
  myStyle.SetNdivisions(510, "XYZ");
  myStyle.SetPadTickX(1);  // To get tick marks on the opposite side of the frame
  myStyle.SetPadTickY(1);

  // Change for log plots:
  myStyle.SetOptLogx(0);
  myStyle.SetOptLogy(0);
  myStyle.SetOptLogz(0);

  // Postscript options:
  myStyle.SetPaperSize(20., 20.);
  // myStyle.SetLineScalePS(Float_t scale = 3);
  // myStyle.SetLineStyleString(Int_t i, const char* text);
  // myStyle.SetHeaderPS(const char* header);
  // myStyle.SetTitlePS(const char* pstitle);

  // myStyle.SetBarOffset(Float_t baroff = 0.5);
  // myStyle.SetBarWidth(Float_t barwidth = 0.5);
  // myStyle.SetPaintTextFormat(const char* format = "g");
  // myStyle.SetPalette(Int_t ncolors = 0, Int_t* colors = 0);
  // myStyle.SetTimeOffset(Double_t toffset);
  // myStyle.SetHistMinimumZero(kTRUE);

  return myStyle;
}

#endif  // STYLE__H
