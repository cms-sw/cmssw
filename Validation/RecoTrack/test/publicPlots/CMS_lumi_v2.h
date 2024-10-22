#include "TPad.h"
#include "TLatex.h"
#include "TLine.h"
#include "TBox.h"
#include "TASImage.h"

//
// Global variables
//

//TString cmsText     = "CMS Phase II Simulation";
//TString cmsText     = "CMS Phase I/II Simulation";
//TString cmsText     = "CMS Simulation";
//TString cmsText     = "CMS Simulation";
TString cmsText     = "CMS";
float cmsTextFont   = 61;  // default is helvetic-bold

//bool writeExtraText = false;
bool writeExtraText = true;
//TString extraText   = "Preliminary";
TString extraText   = "Simulation preliminary";
//TString extraText   = "Simulation private";
float extraTextFont = 52;  // default is helvetica-italics

// text sizes and text offsets with respect to the top frame
// in unit of the top margin size
float lumiTextSize     = 0.6;
float lumiTextOffset   = 0.2;
float cmsTextSize      = 0.75;
float cmsTextOffset    = 0.1;  // only used in outOfFrame version

float relPosX    = 0.045;
float relPosY    = 0.035;
float relExtraDY = 1.2;

// ratio of "CMS" and extra text size
float extraOverCmsTextSize  = 0.76;

//TString lumi_13TeV = "20.1 fb^{-1}";
//TString lumi_13TeV = "PU = 35";
TString lumi_13TeV = "";
TString lumi_8TeV  = "19.7 fb^{-1}";
TString lumi_7TeV  = "5.1 fb^{-1}";
//TString lumi_14TeV = "3000 fb^{-1}, PU = 140";
//TString lumi_14TeV = "PU = 140";
TString lumi_14TeV = "";

bool drawLogo      = false;

void CMS_lumi_v2( TPad* pad, int iPeriod=3, int iPosX=10 );

