//
// Overlay RecHits plots from a pair of DQM validation files
//  Strips and Phase 0 or Phase 1 pixels
// Loosely based on Validation/TrackerRecHits/test/SiStripRecHitsCompare.C
//  Bill Ford  10 Oct 2017
//
#include "TStyle.h"
#include "TROOT.h"
#include "TH1F.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TText.h"

#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
using std::cout;
using std::endl;

#include <fstream>
using std::ifstream;

#include <cstring>

#include "HistoCompare.C"

// static const char* outfiletype = "pdf";
static const char* outfiletype = "gif";

void SetUpHistograms(TH1F* h1, TH1F* h2, float bincorr=1.0)
{
  float scale1 = -9999.9;
  float scale2 = -9999.9;

  if ( h1->Integral() != 0 && h2->Integral() != 0 ) {
    if (bincorr > 0.0) {
      scale1 = 1.0/(float)h1->Integral();
    // In case the bin width is different between reference and new files:
      scale2 = bincorr/(float)h2->Integral();
//     scale2 = scale1;
//     scale2 *= 1000.0/936.0;
      h1->Sumw2();
      h2->Sumw2();
      h1->Scale(scale1);
      h2->Scale(scale2);
    } else {
      h1->SetLineStyle(3);
    }

    h1->SetLineWidth(1);
    h2->SetLineWidth(1);
    h1->SetLineColor(4);
    h2->SetLineColor(2);
  }
}  // -----------------------------------------------------------

void TIBcompare(HistoCompare* myPV, TDirectory* rdir, TDirectory* sdir, const char* varname,
		const char* historoot, const Int_t logy, const float bincorr = 1.0) {
  TH1F* refplotsTIB[6];
  TH1F* newplotsTIB[6];
  char objname[80], histoname[80];
  Int_t layer, plotidx;
  TText* te = new TText;

  plotidx = 0;
  for (layer = 1; layer < 5; ++layer) {
    sprintf(objname, "TIB/layer_%1d/%s_rphi__TIB__layer__%1d", layer, varname, layer);
    rdir->GetObject(objname, refplotsTIB[plotidx]);
    sdir->GetObject(objname, newplotsTIB[plotidx]);
    plotidx++;
  }
  for (layer = 1; layer < 3; ++layer) {
    sprintf(objname, "TIB/layer_%1d/%s_stereo__TIB__layer__%1d", layer, varname, layer);
    rdir->GetObject(objname, refplotsTIB[plotidx]);
    sdir->GetObject(objname, newplotsTIB[plotidx]);
    plotidx++;
  }

  TCanvas* Strip_TIB = new TCanvas("Strip_TIB","Strip_TIB",1000,1000);
  Strip_TIB->Divide(2,3);
  for (Int_t i=0; i<6; ++i) {
    if (refplotsTIB[i]->GetEntries() == 0 || newplotsTIB[i]->GetEntries() == 0) continue;
    Strip_TIB->cd(i+1);
    SetUpHistograms(refplotsTIB[i], newplotsTIB[i], bincorr);
    refplotsTIB[i]->Draw();
    newplotsTIB[i]->Draw("sames");
    myPV->PVCompute(refplotsTIB[i], newplotsTIB[i], te);
    gPad->SetLogy(logy);
  }

  sprintf(histoname, "%sTIBCompare.%s", historoot, outfiletype);  Strip_TIB->Print(histoname);

}  // ------------------------------------------------------------------------

void TOBcompare(HistoCompare* myPV, TDirectory* rdir, TDirectory* sdir, const char* varname,
		const char* historoot, const Int_t logy, const float bincorr = 1.0) {
  TH1F* refplotsTOB[8];
  TH1F* newplotsTOB[8];
  char objname[80], histoname[80];
  Int_t layer, plotidx;
  TText* te = new TText;

  plotidx = 0;
  for (layer = 1; layer < 7; ++layer) {
    sprintf(objname, "TOB/layer_%1d/%s_rphi__TOB__layer__%1d", layer, varname, layer);
    rdir->GetObject(objname, refplotsTOB[plotidx]);
    sdir->GetObject(objname, newplotsTOB[plotidx]);
    plotidx++;
  }
  for (layer = 1; layer < 3; ++layer) {
    sprintf(objname, "TOB/layer_%1d/%s_stereo__TOB__layer__%1d", layer, varname, layer);
    rdir->GetObject(objname, refplotsTOB[plotidx]);
    sdir->GetObject(objname, newplotsTOB[plotidx]);
    plotidx++;
  }

  TCanvas* Strip_TOB = new TCanvas("Strip_TOB","Strip_TOB",1000,1000);
  Strip_TOB->Divide(3,3);
  for (Int_t i=0; i<8; ++i) {
    if (refplotsTOB[i]->GetEntries() == 0 || newplotsTOB[i]->GetEntries() == 0) continue;
    Strip_TOB->cd(i+1);
    SetUpHistograms(refplotsTOB[i],newplotsTOB[i], bincorr);
    refplotsTOB[i]->Draw();
    newplotsTOB[i]->Draw("sames");
    myPV->PVCompute(refplotsTOB[i], newplotsTOB[i], te);
    gPad->SetLogy(logy);
  }

  sprintf(histoname, "%sTOBCompare.%s", historoot, outfiletype);  Strip_TOB->Print(histoname);

}  // ------------------------------------------------------------------------

void TIDcompare(HistoCompare* myPV, TDirectory* rdir, TDirectory* sdir, const char* varname,
		const char* historoot, const Int_t logy, const float bincorr = 1.0) {
  TH1F* refplotsTID[5];
  TH1F* newplotsTID[5];
  char objname[80], histoname[80];
  Int_t layer, plotidx;
  TText* te = new TText;

  plotidx = 0;
  for (layer = 1; layer < 4; ++layer) {
    sprintf(objname, "TID/MINUS/ring_%1d/%s_rphi__TID__MINUS__ring__%1d", layer, varname, layer);
    rdir->GetObject(objname, refplotsTID[plotidx]);
     sdir->GetObject(objname, newplotsTID[plotidx]);
   plotidx++;
  }
  for (layer = 1; layer < 3; ++layer) {
    sprintf(objname, "TID/MINUS/ring_%1d/%s_stereo__TID__MINUS__ring__%1d", layer, varname, layer);
    rdir->GetObject(objname, refplotsTID[plotidx]);
    sdir->GetObject(objname, newplotsTID[plotidx]);
    plotidx++;
  }

  TCanvas* Strip_TID = new TCanvas("Strip_TID","Strip_TID",1000,1000);
  Strip_TID->Divide(2,3);
  for (Int_t i=0; i<5; ++i) {
    if (refplotsTID[i]->GetEntries() == 0 || newplotsTID[i]->GetEntries() == 0) continue;
    Strip_TID->cd(i+1);
    SetUpHistograms(refplotsTID[i],newplotsTID[i], bincorr);
    refplotsTID[i]->Draw();
    newplotsTID[i]->Draw("sames");
    myPV->PVCompute(refplotsTID[i], newplotsTID[i], te);
    gPad->SetLogy(logy);
  }

  sprintf(histoname, "%sTIDCompare.%s", historoot, outfiletype);  Strip_TID->Print(histoname);

}  // ------------------------------------------------------------------------

void TECcompare(HistoCompare* myPV, TDirectory* rdir, TDirectory* sdir, const char* varname,
		const char* historoot, const Int_t logy, const float bincorr = 1.0) {
  TH1F* refplotsTEC[10];
  TH1F* newplotsTEC[10];
  char objname[80], histoname[80];
  Int_t layer[3] = {1, 2, 5}, lidx, plotidx;
  TText* te = new TText;

  plotidx = 0;
  for (lidx = 1; lidx < 8; ++lidx) {
    sprintf(objname, "TEC/MINUS/ring_%1d/%s_rphi__TEC__MINUS__ring__%1d", lidx, varname, lidx);
    rdir->GetObject(objname, refplotsTEC[plotidx]);
    sdir->GetObject(objname, newplotsTEC[plotidx]);
    plotidx++;
  }
  for (lidx = 1; lidx < 3; ++lidx) {
    sprintf(objname, "TEC/MINUS/ring_%1d/%s_stereo__TEC__MINUS__ring__%1d",
	    layer[lidx], varname, layer[lidx]);
    rdir->GetObject(objname, refplotsTEC[plotidx]);
    sdir->GetObject(objname, newplotsTEC[plotidx]);
    plotidx++;
  }

  TCanvas* Strip_TEC = new TCanvas("Strip_TEC","Strip_TEC",1000,1000);
  Strip_TEC->Divide(2,3);
  for (Int_t i=0; i<5; ++i) {
    if (refplotsTEC[i]->GetEntries() == 0 || newplotsTEC[i]->GetEntries() == 0) continue;
    Strip_TEC->cd(i+1);
    SetUpHistograms(refplotsTEC[i],newplotsTEC[i], bincorr);
    refplotsTEC[i]->Draw();
    newplotsTEC[i]->Draw("sames");
    myPV->PVCompute(refplotsTEC[i], newplotsTEC[i], te);
    gPad->SetLogy(logy);
  }

  sprintf(histoname, "%sTECCompare.%s", historoot, outfiletype);  Strip_TEC->Print(histoname);

}  // ------------------------------------------------------------------------

void BPIXcompare(HistoCompare* myPV, TDirectory* rdir, TDirectory* sdir, const char* varname, int layer, const char* historoot) {

  TH1F* refplotsBPIX[11];
  TH1F* newplotsBPIX[11];
  char objname[80], histoname[80];
  std::string dirroot("");
  Int_t module, plotidx;
  TText* te = new TText;

  if (strstr(varname, "Pull")) dirroot = "Pulls";

  plotidx = 0;
  sprintf(objname, "recHit%sBPIX/RecHit_NsimHit_Layer%1d", "", layer);
  rdir->GetObject(objname, refplotsBPIX[plotidx]);
  sdir->GetObject(objname, newplotsBPIX[plotidx]);
  plotidx++;
  sprintf(objname, "recHit%sBPIX/RecHit_X%s_FlippedLadder_Layer%1d", dirroot.c_str(), varname, layer);
  rdir->GetObject(objname, refplotsBPIX[plotidx]);
  sdir->GetObject(objname, newplotsBPIX[plotidx]);
  plotidx++;
  sprintf(objname, "recHit%sBPIX/RecHit_X%s_UnFlippedLadder_Layer%1d", dirroot.c_str(), varname, layer);
  rdir->GetObject(objname, refplotsBPIX[plotidx]);
  sdir->GetObject(objname, newplotsBPIX[plotidx]);
  plotidx++;
  for (module = 0; module < 8; ++module) {
    sprintf(objname, "recHit%sBPIX/RecHit_Y%s_Layer%1d_Module%1d", dirroot.c_str(), varname, layer, module+1);
    rdir->GetObject(objname, refplotsBPIX[plotidx]);
    sdir->GetObject(objname, newplotsBPIX[plotidx]);
    plotidx++;
  }
  int nplots = plotidx;

  TCanvas* Pixel = new TCanvas("Pixel", "Pixel", 1000, 1000);
  Pixel->Divide(3,4);
  for (Int_t i=0; i<nplots; ++i) {
    if (refplotsBPIX[i]->GetEntries() == 0 || newplotsBPIX[i]->GetEntries() == 0) continue;
    Pixel->cd(i+1);
    SetUpHistograms(refplotsBPIX[i],newplotsBPIX[i]);
    refplotsBPIX[i]->Draw();
    newplotsBPIX[i]->Draw("sames");
    myPV->PVCompute(refplotsBPIX[i], newplotsBPIX[i], te);
  }
  TPad *p1 = (TPad *)(Pixel->cd(1));  p1->SetLogy(1);
  sprintf(histoname, "%sBPIX_Layer_%1dCompare.%s", historoot, layer, outfiletype);  Pixel->Print(histoname);

}  // ------------------------------------------------------------------------

void FPIXcompare(HistoCompare* myPV, TDirectory* rdir, TDirectory* sdir, const char* varname, int disk, const char* historoot) {

  TH1F* refplotsFPIX[8];
  TH1F* newplotsFPIX[8];
  char objname[80], histoname[80];
  std::string dirroot("");
  Int_t plaquette, plotidx;
  TText* te = new TText;

  if (strstr(varname, "Pull")) dirroot = "Pulls";

  plotidx = 0;
  sprintf(objname, "recHit%sFPIX/RecHit_NsimHit_Disk%1d", "", disk);
  rdir->GetObject(objname, refplotsFPIX[plotidx]);
  sdir->GetObject(objname, newplotsFPIX[plotidx]);
  plotidx++;
  for (plaquette = 0; plaquette < 7; ++plaquette) {
    sprintf(objname, "recHit%sFPIX/RecHit_Y%s_Disk%1d_Plaquette%1d", dirroot.c_str(), varname, disk, plaquette+1);
    rdir->GetObject(objname, refplotsFPIX[plotidx]);
    sdir->GetObject(objname, newplotsFPIX[plotidx]);
    plotidx++;
  }
  int nplots = plotidx;

  TCanvas* Pixel = new TCanvas("Pixel","Pixel",1000,1000);
  Pixel->Divide(3,3);
  for (Int_t i=0; i<nplots; ++i) {
    if (refplotsFPIX[i]->GetEntries() == 0 || newplotsFPIX[i]->GetEntries() == 0) continue;
    Pixel->cd(i+1);
    SetUpHistograms(refplotsFPIX[i],newplotsFPIX[i]);
    refplotsFPIX[i]->Draw();
    newplotsFPIX[i]->Draw("sames");
    myPV->PVCompute(refplotsFPIX[i], newplotsFPIX[i], te);
  }

  TPad *p1 = (TPad *)(Pixel->cd(1));  p1->SetLogy(1);
  sprintf(histoname, "%sFPIX_Disk_%1dCompare.%s", historoot, disk, outfiletype);  Pixel->Print(histoname);

}  // ------------------------------------------------------------------------

void Phase1PIXcompare(HistoCompare* myPV, TDirectory* rdir, TDirectory* sdir, const Int_t side, const Int_t layerdisk) {

  TH1F* refplots[6];
  TH1F* newplots[6];
  char objname[80], histoname[80];
  std::string subdir(""), LayerDisk(""), Side("");
  Int_t abslayerdisk, plotidx;
  TText* te = new TText;
  if (side == 0) {
    subdir = "PXBarrel";
    LayerDisk = "Layer";
    abslayerdisk = layerdisk;
    Side = "";
  } else {
    subdir = "PXForward";
    LayerDisk = "Disk";
    abslayerdisk = layerdisk > 0 ? layerdisk : -layerdisk;
    Side = side > 0 ? "+" : "-";
  }

  plotidx = 0;
  sprintf(objname, "%s/res_x_PX%s_%s%1d", subdir.c_str(), LayerDisk.c_str(), Side.c_str(), abslayerdisk);
  rdir->GetObject(objname, refplots[plotidx]);
  sdir->GetObject(objname, newplots[plotidx]);
  plotidx++;
  sprintf(objname, "%s/res_y_PX%s_%s%1d", subdir.c_str(), LayerDisk.c_str(), Side.c_str(), abslayerdisk);
  rdir->GetObject(objname, refplots[plotidx]);
  sdir->GetObject(objname, newplots[plotidx]);
  plotidx++;
  sprintf(objname, "%s/pull_x_PX%s_%s%1d", subdir.c_str(), LayerDisk.c_str(), Side.c_str(), abslayerdisk);
  rdir->GetObject(objname, refplots[plotidx]);
  sdir->GetObject(objname, newplots[plotidx]);
  plotidx++;
  sprintf(objname, "%s/pull_y_PX%s_%s%1d", subdir.c_str(), LayerDisk.c_str(), Side.c_str(), abslayerdisk);
  rdir->GetObject(objname, refplots[plotidx]);
  sdir->GetObject(objname, newplots[plotidx]);
  plotidx++;
  sprintf(objname, "%s/nsimhits_PX%s_%s%1d", subdir.c_str(), LayerDisk.c_str(), Side.c_str(), abslayerdisk);
  rdir->GetObject(objname, refplots[plotidx]);
  sdir->GetObject(objname, newplots[plotidx]);
  plotidx++;
  int nplots = plotidx;

  TCanvas* Phase1Pix = new TCanvas("Phase1Pix", "Phase1Pix", 1000, 1000);
  Phase1Pix->Divide(2,3);
  for (Int_t i=0; i<nplots; ++i) {
    if (refplots[i]->GetEntries() == 0 || newplots[i]->GetEntries() == 0) continue;
    Phase1Pix->cd(i+1);
    SetUpHistograms(refplots[i],newplots[i]);
    refplots[i]->Draw();
    newplots[i]->Draw("sames");
    myPV->PVCompute(refplots[i], newplots[i], te);
  }
  TPad *p5 = (TPad *)(Phase1Pix->cd(5));  p5->SetLogy(1);
  sprintf(histoname, "%s_%s_%s%1d_Compare.%s", subdir.c_str(), LayerDisk.c_str(), Side.c_str(), abslayerdisk, outfiletype);
  Phase1Pix->Print(histoname);

}  // ------------------------------------------------------------------------

void RecHitsCompare(
		    const char* rfilename = "DQM_V0001_R000000001__my__relVal__tracker.root",
		    const char* sfilename = "DQM_V0001_R000000001__my__relVal__tracker.root"
		    )
{
  //  color 2 = red  = sfile = new file
  //  color 4 = blue = rfile = reference file

  gROOT ->Reset();
  // gROOT->ProcessLine(".L ../HistoCompare.C");

  delete gROOT->GetListOfFiles()->FindObject(rfilename);
  delete gROOT->GetListOfFiles()->FindObject(sfilename);;
  TFile * rfile = new TFile(rfilename);
  TFile * sfile = new TFile(sfilename);
  TDirectory * rdir=gDirectory;
  TDirectory * sdir=gDirectory;

  HistoCompare * myPV = new HistoCompare();
  TText* te = new TText();
  char histoname[80];
 
  // Strips

  if (rfile->cd("DQMData/Run 1/SiStrip/Run summary/RecHitsValidation/StiffTrackingRecHits/MechanicalView")) {
    rdir = gDirectory;
    if (sfile->cd("DQMData/Run 1/SiStrip/Run summary/RecHitsValidation/StiffTrackingRecHits/MechanicalView")) {
      sdir=gDirectory;

      //=============================================================== 
      // TIB

      // rphi, stereo layers
      TIBcompare(myPV, rdir, sdir, "Pull_LF", "PullLF", 0);
      TIBcompare(myPV, rdir, sdir, "Pull_MF", "PullMF", 0);
      TIBcompare(myPV, rdir, sdir, "Resolx",  "Resolx", 0);
      TIBcompare(myPV, rdir, sdir, "Wclus", "Wclus", 0);
      TIBcompare(myPV, rdir, sdir, "Adc", "Adc", 0);
      TIBcompare(myPV, rdir, sdir, "Posx", "Pos", 0);
      TIBcompare(myPV, rdir, sdir, "Res", "Res", 0);
      TIBcompare(myPV, rdir, sdir, "Chi2", "Chi2", 0);
      TIBcompare(myPV, rdir, sdir, "NsimHit", "NsimHit", 1, -1.0);
 
      // matched layers
      TH1F* matchedtib[16];
      TH1F* newmatchedtib[16];

      rdir->GetObject("TIB/layer_1/Posx_matched__TIB__layer__1",matchedtib[0]);
      rdir->GetObject("TIB/layer_1/Posy_matched__TIB__layer__1",matchedtib[1]);
      rdir->GetObject("TIB/layer_2/Posx_matched__TIB__layer__2",matchedtib[2]);
      rdir->GetObject("TIB/layer_2/Posy_matched__TIB__layer__2",matchedtib[3]);
      rdir->GetObject("TIB/layer_1/Resolx_matched__TIB__layer__1",matchedtib[4]);
      rdir->GetObject("TIB/layer_1/Resoly_matched__TIB__layer__1",matchedtib[5]);
      rdir->GetObject("TIB/layer_2/Resolx_matched__TIB__layer__2",matchedtib[6]);
      rdir->GetObject("TIB/layer_2/Resoly_matched__TIB__layer__2",matchedtib[7]);
      rdir->GetObject("TIB/layer_1/Resx_matched__TIB__layer__1",matchedtib[8]);
      rdir->GetObject("TIB/layer_1/Resy_matched__TIB__layer__1",matchedtib[9]);
      rdir->GetObject("TIB/layer_2/Resx_matched__TIB__layer__2",matchedtib[10]);
      rdir->GetObject("TIB/layer_2/Resy_matched__TIB__layer__2",matchedtib[11]);
      rdir->GetObject("TIB/layer_1/Chi2_matched__TIB__layer__1",matchedtib[12]);
      rdir->GetObject("TIB/layer_2/Chi2_matched__TIB__layer__2",matchedtib[13]);
      rdir->GetObject("TIB/layer_1/NsimHit_matched__TIB__layer__1",matchedtib[14]);
      rdir->GetObject("TIB/layer_2/NsimHit_matched__TIB__layer__2",matchedtib[15]);

      sdir->GetObject("TIB/layer_1/Posx_matched__TIB__layer__1",newmatchedtib[0]);
      sdir->GetObject("TIB/layer_1/Posy_matched__TIB__layer__1",newmatchedtib[1]);
      sdir->GetObject("TIB/layer_2/Posx_matched__TIB__layer__2",newmatchedtib[2]);
      sdir->GetObject("TIB/layer_2/Posy_matched__TIB__layer__2",newmatchedtib[3]);
      sdir->GetObject("TIB/layer_1/Resolx_matched__TIB__layer__1",newmatchedtib[4]);
      sdir->GetObject("TIB/layer_1/Resoly_matched__TIB__layer__1",newmatchedtib[5]);
      sdir->GetObject("TIB/layer_2/Resolx_matched__TIB__layer__2",newmatchedtib[6]);
      sdir->GetObject("TIB/layer_2/Resoly_matched__TIB__layer__2",newmatchedtib[7]);
      sdir->GetObject("TIB/layer_1/Resx_matched__TIB__layer__1",newmatchedtib[8]);
      sdir->GetObject("TIB/layer_1/Resy_matched__TIB__layer__1",newmatchedtib[9]);
      sdir->GetObject("TIB/layer_2/Resx_matched__TIB__layer__2",newmatchedtib[10]);
      sdir->GetObject("TIB/layer_2/Resy_matched__TIB__layer__2",newmatchedtib[11]);
      sdir->GetObject("TIB/layer_1/Chi2_matched__TIB__layer__1",newmatchedtib[12]);
      sdir->GetObject("TIB/layer_2/Chi2_matched__TIB__layer__2",newmatchedtib[13]);
      sdir->GetObject("TIB/layer_1/NsimHit_matched__TIB__layer__1",newmatchedtib[14]);
      sdir->GetObject("TIB/layer_2/NsimHit_matched__TIB__layer__2",newmatchedtib[15]);

      TCanvas* Strip_TIB_matched = new TCanvas("Strip_TIB_matched","Strip_TIB_matched",1000,1000);
      Strip_TIB_matched->Divide(4,4);
      for (Int_t i=0; i<16; ++i) {
        if (matchedtib[i]->GetEntries() == 0 || newmatchedtib[i]->GetEntries() == 0) continue;
        Strip_TIB_matched->cd(i+1);
        if (i == 14 || i == 15) SetUpHistograms(matchedtib[i],newmatchedtib[i], -1.0);
        else SetUpHistograms(matchedtib[i],newmatchedtib[i]);
        matchedtib[i]->Draw();
        newmatchedtib[i]->Draw("sames");
        myPV->PVCompute(matchedtib[i] , newmatchedtib[i] , te );
      }
      // TPad *p15 = (TPad *)(Strip_TIB_matched->cd(15));  p15->SetLogy(1);
      // TPad *p16 = (TPad *)(Strip_TIB_matched->cd(16));  p16->SetLogy(1);

      sprintf(histoname, "MatchedTIBCompare.%s", outfiletype);  Strip_TIB_matched->Print(histoname);

      //======================================================================================================
      // TOB

      // rphi, stereo layers
      TOBcompare(myPV, rdir, sdir, "Pull_LF", "PullLF", 0);
      TOBcompare(myPV, rdir, sdir, "Pull_MF", "PullMF", 0);
      TOBcompare(myPV, rdir, sdir, "Resolx",  "Resolx", 0);
      TOBcompare(myPV, rdir, sdir, "Wclus", "Wclus", 0);
      TOBcompare(myPV, rdir, sdir, "Adc", "Adc", 0);
      TOBcompare(myPV, rdir, sdir, "Posx", "Pos", 0);
      TOBcompare(myPV, rdir, sdir, "Res", "Res", 0);
      TOBcompare(myPV, rdir, sdir, "Chi2", "Chi2", 0);
      TOBcompare(myPV, rdir, sdir, "NsimHit", "NsimHit", 1, -1.0);

      // matched layers
      TH1F* matchedtob[16];
      TH1F* newmatchedtob[16];

      rdir->GetObject("TOB/layer_1/Posx_matched__TOB__layer__1",matchedtob[0]);
      rdir->GetObject("TOB/layer_1/Posy_matched__TOB__layer__1",matchedtob[1]);
      rdir->GetObject("TOB/layer_2/Posx_matched__TOB__layer__2",matchedtob[2]);
      rdir->GetObject("TOB/layer_2/Posy_matched__TOB__layer__2",matchedtob[3]);
      rdir->GetObject("TOB/layer_1/Resolx_matched__TOB__layer__1",matchedtob[4]);
      rdir->GetObject("TOB/layer_1/Resoly_matched__TOB__layer__1",matchedtob[5]);
      rdir->GetObject("TOB/layer_2/Resolx_matched__TOB__layer__2",matchedtob[6]);
      rdir->GetObject("TOB/layer_2/Resoly_matched__TOB__layer__2",matchedtob[7]);
      rdir->GetObject("TOB/layer_1/Resx_matched__TOB__layer__1",matchedtob[8]);
      rdir->GetObject("TOB/layer_1/Resy_matched__TOB__layer__1",matchedtob[9]);
      rdir->GetObject("TOB/layer_2/Resx_matched__TOB__layer__2",matchedtob[10]);
      rdir->GetObject("TOB/layer_2/Resy_matched__TOB__layer__2",matchedtob[11]);
      rdir->GetObject("TOB/layer_1/Chi2_matched__TOB__layer__1",matchedtob[12]);
      rdir->GetObject("TOB/layer_2/Chi2_matched__TOB__layer__2",matchedtob[13]);
      rdir->GetObject("TOB/layer_1/NsimHit_matched__TOB__layer__1",matchedtob[14]);
      rdir->GetObject("TOB/layer_2/NsimHit_matched__TOB__layer__2",matchedtob[15]);

      sdir->GetObject("TOB/layer_1/Posx_matched__TOB__layer__1",newmatchedtob[0]);
      sdir->GetObject("TOB/layer_1/Posy_matched__TOB__layer__1",newmatchedtob[1]);
      sdir->GetObject("TOB/layer_2/Posx_matched__TOB__layer__2",newmatchedtob[2]);
      sdir->GetObject("TOB/layer_2/Posy_matched__TOB__layer__2",newmatchedtob[3]);
      sdir->GetObject("TOB/layer_1/Resolx_matched__TOB__layer__1",newmatchedtob[4]);
      sdir->GetObject("TOB/layer_1/Resoly_matched__TOB__layer__1",newmatchedtob[5]);
      sdir->GetObject("TOB/layer_2/Resolx_matched__TOB__layer__2",newmatchedtob[6]);
      sdir->GetObject("TOB/layer_2/Resoly_matched__TOB__layer__2",newmatchedtob[7]);
      sdir->GetObject("TOB/layer_1/Resx_matched__TOB__layer__1",newmatchedtob[8]);
      sdir->GetObject("TOB/layer_1/Resy_matched__TOB__layer__1",newmatchedtob[9]);
      sdir->GetObject("TOB/layer_2/Resx_matched__TOB__layer__2",newmatchedtob[10]);
      sdir->GetObject("TOB/layer_2/Resy_matched__TOB__layer__2",newmatchedtob[11]);
      sdir->GetObject("TOB/layer_1/Chi2_matched__TOB__layer__1",newmatchedtob[12]);
      sdir->GetObject("TOB/layer_2/Chi2_matched__TOB__layer__2",newmatchedtob[13]);
      sdir->GetObject("TOB/layer_1/NsimHit_matched__TOB__layer__1",newmatchedtob[14]);
      sdir->GetObject("TOB/layer_2/NsimHit_matched__TOB__layer__2",newmatchedtob[15]);

      TCanvas* Strip_TOB_matched = new TCanvas("Strip_TOB_matched","Strip_TOB_matched",1000,1000);
      Strip_TOB_matched->Divide(4,4);
      for (Int_t i=0; i<16; ++i) {
        if (matchedtob[i]->GetEntries() == 0 || newmatchedtob[i]->GetEntries() == 0) continue;
        Strip_TOB_matched->cd(i+1);
        if (i == 14 || i == 15) SetUpHistograms(matchedtob[i],newmatchedtob[i], -1.0);
        else SetUpHistograms(matchedtob[i],newmatchedtob[i]);
        matchedtob[i]->Draw();
        newmatchedtob[i]->Draw("sames");
        myPV->PVCompute(matchedtob[i] , newmatchedtob[i] , te );
      }
      // TPad *p15 = (TPad *)(Strip_TOB_matched->cd(15));  p15->SetLogy(1);
      // TPad *p16 = (TPad *)(Strip_TOB_matched->cd(16));  p16->SetLogy(1);
 
      sprintf(histoname, "MatchedTOBCompare.%s", outfiletype);  Strip_TOB_matched->Print(histoname);

      //=============================================================== 
      // TID

      // rphi, stereo layers
      TIDcompare(myPV, rdir, sdir, "Pull_LF", "PullLF", 0);
      TIDcompare(myPV, rdir, sdir, "Pull_MF", "PullMF", 0);
      TIDcompare(myPV, rdir, sdir, "Resolx",  "Resolx", 0);
      TIDcompare(myPV, rdir, sdir, "Wclus", "Wclus", 0);
      TIDcompare(myPV, rdir, sdir, "Adc", "Adc", 0);
      TIDcompare(myPV, rdir, sdir, "Posx", "Pos", 0);
      TIDcompare(myPV, rdir, sdir, "Res", "Res", 0);
      TIDcompare(myPV, rdir, sdir, "Chi2", "Chi2", 0);
      TIDcompare(myPV, rdir, sdir, "NsimHit", "NsimHit", 1, -1.0);

      // matched layers
      TH1F* matchedtid[16];
      TH1F* newmatchedtid[16];

      rdir->GetObject("TID/MINUS/ring_1/Posx_matched__TID__MINUS__ring__1",matchedtid[0]);
      rdir->GetObject("TID/MINUS/ring_1/Posy_matched__TID__MINUS__ring__1",matchedtid[1]);
      rdir->GetObject("TID/MINUS/ring_2/Posx_matched__TID__MINUS__ring__2",matchedtid[2]);
      rdir->GetObject("TID/MINUS/ring_2/Posy_matched__TID__MINUS__ring__2",matchedtid[3]);
      rdir->GetObject("TID/MINUS/ring_1/Resolx_matched__TID__MINUS__ring__1",matchedtid[4]);
      rdir->GetObject("TID/MINUS/ring_1/Resoly_matched__TID__MINUS__ring__1",matchedtid[5]);
      rdir->GetObject("TID/MINUS/ring_2/Resolx_matched__TID__MINUS__ring__2",matchedtid[6]);
      rdir->GetObject("TID/MINUS/ring_2/Resoly_matched__TID__MINUS__ring__2",matchedtid[7]);
      rdir->GetObject("TID/MINUS/ring_1/Resx_matched__TID__MINUS__ring__1",matchedtid[8]);
      rdir->GetObject("TID/MINUS/ring_1/Resy_matched__TID__MINUS__ring__1",matchedtid[9]);
      rdir->GetObject("TID/MINUS/ring_2/Resx_matched__TID__MINUS__ring__2",matchedtid[10]);
      rdir->GetObject("TID/MINUS/ring_2/Resy_matched__TID__MINUS__ring__2",matchedtid[11]);
      rdir->GetObject("TID/MINUS/ring_1/Chi2_matched__TID__MINUS__ring__1",matchedtid[12]);
      rdir->GetObject("TID/MINUS/ring_1/Chi2_matched__TID__MINUS__ring__1",matchedtid[13]);
      rdir->GetObject("TID/MINUS/ring_2/NsimHit_matched__TID__MINUS__ring__2",matchedtid[14]);
      rdir->GetObject("TID/MINUS/ring_2/NsimHit_matched__TID__MINUS__ring__2",matchedtid[15]);

      sdir->GetObject("TID/MINUS/ring_1/Posx_matched__TID__MINUS__ring__1",newmatchedtid[0]);
      sdir->GetObject("TID/MINUS/ring_1/Posy_matched__TID__MINUS__ring__1",newmatchedtid[1]);
      sdir->GetObject("TID/MINUS/ring_2/Posx_matched__TID__MINUS__ring__2",newmatchedtid[2]);
      sdir->GetObject("TID/MINUS/ring_2/Posy_matched__TID__MINUS__ring__2",newmatchedtid[3]);
      sdir->GetObject("TID/MINUS/ring_1/Resolx_matched__TID__MINUS__ring__1",newmatchedtid[4]);
      sdir->GetObject("TID/MINUS/ring_1/Resoly_matched__TID__MINUS__ring__1",newmatchedtid[5]);
      sdir->GetObject("TID/MINUS/ring_2/Resolx_matched__TID__MINUS__ring__2",newmatchedtid[6]);
      sdir->GetObject("TID/MINUS/ring_2/Resoly_matched__TID__MINUS__ring__2",newmatchedtid[7]);
      sdir->GetObject("TID/MINUS/ring_1/Resx_matched__TID__MINUS__ring__1",newmatchedtid[8]);
      sdir->GetObject("TID/MINUS/ring_1/Resy_matched__TID__MINUS__ring__1",newmatchedtid[9]);
      sdir->GetObject("TID/MINUS/ring_2/Resx_matched__TID__MINUS__ring__2",newmatchedtid[10]);
      sdir->GetObject("TID/MINUS/ring_2/Resy_matched__TID__MINUS__ring__2",newmatchedtid[11]);
      sdir->GetObject("TID/MINUS/ring_1/Chi2_matched__TID__MINUS__ring__1",newmatchedtid[12]);
      sdir->GetObject("TID/MINUS/ring_2/Chi2_matched__TID__MINUS__ring__2",newmatchedtid[13]);
      sdir->GetObject("TID/MINUS/ring_1/NsimHit_matched__TID__MINUS__ring__1",newmatchedtid[14]);
      sdir->GetObject("TID/MINUS/ring_2/NsimHit_matched__TID__MINUS__ring__2",newmatchedtid[15]);

      TCanvas* Strip_TID_matched = new TCanvas("Strip_TID_matched","Strip_TID_matched",1000,1000);
      Strip_TID_matched->Divide(4,4);
      for (Int_t i=0; i<16; ++i) {
        if (matchedtid[i]->GetEntries() == 0 || newmatchedtid[i]->GetEntries() == 0) continue;
        Strip_TID_matched->cd(i+1);
        if (i == 14 || i == 15) SetUpHistograms(matchedtid[i],newmatchedtid[i], -1.0);
        else SetUpHistograms(matchedtid[i],newmatchedtid[i]);
        matchedtid[i]->Draw();
        newmatchedtid[i]->Draw("sames");
        myPV->PVCompute(matchedtid[i] , newmatchedtid[i] , te );
      }
      // TPad *p15 = (TPad *)(Strip_TID_matched->cd(15));  p15->SetLogy(1);
      // TPad *p16 = (TPad *)(Strip_TID_matched->cd(16));  p16->SetLogy(1);
 
      sprintf(histoname, "MatchedTIDCompare.%s", outfiletype);  Strip_TID_matched->Print(histoname);

      //======================================================================================================
      // TEC

      // rphi, stereo layers
      TECcompare(myPV, rdir, sdir, "Pull_LF", "PullLF", 0);
      TECcompare(myPV, rdir, sdir, "Pull_MF", "PullMF", 0);
      TECcompare(myPV, rdir, sdir, "Resolx",  "Resolx", 0);
      TECcompare(myPV, rdir, sdir, "Wclus", "Wclus", 0);
      TECcompare(myPV, rdir, sdir, "Adc", "Adc", 0);
      TECcompare(myPV, rdir, sdir, "Posx", "Pos", 0);
      TECcompare(myPV, rdir, sdir, "Res", "Res", 0);
      TECcompare(myPV, rdir, sdir, "Chi2", "Chi2", 0);
      TECcompare(myPV, rdir, sdir, "NsimHit", "NsimHit", 1, -1.0);

      // matched layers
      TH1F* matchedtec[24];
      TH1F* newmatchedtec[24];

      rdir->GetObject("TEC/MINUS/ring_1/Posx_matched__TEC__MINUS__ring__1",matchedtec[0]);
      rdir->GetObject("TEC/MINUS/ring_1/Posy_matched__TEC__MINUS__ring__1",matchedtec[1]);
      rdir->GetObject("TEC/MINUS/ring_2/Posx_matched__TEC__MINUS__ring__2",matchedtec[2]);
      rdir->GetObject("TEC/MINUS/ring_2/Posy_matched__TEC__MINUS__ring__2",matchedtec[3]);
      rdir->GetObject("TEC/MINUS/ring_5/Posx_matched__TEC__MINUS__ring__5",matchedtec[4]);
      rdir->GetObject("TEC/MINUS/ring_5/Posy_matched__TEC__MINUS__ring__5",matchedtec[5]);
      rdir->GetObject("TEC/MINUS/ring_1/Resolx_matched__TEC__MINUS__ring__1",matchedtec[6]);
      rdir->GetObject("TEC/MINUS/ring_1/Resoly_matched__TEC__MINUS__ring__1",matchedtec[7]);
      rdir->GetObject("TEC/MINUS/ring_2/Resolx_matched__TEC__MINUS__ring__2",matchedtec[8]);
      rdir->GetObject("TEC/MINUS/ring_2/Resoly_matched__TEC__MINUS__ring__2",matchedtec[9]);
      rdir->GetObject("TEC/MINUS/ring_5/Resolx_matched__TEC__MINUS__ring__5",matchedtec[10]);
      rdir->GetObject("TEC/MINUS/ring_5/Resoly_matched__TEC__MINUS__ring__5",matchedtec[11]);
      rdir->GetObject("TEC/MINUS/ring_1/Resx_matched__TEC__MINUS__ring__1",matchedtec[12]);
      rdir->GetObject("TEC/MINUS/ring_1/Resy_matched__TEC__MINUS__ring__1",matchedtec[13]);
      rdir->GetObject("TEC/MINUS/ring_2/Resx_matched__TEC__MINUS__ring__2",matchedtec[14]);
      rdir->GetObject("TEC/MINUS/ring_2/Resy_matched__TEC__MINUS__ring__2",matchedtec[15]);
      rdir->GetObject("TEC/MINUS/ring_5/Resx_matched__TEC__MINUS__ring__5",matchedtec[16]);
      rdir->GetObject("TEC/MINUS/ring_5/Resy_matched__TEC__MINUS__ring__5",matchedtec[17]);
      rdir->GetObject("TEC/MINUS/ring_1/Chi2_matched__TEC__MINUS__ring__1",matchedtec[18]);
      rdir->GetObject("TEC/MINUS/ring_2/Chi2_matched__TEC__MINUS__ring__2",matchedtec[19]);
      rdir->GetObject("TEC/MINUS/ring_5/Chi2_matched__TEC__MINUS__ring__5",matchedtec[20]);
      rdir->GetObject("TEC/MINUS/ring_1/NsimHit_matched__TEC__MINUS__ring__1",matchedtec[21]);
      rdir->GetObject("TEC/MINUS/ring_2/NsimHit_matched__TEC__MINUS__ring__2",matchedtec[22]);
      rdir->GetObject("TEC/MINUS/ring_5/NsimHit_matched__TEC__MINUS__ring__5",matchedtec[23]);

      sdir->GetObject("TEC/MINUS/ring_1/Posx_matched__TEC__MINUS__ring__1",newmatchedtec[0]);
      sdir->GetObject("TEC/MINUS/ring_1/Posy_matched__TEC__MINUS__ring__1",newmatchedtec[1]);
      sdir->GetObject("TEC/MINUS/ring_2/Posx_matched__TEC__MINUS__ring__2",newmatchedtec[2]);
      sdir->GetObject("TEC/MINUS/ring_2/Posy_matched__TEC__MINUS__ring__2",newmatchedtec[3]);
      sdir->GetObject("TEC/MINUS/ring_5/Posx_matched__TEC__MINUS__ring__5",newmatchedtec[4]);
      sdir->GetObject("TEC/MINUS/ring_5/Posy_matched__TEC__MINUS__ring__5",newmatchedtec[5]);
      sdir->GetObject("TEC/MINUS/ring_1/Resolx_matched__TEC__MINUS__ring__1",newmatchedtec[6]);
      sdir->GetObject("TEC/MINUS/ring_1/Resoly_matched__TEC__MINUS__ring__1",newmatchedtec[7]);
      sdir->GetObject("TEC/MINUS/ring_2/Resolx_matched__TEC__MINUS__ring__2",newmatchedtec[8]);
      sdir->GetObject("TEC/MINUS/ring_2/Resoly_matched__TEC__MINUS__ring__2",newmatchedtec[9]);
      sdir->GetObject("TEC/MINUS/ring_5/Resolx_matched__TEC__MINUS__ring__5",newmatchedtec[10]);
      sdir->GetObject("TEC/MINUS/ring_5/Resoly_matched__TEC__MINUS__ring__5",newmatchedtec[11]);
      sdir->GetObject("TEC/MINUS/ring_1/Resx_matched__TEC__MINUS__ring__1",newmatchedtec[12]);
      sdir->GetObject("TEC/MINUS/ring_1/Resy_matched__TEC__MINUS__ring__1",newmatchedtec[13]);
      sdir->GetObject("TEC/MINUS/ring_2/Resx_matched__TEC__MINUS__ring__2",newmatchedtec[14]);
      sdir->GetObject("TEC/MINUS/ring_2/Resy_matched__TEC__MINUS__ring__2",newmatchedtec[15]);
      sdir->GetObject("TEC/MINUS/ring_5/Resx_matched__TEC__MINUS__ring__5",newmatchedtec[16]);
      sdir->GetObject("TEC/MINUS/ring_5/Resy_matched__TEC__MINUS__ring__5",newmatchedtec[17]);
      sdir->GetObject("TEC/MINUS/ring_1/Chi2_matched__TEC__MINUS__ring__1",newmatchedtec[18]);
      sdir->GetObject("TEC/MINUS/ring_2/Chi2_matched__TEC__MINUS__ring__2",newmatchedtec[19]);
      sdir->GetObject("TEC/MINUS/ring_5/Chi2_matched__TEC__MINUS__ring__5",newmatchedtec[20]);
      sdir->GetObject("TEC/MINUS/ring_1/NsimHit_matched__TEC__MINUS__ring__1",newmatchedtec[21]);
      sdir->GetObject("TEC/MINUS/ring_2/NsimHit_matched__TEC__MINUS__ring__2",newmatchedtec[22]);
      sdir->GetObject("TEC/MINUS/ring_5/NsimHit_matched__TEC__MINUS__ring__5",newmatchedtec[23]);

      TCanvas* Strip_TEC_matched = new TCanvas("Strip_TEC_matched","Strip_TEC_matched",1000,1000);
      Strip_TEC_matched->Divide(5,5);
      for (Int_t i=0; i<24; ++i) {
        if (matchedtec[i]->GetEntries() == 0 || newmatchedtec[i]->GetEntries() == 0) continue;
        Strip_TEC_matched->cd(i+1);
        if (i == 21 || i == 22 || i == 23) SetUpHistograms(matchedtec[i],newmatchedtec[i], -1.0);
        else SetUpHistograms(matchedtec[i],newmatchedtec[i]);
        matchedtec[i]->Draw();
        newmatchedtec[i]->Draw("sames");
        myPV->PVCompute(matchedtec[i] , newmatchedtec[i] , te );
      }
      // TPad *p22 = (TPad *)(Strip_TEC_matched->cd(22));  p22->SetLogy(1);
      // TPad *p23 = (TPad *)(Strip_TEC_matched->cd(23));  p23->SetLogy(1);
      // TPad *p24 = (TPad *)(Strip_TEC_matched->cd(24));  p24->SetLogy(1);
 
      sprintf(histoname, "MatchedTECCompare.%s", outfiletype);  Strip_TEC_matched->Print(histoname);
    }
  }

  //===============================================================
  // Phase 0 pixels 

  if (rfile->cd("DQMData/Run 1/TrackerRecHitsV/Run summary/TrackerRecHits/Pixel")) {
    rdir = gDirectory;
    if (sfile->cd("DQMData/Run 1/TrackerRecHitsV/Run summary/TrackerRecHits/Pixel")) {
      sdir = gDirectory;

      // BPIX
      for (int layer=1; layer<=3; ++layer) {
	BPIXcompare(myPV, rdir, sdir, "Res", layer, "Res");
	BPIXcompare(myPV, rdir, sdir, "Pull", layer, "Pull");
      }

      // FPIX
      for (int disk=1; disk<=2; ++disk) {
	FPIXcompare(myPV, rdir, sdir, "Res", disk, "Res");
	FPIXcompare(myPV, rdir, sdir, "Pull", disk, "Pull");
      }
    }
  }

  //===============================================================
  // Phase 1 pixels 

  if (rfile->cd("DQMData/Run 1/PixelPhase1V/Run summary/RecHits")) {
    rdir = gDirectory;
    if (sfile->cd("DQMData/Run 1/PixelPhase1V/Run summary/RecHits")) {
      sdir = gDirectory;

      // BPIX by layer
      for (int layer=1; layer<=4; ++layer) {
	Phase1PIXcompare(myPV, rdir, sdir, 0, layer);
      }

      // FPIX by disk
      for (int disk=1; disk<=3; ++disk) {
	Phase1PIXcompare(myPV, rdir, sdir, 1, disk);
	Phase1PIXcompare(myPV, rdir, sdir, -1, disk);
      }

      // Combined layers
      TH1F* refplots[6];
      TH1F* newplots[6];
      Int_t plotidx = 0;
      rdir->GetObject("res_x_PXBarrel", refplots[plotidx]);
      sdir->GetObject("res_x_PXBarrel", newplots[plotidx]);
      plotidx++;
      rdir->GetObject("res_y_PXBarrel", refplots[plotidx]);
      sdir->GetObject("res_y_PXBarrel", newplots[plotidx]);
      plotidx++;
      rdir->GetObject("res_x_PXForward", refplots[plotidx]);
      sdir->GetObject("res_x_PXForward", newplots[plotidx]);
      plotidx++;
      rdir->GetObject("res_y_PXForward", refplots[plotidx]);
      sdir->GetObject("res_y_PXForward", newplots[plotidx]);
      plotidx++;
      rdir->GetObject("rechiterror_x", refplots[plotidx]);
      sdir->GetObject("rechiterror_x", newplots[plotidx]);
      plotidx++;
      rdir->GetObject("rechiterror_y", refplots[plotidx]);
      sdir->GetObject("rechiterror_y", newplots[plotidx]);
      plotidx++;
      int nplots = plotidx;

      TCanvas* Phase1Pix = new TCanvas("Phase1Pix", "Phase1Pix", 1000, 1000);
      Phase1Pix->Divide(2,3);
      for (Int_t i=0; i<nplots; ++i) {
	if (refplots[i]->GetEntries() == 0 || newplots[i]->GetEntries() == 0) continue;
	Phase1Pix->cd(i+1);
	SetUpHistograms(refplots[i],newplots[i]);
	refplots[i]->Draw();
	newplots[i]->Draw("sames");
	myPV->PVCompute(refplots[i], newplots[i], te);
      }
      sprintf(histoname, "PX_Compare.%s", outfiletype);
      Phase1Pix->Print(histoname);
    }
  }
}  // ------------------------------------------------------------------------
