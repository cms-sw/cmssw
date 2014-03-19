void SetUpHistograms(TH1F* h1, TH1F* h2, float bincorr=1.0)
{
  float scale1 = -9999.9;
  float scale2 = -9999.9;

  if ( h1->Integral() != 0 && h2->Integral() != 0 ) {
    scale1 = 1.0/(float)h1->Integral();
    // In case the bin width is different between reference and new files:
    scale2 = bincorr/(float)h2->Integral();
    
    h1->Sumw2();
    h2->Sumw2();
    h1->Scale(scale1);
    h2->Scale(scale2);

    h1->SetLineWidth(1);
    h2->SetLineWidth(1);
    h1->SetLineColor(4);
    h1->SetLineStyle(2);  
    h2->SetLineColor(2);
  }
}

void TIBcompare(TDirectory* rdir, TDirectory* sdir, const char* varname, const char* historoot) {

  HistoCompare_Strips* myPV = new HistoCompare_Strips();
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

  Strip = new TCanvas("Strip","Strip",1000,1000);
  Strip->Divide(2,3);
  for (Int_t i=0; i<6; i++) {
    if (refplotsTIB[i]->GetEntries() == 0 || newplotsTIB[i]->GetEntries() == 0) continue;
    Strip->cd(i+1);
    SetUpHistograms(refplotsTIB[i],newplotsTIB[i]);
    refplotsTIB[i]->Draw();
    newplotsTIB[i]->Draw("sames");
    myPV->PVCompute(refplotsTIB[i], newplotsTIB[i], te);
  }

  sprintf(histoname, "%sTIBCompare.pdf", historoot);  Strip->Print(histoname);
  sprintf(histoname, "%sTIBCompare.gif", historoot);  Strip->Print(histoname);

}  // ------------------------------------------------------------------------

void TOBcompare(TDirectory* rdir, TDirectory* sdir, const char* varname, const char* historoot) {

  HistoCompare_Strips* myPV = new HistoCompare_Strips();
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

  Strip = new TCanvas("Strip","Strip",1000,1000);
  Strip->Divide(3,3);
  for (Int_t i=0; i<8; i++) {
    if (refplotsTOB[i]->GetEntries() == 0 || newplotsTOB[i]->GetEntries() == 0) continue;
    Strip->cd(i+1);
    SetUpHistograms(refplotsTOB[i],newplotsTOB[i]);
    refplotsTOB[i]->Draw();
    newplotsTOB[i]->Draw("sames");
    myPV->PVCompute(refplotsTOB[i], newplotsTOB[i], te);
  }

  sprintf(histoname, "%sTOBCompare.pdf", historoot);  Strip->Print(histoname);
  sprintf(histoname, "%sTOBCompare.gif", historoot);  Strip->Print(histoname);

}  // ------------------------------------------------------------------------

void TIDcompare(TDirectory* rdir, TDirectory* sdir, const char* varname, const char* historoot) {

  HistoCompare_Strips* myPV = new HistoCompare_Strips();
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

  Strip = new TCanvas("Strip","Strip",1000,1000);
  Strip->Divide(2,3);
  for (Int_t i=0; i<5; i++) {
    if (refplotsTID[i]->GetEntries() == 0 || newplotsTID[i]->GetEntries() == 0) continue;
    Strip->cd(i+1);
    SetUpHistograms(refplotsTID[i],newplotsTID[i]);
    refplotsTID[i]->Draw();
    newplotsTID[i]->Draw("sames");
    myPV->PVCompute(refplotsTID[i], newplotsTID[i], te);
  }

  sprintf(histoname, "%sTIDCompare.pdf", historoot);  Strip->Print(histoname);
  sprintf(histoname, "%sTIDCompare.gif", historoot);  Strip->Print(histoname);

}  // ------------------------------------------------------------------------

void TECcompare(TDirectory* rdir, TDirectory* sdir, const char* varname, const char* historoot) {

  HistoCompare_Strips* myPV = new HistoCompare_Strips();
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

  Strip = new TCanvas("Strip","Strip",1000,1000);
  Strip->Divide(2,3);
  for (Int_t i=0; i<5; i++) {
    if (refplotsTEC[i]->GetEntries() == 0 || newplotsTEC[i]->GetEntries() == 0) continue;
    Strip->cd(i+1);
    SetUpHistograms(refplotsTEC[i],newplotsTEC[i]);
    refplotsTEC[i]->Draw();
    newplotsTEC[i]->Draw("sames");
    myPV->PVCompute(refplotsTEC[i], newplotsTEC[i], te);
  }

  sprintf(histoname, "%sTECCompare.pdf", historoot);  Strip->Print(histoname);
  sprintf(histoname, "%sTECCompare.gif", historoot);  Strip->Print(histoname);

}  // ------------------------------------------------------------------------

void SiStripTrackingRecHitsCompare()
{
  //color 2 = red  = sfile = new file
  //color 4 = blue = rfile = reference file


 gROOT ->Reset();

 char*  rfilename = "ref/sistriprechitshisto.root";
 char*  sfilename = "sistriprechitshisto.root";

 delete gROOT->GetListOfFiles()->FindObject(rfilename);
 delete gROOT->GetListOfFiles()->FindObject(sfilename); 
 gROOT->ProcessLine(".L HistoCompare_Strips.C");

 TText* te = new TText();
 TFile * rfile = new TFile(rfilename);
 TFile * sfile = new TFile(sfilename);
 TDirectory * rdir=gDirectory; 
 TDirectory * sdir=gDirectory; 
   

 bool rgood=true;
 if(rfile->cd("DQMData/Run 1/RecoTrackV"))rfile->cd("DQMData/Run 1/RecoTrackV/Run summary/TrackingRecHits/Strip");
 else if (rfile->cd("DQMData/TrackerRecHitsV/TrackerRecHits/Strip")) rfile->cd("DQMData/TrackerRecHitsV/TrackerRecHits/Strip");
 else if (rfile->cd("DQMData/SiStrip/RecHitsValidation/StiffTrackingRecHits/MechanicalView")) rfile->cd("DQMData/SiStrip/RecHitsValidation/StiffTrackingRecHits/MechanicalView");
 else {cout << "REFERENCE HISTOS: no RecoTrackV directory found! STOP" << endl; rgood=false;}

 if (rgood) rdir=gDirectory;
 else break;

 bool sgood=true;
 if(sfile->cd("DQMData/Run 1/RecoTrackV"))sfile->cd("DQMData/Run 1/RecoTrackV/Run summary/TrackingRecHits/Strip");
 else if (sfile->cd("DQMData/TrackerRecHitsV/TrackerRecHits/Strip")) sfile->cd("DQMData/TrackerRecHitsV/TrackerRecHits/Strip");
 else if (sfile->cd("DQMData/SiStrip/RecHitsValidation/StiffTrackingRecHits/MechanicalView")) sfile->cd("DQMData/SiStrip/RecHitsValidation/StiffTrackingRecHits/MechanicalView");
 else {cout << "NEW HISTOS: no RecoTrackV directory found! STOP" << endl; sgood=false;}

 if (sgood) sdir=gDirectory; 
 else break;

 Char_t histo[200];

 HistoCompare_Strips * myPV = new HistoCompare_Strips();

 TCanvas *Strip;


 //=============================================================== 
 // TIB

 void TIBcompare(TDirectory* rdir, TDirectory* sdir, const char* varname, const char* historoot);

 TIBcompare(rdir, sdir, "Pull_LF", "PullLF");
 TIBcompare(rdir, sdir, "Pull_MF", "PullMF");
 TIBcompare(rdir, sdir, "Resolx",  "Resolx");
 TIBcompare(rdir, sdir, "Wclus", "Wclus");
 TIBcompare(rdir, sdir, "Adc", "Adc");
 TIBcompare(rdir, sdir, "Posx", "Pos");
 TIBcompare(rdir, sdir, "Res", "Res");
 TIBcompare(rdir, sdir, "Chi2", "Chi2");

 TH1F* matchedtib[14];
 TH1F* newmatchedtib[14];

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

 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(4,4);
 for (Int_t i=0; i<14; i++) {
   if (matchedtib[i]->GetEntries() == 0 || newmatchedtib[i]->GetEntries() == 0) continue;
   Strip->cd(i+1);
   SetUpHistograms(matchedtib[i],newmatchedtib[i]);
   matchedtib[i]->Draw();
   newmatchedtib[i]->Draw("sames");
   myPV->PVCompute(matchedtib[i] , newmatchedtib[i] , te );
 }
 
 Strip->Print("MatchedTIBCompare.pdf");
 Strip->Print("MatchedTIBCompare.gif");

 //======================================================================================================
// TOB

 void TOBcompare(TDirectory* rdir, TDirectory* sdir, const char* varname, const char* historoot);

 TOBcompare(rdir, sdir, "Pull_LF", "PullLF");
 TOBcompare(rdir, sdir, "Pull_MF", "PullMF");
 TOBcompare(rdir, sdir, "Resolx",  "Resolx");
 TOBcompare(rdir, sdir, "Wclus", "Wclus");
 TOBcompare(rdir, sdir, "Adc", "Adc");
 TOBcompare(rdir, sdir, "Posx", "Pos");
 TOBcompare(rdir, sdir, "Res", "Res");
 TOBcompare(rdir, sdir, "Chi2", "Chi2");

 TH1F* matchedtob[14];
 TH1F* newmatchedtob[14];

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

 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(4,4);
 for (Int_t i=0; i<14; i++) {
   if (matchedtob[i]->GetEntries() == 0 || newmatchedtob[i]->GetEntries() == 0) continue;
   Strip->cd(i+1);
   SetUpHistograms(matchedtob[i],newmatchedtob[i]);
   matchedtob[i]->Draw();
   newmatchedtob[i]->Draw("sames");
   myPV->PVCompute(matchedtob[i] , newmatchedtob[i] , te );
 }
 
 Strip->Print("MatchedTOBCompare.pdf");
 Strip->Print("MatchedTOBCompare.gif");

 //=============================================================== 
// TID

 void TIDcompare(TDirectory* rdir, TDirectory* sdir, const char* varname, const char* historoot);

 TIDcompare(rdir, sdir, "Pull_LF", "PullLF");
 TIDcompare(rdir, sdir, "Pull_MF", "PullMF");
 TIDcompare(rdir, sdir, "Resolx",  "Resolx");
 TIDcompare(rdir, sdir, "Wclus", "Wclus");
 TIDcompare(rdir, sdir, "Adc", "Adc");
 TIDcompare(rdir, sdir, "Posx", "Pos");
 TIDcompare(rdir, sdir, "Res", "Res");
 TIDcompare(rdir, sdir, "Chi2", "Chi2");

 TH1F* matchedtid[14];
 TH1F* newmatchedtid[14];

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
 rdir->GetObject("TID/MINUS/ring_2/Chi2_matched__TID__MINUS__ring__2",matchedtid[13]);

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

 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(4,4);
 for (Int_t i=0; i<14; i++) {
   if (matchedtid[i]->GetEntries() == 0 || newmatchedtid[i]->GetEntries() == 0) continue;
   Strip->cd(i+1);
   SetUpHistograms(matchedtid[i],newmatchedtid[i]);
   matchedtid[i]->Draw();
   newmatchedtid[i]->Draw("sames");
   myPV->PVCompute(matchedtid[i] , newmatchedtid[i] , te );
 }
 
 Strip->Print("MatchedTIDCompare.pdf");
 Strip->Print("MatchedTIDCompare.gif");

 //======================================================================================================
// TEC

 void TECcompare(TDirectory* rdir, TDirectory* sdir, const char* varname, const char* historoot);

 TECcompare(rdir, sdir, "Pull_LF", "PullLF");
 TECcompare(rdir, sdir, "Pull_MF", "PullMF");
 TECcompare(rdir, sdir, "Resolx",  "Resolx");
 TECcompare(rdir, sdir, "Wclus", "Wclus");
 TECcompare(rdir, sdir, "Adc", "Adc");
 TECcompare(rdir, sdir, "Posx", "Pos");
 TECcompare(rdir, sdir, "Res", "Res");
 TECcompare(rdir, sdir, "Chi2", "Chi2");

 TH1F* matchedtec[21];
 TH1F* newmatchedtec[21];

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

 Strip = new TCanvas("Strip","Strip",1000,1000);
 Strip->Divide(5,5);
 for (Int_t i=0; i<21; i++) {
   if (matchedtec[i]->GetEntries() == 0 || newmatchedtec[i]->GetEntries() == 0) continue;
   Strip->cd(i+1);
   SetUpHistograms(matchedtec[i],newmatchedtec[i]);
   matchedtec[i]->Draw();
   newmatchedtec[i]->Draw("sames");
   myPV->PVCompute(matchedtec[i] , newmatchedtec[i] , te );
 }
 
 Strip->Print("MatchedTECCompare.pdf");
 Strip->Print("MatchedTECCompare.gif");

}

