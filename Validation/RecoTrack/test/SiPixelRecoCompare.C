void SetUpHistograms(TH1F* h1, TH1F* h2, const char* xtitle, TLegend* leg = 0)
{
  float scale1 = -9999.9;
  float scale2 = -9999.9;

  if ( h1->Integral() != 0 && h2->Integral() != 0 )
    {
      scale1 = 1.0/(float)h1->Integral();
      scale2 = 1.0/(float)h2->Integral();
      
      h1->Sumw2();
      h2->Sumw2();
      h1->Scale(scale1);
      h2->Scale(scale2);
  
      h1->SetLineWidth(2);
      h2->SetLineWidth(2);
      h1->SetLineColor(2);
      h2->SetLineColor(4);
      h2->SetLineStyle(2);  
    }

  h1->SetXTitle(xtitle);

  if ( leg != 0 )
    {
      leg->SetBorderSize(0);
      leg->AddEntry(h1, "reference  ", "l");
      leg->AddEntry(h2, "new release", "l");
    }
}

void SetUpProfileHistograms( TProfile* h1, TProfile* h2, 
			     const char* xtitle, const char* ytitle, 
			     double ymin, double ymax, 
			     TLegend* leg = 0 )
{
  h1->SetLineWidth(2);
  h2->SetLineWidth(2);
  h1->SetLineColor(2);
  h2->SetLineColor(4);
  h2->SetLineStyle(2);  

  h1->SetXTitle(xtitle);
  h1->SetYTitle(ytitle);
  h1->SetTitleOffset(2.5, "Y");

  h1->SetMinimum(ymin);
  h1->SetMaximum(ymax);

  if ( leg != 0 )
    {
      leg->SetBorderSize(0);
      leg->AddEntry(h1, "reference  ", "l");
      leg->AddEntry(h2, "new release", "l");
    }
}

void SiPixelRecoCompare(char* originalName="DQM_V0001_R000000001__CMSSW_3_1_5__RelVal__Validation.root")
{
  gROOT ->Reset();
    
  char*  sfilename = "pixeltrackingrechitshist.root";
  char*  rfilename = "../pixeltrackingrechitshist.root";
  /// WARNING: in other validation macros, rfilename is "new" and sfilename is "reference", while here is inverted.

  delete gROOT->GetListOfFiles()->FindObject(rfilename);
  delete gROOT->GetListOfFiles()->FindObject(sfilename);
  
  TText* te = new TText();
  TFile* rfile = new TFile(rfilename);
  TDirectory * rdir=gDirectory; 
  TFile * sfile = new TFile(sfilename);
  TDirectory * sdir=gDirectory; 
  
  char path[500];
  sprintf(path,"DQMData/Run 1/%s/DQMData/Run 1/RecoTrackV/Run summary/TrackingRecHits/Pixel",originalName);
  cout << "path = " << path << endl;

  if(rfile->cd("DQMData/Run 1/RecoTrackV"))rfile->cd("DQMData/Run 1/RecoTrackV/Run summary/TrackingRecHits/Pixel");
  else if(rfile->cd("DQMData/Run 1/Tracking/Run summary/TrackingRecHits"))rfile->cd("DQMData/Run 1/Tracking/Run summary/TrackingRecHits/Pixel");
  else if (rfile->cd("DQMData/RecoTrackV")) rfile->cd("DQMData/RecoTrackV/TrackingRecHits/Pixel");
  else if (rfile->cd(path)) rfile->cd(path);
  rdir=gDirectory;

  if(sfile->cd("DQMData/Run 1/RecoTrackV"))sfile->cd("DQMData/Run 1/RecoTrackV/Run summary/TrackingRecHits/Pixel");
  else if(sfile->cd("DQMData/Run 1/Tracking/Run summary/TrackingRecHits"))sfile->cd("DQMData/Run 1/Tracking/Run summary/TrackingRecHits/Pixel");
  else if (sfile->cd("DQMData/RecoTrackV")) sfile->cd("DQMData/RecoTrackV/TrackingRecHits/Pixel");
  else if (sfile->cd(path)) sfile->cd(path);
  sdir=gDirectory; 

  Char_t histo[200];
    
  gROOT->ProcessLine(".x HistoCompare_Pixels.C");
  HistoCompare_Pixels* myPV = new HistoCompare_Pixels("RecoTrack_SiPixelRecoCompare.txt");
  //myPV->setName("RecoTrack_SiPixelRecoCompare");

  int n_bins = 194;
  double  low = 0.5;
  double high = (double)n_bins + 0.5;
  TH1F* h_pv = new TH1F("h_pv", "#Chi^{2} results for each distribution", n_bins, low, high);
  int bin = 0;
  double value_pv = -9999.9;


  if (1) 
    {
      TCanvas* can_meControl = new TCanvas("can_meControl", "can_meControl", 1000, 500);
      can_meControl->Divide(2,1);
      
  
      TH1F* meTracksPerEvent;
      TH1F* mePixRecHitsPerTrack;
      
      TH1F* newmeTracksPerEvent;
      TH1F* newmePixRecHitsPerTrack;
      
      rdir->GetObject("Histograms_all/meTracksPerEvent", meTracksPerEvent);
      rdir->GetObject("Histograms_all/mePixRecHitsPerTrack", mePixRecHitsPerTrack );
      
      sdir->GetObject("Histograms_all/meTracksPerEvent", newmeTracksPerEvent );
      sdir->GetObject("Histograms_all/mePixRecHitsPerTrack", newmePixRecHitsPerTrack );
      
      
      TLegend* leg1 = new TLegend(0.6, 0.5, .89, 0.7); 
      can_meControl->cd(1);
      //gPad->SetLogy();
      SetUpHistograms(meTracksPerEvent, newmeTracksPerEvent, "tracks per event", leg1 );
      Float_t refMax = 1.15*meTracksPerEvent->GetMaximum();
      Float_t newMax = 1.15*newmeTracksPerEvent->GetMaximum();
      if refMax > newMax
      {    
          meTracksPerEvent->SetMaximum(refMax);
      }
      else
      {    
          meTracksPerEvent->SetMaximum(newMax);
      }
      meTracksPerEvent->SetName("Reference");
      newmeTracksPerEvent->SetName("New Release");
      meTracksPerEvent->Draw("he");      
      newmeTracksPerEvent->Draw("hesameS"); 
      gPad->Update();      
      TPaveStats *s1 = (TPaveStats*)meTracksPerEvent->GetListOfFunctions()->FindObject("stats");
      if (s1) {
	s1->SetX1NDC (0.55); //new x start position
	s1->SetX2NDC (0.75); //new x end position   
      }
      myPV->PVCompute(meTracksPerEvent, newmeTracksPerEvent, te);
      h_pv->SetBinContent(++bin, myPV->getPV());
      leg1->Draw();   
      
      
		
      can_meControl->cd(2);
      //gPad->SetLogy();
      SetUpHistograms(mePixRecHitsPerTrack, newmePixRecHitsPerTrack, "pixel hits per track" );     
      Float_t refMax = 1.15*mePixRecHitsPerTrack->GetMaximum();
      Float_t newMax = 1.15*newmePixRecHitsPerTrack->GetMaximum();
      if refMax > newMax
      {
          mePixRecHitsPerTrack->SetMaximum(refMax);
      }
      else
      {
          mePixRecHitsPerTrack->SetMaximum(newMax);
      }
      mePixRecHitsPerTrack->SetName("Reference");
      newmePixRecHitsPerTrack->SetName("New Release");
      mePixRecHitsPerTrack->Draw("he");
      newmePixRecHitsPerTrack->Draw("hesameS");
      gPad->Update();
      TPaveStats *s2 = (TPaveStats*)mePixRecHitsPerTrack->GetListOfFunctions()->FindObject("stats");
      if (s2) {
	s2->SetX1NDC (0.55); //new x start position
	s2->SetX2NDC (0.75); //new x end position
      }
      myPV->PVCompute(mePixRecHitsPerTrack, newmePixRecHitsPerTrack, te, 0.15, 0.8 );
      h_pv->SetBinContent(++bin, myPV->getPV());

      can_meControl->SaveAs("meControl_compare.eps");
      can_meControl->SaveAs("meControl_compare.gif");

    }


  if (1) 
    {
      TCanvas* can_meCharge = new TCanvas("can_meCharge", "can_meCharge", 1200, 800);
      can_meCharge->Divide(3,2);
      
      TH1F* meChargeBarrel;
      TH1F* meChargeZmPanel1;
      TH1F* meChargeZmPanel2;
      TH1F* meChargeZpPanel1;
      TH1F* meChargeZpPanel2;
      
      TH1F* newmeChargeBarrel;
      TH1F* newmeChargeZmPanel1;
      TH1F* newmeChargeZmPanel2;
      TH1F* newmeChargeZpPanel1;
      TH1F* newmeChargeZpPanel2;
      
      rdir->GetObject("Histograms_all/meChargeBarrel"  , meChargeBarrel  );
      rdir->GetObject("Histograms_all/meChargeZmPanel1", meChargeZmPanel1);
      rdir->GetObject("Histograms_all/meChargeZmPanel2", meChargeZmPanel2);
      rdir->GetObject("Histograms_all/meChargeZpPanel1", meChargeZpPanel1);
      rdir->GetObject("Histograms_all/meChargeZpPanel2", meChargeZpPanel2);
      
      sdir->GetObject("Histograms_all/meChargeBarrel"  , newmeChargeBarrel  ); 
      sdir->GetObject("Histograms_all/meChargeZmPanel1", newmeChargeZmPanel1);
      sdir->GetObject("Histograms_all/meChargeZmPanel2", newmeChargeZmPanel2);
      sdir->GetObject("Histograms_all/meChargeZpPanel1", newmeChargeZpPanel1);
      sdir->GetObject("Histograms_all/meChargeZpPanel2", newmeChargeZpPanel2);
      TLegend* leg2 = new TLegend(0.65, 0.45, 0.89, 0.6);
      can_meCharge->cd(1);
      //gPad->SetLogy();
      SetUpHistograms(meChargeBarrel, newmeChargeBarrel, "barrel, cluster charge (elec) ", leg2 );
      
      Float_t refMax = 1.2*meChargeBarrel->GetMaximum();
      Float_t newMax = 1.2*newmeChargeBarrel->GetMaximum();
      if refMax > newMax
      {
          meChargeBarrel->SetMaximum(refMax);
      }
      else
      {
          meChargeBarrel->SetMaximum(newMax);
      }
      meChargeBarrel->SetName("Reference");
      newmeChargeBarrel->SetName("New Release");
      meChargeBarrel->Draw("he");
      newmeChargeBarrel->Draw("hesameS"); 
      myPV->PVCompute(meChargeBarrel, newmeChargeBarrel, te );
      h_pv->SetBinContent(++bin, myPV->getPV());
      leg2->Draw();
      gPad->Update();      
      TPaveStats *s3 = (TPaveStats*)meChargeBarrel->GetListOfFunctions()->FindObject("stats");
      if (s3) {
	s3->SetX1NDC (0.55); //new x start position
	s3->SetX2NDC (0.75); //new x end position  
      }
      
      can_meCharge->cd(2);
      //gPad->SetLogy();
      SetUpHistograms(meChargeZmPanel1, newmeChargeZmPanel1, "panel1, z<0, cluster charge (elec)" );
      Float_t refMax = 1.2*meChargeZmPanel1->GetMaximum();
      Float_t newMax = 1.2*newmeChargeZmPanel1->GetMaximum();
      
      if refMax > newMax
      {
          meChargeZmPanel1->SetMaximum(refMax);
      }
      else
      {
          meChargeZmPanel1->SetMaximum(newMax);
      }
      meChargeZmPanel1->SetName("Reference");
      newmeChargeZmPanel1->SetName("New Release");
      meChargeZmPanel1->Draw("he");
      newmeChargeZmPanel1->Draw("hesameS"); 
      myPV->PVCompute(meChargeZmPanel1, newmeChargeZmPanel1, te );
      h_pv->SetBinContent(++bin, myPV->getPV());
      gPad->Update();      
      TPaveStats *s4 = (TPaveStats*)meChargeZmPanel1->GetListOfFunctions()->FindObject("stats");
      if (s4) {
	s4->SetX1NDC (0.55); //new x start position
	s4->SetX2NDC (0.75); //new x end position 
      }

      can_meCharge->cd(3);
      //gPad->SetLogy();
      SetUpHistograms(meChargeZmPanel2, newmeChargeZmPanel2, "panel2, z<0, cluster charge (elec)" );
      Float_t refMax = 1.2*meChargeZmPanel2->GetMaximum();
      Float_t newMax = 1.2*newmeChargeZmPanel2->GetMaximum();
      if refMax > newMax
      {
          meChargeZmPanel2->SetMaximum(refMax);
      }
      else
      {
          meChargeZmPanel2->SetMaximum(newMax);
      }
      meChargeZmPanel2->SetName("Reference");
      newmeChargeZmPanel2->SetName("New Release");
      meChargeZmPanel2->Draw("he");
      newmeChargeZmPanel2->Draw("hesameS"); 
      myPV->PVCompute(meChargeZmPanel2, newmeChargeZmPanel2, te );
      h_pv->SetBinContent(++bin, myPV->getPV());
      gPad->Update();      
      TPaveStats *s5 = (TPaveStats*)meChargeZmPanel2->GetListOfFunctions()->FindObject("stats");
      if (s5) {
	s5->SetX1NDC (0.55); //new x start position
	s5->SetX2NDC (0.75); //new x end position 
      }

      can_meCharge->cd(5);
      //gPad->SetLogy();
      SetUpHistograms(meChargeZpPanel1, newmeChargeZpPanel1, "panel1, z>0, cluster charge (elec)" );
      Float_t refMax = 1.2*meChargeZpPanel1->GetMaximum();
      Float_t newMax = 1.2*newmeChargeZpPanel1->GetMaximum();
      if refMax > newMax
      {
          meChargeZpPanel1->SetMaximum(refMax);
      }
      else
      {
          meChargeZpPanel1->SetMaximum(newMax);
      }
      meChargeZpPanel1->SetName("Reference");
      newmeChargeZpPanel1->SetName("New Release");
      meChargeZpPanel1->Draw("he");
      newmeChargeZpPanel1->Draw("hesameS"); 
      myPV->PVCompute(meChargeZpPanel1, newmeChargeZpPanel1, te );
      h_pv->SetBinContent(++bin, myPV->getPV());
      gPad->Update();      
      TPaveStats *s6 = (TPaveStats*)meChargeZpPanel1->GetListOfFunctions()->FindObject("stats");
      if (s6) {
	s6->SetX1NDC (0.55); //new x start position
	s6->SetX2NDC (0.75); //new x end position 
      }

      can_meCharge->cd(6);
      //gPad->SetLogy();
      SetUpHistograms(meChargeZpPanel2, newmeChargeZpPanel2, "panel2, z>0, cluster charge (elec)" );  
      Float_t refMax = 1.2*meChargeZpPanel2->GetMaximum();
      Float_t newMax = 1.2*newmeChargeZpPanel2->GetMaximum();
      if refMax > newMax
      {
          meChargeZpPanel2->SetMaximum(refMax);
      }
      else
      {
          meChargeZpPanel2->SetMaximum(newMax);
      }
      meChargeZpPanel2->SetName("Reference");
      newmeChargeZpPanel2->SetName("New Release");
      meChargeZpPanel2->Draw("he");
      newmeChargeZpPanel2->Draw("hesameS"); 
      myPV->PVCompute(meChargeZpPanel2, newmeChargeZpPanel2, te );
      h_pv->SetBinContent(++bin, myPV->getPV());
      gPad->Update();      
      TPaveStats *s7 = (TPaveStats*)meChargeZpPanel2->GetListOfFunctions()->FindObject("stats");
      if (s7) {
	s7->SetX1NDC (0.55); //new x start position
	s7->SetX2NDC (0.75); //new x end position 
      }
      
      can_meCharge->SaveAs("meCharge_compare.eps");
      can_meCharge->SaveAs("meCharge_compare.gif");
    }
  
  if (1) 
    {
      TCanvas* can_Errx = new TCanvas("can_Errx", "can_Errx", 1200, 800);
      can_Errx->Divide(3,2);
      
      TH1F* meErrxBarrel;
      TH1F* meErrxZmPanel1;
      TH1F* meErrxZmPanel2;
      TH1F* meErrxZpPanel1;
      TH1F* meErrxZpPanel2;
      
      TH1F* newmeErrxBarrel;
      TH1F* newmeErrxZmPanel1;
      TH1F* newmeErrxZmPanel2;
      TH1F* newmeErrxZpPanel1;
      TH1F* newmeErrxZpPanel2;
      
      rdir->GetObject("Histograms_all/meErrxBarrel"  , meErrxBarrel  );
      rdir->GetObject("Histograms_all/meErrxZmPanel1", meErrxZmPanel1);
      rdir->GetObject("Histograms_all/meErrxZmPanel2", meErrxZmPanel2);
      rdir->GetObject("Histograms_all/meErrxZpPanel1", meErrxZpPanel1);
      rdir->GetObject("Histograms_all/meErrxZpPanel2", meErrxZpPanel2);
      
      sdir->GetObject("Histograms_all/meErrxBarrel"  , newmeErrxBarrel  ); 
      sdir->GetObject("Histograms_all/meErrxZmPanel1", newmeErrxZmPanel1);
      sdir->GetObject("Histograms_all/meErrxZmPanel2", newmeErrxZmPanel2);
      sdir->GetObject("Histograms_all/meErrxZpPanel1", newmeErrxZpPanel1);
      sdir->GetObject("Histograms_all/meErrxZpPanel2", newmeErrxZpPanel2);
      
      TLegend* leg3 = new TLegend(0.65, 0.55, 0.89, 0.7);
      can_Errx->cd(1);
      //gPad->SetLogy();
      SetUpHistograms(meErrxBarrel, newmeErrxBarrel, "barrel, x position error (cm)", leg3 );
      Float_t refMax = 1.2*meErrxBarrel->GetMaximum();
      Float_t newMax = 1.2*newmeErrxBarrel->GetMaximum();
      if refMax > newMax
      {
          meErrxBarrel->SetMaximum(refMax);
      }
      else
      {
          meErrxBarrel->SetMaximum(newMax);
      }
      meErrxBarrel->SetName("Reference");
      newmeErrxBarrel->SetName("New Release");
      meErrxBarrel->Draw("he");
      newmeErrxBarrel->Draw("hesameS"); 
      myPV->PVCompute(meErrxBarrel, newmeErrxBarrel, te );
      leg3->Draw();
      h_pv->SetBinContent(++bin, myPV->getPV());
      gPad->Update();      
      TPaveStats *s8 = (TPaveStats*)meErrxBarrel->GetListOfFunctions()->FindObject("stats");
      if (s8) {
	s8->SetX1NDC (0.55); //new x start position
	s8->SetX2NDC (0.75); //new x end position 
      }

      can_Errx->cd(2);
      //gPad->SetLogy();
      SetUpHistograms(meErrxZmPanel1, newmeErrxZmPanel1, "panel1, z<0, x position error (cm)" );
      Float_t refMax = 1.2*meErrxZmPanel1->GetMaximum();
      Float_t newMax = 1.2*newmeErrxZmPanel1->GetMaximum();
      if refMax > newMax
      {
          meErrxZmPanel1->SetMaximum(refMax);
      }
      else
      {
          meErrxZmPanel1->SetMaximum(newMax);
      }
      meErrxZmPanel1->SetName("Reference");
      newmeErrxZmPanel1->SetName("New Release");
      meErrxZmPanel1->Draw("he");
      newmeErrxZmPanel1->Draw("hesameS"); 
      myPV->PVCompute(meErrxZmPanel1, newmeErrxZmPanel1, te );
      h_pv->SetBinContent(++bin, myPV->getPV());
      gPad->Update();      
      TPaveStats *s9 = (TPaveStats*)meErrxZmPanel1->GetListOfFunctions()->FindObject("stats");
      if (s9) {
	s9->SetX1NDC (0.55); //new x start position
	s9->SetX2NDC (0.75); //new x end position 
      }

      can_Errx->cd(3);
      //gPad->SetLogy();
      SetUpHistograms(meErrxZmPanel2, newmeErrxZmPanel2, "panel2, z<0, x position error (cm)" );
      Float_t refMax = 1.2*meErrxZmPanel2->GetMaximum();
      Float_t newMax = 1.2*newmeErrxZmPanel2->GetMaximum();
      if refMax > newMax
      {
          meErrxZmPanel2->SetMaximum(refMax);
      }
      else
      {
          meErrxZmPanel2->SetMaximum(newMax);
      }
      meErrxZmPanel2->SetName("Reference");
      newmeErrxZmPanel2->SetName("New Release");
      meErrxZmPanel2->Draw("he");
      newmeErrxZmPanel2->Draw("hesameS"); 
      myPV->PVCompute(meErrxZmPanel2, newmeErrxZmPanel2, te );
      h_pv->SetBinContent(++bin, myPV->getPV());
      gPad->Update();      
      TPaveStats *s10 = (TPaveStats*)meErrxZmPanel2->GetListOfFunctions()->FindObject("stats");
      if (s10) {
	s10->SetX1NDC (0.55); //new x start position
	s10->SetX2NDC (0.75); //new x end position 
      }

      can_Errx->cd(5);
      //gPad->SetLogy();
      SetUpHistograms(meErrxZpPanel1, newmeErrxZpPanel1, "panel1, z>0, x position error (cm)" );
      Float_t refMax = 1.2*meErrxZpPanel1->GetMaximum();
      Float_t newMax = 1.2*newmeErrxZpPanel1->GetMaximum();
      if refMax > newMax
      {
          meErrxZpPanel1->SetMaximum(refMax);
      }
      else
      {
          meErrxZpPanel1->SetMaximum(newMax);
      }
      meErrxZpPanel1->SetName("Reference");
      newmeErrxZpPanel1->SetName("New Release");
      meErrxZpPanel1->Draw("he");
      newmeErrxZpPanel1->Draw("hesameS"); 
      myPV->PVCompute(meErrxZpPanel1, newmeErrxZpPanel1, te );
      h_pv->SetBinContent(++bin, myPV->getPV());
      gPad->Update();      
      TPaveStats *s11 = (TPaveStats*)meErrxZpPanel1->GetListOfFunctions()->FindObject("stats");
      if (s11) {
	s11->SetX1NDC (0.55); //new x start position
	s11->SetX2NDC (0.75); //new x end position 
      }

      can_Errx->cd(6);
      //gPad->SetLogy();
      SetUpHistograms(meErrxZpPanel2, newmeErrxZpPanel2, "panel2, z>0, x position error (cm)" );
      Float_t refMax = 1.2*meErrxZpPanel2->GetMaximum();
      Float_t newMax = 1.2*newmeErrxZpPanel2->GetMaximum();
      if refMax > newMax
      {
          meErrxZpPanel2->SetMaximum(refMax);
      }
      else
      {
          meErrxZpPanel2->SetMaximum(newMax);
      }
      meErrxZpPanel2->SetName("Reference");
      newmeErrxZpPanel2->SetName("New Release");
      meErrxZpPanel2->Draw("he");
      newmeErrxZpPanel2->Draw("hesameS"); 
      myPV->PVCompute(meErrxZpPanel2, newmeErrxZpPanel2, te );
      h_pv->SetBinContent(++bin, myPV->getPV());
      gPad->Update();      
      TPaveStats *s12 = (TPaveStats*)meErrxZpPanel2->GetListOfFunctions()->FindObject("stats");
      if (s12) {
	s12->SetX1NDC (0.55); //new x start position
	s12->SetX2NDC (0.75); //new x end position 
      }

      can_Errx->SaveAs("meErrx_compare.eps");
      can_Errx->SaveAs("meErrx_compare.gif");
    }
    
  if (1) 
    {
      TCanvas* can_Erry = new TCanvas("can_Erry", "can_Erry", 1200, 800);
      can_Erry->Divide(3,2);
      
      TH1F* meErryBarrel;
      TH1F* meErryZmPanel1;
      TH1F* meErryZmPanel2;
      TH1F* meErryZpPanel1;
      TH1F* meErryZpPanel2;
      
      TH1F* newmeErryBarrel;
      TH1F* newmeErryZmPanel1;
      TH1F* newmeErryZmPanel2;
      TH1F* newmeErryZpPanel1;
      TH1F* newmeErryZpPanel2;
      
      rdir->GetObject("Histograms_all/meErryBarrel"  , meErryBarrel  );
      rdir->GetObject("Histograms_all/meErryZmPanel1", meErryZmPanel1);
      rdir->GetObject("Histograms_all/meErryZmPanel2", meErryZmPanel2);
      rdir->GetObject("Histograms_all/meErryZpPanel1", meErryZpPanel1);
      rdir->GetObject("Histograms_all/meErryZpPanel2", meErryZpPanel2);
      
      sdir->GetObject("Histograms_all/meErryBarrel"  , newmeErryBarrel  ); 
      sdir->GetObject("Histograms_all/meErryZmPanel1", newmeErryZmPanel1);
      sdir->GetObject("Histograms_all/meErryZmPanel2", newmeErryZmPanel2);
      sdir->GetObject("Histograms_all/meErryZpPanel1", newmeErryZpPanel1);
      sdir->GetObject("Histograms_all/meErryZpPanel2", newmeErryZpPanel2);
      
      TLegend* leg4 = new TLegend(0.65, 0.5, 0.89, 0.65);
      can_Erry->cd(1);
      //gPad->SetLogy();
      SetUpHistograms(meErryBarrel, newmeErryBarrel, "barrel, y position error (cm)", leg4 );
      Float_t refMax = 1.2*meErryBarrel->GetMaximum();
      Float_t newMax = 1.2*newmeErryBarrel->GetMaximum();
      if refMax > newMax
      {
          meErryBarrel->SetMaximum(refMax);
      }
      else
      {
          meErryBarrel->SetMaximum(newMax);
      }
      meErryBarrel->SetName("Reference");
      newmeErryBarrel->SetName("New Release");
      meErryBarrel->Draw("he");
      newmeErryBarrel->Draw("hesameS"); 
      myPV->PVCompute(meErryBarrel, newmeErryBarrel, te );
      leg4->Draw();
      h_pv->SetBinContent(++bin, myPV->getPV());
      gPad->Update();      
      TPaveStats *s13 = (TPaveStats*)meErryBarrel->GetListOfFunctions()->FindObject("stats");
      if (s13) {
	s13->SetX1NDC (0.55); //new x start position
	s13->SetX2NDC (0.75); //new x end position 
      }

      can_Erry->cd(2);
      //gPad->SetLogy();
      SetUpHistograms(meErryZmPanel1, newmeErryZmPanel1, "panel1, z<0, y position error (cm)"  );
      Float_t refMax = 1.2*meErryZmPanel1->GetMaximum();
      Float_t newMax = 1.2*newmeErryZmPanel1->GetMaximum();
      if refMax > newMax
      {
          meErryZmPanel1->SetMaximum(refMax);
      }
      else
      {
          meErryZmPanel1->SetMaximum(newMax);
      }
      meErryZmPanel1->SetName("Reference");
      newmeErryZmPanel1->SetName("New Release");
      meErryZmPanel1->Draw("he");
      newmeErryZmPanel1->Draw("hesameS"); 
      myPV->PVCompute(meErryZmPanel1, newmeErryZmPanel1, te );
      h_pv->SetBinContent(++bin, myPV->getPV());
      gPad->Update();      
      TPaveStats *s14 = (TPaveStats*)meErryZmPanel1->GetListOfFunctions()->FindObject("stats");
      if (s14) {
	s14->SetX1NDC (0.55); //new x start position
	s14->SetX2NDC (0.75); //new x end position 
      }

      can_Erry->cd(3);
      //gPad->SetLogy();
      SetUpHistograms(meErryZmPanel2, newmeErryZmPanel2, "panel2, z<0, y position error (cm)" );
      Float_t refMax = 1.2*meErryZmPanel2->GetMaximum();
      Float_t newMax = 1.2*newmeErryZmPanel2->GetMaximum();      
      if refMax > newMax
      {
          meErryZmPanel2->SetMaximum(refMax);
      }
      else
      {
          meErryZmPanel2->SetMaximum(newMax);
      }      
      meErryZmPanel2->SetName("Reference");
      newmeErryZmPanel2->SetName("New Release");
      meErryZmPanel2->Draw("he");
      newmeErryZmPanel2->Draw("hesameS"); 
      myPV->PVCompute(meErryZmPanel2, newmeErryZmPanel2, te );
      h_pv->SetBinContent(++bin, myPV->getPV());      
      gPad->Update();      
      TPaveStats *s15 = (TPaveStats*)meErryZmPanel2->GetListOfFunctions()->FindObject("stats");
      if (s15) {
	s15->SetX1NDC (0.55); //new x start position
	s15->SetX2NDC (0.75); //new x end position
      }

      can_Erry->cd(5);
      //gPad->SetLogy();
      SetUpHistograms(meErryZpPanel1, newmeErryZpPanel1, "panel1, z>0, y position error (cm)" );
      Float_t refMax = 1.2*meErryZpPanel1->GetMaximum();
      Float_t newMax = 1.2*newmeErryZpPanel1->GetMaximum();      
      if refMax > newMax
      {
          meErryZpPanel1->SetMaximum(refMax);
      }
      else
      {
          meErryZpPanel1->SetMaximum(newMax);
      }      
      meErryZpPanel1->SetName("Reference");
      newmeErryZpPanel1->SetName("New Release");
      meErryZpPanel1->Draw("he");
      newmeErryZpPanel1->Draw("hesameS"); 
      myPV->PVCompute(meErryZpPanel1, newmeErryZpPanel1, te );
      h_pv->SetBinContent(++bin, myPV->getPV());
      gPad->Update();      
      TPaveStats *s16 = (TPaveStats*)meErryZpPanel1->GetListOfFunctions()->FindObject("stats");
      if (s16) {
	s16->SetX1NDC (0.55); //new x start position
	s16->SetX2NDC (0.75); //new x end position
      }

      can_Erry->cd(6);
      //gPad->SetLogy();
      SetUpHistograms(meErryZpPanel2, newmeErryZpPanel2, "panel2, z>0, y position error (cm)" );
      Float_t refMax = 1.2*meErryZpPanel2->GetMaximum();
      Float_t newMax = 1.2*newmeErryZpPanel2->GetMaximum();
      if refMax > newMax
      {
          meErryZpPanel2->SetMaximum(refMax);
      }
      else
      {
          meErryZpPanel2->SetMaximum(newMax);
      }
      meErryZpPanel2->SetName("Reference");
      newmeErryZpPanel2->SetName("New Release");
      meErryZpPanel2->Draw("he");
      newmeErryZpPanel2->Draw("hesameS"); 
      myPV->PVCompute(meErryZpPanel2, newmeErryZpPanel2, te );
      h_pv->SetBinContent(++bin, myPV->getPV());
      gPad->Update();      
      TPaveStats *s17 = (TPaveStats*)meErryZpPanel2->GetListOfFunctions()->FindObject("stats");
      if (s17) {
	s17->SetX1NDC (0.55); //new x start position
	s17->SetX2NDC (0.75); //new x end position
      }

      can_Erry->SaveAs("meErry_compare.eps");
      can_Erry->SaveAs("meErry_compare.gif");
    }
  
      
  if (1) 
    {
      TCanvas* can_Npix = new TCanvas("can_Npix", "can_Npix", 1200, 800);
      can_Npix->Divide(3,2);
      
      TH1F* meNpixBarrel;
      TH1F* meNpixZmPanel1;
      TH1F* meNpixZmPanel2;
      TH1F* meNpixZpPanel1;
      TH1F* meNpixZpPanel2;
      
      TH1F* newmeNpixBarrel;
      TH1F* newmeNpixZmPanel1;
      TH1F* newmeNpixZmPanel2;
      TH1F* newmeNpixZpPanel1;
      TH1F* newmeNpixZpPanel2;
      
      rdir->GetObject("Histograms_all/meNpixBarrel"  , meNpixBarrel  );
      rdir->GetObject("Histograms_all/meNpixZmPanel1", meNpixZmPanel1);
      rdir->GetObject("Histograms_all/meNpixZmPanel2", meNpixZmPanel2);
      rdir->GetObject("Histograms_all/meNpixZpPanel1", meNpixZpPanel1);
      rdir->GetObject("Histograms_all/meNpixZpPanel2", meNpixZpPanel2);
      
      sdir->GetObject("Histograms_all/meNpixBarrel"  , newmeNpixBarrel  ); 
      sdir->GetObject("Histograms_all/meNpixZmPanel1", newmeNpixZmPanel1);
      sdir->GetObject("Histograms_all/meNpixZmPanel2", newmeNpixZmPanel2);
      sdir->GetObject("Histograms_all/meNpixZpPanel1", newmeNpixZpPanel1);
      sdir->GetObject("Histograms_all/meNpixZpPanel2", newmeNpixZpPanel2);
      
      TLegend* leg5 = new TLegend(0.65, 0.5, 0.89, 0.65);
      can_Npix->cd(1);
      //gPad->SetLogy();
      SetUpHistograms(meNpixBarrel, newmeNpixBarrel, "barrel, cluster size (pixels)", leg5 );
      Float_t refMax = 1.2*meNpixBarrel->GetMaximum();
      Float_t newMax = 1.2*newmeNpixBarrel->GetMaximum();
      if refMax > newMax
      {
          meNpixBarrel->SetMaximum(refMax);
      }
      else
      {
          meNpixBarrel->SetMaximum(newMax);
      }
      meNpixBarrel->SetName("Reference");
      newmeNpixBarrel->SetName("New Release");
      meNpixBarrel->Draw("he");
      newmeNpixBarrel->Draw("hesameS"); 
      myPV->PVCompute(meNpixBarrel, newmeNpixBarrel, te );
      leg5->Draw();
      h_pv->SetBinContent(++bin, myPV->getPV());
      gPad->Update();      
      TPaveStats *s18 = (TPaveStats*)meNpixBarrel->GetListOfFunctions()->FindObject("stats");
      if (s18) {
	s18->SetX1NDC (0.55); //new x start position
	s18->SetX2NDC (0.75); //new x end position
      }

      can_Npix->cd(2);
      //gPad->SetLogy();
      SetUpHistograms(meNpixZmPanel1, newmeNpixZmPanel1, "panel1, z<0, cluster size (pixels)"  );
      Float_t refMax = 1.2*meNpixZmPanel1->GetMaximum();
      Float_t newMax = 1.2*newmeNpixZmPanel1->GetMaximum();
      if refMax > newMax
      {
          meNpixZmPanel1->SetMaximum(refMax);
      }
      else
      {
          meNpixZmPanel1->SetMaximum(newMax);
      }
      meNpixZmPanel1->SetName("Reference");
      newmeNpixZmPanel1->SetName("New Release");
      meNpixZmPanel1->Draw("he");
      newmeNpixZmPanel1->Draw("hesameS"); 
      myPV->PVCompute(meNpixZmPanel1, newmeNpixZmPanel1, te );
      h_pv->SetBinContent(++bin, myPV->getPV());
      gPad->Update();      
      TPaveStats *s19 = (TPaveStats*)meNpixZmPanel1->GetListOfFunctions()->FindObject("stats");
      if (s19) {
	s19->SetX1NDC (0.55); //new x start position
	s19->SetX2NDC (0.75); //new x end position
      }

      can_Npix->cd(3);
      //gPad->SetLogy();
      SetUpHistograms(meNpixZmPanel2, newmeNpixZmPanel2, "panel2, z<0, cluster size (pixels)" );
      Float_t refMax = 1.2*meNpixZmPanel2->GetMaximum();
      Float_t newMax = 1.2*newmeNpixZmPanel2->GetMaximum();
      if refMax > newMax
      {
          meNpixZmPanel2->SetMaximum(refMax);
      }
      else
      {
          meNpixZmPanel2->SetMaximum(newMax);
      }
      meNpixZmPanel2->SetName("Reference");
      newmeNpixZmPanel2->SetName("New Release");
      meNpixZmPanel2->Draw("he");
      newmeNpixZmPanel2->Draw("hesameS"); 
      myPV->PVCompute(meNpixZmPanel2, newmeNpixZmPanel2, te );
      h_pv->SetBinContent(++bin, myPV->getPV());
      gPad->Update();      
      TPaveStats *s20 = (TPaveStats*)meNpixZmPanel2->GetListOfFunctions()->FindObject("stats");
      if (s20) {
	s20->SetX1NDC (0.55); //new x start position
	s20->SetX2NDC (0.75); //new x end position
      }

      can_Npix->cd(5);
      //gPad->SetLogy();
      SetUpHistograms(meNpixZpPanel1, newmeNpixZpPanel1, "panel1, z>0, cluster size (pixels)" );
      Float_t refMax = 1.2*meNpixZpPanel1->GetMaximum();
      Float_t newMax = 1.2*newmeNpixZpPanel1->GetMaximum();
      if refMax > newMax
      {
          meNpixZpPanel1->SetMaximum(refMax);
      }
      else
      {
          meNpixZpPanel1->SetMaximum(newMax);
      }
      meNpixZpPanel1->SetName("Reference");
      newmeNpixZpPanel1->SetName("New Release");
      meNpixZpPanel1->Draw("he");
      newmeNpixZpPanel1->Draw("hesameS"); 
      myPV->PVCompute(meNpixZpPanel1, newmeNpixZpPanel1, te );
      h_pv->SetBinContent(++bin, myPV->getPV());
      gPad->Update();      
      TPaveStats *s21 = (TPaveStats*)meNpixZpPanel1->GetListOfFunctions()->FindObject("stats");
      if (s21) {
	s21->SetX1NDC (0.55); //new x start position
	s21->SetX2NDC (0.75); //new x end position
      }

      can_Npix->cd(6);
      //gPad->SetLogy();
      SetUpHistograms(meNpixZpPanel2, newmeNpixZpPanel2, "panel2, z>0, cluster size (pixels)" );
      Float_t refMax = 1.2*meNpixZpPanel2->GetMaximum();
      Float_t newMax = 1.2*newmeNpixZpPanel2->GetMaximum();
      if refMax > newMax
      {
          meNpixZpPanel2->SetMaximum(refMax);
      }
      else
      {
          meNpixZpPanel2->SetMaximum(newMax);
      }
      meNpixZpPanel2->SetName("Reference");
      newmeNpixZpPanel2->SetName("New Release");
      meNpixZpPanel2->Draw("he");
      newmeNpixZpPanel2->Draw("hesameS"); 
      myPV->PVCompute(meNpixZpPanel2, newmeNpixZpPanel2, te );
      h_pv->SetBinContent(++bin, myPV->getPV());
      gPad->Update();      
      TPaveStats *s22 = (TPaveStats*)meNpixZpPanel2->GetListOfFunctions()->FindObject("stats");
      if (s22) {
	s22->SetX1NDC (0.55); //new x start position
	s22->SetX2NDC (0.75); //new x end position
      }

      can_Npix->SaveAs("meNpix_compare.eps");
      can_Npix->SaveAs("meNpix_compare.gif");
    }

  if (1) 
    {
      TCanvas* can_Nxpix = new TCanvas("can_Nxpix", "can_Nxpix", 1200, 800);
      can_Nxpix->Divide(3,2);
      
      TH1F* meNxpixBarrel;
      TH1F* meNxpixZmPanel1;
      TH1F* meNxpixZmPanel2;
      TH1F* meNxpixZpPanel1;
      TH1F* meNxpixZpPanel2;
      
      TH1F* newmeNxpixBarrel;
      TH1F* newmeNxpixZmPanel1;
      TH1F* newmeNxpixZmPanel2;
      TH1F* newmeNxpixZpPanel1;
      TH1F* newmeNxpixZpPanel2;
      
      rdir->GetObject("Histograms_all/meNxpixBarrel"  , meNxpixBarrel  );
      rdir->GetObject("Histograms_all/meNxpixZmPanel1", meNxpixZmPanel1);
      rdir->GetObject("Histograms_all/meNxpixZmPanel2", meNxpixZmPanel2);
      rdir->GetObject("Histograms_all/meNxpixZpPanel1", meNxpixZpPanel1);
      rdir->GetObject("Histograms_all/meNxpixZpPanel2", meNxpixZpPanel2);
      
      sdir->GetObject("Histograms_all/meNxpixBarrel"  , newmeNxpixBarrel  ); 
      sdir->GetObject("Histograms_all/meNxpixZmPanel1", newmeNxpixZmPanel1);
      sdir->GetObject("Histograms_all/meNxpixZmPanel2", newmeNxpixZmPanel2);
      sdir->GetObject("Histograms_all/meNxpixZpPanel1", newmeNxpixZpPanel1);
      sdir->GetObject("Histograms_all/meNxpixZpPanel2", newmeNxpixZpPanel2);
      
      TLegend* leg6 = new TLegend(0.65, 0.5, 0.89, 0.65);
      can_Nxpix->cd(1);
      //gPad->SetLogy();
      SetUpHistograms(meNxpixBarrel, newmeNxpixBarrel, "barrel, cluster x size (pixels)", leg6 );
      Float_t refMax = 1.2*meNxpixBarrel->GetMaximum();
      Float_t newMax = 1.2*newmeNxpixBarrel->GetMaximum();
      if refMax > newMax
      {
          meNxpixBarrel->SetMaximum(refMax);
      }
      else
      {
          meNxpixBarrel->SetMaximum(newMax);
      }
      meNxpixBarrel->SetName("Reference");
      newmeNxpixBarrel->SetName("New Release");
      meNxpixBarrel->Draw("he");
      newmeNxpixBarrel->Draw("hesameS"); 
      myPV->PVCompute(meNxpixBarrel, newmeNxpixBarrel, te );
      leg6->Draw();
      h_pv->SetBinContent(++bin, myPV->getPV());
      gPad->Update();      
      TPaveStats *s23 = (TPaveStats*)meNxpixBarrel->GetListOfFunctions()->FindObject("stats");
      if (s23) {
	s23->SetX1NDC (0.55); //new x start position
	s23->SetX2NDC (0.75); //new x end position
      }

      can_Nxpix->cd(2);
      //gPad->SetLogy();
      SetUpHistograms(meNxpixZmPanel1, newmeNxpixZmPanel1, "panel1, z<0, cluster x size (pixels)" );
      Float_t refMax = 1.2*meNxpixZmPanel1->GetMaximum();
      Float_t newMax = 1.2*newmeNxpixZmPanel1->GetMaximum();
      if refMax > newMax
      {
          meNxpixZmPanel1->SetMaximum(refMax);
      }
      else
      {
          meNxpixZmPanel1->SetMaximum(newMax);
      }
      meNxpixZmPanel1->SetName("Reference");
      newmeNxpixZmPanel1->SetName("New Release");
      meNxpixZmPanel1->Draw("he");
      newmeNxpixZmPanel1->Draw("hesameS"); 
      myPV->PVCompute(meNxpixZmPanel1, newmeNxpixZmPanel1, te );
      h_pv->SetBinContent(++bin, myPV->getPV());
      gPad->Update();      
      TPaveStats *s24 = (TPaveStats*)meNxpixZmPanel1->GetListOfFunctions()->FindObject("stats");
      if (s24) {
	s24->SetX1NDC (0.55); //new x start position
	s24->SetX2NDC (0.75); //new x end position
      }

      can_Nxpix->cd(3);
      //gPad->SetLogy();
      SetUpHistograms(meNxpixZmPanel2, newmeNxpixZmPanel2, "panel2, z<0, cluster x size (pixels)" );
      Float_t refMax = 1.2*meNxpixZmPanel2->GetMaximum();
      Float_t newMax = 1.2*newmeNxpixZmPanel2->GetMaximum();
      if refMax > newMax
      {
          meNxpixZmPanel2->SetMaximum(refMax);
      }
      else
      {
          meNxpixZmPanel2->SetMaximum(newMax);
      }
      meNxpixZmPanel2->SetName("Reference");
      newmeNxpixZmPanel2->SetName("New Release");
      meNxpixZmPanel2->Draw("he");
      newmeNxpixZmPanel2->Draw("hesameS"); 
      myPV->PVCompute(meNxpixZmPanel2, newmeNxpixZmPanel2, te );
      h_pv->SetBinContent(++bin, myPV->getPV());
      
      gPad->Update();      
      TPaveStats *s25 = (TPaveStats*)meNxpixZmPanel2->GetListOfFunctions()->FindObject("stats");
      if (s25) {
	s25->SetX1NDC (0.55); //new x start position
	s25->SetX2NDC (0.75); //new x end position
      }

      can_Nxpix->cd(5);
      //gPad->SetLogy();
      SetUpHistograms(meNxpixZpPanel1, newmeNxpixZpPanel1, "panel1, z>0, cluster x size (pixels)" );
      Float_t refMax = 1.2*meNxpixZpPanel1->GetMaximum();
      Float_t newMax = 1.2*newmeNxpixZpPanel1->GetMaximum();
      if refMax > newMax
      {
          meNxpixZpPanel1->SetMaximum(refMax);
      }
      else
      {
          meNxpixZpPanel1->SetMaximum(newMax);
      }
      meNxpixZpPanel1->SetName("Reference");
      newmeNxpixZpPanel1->SetName("New Release");
      meNxpixZpPanel1->Draw("he");
      newmeNxpixZpPanel1->Draw("hesameS"); 
      myPV->PVCompute(meNxpixZpPanel1, newmeNxpixZpPanel1, te );
      h_pv->SetBinContent(++bin, myPV->getPV());
      
      gPad->Update();      
      TPaveStats *s26 = (TPaveStats*)meNxpixZpPanel1->GetListOfFunctions()->FindObject("stats");
      if (s26) {
	s26->SetX1NDC (0.55); //new x start position
	s26->SetX2NDC (0.75); //new x end position
      }

      can_Nxpix->cd(6);
      //gPad->SetLogy();
      SetUpHistograms(meNxpixZpPanel2, newmeNxpixZpPanel2, "panel2, z>0, cluster x size (pixels)" );
      Float_t refMax = 1.2*meNxpixZpPanel2->GetMaximum();
      Float_t newMax = 1.2*newmeNxpixZpPanel2->GetMaximum();
      if refMax > newMax
      {
          meNxpixZpPanel2->SetMaximum(refMax);
      }
      else
      {
          meNxpixZpPanel2->SetMaximum(newMax);
      }
      meNxpixZpPanel2->SetName("Reference");
      newmeNxpixZpPanel2->SetName("New Release");
      meNxpixZpPanel2->Draw("he");
      newmeNxpixZpPanel2->Draw("hesameS"); 
      myPV->PVCompute(meNxpixZpPanel2, newmeNxpixZpPanel2, te );
      h_pv->SetBinContent(++bin, myPV->getPV());
      gPad->Update();      
      TPaveStats *s27 = (TPaveStats*)meNxpixZpPanel2->GetListOfFunctions()->FindObject("stats");
      if (s27) {
	s27->SetX1NDC (0.55); //new x start position
	s27->SetX2NDC (0.75); //new x end position
      }

      can_Nxpix->SaveAs("meNxpix_compare.eps");
      can_Nxpix->SaveAs("meNxpix_compare.gif");
    }
    

  if (1) 
    {
      TCanvas* can_Nypix = new TCanvas("can_Nypix", "can_Nypix", 1200, 800);
      can_Nypix->Divide(3,2);
      
      TH1F* meNypixBarrel;
      TH1F* meNypixZmPanel1;
      TH1F* meNypixZmPanel2;
      TH1F* meNypixZpPanel1;
      TH1F* meNypixZpPanel2;
      
      TH1F* newmeNypixBarrel;
      TH1F* newmeNypixZmPanel1;
      TH1F* newmeNypixZmPanel2;
      TH1F* newmeNypixZpPanel1;
      TH1F* newmeNypixZpPanel2;
      
      rdir->GetObject("Histograms_all/meNypixBarrel"  , meNypixBarrel  );
      rdir->GetObject("Histograms_all/meNypixZmPanel1", meNypixZmPanel1);
      rdir->GetObject("Histograms_all/meNypixZmPanel2", meNypixZmPanel2);
      rdir->GetObject("Histograms_all/meNypixZpPanel1", meNypixZpPanel1);
      rdir->GetObject("Histograms_all/meNypixZpPanel2", meNypixZpPanel2);
      
      sdir->GetObject("Histograms_all/meNypixBarrel"  , newmeNypixBarrel  ); 
      sdir->GetObject("Histograms_all/meNypixZmPanel1", newmeNypixZmPanel1);
      sdir->GetObject("Histograms_all/meNypixZmPanel2", newmeNypixZmPanel2);
      sdir->GetObject("Histograms_all/meNypixZpPanel1", newmeNypixZpPanel1);
      sdir->GetObject("Histograms_all/meNypixZpPanel2", newmeNypixZpPanel2);
      
      TLegend* leg7 = new TLegend(0.65, 0.55, 0.89, 0.7);
      can_Nypix->cd(1);
      //gPad->SetLogy();
      SetUpHistograms(meNypixBarrel, newmeNypixBarrel, "barrel, cluster y size (pixels)", leg7 );
      Float_t refMax = 1.2*meNypixBarrel->GetMaximum();
      Float_t newMax = 1.2*newmeNypixBarrel->GetMaximum();
      if refMax > newMax
      {
          meNypixBarrel->SetMaximum(refMax);
      }
      else
      {
          meNypixBarrel->SetMaximum(newMax);
      }
      meNypixBarrel->SetName("Reference");
      newmeNypixBarrel->SetName("New Release");
      meNypixBarrel->Draw("he");
      newmeNypixBarrel->Draw("hesameS"); 
      myPV->PVCompute(meNypixBarrel, newmeNypixBarrel, te );
      leg7->Draw();
      h_pv->SetBinContent(++bin, myPV->getPV());
      gPad->Update();      
      TPaveStats *s28 = (TPaveStats*)meNypixBarrel->GetListOfFunctions()->FindObject("stats");
      if (s28) {
	s28->SetX1NDC (0.55); //new x start position
	s28->SetX2NDC (0.75); //new x end position
      }

      can_Nypix->cd(2);
      //gPad->SetLogy();
      SetUpHistograms(meNypixZmPanel1, newmeNypixZmPanel1, "panel1, z<0, cluster y size (pixels)" );
      Float_t refMax = 1.2*meNypixZmPanel1->GetMaximum();
      Float_t newMax = 1.2*newmeNypixZmPanel1->GetMaximum();
      if refMax > newMax
      {
          meNypixZmPanel1->SetMaximum(refMax);
      }
      else
      {
          meNypixZmPanel1->SetMaximum(newMax);
      }
      meNypixZmPanel1->SetName("Reference");
      newmeNypixZmPanel1->SetName("New Release");
      meNypixZmPanel1->Draw("he");
      newmeNypixZmPanel1->Draw("hesameS"); 
      myPV->PVCompute(meNypixZmPanel1, newmeNypixZmPanel1, te );
      h_pv->SetBinContent(++bin, myPV->getPV());
      gPad->Update();      
      TPaveStats *s29 = (TPaveStats*)meNypixZmPanel1->GetListOfFunctions()->FindObject("stats");
      if (s29) {
	s29->SetX1NDC (0.55); //new x start position
	s29->SetX2NDC (0.75); //new x end position
      }

      can_Nypix->cd(3);
      //gPad->SetLogy();
      SetUpHistograms(meNypixZmPanel2, newmeNypixZmPanel2, "panel2, z<0, cluster y size (pixels)" );
      Float_t refMax = 1.2*meNypixZmPanel2->GetMaximum();
      Float_t newMax = 1.2*newmeNypixZmPanel2->GetMaximum();
      if refMax > newMax
      {
          meNypixZmPanel2->SetMaximum(refMax);
      }
      else
      {
          meNypixZmPanel2->SetMaximum(newMax);
      }
      meNypixZmPanel2->SetName("Reference");
      newmeNypixZmPanel2->SetName("New Release");
      meNypixZmPanel2->Draw("he");
      newmeNypixZmPanel2->Draw("hesameS"); 
      myPV->PVCompute(meNypixZmPanel2, newmeNypixZmPanel2, te );
      h_pv->SetBinContent(++bin, myPV->getPV());
      gPad->Update();      
      TPaveStats *s30 = (TPaveStats*)meNypixZmPanel2->GetListOfFunctions()->FindObject("stats");
      if (s30) {
	s30->SetX1NDC (0.55); //new x start position
	s30->SetX2NDC (0.75); //new x end position
      }

      can_Nypix->cd(5);
      //gPad->SetLogy();
      SetUpHistograms(meNypixZpPanel1, newmeNypixZpPanel1, "panel1, z>0, cluster y size (pixels)" );
      Float_t refMax = 1.2*meNypixZpPanel1->GetMaximum();
      Float_t newMax = 1.2*newmeNypixZpPanel1->GetMaximum();
      if refMax > newMax
      {
          meNypixZpPanel1->SetMaximum(refMax);
      }
      else
      {
          meNypixZpPanel1->SetMaximum(newMax);
      }
      meNypixZpPanel1->SetName("Reference");
      newmeNypixZpPanel1->SetName("New Release");
      meNypixZpPanel1->Draw("he");
      newmeNypixZpPanel1->Draw("hesameS"); 
      myPV->PVCompute(meNypixZpPanel1, newmeNypixZpPanel1, te );
      h_pv->SetBinContent(++bin, myPV->getPV());
      gPad->Update();      
      TPaveStats *s31 = (TPaveStats*)meNypixZpPanel1->GetListOfFunctions()->FindObject("stats");
      if (s31) {
	s31->SetX1NDC (0.55); //new x start position
	s31->SetX2NDC (0.75); //new x end position
      }

      can_Nypix->cd(6);
      //gPad->SetLogy();
      SetUpHistograms(meNypixZpPanel2, newmeNypixZpPanel2, "panel2, z>0, cluster y size (pixels)" );
      Float_t refMax = 1.2*meNypixZpPanel2->GetMaximum();
      Float_t newMax = 1.2*newmeNypixZpPanel2->GetMaximum();
      if refMax > newMax
      {
          meNypixZpPanel2->SetMaximum(refMax);
      }
      else
      {
          meNypixZpPanel2->SetMaximum(newMax);
      }      
      meNypixZpPanel2->SetName("Reference");
      newmeNypixZpPanel2->SetName("New Release");
      meNypixZpPanel2->Draw("he");
      newmeNypixZpPanel2->Draw("hesameS"); 
      myPV->PVCompute(meNypixZpPanel2, newmeNypixZpPanel2, te );
      h_pv->SetBinContent(++bin, myPV->getPV());      
      gPad->Update();      
      TPaveStats *s32 = (TPaveStats*)meNypixZpPanel2->GetListOfFunctions()->FindObject("stats");
      if (s32) {
	s32->SetX1NDC (0.55); //new x start position
	s32->SetX2NDC (0.75); //new x end position
      }

      can_Nypix->SaveAs("meNypix_compare.eps");
      can_Nypix->SaveAs("meNypix_compare.gif");
    }
  
  if (1) 
    {
      TCanvas* can_Posx = new TCanvas("can_Posx", "can_Posx", 1200, 800);
      can_Posx->Divide(3,2);
      
      TH1F* mePosxBarrel;
      TH1F* mePosxZmPanel1;
      TH1F* mePosxZmPanel2;
      TH1F* mePosxZpPanel1;
      TH1F* mePosxZpPanel2;
      
      TH1F* newmePosxBarrel;
      TH1F* newmePosxZmPanel1;
      TH1F* newmePosxZmPanel2;
      TH1F* newmePosxZpPanel1;
      TH1F* newmePosxZpPanel2;
      
      rdir->GetObject("Histograms_all/mePosxBarrel"  , mePosxBarrel  );
      rdir->GetObject("Histograms_all/mePosxZmPanel1", mePosxZmPanel1);
      rdir->GetObject("Histograms_all/mePosxZmPanel2", mePosxZmPanel2);
      rdir->GetObject("Histograms_all/mePosxZpPanel1", mePosxZpPanel1);
      rdir->GetObject("Histograms_all/mePosxZpPanel2", mePosxZpPanel2);
      
      sdir->GetObject("Histograms_all/mePosxBarrel"  , newmePosxBarrel  ); 
      sdir->GetObject("Histograms_all/mePosxZmPanel1", newmePosxZmPanel1);
      sdir->GetObject("Histograms_all/mePosxZmPanel2", newmePosxZmPanel2);
      sdir->GetObject("Histograms_all/mePosxZpPanel1", newmePosxZpPanel1);
      sdir->GetObject("Histograms_all/mePosxZpPanel2", newmePosxZpPanel2);
      
      TLegend* leg8 = new TLegend(0.3, 0.2, 0.6, 0.4);
      can_Posx->cd(1);
      //gPad->SetLogy();
      SetUpHistograms(mePosxBarrel, newmePosxBarrel, "barrel, x (cm)", leg8 );
      Float_t refMax = 1.5*mePosxBarrel->GetMaximum();
      Float_t newMax = 1.5*newmePosxBarrel->GetMaximum();
      if refMax > newMax
      {
          mePosxBarrel->SetMaximum(refMax);
      }
      else
      {
          mePosxBarrel->SetMaximum(newMax);
      }      
      mePosxBarrel->SetName("Reference");
      newmePosxBarrel->SetName("New Release");
      mePosxBarrel->Draw("he");
      newmePosxBarrel->Draw("heSameS"); 
      myPV->PVCompute(mePosxBarrel, newmePosxBarrel, te, 0.6, 0.75 );
      leg8->Draw();
      h_pv->SetBinContent(++bin, myPV->getPV());      
      gPad->Update();      
      TPaveStats *s33 = (TPaveStats*)newmePosxBarrel->GetListOfFunctions()->FindObject("stats");
      if (s33) {
	s33->SetX1NDC (0.55); //new x start position
	s33->SetX2NDC (0.75); //new x end position
      }

      can_Posx->cd(2);
      //gPad->SetLogy();
      SetUpHistograms(mePosxZmPanel1, newmePosxZmPanel1, "panel1, z<0, x (cm)" );
      Float_t refMax = 1.5*mePosxZmPanel1->GetMaximum();
      Float_t newMax = 1.5*newmePosxZmPanel1->GetMaximum();
      if refMax > newMax
      {
          mePosxZmPanel1->SetMaximum(refMax);
      }
      else
      {
          mePosxZmPanel1->SetMaximum(newMax);
      }      
      mePosxZmPanel1->SetName("Reference");
      newmePosxZmPanel1->SetName("New Release");
      mePosxZmPanel1->Draw("he");
      newmePosxZmPanel1->Draw("hesameS"); 
      myPV->PVCompute(mePosxZmPanel1, newmePosxZmPanel1, te, 0.6, 0.75 );
      h_pv->SetBinContent(++bin, myPV->getPV());      
      gPad->Update();      
      TPaveStats *s34 = (TPaveStats*)mePosxZmPanel1->GetListOfFunctions()->FindObject("stats");
      if (s34) {
	s34->SetX1NDC (0.55); //new x start position
	s34->SetX2NDC (0.75); //new x end position
      }

      can_Posx->cd(3);
      //gPad->SetLogy();
      SetUpHistograms(mePosxZmPanel2, newmePosxZmPanel2, "panel2, z>0, x (cm)" );
      Float_t refMax = 1.5*mePosxZmPanel2->GetMaximum();
      Float_t newMax = 1.5*newmePosxZmPanel2->GetMaximum();
      if refMax > newMax
      {
          mePosxZmPanel2->SetMaximum(refMax);
      }
      else
      {
          mePosxZmPanel2->SetMaximum(newMax);
      }      
      mePosxZmPanel2->SetName("Reference");
      newmePosxZmPanel2->SetName("New Release");
      mePosxZmPanel2->Draw("he");
      newmePosxZmPanel2->Draw("hesameS"); 
      myPV->PVCompute(mePosxZmPanel2, newmePosxZmPanel2, te, 0.6, 0.75 );
      h_pv->SetBinContent(++bin, myPV->getPV());     
      gPad->Update();      
      TPaveStats *s35 = (TPaveStats*)mePosxZmPanel2->GetListOfFunctions()->FindObject("stats");
      if (s35) {
	s35->SetX1NDC (0.55); //new x start position
	s35->SetX2NDC (0.75); //new x end position
      }

      can_Posx->cd(5);
      //gPad->SetLogy();
      SetUpHistograms(mePosxZpPanel1, newmePosxZpPanel1, "panel1, z<0, x (cm)" );
      Float_t refMax = 1.5*mePosxZpPanel1->GetMaximum();
      Float_t newMax = 1.5*newmePosxZpPanel1->GetMaximum();
      if refMax > newMax
      {
          mePosxZpPanel1->SetMaximum(refMax);
      }
      else
      {
          mePosxZpPanel1->SetMaximum(newMax);
      }
      mePosxZpPanel1->SetName("Reference");
      newmePosxZpPanel1->SetName("New Release");
      mePosxZpPanel1->Draw("he");
      newmePosxZpPanel1->Draw("hesameS"); 
      myPV->PVCompute(mePosxZpPanel1, newmePosxZpPanel1, te, 0.6, 0.75  );
      h_pv->SetBinContent(++bin, myPV->getPV());
      gPad->Update();      
      TPaveStats *s36 = (TPaveStats*)mePosxZpPanel1->GetListOfFunctions()->FindObject("stats");
      if (s36) {
	s36->SetX1NDC (0.55); //new x start position
	s36->SetX2NDC (0.75); //new x end position
      }

      can_Posx->cd(6);
      //gPad->SetLogy();
      SetUpHistograms(mePosxZpPanel2, newmePosxZpPanel2, "panel2, z>0, x (cm)" );
      Float_t refMax = 1.5*mePosxZpPanel2->GetMaximum();
      Float_t newMax = 1.5*newmePosxZpPanel2->GetMaximum();
      if refMax > newMax
      {
          mePosxZpPanel2->SetMaximum(refMax);
      }
      else
      {
          mePosxZpPanel2->SetMaximum(newMax);
      }
      mePosxZpPanel2->SetName("Reference");
      newmePosxZpPanel2->SetName("New Release");
      mePosxZpPanel2->Draw("he");
      newmePosxZpPanel2->Draw("hesameS"); 
      myPV->PVCompute(mePosxZpPanel2, newmePosxZpPanel2, te, 0.6, 0.75  );
      h_pv->SetBinContent(++bin, myPV->getPV());      
      gPad->Update();      
      TPaveStats *s37 = (TPaveStats*)mePosxZpPanel2->GetListOfFunctions()->FindObject("stats");
      if (s37) {
	s37->SetX1NDC (0.55); //new x start position
	s37->SetX2NDC (0.75); //new x end position
      }

      can_Posx->SaveAs("mePosx_compare.eps");
      can_Posx->SaveAs("mePosx_compare.gif");
  }

  if (1) 
    {
      TCanvas* can_Posy = new TCanvas("can_Posy", "can_Posy", 1200, 800);
      can_Posy->Divide(3,2);
      
      TH1F* mePosyBarrel;
      TH1F* mePosyZmPanel1;
      TH1F* mePosyZmPanel2;
      TH1F* mePosyZpPanel1;
      TH1F* mePosyZpPanel2;
      
      TH1F* newmePosyBarrel;
      TH1F* newmePosyZmPanel1;
      TH1F* newmePosyZmPanel2;
      TH1F* newmePosyZpPanel1;
      TH1F* newmePosyZpPanel2;
      
      rdir->GetObject("Histograms_all/mePosyBarrel"  , mePosyBarrel  );
      rdir->GetObject("Histograms_all/mePosyZmPanel1", mePosyZmPanel1);
      rdir->GetObject("Histograms_all/mePosyZmPanel2", mePosyZmPanel2);
      rdir->GetObject("Histograms_all/mePosyZpPanel1", mePosyZpPanel1);
      rdir->GetObject("Histograms_all/mePosyZpPanel2", mePosyZpPanel2);
      
      sdir->GetObject("Histograms_all/mePosyBarrel"  , newmePosyBarrel  ); 
      sdir->GetObject("Histograms_all/mePosyZmPanel1", newmePosyZmPanel1);
      sdir->GetObject("Histograms_all/mePosyZmPanel2", newmePosyZmPanel2);
      sdir->GetObject("Histograms_all/mePosyZpPanel1", newmePosyZpPanel1);
      sdir->GetObject("Histograms_all/mePosyZpPanel2", newmePosyZpPanel2);
      
      TLegend* leg9 = new TLegend(0.3, 0.2, 0.6, 0.4);
      can_Posy->cd(1);
      //gPad->SetLogy();
      SetUpHistograms(mePosyBarrel, newmePosyBarrel, "barrel, y (cm)", leg9 );
      Float_t refMax = 1.2*mePosyBarrel->GetMaximum();
      Float_t newMax = 1.2*newmePosyBarrel->GetMaximum();
      if refMax > newMax
      {
          mePosyBarrel->SetMaximum(refMax);
      }
      else
      {
          mePosyBarrel->SetMaximum(newMax);
      }      
      mePosyBarrel->SetName("Reference");
      newmePosyBarrel->SetName("New Release");
      mePosyBarrel->Draw("he");
      newmePosyBarrel->Draw("heSameS"); 
      myPV->PVCompute(mePosyBarrel, newmePosyBarrel, te, 0.3, 0.4 );
      leg9->Draw();
      h_pv->SetBinContent(++bin, myPV->getPV());
      gPad->Update();      
      TPaveStats *s38 = (TPaveStats*)mePosyBarrel->GetListOfFunctions()->FindObject("stats");
      if (s38) {
	s38->SetX1NDC (0.55); //new x start position
	s38->SetX2NDC (0.75); //new x end position
      }

      can_Posy->cd(2);
      //gPad->SetLogy();
      SetUpHistograms(mePosyZmPanel1,  newmePosyZmPanel1, "panel1, z<0, y (cm)" );
      Float_t refMax = 1.5*mePosyZmPanel1->GetMaximum();
      Float_t newMax = 1.5*newmePosyZmPanel1->GetMaximum();
      if refMax > newMax
      {
          mePosyZmPanel1->SetMaximum(refMax);
      }
      else
      {
          mePosyZmPanel1->SetMaximum(newMax);
      }      
      mePosyZmPanel1->SetName("Reference");
      newmePosyZmPanel1->SetName("New Release");
      mePosyZmPanel1->Draw("he");
      newmePosyZmPanel1->Draw("hesameS"); 
      myPV->PVCompute(mePosyZmPanel1, newmePosyZmPanel1, te, 0.6, 0.75  );
      h_pv->SetBinContent(++bin, myPV->getPV());
      gPad->Update();      
      TPaveStats *s39 = (TPaveStats*)mePosyZmPanel1->GetListOfFunctions()->FindObject("stats");
      if (s39) {
	s39->SetX1NDC (0.55); //new x start position
	s39->SetX2NDC (0.75); //new x end position
      }

      can_Posy->cd(3);
      //gPad->SetLogy();
      SetUpHistograms(mePosyZmPanel2, newmePosyZmPanel2, "panel2, z<0, y (cm)" );
      Float_t refMax = 1.5*mePosyZmPanel2->GetMaximum();
      Float_t newMax = 1.5*newmePosyZmPanel2->GetMaximum();
      if refMax > newMax
      {
          mePosyZmPanel2->SetMaximum(refMax);
      }
      else
      {
          mePosyZmPanel2->SetMaximum(newMax);
      }      
      mePosyZmPanel2->SetName("Reference");
      newmePosyZmPanel2->SetName("New Release");
      mePosyZmPanel2->Draw("he");
      newmePosyZmPanel2->Draw("hesameS"); 
      myPV->PVCompute(mePosyZmPanel2, newmePosyZmPanel2, te, 0.6, 0.75  );
      h_pv->SetBinContent(++bin, myPV->getPV());
      gPad->Update();      
      TPaveStats *s40 = (TPaveStats*)mePosyZmPanel2->GetListOfFunctions()->FindObject("stats");
      if (s40) {
	s40->SetX1NDC (0.55); //new x start position
	s40->SetX2NDC (0.75); //new x end position
      }

      can_Posy->cd(5);
      //gPad->SetLogy();
      SetUpHistograms(mePosyZpPanel1, newmePosyZpPanel1, "panel1, z>0, y (cm)" );
      Float_t refMax = 1.5*mePosyZpPanel1->GetMaximum();
      Float_t newMax = 1.5*newmePosyZpPanel1->GetMaximum();
      if refMax > newMax
      {
          mePosyZpPanel1->SetMaximum(refMax);
      }
      else
      {
          mePosyZpPanel1->SetMaximum(newMax);
      }      
      mePosyZpPanel1->SetName("Reference");
      newmePosyZpPanel1->SetName("New Release");
      mePosyZpPanel1->Draw("he");
      newmePosyZpPanel1->Draw("hesameS"); 
      myPV->PVCompute(mePosyZpPanel1, newmePosyZpPanel1, te, 0.6, 0.75 );
      h_pv->SetBinContent(++bin, myPV->getPV());
      gPad->Update();      
      TPaveStats *s41 = (TPaveStats*)mePosyZpPanel1->GetListOfFunctions()->FindObject("stats");
      if (s41) {
	s41->SetX1NDC (0.55); //new x start position
	s41->SetX2NDC (0.75); //new x end position
      }

      can_Posy->cd(6);
      //gPad->SetLogy();
      SetUpHistograms(mePosyZpPanel2, newmePosyZpPanel2, "panel2, z>0, y (cm)" );
      Float_t refMax = 1.5*mePosyZpPanel2->GetMaximum();
      Float_t newMax = 1.5*newmePosyZpPanel2->GetMaximum();
      if refMax > newMax
      {
          mePosyZpPanel2->SetMaximum(refMax);
      }
      else
      {
          mePosyZpPanel2->SetMaximum(newMax);
      }      
      mePosyZpPanel2->SetName("Reference");
      newmePosyZpPanel2->SetName("New Release");
      mePosyZpPanel2->Draw("he");
      newmePosyZpPanel2->Draw("hesameS"); 
      myPV->PVCompute(mePosyZpPanel2, newmePosyZpPanel2, te, 0.6, 0.75 );
      h_pv->SetBinContent(++bin, myPV->getPV());
      gPad->Update();      
      TPaveStats *s42 = (TPaveStats*)mePosyZpPanel2->GetListOfFunctions()->FindObject("stats");
      if (s42) {
	s42->SetX1NDC (0.55); //new x start position
	s42->SetX2NDC (0.75); //new x end position
      }

      can_Posy->SaveAs("mePosy_compare.eps");
      can_Posy->SaveAs("mePosy_compare.gif");
    }

  double lpull = -1.0;
  double hpull =  1.0;

  double lwpull = 0.0;
  double hwpull = 2.0;
  
  //if (   0   ) 
  if (   1   ) 
    {
      TCanvas* can_PullXvsAlpha = new TCanvas("can_PullXvsAlpha", "can_PullXvsAlpha", 1200, 800);
      can_PullXvsAlpha->Divide(3,2);
      
      TProfile* mePullXvsAlphaBarrel;
      TProfile* mePullXvsAlphaZmPanel1;
      TProfile* mePullXvsAlphaZmPanel2;
      TProfile* mePullXvsAlphaZpPanel1;
      TProfile* mePullXvsAlphaZpPanel2;
      
      TProfile* newmePullXvsAlphaBarrel;
      TProfile* newmePullXvsAlphaZmPanel1;
      TProfile* newmePullXvsAlphaZmPanel2;
      TProfile* newmePullXvsAlphaZpPanel1;
      TProfile* newmePullXvsAlphaZpPanel2;
      
      rdir->GetObject("Histograms_all/mePullXvsAlphaBarrel"  , mePullXvsAlphaBarrel  );
      rdir->GetObject("Histograms_all/mePullXvsAlphaZmPanel1", mePullXvsAlphaZmPanel1);
      rdir->GetObject("Histograms_all/mePullXvsAlphaZmPanel2", mePullXvsAlphaZmPanel2);
      rdir->GetObject("Histograms_all/mePullXvsAlphaZpPanel1", mePullXvsAlphaZpPanel1);
      rdir->GetObject("Histograms_all/mePullXvsAlphaZpPanel2", mePullXvsAlphaZpPanel2);
      
      sdir->GetObject("Histograms_all/mePullXvsAlphaBarrel"  , newmePullXvsAlphaBarrel  ); 
      sdir->GetObject("Histograms_all/mePullXvsAlphaZmPanel1", newmePullXvsAlphaZmPanel1);
      sdir->GetObject("Histograms_all/mePullXvsAlphaZmPanel2", newmePullXvsAlphaZmPanel2);
      sdir->GetObject("Histograms_all/mePullXvsAlphaZpPanel1", newmePullXvsAlphaZpPanel1);
      sdir->GetObject("Histograms_all/mePullXvsAlphaZpPanel2", newmePullXvsAlphaZpPanel2);
      
      TLegend* leg10 = new TLegend(0.3, 0.2, 0.6, 0.4);
      can_PullXvsAlpha->cd(1);
      //gPad->SetLogy();
      SetUpProfileHistograms(mePullXvsAlphaBarrel, newmePullXvsAlphaBarrel, "barrel, |alpha| (deg)", "pull x", lpull, hpull, leg10 );
      Float_t refMax = 0.5 + mePullXvsAlphaBarrel->GetMaximum();
      Float_t newMax = 0.5 + newmePullXvsAlphaBarrel->GetMaximum();
      if refMax > newMax
      {
          mePullXvsAlphaBarrel->SetMaximum(refMax);
      }
      else
      {
          mePullXvsAlphaBarrel->SetMaximum(newMax);
      }      
      mePullXvsAlphaBarrel->SetName("Reference");
      newmePullXvsAlphaBarrel->SetName("New Release");
      mePullXvsAlphaBarrel->Draw("e");
      newmePullXvsAlphaBarrel->Draw("eSameS"); 
      myPV->PVCompute(mePullXvsAlphaBarrel, newmePullXvsAlphaBarrel, te, 0.3, 0.4  );
      leg10->Draw();
      h_pv->SetBinContent(++bin, myPV->getPV());
      gPad->Update();      
      TPaveStats *s43 = (TPaveStats*)mePullXvsAlphaBarrel->GetListOfFunctions()->FindObject("stats");
      if (s43) {
	s43->SetX1NDC (0.55); //new x start position
	s43->SetX2NDC (0.75); //new x end position
      }

      can_PullXvsAlpha->cd(2);
      //gPad->SetLogy();
      SetUpProfileHistograms(mePullXvsAlphaZmPanel1, newmePullXvsAlphaZmPanel1, "panel1, z<0, |alpha| (deg)", "pull x", lpull, hpull );
      Float_t refMax = 0.5+1.5*mePullXvsAlphaZmPanel1->GetMaximum();
      Float_t newMax = 0.5+1.5*newmePullXvsAlphaZmPanel1->GetMaximum();
      if refMax > newMax
      {
          mePullXvsAlphaZmPanel1->SetMaximum(refMax);
      }
      else
      {
          mePullXvsAlphaZmPanel1->SetMaximum(newMax);
      }      
      mePullXvsAlphaZmPanel1->SetName("Reference");
      newmePullXvsAlphaZmPanel1->SetName("New Release");
      mePullXvsAlphaZmPanel1->Draw("e");
      newmePullXvsAlphaZmPanel1->Draw("esameS"); 
      myPV->PVCompute(mePullXvsAlphaZmPanel1, newmePullXvsAlphaZmPanel1, te, 0.2, 0.2);
      h_pv->SetBinContent(++bin, myPV->getPV());
      gPad->Update();      
      TPaveStats *s44 = (TPaveStats*)mePullXvsAlphaZmPanel1->GetListOfFunctions()->FindObject("stats");
      if (s44) {
	s44->SetX1NDC (0.55); //new x start position
	s44->SetX2NDC (0.75); //new x end position
      }

      can_PullXvsAlpha->cd(3);
      //gPad->SetLogy();
      SetUpProfileHistograms(mePullXvsAlphaZmPanel2, newmePullXvsAlphaZmPanel2, "panel2, z<0, |alpha| (deg)", "pull x", lpull, hpull );
      Float_t refMax = 0.5+1.5*mePullXvsAlphaZmPanel2->GetMaximum();
      Float_t newMax = 0.5+1.5*newmePullXvsAlphaZmPanel2->GetMaximum();
      if refMax > newMax
      {
          mePullXvsAlphaZmPanel2->SetMaximum(refMax);
      }
      else
      {
          mePullXvsAlphaZmPanel2->SetMaximum(newMax);
      }      
      mePullXvsAlphaZmPanel2->SetName("Reference");
      newmePullXvsAlphaZmPanel2->SetName("New Release");
      mePullXvsAlphaZmPanel2->Draw("e");
      newmePullXvsAlphaZmPanel2->Draw("esameS"); 
      myPV->PVCompute(mePullXvsAlphaZmPanel2, newmePullXvsAlphaZmPanel2, te, 0.2, 0.2);
      h_pv->SetBinContent(++bin, myPV->getPV());
      gPad->Update();      
      TPaveStats *s45 = (TPaveStats*)mePullXvsAlphaZmPanel2->GetListOfFunctions()->FindObject("stats");
      if (s45) {
	s45->SetX1NDC (0.55); //new x start position
	s45->SetX2NDC (0.75); //new x end position
      }

      can_PullXvsAlpha->cd(5);
      //gPad->SetLogy();
      SetUpProfileHistograms(mePullXvsAlphaZpPanel1, newmePullXvsAlphaZpPanel1, "panel1, z>0, |alpha| (deg)", "pull x", lpull, hpull );
      Float_t refMax = 0.5+1.5*mePullXvsAlphaZpPanel1->GetMaximum();
      Float_t newMax = 0.5+1.5*newmePullXvsAlphaZpPanel1->GetMaximum();
      if refMax > newMax
      {
          mePullXvsAlphaZpPanel1->SetMaximum(refMax);
      }
      else
      {
          mePullXvsAlphaZpPanel1->SetMaximum(newMax);
      }      
      mePullXvsAlphaZpPanel1->SetName("Reference");
      newmePullXvsAlphaZpPanel1->SetName("New Release");
      mePullXvsAlphaZpPanel1->Draw("e");
      newmePullXvsAlphaZpPanel1->Draw("esameS"); 
      myPV->PVCompute(mePullXvsAlphaZpPanel1, newmePullXvsAlphaZpPanel1, te, 0.2, 0.2);
      h_pv->SetBinContent(++bin, myPV->getPV());
      gPad->Update();      
      TPaveStats *s46 = (TPaveStats*)mePullXvsAlphaZpPanel1->GetListOfFunctions()->FindObject("stats");
      if (s46) {
	s46->SetX1NDC (0.55); //new x start position
	s46->SetX2NDC (0.75); //new x end position
      }

      can_PullXvsAlpha->cd(6);
      //gPad->SetLogy();
      SetUpProfileHistograms(mePullXvsAlphaZpPanel2, newmePullXvsAlphaZpPanel2, "panel2, z>0, |alpha| (deg)", "pull x", lpull, hpull );
      Float_t refMax = 0.5+1.5*mePullXvsAlphaZpPanel2->GetMaximum();
      Float_t newMax = 0.5+1.5*newmePullXvsAlphaZpPanel2->GetMaximum();
      if refMax > newMax
      {
          mePullXvsAlphaZpPanel2->SetMaximum(refMax);
      }
      else
      {
          mePullXvsAlphaZpPanel2->SetMaximum(newMax);
      }      
      mePullXvsAlphaZpPanel2->SetName("Reference");
      newmePullXvsAlphaZpPanel2->SetName("New Release");
      mePullXvsAlphaZpPanel2->Draw("e");
      newmePullXvsAlphaZpPanel2->Draw("esameS"); 
      myPV->PVCompute(mePullXvsAlphaZpPanel2, newmePullXvsAlphaZpPanel2, te , 0.2, 0.2);
      h_pv->SetBinContent(++bin, myPV->getPV());
      gPad->Update();      
      TPaveStats *s47 = (TPaveStats*)mePullXvsAlphaZpPanel2->GetListOfFunctions()->FindObject("stats");
      if (s47) {
	s47->SetX1NDC (0.55); //new x start position
	s47->SetX2NDC (0.75); //new x end position
      }

      can_PullXvsAlpha->SaveAs("mePullXvsAlpha_compare.eps");
      can_PullXvsAlpha->SaveAs("mePullXvsAlpha_compare.gif");
    }

  //if (   0   ) 
  if (   1   ) 
    {
      TCanvas* can_PullXvsBeta = new TCanvas("can_PullXvsBeta", "can_PullXvsBeta", 1200, 800);
      can_PullXvsBeta->Divide(3,2);
      
      TProfile* mePullXvsBetaBarrel;
      TProfile* mePullXvsBetaZmPanel1;
      TProfile* mePullXvsBetaZmPanel2;
      TProfile* mePullXvsBetaZpPanel1;
      TProfile* mePullXvsBetaZpPanel2;
      
      TProfile* newmePullXvsBetaBarrel;
      TProfile* newmePullXvsBetaZmPanel1;
      TProfile* newmePullXvsBetaZmPanel2;
      TProfile* newmePullXvsBetaZpPanel1;
      TProfile* newmePullXvsBetaZpPanel2;
      
      rdir->GetObject("Histograms_all/mePullXvsBetaBarrel"  , mePullXvsBetaBarrel  );
      rdir->GetObject("Histograms_all/mePullXvsBetaZmPanel1", mePullXvsBetaZmPanel1);
      rdir->GetObject("Histograms_all/mePullXvsBetaZmPanel2", mePullXvsBetaZmPanel2);
      rdir->GetObject("Histograms_all/mePullXvsBetaZpPanel1", mePullXvsBetaZpPanel1);
      rdir->GetObject("Histograms_all/mePullXvsBetaZpPanel2", mePullXvsBetaZpPanel2);
      
      sdir->GetObject("Histograms_all/mePullXvsBetaBarrel"  , newmePullXvsBetaBarrel  ); 
      sdir->GetObject("Histograms_all/mePullXvsBetaZmPanel1", newmePullXvsBetaZmPanel1);
      sdir->GetObject("Histograms_all/mePullXvsBetaZmPanel2", newmePullXvsBetaZmPanel2);
      sdir->GetObject("Histograms_all/mePullXvsBetaZpPanel1", newmePullXvsBetaZpPanel1);
      sdir->GetObject("Histograms_all/mePullXvsBetaZpPanel2", newmePullXvsBetaZpPanel2);
      
      TLegend* leg11 = new TLegend(0.3, 0.2, 0.6, 0.4);
      can_PullXvsBeta->cd(1);
      //gPad->SetLogy();
      SetUpProfileHistograms(mePullXvsBetaBarrel, newmePullXvsBetaBarrel, "barrel, |beta| (deg)", "pull x", lpull, hpull, leg11 );
      Float_t refMax = 0.3+1.2*mePullXvsBetaBarrel->GetMaximum();
      Float_t newMax = 0.3+1.2*newmePullXvsBetaBarrel->GetMaximum();
      if refMax > newMax
      {
          mePullXvsBetaBarrel->SetMaximum(refMax);
      }
      else
      {
          mePullXvsBetaBarrel->SetMaximum(newMax);
      }      
      mePullXvsBetaBarrel->SetName("Reference");
      newmePullXvsBetaBarrel->SetName("New Release");
      mePullXvsBetaBarrel->Draw("e");
      newmePullXvsBetaBarrel->Draw("eSameS"); 
      myPV->PVCompute(mePullXvsBetaBarrel, newmePullXvsBetaBarrel, te, 0.3, 0.4  );
      leg11->Draw();
      h_pv->SetBinContent(++bin, myPV->getPV());
      gPad->Update();      
      TPaveStats *s48 = (TPaveStats*)mePullXvsBetaBarrel->GetListOfFunctions()->FindObject("stats");
      if (s48) {
	s48->SetX1NDC (0.55); //new x start position
	s48->SetX2NDC (0.75); //new x end position
      }

      can_PullXvsBeta->cd(2);
      //gPad->SetLogy();
      SetUpProfileHistograms(mePullXvsBetaZmPanel1, newmePullXvsBetaZmPanel1, "panel1, z<0, |beta| (deg)", "pull x", lpull, hpull );
      Float_t refMax = 0.5+1.2*mePullXvsBetaZmPanel1->GetMaximum();
      Float_t newMax = 0.5+1.2*newmePullXvsBetaZmPanel1->GetMaximum();
      if refMax > newMax
      {
          mePullXvsBetaZmPanel1->SetMaximum(refMax);
      }
      else
      {
          mePullXvsBetaZmPanel1->SetMaximum(newMax);
      }      
      mePullXvsBetaZmPanel1->SetName("Reference");
      newmePullXvsBetaZmPanel1->SetName("New Release");
      mePullXvsBetaZmPanel1->Draw("e");
      newmePullXvsBetaZmPanel1->Draw("esameS"); 
      myPV->PVCompute(mePullXvsBetaZmPanel1, newmePullXvsBetaZmPanel1, te, 0.2, 0.2);
      h_pv->SetBinContent(++bin, myPV->getPV());
      gPad->Update();      
      TPaveStats *s49 = (TPaveStats*)mePullXvsBetaZmPanel1->GetListOfFunctions()->FindObject("stats");
      if (s49) {
	s49->SetX1NDC (0.55); //new x start position
	s49->SetX2NDC (0.75); //new x end position
      }

      can_PullXvsBeta->cd(3);
      //gPad->SetLogy();
      SetUpProfileHistograms(mePullXvsBetaZmPanel2, newmePullXvsBetaZmPanel2, "panel2, z<0, |beta| (deg)", "pull x", lpull, hpull );
      Float_t refMax = 0.5+1.2*mePullXvsBetaZmPanel2->GetMaximum();
      Float_t newMax = 0.5+1.2*newmePullXvsBetaZmPanel2->GetMaximum();
      if refMax > newMax
      {
          mePullXvsBetaZmPanel2->SetMaximum(refMax);
      }
      else
      {
          mePullXvsBetaZmPanel2->SetMaximum(newMax);
      }      
      mePullXvsBetaZmPanel2->SetName("Reference");
      newmePullXvsBetaZmPanel2->SetName("New Release");
      mePullXvsBetaZmPanel2->Draw("e");
      newmePullXvsBetaZmPanel2->Draw("esameS"); 
      myPV->PVCompute(mePullXvsBetaZmPanel2, newmePullXvsBetaZmPanel2, te, 0.2, 0.2);
      h_pv->SetBinContent(++bin, myPV->getPV());
      gPad->Update();      
      TPaveStats *s50 = (TPaveStats*)mePullXvsBetaZmPanel2->GetListOfFunctions()->FindObject("stats");
      if (s50) {
	s50->SetX1NDC (0.55); //new x start position
	s50->SetX2NDC (0.75); //new x end position
      }

      can_PullXvsBeta->cd(5);
      //gPad->SetLogy();
      SetUpProfileHistograms(mePullXvsBetaZpPanel1, newmePullXvsBetaZpPanel1, "panel1, z>0, |beta| (deg)", "pull x", lpull, hpull );
      Float_t refMax = 0.5+1.2*mePullXvsBetaZpPanel1->GetMaximum();
      Float_t newMax = 0.5+1.2*newmePullXvsBetaZpPanel1->GetMaximum();
      if refMax > newMax
      {
          mePullXvsBetaZpPanel1->SetMaximum(refMax);
      }
      else
      {
          mePullXvsBetaZpPanel1->SetMaximum(newMax);
      }      
      mePullXvsBetaZpPanel1->SetName("Reference");
      newmePullXvsBetaZpPanel1->SetName("New Release");
      mePullXvsBetaZpPanel1->Draw("e");
      newmePullXvsBetaZpPanel1->Draw("esameS"); 
      myPV->PVCompute(mePullXvsBetaZpPanel1, newmePullXvsBetaZpPanel1, te, 0.2, 0.2 );
      h_pv->SetBinContent(++bin, myPV->getPV());
      gPad->Update();      
      TPaveStats *s51 = (TPaveStats*)mePullXvsBetaZpPanel1->GetListOfFunctions()->FindObject("stats");
      if (s51) {
	s51->SetX1NDC (0.55); //new x start position
	s51->SetX2NDC (0.75); //new x end position
      }

      can_PullXvsBeta->cd(6);
      //gPad->SetLogy();
      SetUpProfileHistograms(mePullXvsBetaZpPanel2, newmePullXvsBetaZpPanel2, "panel2, z>0, |beta| (deg)", "pull x", lpull, hpull );
      Float_t refMax = 0.5+1.2*mePullXvsBetaZpPanel2->GetMaximum();
      Float_t newMax = 0.5+1.2*newmePullXvsBetaZpPanel2->GetMaximum();
      if refMax > newMax
      {
          mePullXvsBetaZpPanel2->SetMaximum(refMax);
      }
      else
      {
          mePullXvsBetaZpPanel2->SetMaximum(newMax);
      }      
      mePullXvsBetaZpPanel2->SetName("Reference");
      newmePullXvsBetaZpPanel2->SetName("New Release");
      mePullXvsBetaZpPanel2->Draw("e");
      newmePullXvsBetaZpPanel2->Draw("esameS"); 
      myPV->PVCompute(mePullXvsBetaZpPanel2, newmePullXvsBetaZpPanel2, te, 0.2, 0.2);
      h_pv->SetBinContent(++bin, myPV->getPV());
      gPad->Update();      
      TPaveStats *s52 = (TPaveStats*)mePullXvsBetaZpPanel2->GetListOfFunctions()->FindObject("stats");
      if (s52) {
	s52->SetX1NDC (0.55); //new x start position
	s52->SetX2NDC (0.75); //new x end position
      }

      can_PullXvsBeta->SaveAs("mePullXvsBeta_compare.eps");
      can_PullXvsBeta->SaveAs("mePullXvsBeta_compare.gif");
    }

  
 if (   1   ) 
    {
      TCanvas* can_WPullXvsAlpha = new TCanvas("can_WPullXvsAlpha", "can_WPullXvsAlpha", 1200, 800);
      can_WPullXvsAlpha->Divide(3,2);
      
      TProfile* meWPullXvsAlphaBarrelNFP;
      TProfile* meWPullXvsAlphaBarrelFP;
      TProfile* meWPullXvsAlphaZmPanel1;
      TProfile* meWPullXvsAlphaZmPanel2;
      TProfile* meWPullXvsAlphaZpPanel1;
      TProfile* meWPullXvsAlphaZpPanel2;
      
      TProfile* newmeWPullXvsAlphaBarrelNFP;
      TProfile* newmeWPullXvsAlphaBarrelFP;
      TProfile* newmeWPullXvsAlphaZmPanel1;
      TProfile* newmeWPullXvsAlphaZmPanel2;
      TProfile* newmeWPullXvsAlphaZpPanel1;
      TProfile* newmeWPullXvsAlphaZpPanel2;
      
      rdir->GetObject("Histograms_all/meWPullXvsAlphaBarrelNonFlippedLadders", meWPullXvsAlphaBarrelNFP  );
      rdir->GetObject("Histograms_all/meWPullXvsAlphaBarrelFlippedLadders"   , meWPullXvsAlphaBarrelFP   );
      rdir->GetObject("Histograms_all/meWPullXvsAlphaZmPanel1", meWPullXvsAlphaZmPanel1);
      rdir->GetObject("Histograms_all/meWPullXvsAlphaZmPanel2", meWPullXvsAlphaZmPanel2);
      rdir->GetObject("Histograms_all/meWPullXvsAlphaZpPanel1", meWPullXvsAlphaZpPanel1);
      rdir->GetObject("Histograms_all/meWPullXvsAlphaZpPanel2", meWPullXvsAlphaZpPanel2);
      
      sdir->GetObject("Histograms_all/meWPullXvsAlphaBarrelNonFlippedLadders", newmeWPullXvsAlphaBarrelNFP  );
      sdir->GetObject("Histograms_all/meWPullXvsAlphaBarrelFlippedLadders"   , newmeWPullXvsAlphaBarrelFP   );
      sdir->GetObject("Histograms_all/meWPullXvsAlphaZmPanel1", newmeWPullXvsAlphaZmPanel1);
      sdir->GetObject("Histograms_all/meWPullXvsAlphaZmPanel2", newmeWPullXvsAlphaZmPanel2);
      sdir->GetObject("Histograms_all/meWPullXvsAlphaZpPanel1", newmeWPullXvsAlphaZpPanel1);
      sdir->GetObject("Histograms_all/meWPullXvsAlphaZpPanel2", newmeWPullXvsAlphaZpPanel2);
      
      TLegend* leg10 = new TLegend(0.3, 0.2, 0.6, 0.4);
      can_WPullXvsAlpha->cd(1);
      //gPad->SetLogy();
      SetUpProfileHistograms(meWPullXvsAlphaBarrelNFP, newmeWPullXvsAlphaBarrelNFP, "non-flipped  ladders, barrel, |alpha| (deg)", "< | pull x | >", lwpull, hwpull, leg10 );
      Float_t refMax = 0.5+1.2*meWPullXvsAlphaBarrelNFP->GetMaximum();
      Float_t newMax = 0.5+1.2*newmeWPullXvsAlphaBarrelNFP->GetMaximum();
      if refMax > newMax
      {
          meWPullXvsAlphaBarrelNFP->SetMaximum(refMax);
      }
      else
      {
          meWPullXvsAlphaBarrelNFP->SetMaximum(newMax);
      }      
      meWPullXvsAlphaBarrelNFP->SetName("Reference");
      newmeWPullXvsAlphaBarrelNFP->SetName("New Release");
      meWPullXvsAlphaBarrelNFP->Draw("e");
      newmeWPullXvsAlphaBarrelNFP->Draw("eSameS"); 
      myPV->PVCompute(meWPullXvsAlphaBarrelNFP, newmeWPullXvsAlphaBarrelNFP, te, 0.3, 0.4  );
      leg10->Draw();
      h_pv->SetBinContent(++bin, myPV->getPV());
      gPad->Update();      
      TPaveStats *s53 = (TPaveStats*)meWPullXvsAlphaBarrelNFP->GetListOfFunctions()->FindObject("stats");
      if (s53) {
	s53->SetX1NDC (0.55); //new x start position
	s53->SetX2NDC (0.75); //new x end position
      }
      can_WPullXvsAlpha->cd(2);
      //gPad->SetLogy();
      SetUpProfileHistograms(meWPullXvsAlphaZmPanel1, newmeWPullXvsAlphaZmPanel1, "panel1, z<0, |alpha| (deg)", "< | pull x | >", lwpull, hwpull );
      Float_t refMax = 0.5+1.2*meWPullXvsAlphaZmPanel1->GetMaximum();
      Float_t newMax = 0.5+1.2*newmeWPullXvsAlphaZmPanel1->GetMaximum();
      if refMax > newMax
      {
          meWPullXvsAlphaZmPanel1->SetMaximum(refMax);
      }
      else
      {
          meWPullXvsAlphaZmPanel1->SetMaximum(newMax);
      }      
      meWPullXvsAlphaZmPanel1->SetName("Reference");
      newmeWPullXvsAlphaZmPanel1->SetName("New Release");
      meWPullXvsAlphaZmPanel1->Draw("e");
      newmeWPullXvsAlphaZmPanel1->Draw("esameS"); 
      myPV->PVCompute(meWPullXvsAlphaZmPanel1, newmeWPullXvsAlphaZmPanel1, te, 0.2, 0.2);
      h_pv->SetBinContent(++bin, myPV->getPV());
      gPad->Update();      
      TPaveStats *s54 = (TPaveStats*)meWPullXvsAlphaZmPanel1->GetListOfFunctions()->FindObject("stats");
      if (s54) {
	s54->SetX1NDC (0.55); //new x start position
	s54->SetX2NDC (0.75); //new x end position
      }
      can_WPullXvsAlpha->cd(3);
      //gPad->SetLogy();
      SetUpProfileHistograms(meWPullXvsAlphaZmPanel2, newmeWPullXvsAlphaZmPanel2, "panel2, z<0, |alpha| (deg)", "< | pull x | >", lwpull, hwpull );
      Float_t refMax = 0.5+1.2*meWPullXvsAlphaZmPanel2->GetMaximum();
      Float_t newMax = 0.5+1.2*newmeWPullXvsAlphaZmPanel2->GetMaximum();
      if refMax > newMax
      {
          meWPullXvsAlphaZmPanel2->SetMaximum(refMax);
      }
      else
      {
          meWPullXvsAlphaZmPanel2->SetMaximum(newMax);
      }      
      meWPullXvsAlphaZmPanel2->SetName("Reference");
      newmeWPullXvsAlphaZmPanel2->SetName("New Release");
      meWPullXvsAlphaZmPanel2->Draw("e");
      newmeWPullXvsAlphaZmPanel2->Draw("esameS"); 
      myPV->PVCompute(meWPullXvsAlphaZmPanel2, newmeWPullXvsAlphaZmPanel2, te, 0.2, 0.2);
      h_pv->SetBinContent(++bin, myPV->getPV());
      gPad->Update();      
      TPaveStats *s55 = (TPaveStats*)meWPullXvsAlphaZmPanel2->GetListOfFunctions()->FindObject("stats");
      if (s55) {
	s55->SetX1NDC (0.55); //new x start position
	s55->SetX2NDC (0.75); //new x end position
      }
      can_WPullXvsAlpha->cd(4);
      //gPad->SetLogy();
      SetUpProfileHistograms(meWPullXvsAlphaBarrelFP, newmeWPullXvsAlphaBarrelFP, "flipped ladders, barrel, |alpha| (deg)", "< | pull x | >", lwpull, hwpull);
      Float_t refMax = 0.5+1.2*meWPullXvsAlphaBarrelFP->GetMaximum();
      Float_t newMax = 0.5+1.2*newmeWPullXvsAlphaBarrelFP->GetMaximum();
      if refMax > newMax
      {
          meWPullXvsAlphaBarrelFP->SetMaximum(refMax);
      }
      else
      {
          meWPullXvsAlphaBarrelFP->SetMaximum(newMax);
      }      
      meWPullXvsAlphaBarrelFP->SetName("Reference");
      newmeWPullXvsAlphaBarrelFP->SetName("New Release");
      meWPullXvsAlphaBarrelFP->Draw("e");
      newmeWPullXvsAlphaBarrelFP->Draw("eSameS"); 
      myPV->PVCompute(meWPullXvsAlphaBarrelFP, newmeWPullXvsAlphaBarrelFP, te, 0.2, 0.2);
      h_pv->SetBinContent(++bin, myPV->getPV());
      gPad->Update();      
      TPaveStats *s56 = (TPaveStats*)meWPullXvsAlphaBarrelFP->GetListOfFunctions()->FindObject("stats");
      if (s56) {
	s56->SetX1NDC (0.55); //new x start position
	s56->SetX2NDC (0.75); //new x end position
      }
      can_WPullXvsAlpha->cd(5);
      //gPad->SetLogy();
      SetUpProfileHistograms(meWPullXvsAlphaZpPanel1, newmeWPullXvsAlphaZpPanel1, "panel1, z>0, |alpha| (deg)", "< | pull x | >", lwpull, hwpull );
      Float_t refMax = 0.5+1.2*meWPullXvsAlphaZpPanel1->GetMaximum();
      Float_t newMax = 0.5+1.2*newmeWPullXvsAlphaZpPanel1->GetMaximum();
      if refMax > newMax
      {
          meWPullXvsAlphaZpPanel1->SetMaximum(refMax);
      }
      else
      {
          meWPullXvsAlphaZpPanel1->SetMaximum(newMax);
      }      
      meWPullXvsAlphaZpPanel1->SetName("Reference");
      newmeWPullXvsAlphaZpPanel1->SetName("New Release");
      meWPullXvsAlphaZpPanel1->Draw("e");
      newmeWPullXvsAlphaZpPanel1->Draw("esameS"); 
      myPV->PVCompute(meWPullXvsAlphaZpPanel1, newmeWPullXvsAlphaZpPanel1, te, 0.2, 0.2);
      h_pv->SetBinContent(++bin, myPV->getPV());
      gPad->Update();      
      TPaveStats *s57 = (TPaveStats*)meWPullXvsAlphaZpPanel1->GetListOfFunctions()->FindObject("stats");
      if (s57) {
	s57->SetX1NDC (0.55); //new x start position
	s57->SetX2NDC (0.75); //new x end position
      }
      can_WPullXvsAlpha->cd(6);
      //gPad->SetLogy();
      SetUpProfileHistograms(meWPullXvsAlphaZpPanel2, newmeWPullXvsAlphaZpPanel2, "panel2, z>0, |alpha| (deg)", "< | pull x | >", lwpull, hwpull );
      Float_t refMax = 0.5+1.2*meWPullXvsAlphaZpPanel2->GetMaximum();
      Float_t newMax = 0.5+1.2*newmeWPullXvsAlphaZpPanel2->GetMaximum();
      if refMax > newMax
      {
          meWPullXvsAlphaZpPanel2->SetMaximum(refMax);
      }
      else
      {
          meWPullXvsAlphaZpPanel2->SetMaximum(newMax);
      }      
      meWPullXvsAlphaZpPanel2->SetName("Reference");
      newmeWPullXvsAlphaZpPanel2->SetName("New Release");
      meWPullXvsAlphaZpPanel2->Draw("e");
      newmeWPullXvsAlphaZpPanel2->Draw("esameS"); 
      myPV->PVCompute(meWPullXvsAlphaZpPanel2, newmeWPullXvsAlphaZpPanel2, te, 0.2, 0.2);
      h_pv->SetBinContent(++bin, myPV->getPV());
      gPad->Update();      
      TPaveStats *s58 = (TPaveStats*)meWPullXvsAlphaZpPanel2->GetListOfFunctions()->FindObject("stats");
      if (s58) {
	s58->SetX1NDC (0.55); //new x start position
	s58->SetX2NDC (0.75); //new x end position
      }
      can_WPullXvsAlpha->SaveAs("meWPullXvsAlpha_compare.eps");
      can_WPullXvsAlpha->SaveAs("meWPullXvsAlpha_compare.gif");
    }
  
 //if (   0   ) 
 if (1) 
 {
    TCanvas* can_PullXvsPhi = new TCanvas("can_PullXvsPhi", "can_PullXvsPhi", 1200, 800);
    can_PullXvsPhi->Divide(3,2);
    
    TProfile* mePullXvsPhiBarrel;
    TProfile* mePullXvsPhiZmPanel1;
    TProfile* mePullXvsPhiZmPanel2;
    TProfile* mePullXvsPhiZpPanel1;
    TProfile* mePullXvsPhiZpPanel2;
    
    TProfile* newmePullXvsPhiBarrel;
    TProfile* newmePullXvsPhiZmPanel1;
    TProfile* newmePullXvsPhiZmPanel2;
    TProfile* newmePullXvsPhiZpPanel1;
    TProfile* newmePullXvsPhiZpPanel2;

    rdir->GetObject("Histograms_all/mePullXvsPhiBarrel"  , mePullXvsPhiBarrel  );
    rdir->GetObject("Histograms_all/mePullXvsPhiZmPanel1", mePullXvsPhiZmPanel1);
    rdir->GetObject("Histograms_all/mePullXvsPhiZmPanel2", mePullXvsPhiZmPanel2);
    rdir->GetObject("Histograms_all/mePullXvsPhiZpPanel1", mePullXvsPhiZpPanel1);
    rdir->GetObject("Histograms_all/mePullXvsPhiZpPanel2", mePullXvsPhiZpPanel2);

    sdir->GetObject("Histograms_all/mePullXvsPhiBarrel"  , newmePullXvsPhiBarrel  ); 
    sdir->GetObject("Histograms_all/mePullXvsPhiZmPanel1", newmePullXvsPhiZmPanel1);
    sdir->GetObject("Histograms_all/mePullXvsPhiZmPanel2", newmePullXvsPhiZmPanel2);
    sdir->GetObject("Histograms_all/mePullXvsPhiZpPanel1", newmePullXvsPhiZpPanel1);
    sdir->GetObject("Histograms_all/mePullXvsPhiZpPanel2", newmePullXvsPhiZpPanel2);
  
    TLegend* leg13 = new TLegend(0.3, 0.2, 0.6, 0.4);
    can_PullXvsPhi->cd(1);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullXvsPhiBarrel, newmePullXvsPhiBarrel, "barrel, phi (deg)", "pull x", lpull, hpull, leg13 );
    Float_t refMax = 0.5+1.2*mePullXvsPhiBarrel->GetMaximum();
    Float_t newMax = 0.5+1.2*newmePullXvsPhiBarrel->GetMaximum();
    if refMax > newMax
    {
        mePullXvsPhiBarrel->SetMaximum(refMax);
    }
    else
    {
        mePullXvsPhiBarrel->SetMaximum(newMax);
    }      
    mePullXvsPhiBarrel->SetName("Reference");
    newmePullXvsPhiBarrel->SetName("New Release");
    mePullXvsPhiBarrel->Draw("e");
    newmePullXvsPhiBarrel->Draw("eSameS"); 
    myPV->PVCompute(mePullXvsPhiBarrel, newmePullXvsPhiBarrel, te, 0.3, 0.4);
    leg13->Draw();
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();      
    TPaveStats *s59 = (TPaveStats*)mePullXvsPhiBarrel->GetListOfFunctions()->FindObject("stats");
    if (s59) {
      s59->SetX1NDC (0.55); //new x start position
      s59->SetX2NDC (0.75); //new x end position
    }
    can_PullXvsPhi->cd(2);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullXvsPhiZmPanel1, newmePullXvsPhiZmPanel1, "panel1, z<0, phi (deg)", "pull x", lpull, hpull );
    Float_t refMax = 0.5+1.2*mePullXvsPhiZmPanel1->GetMaximum();
    Float_t newMax = 0.5+1.2*newmePullXvsPhiZmPanel1->GetMaximum();
    if refMax > newMax
    {
        mePullXvsPhiZmPanel1->SetMaximum(refMax);
    }
    else
    {
        mePullXvsPhiZmPanel1->SetMaximum(newMax);
    }      
    mePullXvsPhiZmPanel1->SetName("Reference");
    newmePullXvsPhiZmPanel1->SetName("New Release");
    mePullXvsPhiZmPanel1->Draw("e");
    newmePullXvsPhiZmPanel1->Draw("esameS"); 
    myPV->PVCompute(mePullXvsPhiZmPanel1, newmePullXvsPhiZmPanel1, te, 0.2, 0.2);
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();      
    TPaveStats *s60 = (TPaveStats*)mePullXvsPhiZmPanel1->GetListOfFunctions()->FindObject("stats");
    if (s60) {
      s60->SetX1NDC (0.55); //new x start position
      s60->SetX2NDC (0.75); //new x end position
    }
    can_PullXvsPhi->cd(3);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullXvsPhiZmPanel2, newmePullXvsPhiZmPanel2, "panel2, z<0, phi (deg)", "pull x", lpull, hpull );
    Float_t refMax = 0.5+1.2*mePullXvsPhiZmPanel2->GetMaximum();
    Float_t newMax = 0.5+1.2*newmePullXvsPhiZmPanel2->GetMaximum();
    if refMax > newMax
    {
        mePullXvsPhiZmPanel2->SetMaximum(refMax);
    }
    else
    {
        mePullXvsPhiZmPanel2->SetMaximum(newMax);
    }      
    mePullXvsPhiZmPanel2->SetName("Reference");
    newmePullXvsPhiZmPanel2->SetName("New Release");
    mePullXvsPhiZmPanel2->Draw("e");
    newmePullXvsPhiZmPanel2->Draw("esameS"); 
    myPV->PVCompute(mePullXvsPhiZmPanel2, newmePullXvsPhiZmPanel2, te, 0.2, 0.2 );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();      
    TPaveStats *s61 = (TPaveStats*)mePullXvsPhiZmPanel2->GetListOfFunctions()->FindObject("stats");
    if (s61) {
      s61->SetX1NDC (0.55); //new x start position
      s61->SetX2NDC (0.75); //new x end position
    }
    can_PullXvsPhi->cd(5);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullXvsPhiZpPanel1, newmePullXvsPhiZpPanel1, "panel1, z>0, phi (deg)", "pull x", lpull, hpull );
    Float_t refMax = 0.5+1.2*mePullXvsPhiZpPanel1->GetMaximum();
    Float_t newMax = 0.5+1.2*newmePullXvsPhiZpPanel1->GetMaximum();
    if refMax > newMax
    {
        mePullXvsPhiZpPanel1->SetMaximum(refMax);
    }
    else
    {
        mePullXvsPhiZpPanel1->SetMaximum(newMax);
    }      
    mePullXvsPhiZpPanel1->SetName("Reference");
    newmePullXvsPhiZpPanel1->SetName("New Release");
    mePullXvsPhiZpPanel1->Draw("e");
    newmePullXvsPhiZpPanel1->Draw("esameS"); 
    myPV->PVCompute(mePullXvsPhiZpPanel1, newmePullXvsPhiZpPanel1, te, 0.2, 0.2 );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();      
    TPaveStats *s62 = (TPaveStats*)mePullXvsPhiZpPanel1->GetListOfFunctions()->FindObject("stats");
    if (s62) {
      s62->SetX1NDC (0.55); //new x start position
      s62->SetX2NDC (0.75); //new x end position
    }
    can_PullXvsPhi->cd(6);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullXvsPhiZpPanel2, newmePullXvsPhiZpPanel2, "panel2, z>0, phi (deg)", "pull x" , lpull, hpull);
    Float_t refMax = 0.5+1.2*mePullXvsPhiZpPanel2->GetMaximum();
    Float_t newMax = 0.5+1.2*newmePullXvsPhiZpPanel2->GetMaximum();
    if refMax > newMax
    {
        mePullXvsPhiZpPanel2->SetMaximum(refMax);
    }
    else
    {
        mePullXvsPhiZpPanel2->SetMaximum(newMax);
    }      
    mePullXvsPhiZpPanel2->SetName("Reference");
    newmePullXvsPhiZpPanel2->SetName("New Release");
    mePullXvsPhiZpPanel2->Draw("e");
    newmePullXvsPhiZpPanel2->Draw("esameS"); 
    myPV->PVCompute(mePullXvsPhiZpPanel2, newmePullXvsPhiZpPanel2, te, 0.2, 0.2 );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();      
    TPaveStats *s63 = (TPaveStats*)mePullXvsPhiZpPanel2->GetListOfFunctions()->FindObject("stats");
    if (s63) {
      s63->SetX1NDC (0.55); //new x start position
      s63->SetX2NDC (0.75); //new x end position
    }
    can_PullXvsPhi->SaveAs("mePullXvsPhi_compare.eps");
    can_PullXvsPhi->SaveAs("mePullXvsPhi_compare.gif");
  }

 //if (   0   ) 
 if (1) 
 {
    TCanvas* can_PullYvsAlpha = new TCanvas("can_PullYvsAlpha", "can_PullYvsAlpha", 1200, 800);
    can_PullYvsAlpha->Divide(3,2);
    
    TProfile* mePullYvsAlphaBarrel;
    TProfile* mePullYvsAlphaZmPanel1;
    TProfile* mePullYvsAlphaZmPanel2;
    TProfile* mePullYvsAlphaZpPanel1;
    TProfile* mePullYvsAlphaZpPanel2;
    
    TProfile* newmePullYvsAlphaBarrel;
    TProfile* newmePullYvsAlphaZmPanel1;
    TProfile* newmePullYvsAlphaZmPanel2;
    TProfile* newmePullYvsAlphaZpPanel1;
    TProfile* newmePullYvsAlphaZpPanel2;

    rdir->GetObject("Histograms_all/mePullYvsAlphaBarrel"  , mePullYvsAlphaBarrel  );
    rdir->GetObject("Histograms_all/mePullYvsAlphaZmPanel1", mePullYvsAlphaZmPanel1);
    rdir->GetObject("Histograms_all/mePullYvsAlphaZmPanel2", mePullYvsAlphaZmPanel2);
    rdir->GetObject("Histograms_all/mePullYvsAlphaZpPanel1", mePullYvsAlphaZpPanel1);
    rdir->GetObject("Histograms_all/mePullYvsAlphaZpPanel2", mePullYvsAlphaZpPanel2);

    sdir->GetObject("Histograms_all/mePullYvsAlphaBarrel"  , newmePullYvsAlphaBarrel  ); 
    sdir->GetObject("Histograms_all/mePullYvsAlphaZmPanel1", newmePullYvsAlphaZmPanel1);
    sdir->GetObject("Histograms_all/mePullYvsAlphaZmPanel2", newmePullYvsAlphaZmPanel2);
    sdir->GetObject("Histograms_all/mePullYvsAlphaZpPanel1", newmePullYvsAlphaZpPanel1);
    sdir->GetObject("Histograms_all/mePullYvsAlphaZpPanel2", newmePullYvsAlphaZpPanel2);
  
    TLegend* leg14 = new TLegend(0.3, 0.2, 0.6, 0.4);
    can_PullYvsAlpha->cd(1);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsAlphaBarrel, newmePullYvsAlphaBarrel, "barrel, |alpha| (deg)", "pull y", lpull, hpull, leg14 );
    Float_t refMax = 1.4*(0.5+mePullYvsAlphaBarrel->GetMaximum());
    Float_t newMax = 1.4*(0.5+newmePullYvsAlphaBarrel->GetMaximum());
    if refMax > newMax
    {
        mePullYvsAlphaBarrel->SetMaximum(refMax);
    }
    else
    {
        mePullYvsAlphaBarrel->SetMaximum(newMax);
    }      
    mePullYvsAlphaBarrel->SetName("Reference");
    newmePullYvsAlphaBarrel->SetName("New Release");
    mePullYvsAlphaBarrel->Draw("e");
    newmePullYvsAlphaBarrel->Draw("eSameS"); 
    myPV->PVCompute(mePullYvsAlphaBarrel, newmePullYvsAlphaBarrel, te, 0.3, 0.4 );
    leg14->Draw();
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();      
    TPaveStats *s64 = (TPaveStats*)mePullYvsAlphaBarrel->GetListOfFunctions()->FindObject("stats");
    if (s64) {
      s64->SetX1NDC (0.55); //new x start position
      s64->SetX2NDC (0.75); //new x end position
    }
    can_PullYvsAlpha->cd(2);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsAlphaZmPanel1, newmePullYvsAlphaZmPanel1, "panel1, z<0, |alpha| (deg)", "pull y", lpull, hpull );
    Float_t refMax = 1.4*(0.5+mePullYvsAlphaZmPanel1->GetMaximum());
    Float_t newMax = 1.4*(0.5+newmePullYvsAlphaZmPanel1->GetMaximum());
    if refMax > newMax
    {
        mePullYvsAlphaZmPanel1->SetMaximum(refMax);
    }
    else
    {
        mePullYvsAlphaZmPanel1->SetMaximum(newMax);
    }      
    mePullYvsAlphaZmPanel1->SetName("Reference");
    newmePullYvsAlphaZmPanel1->SetName("New Release");
    mePullYvsAlphaZmPanel1->Draw("e");
    newmePullYvsAlphaZmPanel1->Draw("esameS"); 
    myPV->PVCompute(mePullYvsAlphaZmPanel1, newmePullYvsAlphaZmPanel1, te, 0.2, 0.2);
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();      
    TPaveStats *s138 = (TPaveStats*)mePullYvsAlphaZmPanel1->GetListOfFunctions()->FindObject("stats");
    if (s138) {
    	s138->SetX1NDC (0.55); //new x start position
    	s138->SetX2NDC (0.75); //new x end position
    }
    can_PullYvsAlpha->cd(3);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsAlphaZmPanel2, newmePullYvsAlphaZmPanel2, "panel2, z<0, |alpha| (deg)", "pull y", lpull, hpull );
    Float_t refMax = 1.4*(0.5+mePullYvsAlphaZmPanel2->GetMaximum());
    Float_t newMax = 1.4*(0.5+newmePullYvsAlphaZmPanel2->GetMaximum());
    if refMax > newMax
    {
        mePullYvsAlphaZmPanel2->SetMaximum(refMax);
    }
    else
    {
        mePullYvsAlphaZmPanel2->SetMaximum(newMax);
    }      
    mePullYvsAlphaZmPanel2->SetName("Reference");
    newmePullYvsAlphaZmPanel2->SetName("New Release");
    mePullYvsAlphaZmPanel2->Draw("e");
    newmePullYvsAlphaZmPanel2->Draw("esameS"); 
    myPV->PVCompute(mePullYvsAlphaZmPanel2, newmePullYvsAlphaZmPanel2, te, 0.2, 0.2 );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();      
    TPaveStats *s65 = (TPaveStats*)mePullYvsAlphaZmPanel2->GetListOfFunctions()->FindObject("stats");
    if (s65) {
      s65->SetX1NDC (0.55); //new x start position
      s65->SetX2NDC (0.75); //new x end position
    }
    can_PullYvsAlpha->cd(5);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsAlphaZpPanel1, newmePullYvsAlphaZpPanel1, "panel1, z>0, |alpha| (deg)", "pull y", lpull, hpull );
    Float_t refMax = 1.4*(0.5+mePullYvsAlphaZpPanel1->GetMaximum());
    Float_t newMax = 1.4*(0.5+newmePullYvsAlphaZpPanel1->GetMaximum());
    if refMax > newMax
    {
        mePullYvsAlphaZpPanel1->SetMaximum(refMax);
    }
    else
    {
        mePullYvsAlphaZpPanel1->SetMaximum(newMax);
    }      
    mePullYvsAlphaZpPanel1->SetName("Reference");
    newmePullYvsAlphaZpPanel1->SetName("New Release");
    mePullYvsAlphaZpPanel1->Draw("e");
    newmePullYvsAlphaZpPanel1->Draw("esameS"); 
    myPV->PVCompute(mePullYvsAlphaZpPanel1, newmePullYvsAlphaZpPanel1, te, 0.2, 0.2 );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();      
    TPaveStats *s66 = (TPaveStats*)mePullYvsAlphaZpPanel1->GetListOfFunctions()->FindObject("stats");
    if (s66) {
      s66->SetX1NDC (0.55); //new x start position
      s66->SetX2NDC (0.75); //new x end position
    }
    can_PullYvsAlpha->cd(6);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsAlphaZpPanel2, newmePullYvsAlphaZpPanel2, "panel2, z>0, |alpha| (deg)", "pull y" , lpull, hpull);
    Float_t refMax = 1.4*(0.5+mePullYvsAlphaZpPanel2->GetMaximum());
    Float_t newMax = 1.4*(0.5+newmePullYvsAlphaZpPanel2->GetMaximum());
    if refMax > newMax
    {
        mePullYvsAlphaZpPanel2->SetMaximum(refMax);
    }
    else
    {
        mePullYvsAlphaZpPanel2->SetMaximum(newMax);
    }      
    mePullYvsAlphaZpPanel2->SetName("Reference");
    newmePullYvsAlphaZpPanel2->SetName("New Release");
    mePullYvsAlphaZpPanel2->Draw("e");
    newmePullYvsAlphaZpPanel2->Draw("esameS"); 
    myPV->PVCompute(mePullYvsAlphaZpPanel2, newmePullYvsAlphaZpPanel2, te, 0.2, 0.2 );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();      
    TPaveStats *s67 = (TPaveStats*)mePullYvsAlphaZpPanel2->GetListOfFunctions()->FindObject("stats");
    if (s67) {
      s67->SetX1NDC (0.55); //new x start position
      s67->SetX2NDC (0.75); //new x end position
    }
    can_PullYvsAlpha->SaveAs("mePullYvsAlpha_compare.eps");
    can_PullYvsAlpha->SaveAs("mePullYvsAlpha_compare.gif");
  }

//if (   0   ) 
 if (   1   ) 
   {
    TCanvas* can_PullYvsBeta = new TCanvas("can_PullYvsBeta", "can_PullYvsBeta", 1200, 800);
    can_PullYvsBeta->Divide(3,2);
    
    TProfile* mePullYvsBetaBarrel;
    TProfile* mePullYvsBetaZmPanel1;
    TProfile* mePullYvsBetaZmPanel2;
    TProfile* mePullYvsBetaZpPanel1;
    TProfile* mePullYvsBetaZpPanel2;
    
    TProfile* newmePullYvsBetaBarrel;
    TProfile* newmePullYvsBetaZmPanel1;
    TProfile* newmePullYvsBetaZmPanel2;
    TProfile* newmePullYvsBetaZpPanel1;
    TProfile* newmePullYvsBetaZpPanel2;

    rdir->GetObject("Histograms_all/mePullYvsBetaBarrel"  , mePullYvsBetaBarrel  );
    rdir->GetObject("Histograms_all/mePullYvsBetaZmPanel1", mePullYvsBetaZmPanel1);
    rdir->GetObject("Histograms_all/mePullYvsBetaZmPanel2", mePullYvsBetaZmPanel2);
    rdir->GetObject("Histograms_all/mePullYvsBetaZpPanel1", mePullYvsBetaZpPanel1);
    rdir->GetObject("Histograms_all/mePullYvsBetaZpPanel2", mePullYvsBetaZpPanel2);

    sdir->GetObject("Histograms_all/mePullYvsBetaBarrel"  , newmePullYvsBetaBarrel  ); 
    sdir->GetObject("Histograms_all/mePullYvsBetaZmPanel1", newmePullYvsBetaZmPanel1);
    sdir->GetObject("Histograms_all/mePullYvsBetaZmPanel2", newmePullYvsBetaZmPanel2);
    sdir->GetObject("Histograms_all/mePullYvsBetaZpPanel1", newmePullYvsBetaZpPanel1);
    sdir->GetObject("Histograms_all/mePullYvsBetaZpPanel2", newmePullYvsBetaZpPanel2);
  
    TLegend* leg15 = new TLegend(0.3, 0.2, 0.6, 0.4);
    can_PullYvsBeta->cd(1);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsBetaBarrel, newmePullYvsBetaBarrel, "barrel, |beta| (deg)", "pull y", lpull, hpull, leg15 );
    Float_t refMax = 0.5+1.2*mePullYvsBetaBarrel->GetMaximum();
    Float_t newMax = 0.5+1.2*newmePullYvsBetaBarrel->GetMaximum();
    if refMax > newMax
    {
        mePullYvsBetaBarrel->SetMaximum(refMax);
    }
    else
    {
        mePullYvsBetaBarrel->SetMaximum(newMax);
    }      
    mePullYvsBetaBarrel->SetName("Reference");
    newmePullYvsBetaBarrel->SetName("New Release");
    mePullYvsBetaBarrel->Draw("e");
    newmePullYvsBetaBarrel->Draw("esameS"); 
    myPV->PVCompute(mePullYvsBetaBarrel, newmePullYvsBetaBarrel, te, 0.3, 0.4);
    leg15->Draw();
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();      
    TPaveStats *s68 = (TPaveStats*)mePullYvsBetaBarrel->GetListOfFunctions()->FindObject("stats");
    if (s68) {
      s68->SetX1NDC (0.55); //new x start position
      s68->SetX2NDC (0.75); //new x end position
    }
    can_PullYvsBeta->cd(2);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsBetaZmPanel1, newmePullYvsBetaZmPanel1, "panel1, z<0, |beta| (deg)", "pull y", lpull, hpull );
    Float_t refMax = 0.5+1.2*mePullYvsBetaZmPanel1->GetMaximum();
    Float_t newMax = 0.5+1.2*newmePullYvsBetaZmPanel1->GetMaximum();
    if refMax > newMax
    {
        mePullYvsBetaZmPanel1->SetMaximum(refMax);
    }
    else
    {
        mePullYvsBetaZmPanel1->SetMaximum(newMax);
    }      
    mePullYvsBetaZmPanel1->SetName("Reference");
    newmePullYvsBetaZmPanel1->SetName("New Release");
    mePullYvsBetaZmPanel1->Draw("e");
    newmePullYvsBetaZmPanel1->Draw("esameS"); 
    myPV->PVCompute(mePullYvsBetaZmPanel1, newmePullYvsBetaZmPanel1, te, 0.2, 0.2 );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();      
    TPaveStats *s69 = (TPaveStats*)mePullYvsBetaZmPanel1->GetListOfFunctions()->FindObject("stats");
    if (s69) {
      s69->SetX1NDC (0.55); //new x start position
      s69->SetX2NDC (0.75); //new x end position
    }
    can_PullYvsBeta->cd(3);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsBetaZmPanel2, newmePullYvsBetaZmPanel2, "panel2, z<0, |beta| (deg)", "pull y", lpull, hpull );
    Float_t refMax = 0.5+1.2*mePullYvsBetaZmPanel2->GetMaximum();
    Float_t newMax = 0.5+1.2*newmePullYvsBetaZmPanel2->GetMaximum();
    if refMax > newMax
    {
        mePullYvsBetaZmPanel2->SetMaximum(refMax);
    }
    else
    {
        mePullYvsBetaZmPanel2->SetMaximum(newMax);
    }      
    mePullYvsBetaZmPanel2->SetName("Reference");
    newmePullYvsBetaZmPanel2->SetName("New Release");
    mePullYvsBetaZmPanel2->Draw("e");
    newmePullYvsBetaZmPanel2->Draw("esameS"); 
    myPV->PVCompute(mePullYvsBetaZmPanel2, newmePullYvsBetaZmPanel2, te, 0.2, 0.2 );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();      
    TPaveStats *s70 = (TPaveStats*)mePullYvsBetaZmPanel2->GetListOfFunctions()->FindObject("stats");
    if (s70) {
      s70->SetX1NDC (0.55); //new x start position
      s70->SetX2NDC (0.75); //new x end position
    }
    can_PullYvsBeta->cd(5);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsBetaZpPanel1, newmePullYvsBetaZpPanel1, "panel1, z>0, |beta| (deg)", "pull y", lpull, hpull );
    Float_t refMax = 0.5+1.2*mePullYvsBetaZpPanel1->GetMaximum();
    Float_t newMax = 0.5+1.2*newmePullYvsBetaZpPanel1->GetMaximum();
    if refMax > newMax
    {
        mePullYvsBetaZpPanel1->SetMaximum(refMax);
    }
    else
    {
        mePullYvsBetaZpPanel1->SetMaximum(newMax);
    }      
    mePullYvsBetaZpPanel1->SetName("Reference");
    newmePullYvsBetaZpPanel1->SetName("New Release");
    mePullYvsBetaZpPanel1->Draw("e");
    newmePullYvsBetaZpPanel1->Draw("esameS"); 
    myPV->PVCompute(mePullYvsBetaZpPanel1, newmePullYvsBetaZpPanel1, te, 0.2, 0.2 );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();      
    TPaveStats *s71 = (TPaveStats*)mePullYvsBetaZpPanel1->GetListOfFunctions()->FindObject("stats");
    if (s71) {
      s71->SetX1NDC (0.55); //new x start position
      s71->SetX2NDC (0.75); //new x end position
    }
    can_PullYvsBeta->cd(6);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsBetaZpPanel2, newmePullYvsBetaZpPanel2, "panel2, z>0, |beta| (deg)", "pull y", lpull, hpull );
    Float_t refMax = 0.5+1.2*mePullYvsBetaZpPanel2->GetMaximum();
    Float_t newMax = 0.5+1.2*newmePullYvsBetaZpPanel2->GetMaximum();
    if refMax > newMax
    {
        mePullYvsBetaZpPanel2->SetMaximum(refMax);
    }
    else
    {
        mePullYvsBetaZpPanel2->SetMaximum(newMax);
    }      
    mePullYvsBetaZpPanel2->SetName("Reference");
    newmePullYvsBetaZpPanel2->SetName("New Release");
    mePullYvsBetaZpPanel2->Draw("e");
    newmePullYvsBetaZpPanel2->Draw("esameS"); 
    myPV->PVCompute(mePullYvsBetaZpPanel2, newmePullYvsBetaZpPanel2, te, 0.2, 0.2 );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();      
    TPaveStats *s72 = (TPaveStats*)mePullYvsBetaZpPanel2->GetListOfFunctions()->FindObject("stats");
    if (s72) {
      s72->SetX1NDC (0.55); //new x start position
      s72->SetX2NDC (0.75); //new x end position
    }
    can_PullYvsBeta->SaveAs("mePullYvsBeta_compare.eps");
    can_PullYvsBeta->SaveAs("mePullYvsBeta_compare.gif");
  }


if (   1   ) 
  {
    TCanvas* can_WPullYvsBeta = new TCanvas("can_WPullYvsBeta", "can_WPullYvsBeta", 1200, 800);
    can_WPullYvsBeta->Divide(3,2);
    
    TProfile* meWPullYvsBetaBarrelNFP;
    TProfile* meWPullYvsBetaBarrelFP;
    TProfile* meWPullYvsBetaZmPanel1;
    TProfile* meWPullYvsBetaZmPanel2;
    TProfile* meWPullYvsBetaZpPanel1;
    TProfile* meWPullYvsBetaZpPanel2;
    
    TProfile* newmeWPullYvsBetaBarrelNFP;
    TProfile* newmeWPullYvsBetaBarrelFP;
    TProfile* newmeWPullYvsBetaBarrel;
    TProfile* newmeWPullYvsBetaZmPanel1;
    TProfile* newmeWPullYvsBetaZmPanel2;
    TProfile* newmeWPullYvsBetaZpPanel1;
    TProfile* newmeWPullYvsBetaZpPanel2;

    rdir->GetObject("Histograms_all/meWPullYvsBetaBarrelNonFlippedLadders", meWPullYvsBetaBarrelNFP  );
    rdir->GetObject("Histograms_all/meWPullYvsBetaBarrelFlippedLadders"   , meWPullYvsBetaBarrelFP  );
    rdir->GetObject("Histograms_all/meWPullYvsBetaZmPanel1", meWPullYvsBetaZmPanel1);
    rdir->GetObject("Histograms_all/meWPullYvsBetaZmPanel2", meWPullYvsBetaZmPanel2);
    rdir->GetObject("Histograms_all/meWPullYvsBetaZpPanel1", meWPullYvsBetaZpPanel1);
    rdir->GetObject("Histograms_all/meWPullYvsBetaZpPanel2", meWPullYvsBetaZpPanel2);

    sdir->GetObject("Histograms_all/meWPullYvsBetaBarrelNonFlippedLadders", newmeWPullYvsBetaBarrelNFP  );
    sdir->GetObject("Histograms_all/meWPullYvsBetaBarrelFlippedLadders"   , newmeWPullYvsBetaBarrelFP  );
    sdir->GetObject("Histograms_all/meWPullYvsBetaZmPanel1", newmeWPullYvsBetaZmPanel1);
    sdir->GetObject("Histograms_all/meWPullYvsBetaZmPanel2", newmeWPullYvsBetaZmPanel2);
    sdir->GetObject("Histograms_all/meWPullYvsBetaZpPanel1", newmeWPullYvsBetaZpPanel1);
    sdir->GetObject("Histograms_all/meWPullYvsBetaZpPanel2", newmeWPullYvsBetaZpPanel2);
  
    TLegend* leg15 = new TLegend(0.3, 0.2, 0.6, 0.4);
    can_WPullYvsBeta->cd(1);
    //gPad->SetLogy();
    SetUpProfileHistograms(meWPullYvsBetaBarrelNFP, newmeWPullYvsBetaBarrelNFP, "non-flipped ladders, barrel, |beta| (deg)", "< | pull y | > ", lwpull, hwpull, leg15 );
    Float_t refMax = 0.5+1.2*meWPullYvsBetaBarrelNFP->GetMaximum();
    Float_t newMax = 0.5+1.2*newmeWPullYvsBetaBarrelNFP->GetMaximum();
    if refMax > newMax
    {
        meWPullYvsBetaBarrelNFP->SetMaximum(refMax);
    }
    else
    {
        meWPullYvsBetaBarrelNFP->SetMaximum(newMax);
    }      
    meWPullYvsBetaBarrelNFP->SetName("Reference");
    newmeWPullYvsBetaBarrelNFP->SetName("New Release");
    meWPullYvsBetaBarrelNFP->Draw("e");
    newmeWPullYvsBetaBarrelNFP->Draw("eSameS"); 
    myPV->PVCompute(meWPullYvsBetaBarrelNFP, newmeWPullYvsBetaBarrelNFP, te );
    leg15->Draw();
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();    
    TPaveStats *s73 = (TPaveStats*)meWPullYvsBetaBarrelNFP->GetListOfFunctions()->FindObject("stats");
    if (s73) {
      s73->SetX1NDC (0.55); //new x start position
      s73->SetX2NDC (0.75); //new x end position
    }
    can_WPullYvsBeta->cd(2);
    //gPad->SetLogy();
    SetUpProfileHistograms(meWPullYvsBetaZmPanel1, newmeWPullYvsBetaZmPanel1, "panel1, z<0, |beta| (deg)", "< | pull y | > ", lwpull, hwpull );
    Float_t refMax = 0.5+1.2*meWPullYvsBetaZmPanel1->GetMaximum();
    Float_t newMax = 0.5+1.2*newmeWPullYvsBetaZmPanel1->GetMaximum();
    if refMax > newMax
    {
        meWPullYvsBetaZmPanel1->SetMaximum(refMax);
    }
    else
    {
        meWPullYvsBetaZmPanel1->SetMaximum(newMax);
    }      
    meWPullYvsBetaZmPanel1->SetName("Reference");
    newmeWPullYvsBetaZmPanel1->SetName("New Release");
    meWPullYvsBetaZmPanel1->Draw("e");
    newmeWPullYvsBetaZmPanel1->Draw("esameS"); 
    myPV->PVCompute(meWPullYvsBetaZmPanel1, newmeWPullYvsBetaZmPanel1, te, 0.2, 0.2 );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();    
    TPaveStats *s74 = (TPaveStats*)meWPullYvsBetaZmPanel1->GetListOfFunctions()->FindObject("stats");
    if (s74) {
      s74->SetX1NDC (0.55); //new x start position
      s74->SetX2NDC (0.75); //new x end position
    }
    can_WPullYvsBeta->cd(3);
    //gPad->SetLogy();
    SetUpProfileHistograms(meWPullYvsBetaZmPanel2, newmeWPullYvsBetaZmPanel2, "panel2, z<0, |beta| (deg)", "< | pull y | > ", lwpull, hwpull );
    Float_t refMax = 0.5+1.2*meWPullYvsBetaZmPanel2->GetMaximum();
    Float_t newMax = 0.5+1.2*newmeWPullYvsBetaZmPanel2->GetMaximum();
    if refMax > newMax
    {
        meWPullYvsBetaZmPanel2->SetMaximum(refMax);
    }
    else
    {
        meWPullYvsBetaZmPanel2->SetMaximum(newMax);
    }      
    meWPullYvsBetaZmPanel2->SetName("Reference");
    newmeWPullYvsBetaZmPanel2->SetName("New Release");
    meWPullYvsBetaZmPanel2->Draw("e");
    newmeWPullYvsBetaZmPanel2->Draw("esameS"); 
    myPV->PVCompute(meWPullYvsBetaZmPanel2, newmeWPullYvsBetaZmPanel2, te, 0.2, 0.2 );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();    
    TPaveStats *s75 = (TPaveStats*)meWPullYvsBetaZmPanel2->GetListOfFunctions()->FindObject("stats");
    if (s75) {
      s75->SetX1NDC (0.55); //new x start position
      s75->SetX2NDC (0.75); //new x end position
    }
    can_WPullYvsBeta->cd(4);
    //gPad->SetLogy();
    SetUpProfileHistograms(meWPullYvsBetaBarrelFP, newmeWPullYvsBetaBarrelFP, "flipped ladders, barrel, |beta| (deg)", "< | pull y | > ", lwpull, hwpull);
    Float_t refMax = 0.5+1.2*meWPullYvsBetaBarrelFP->GetMaximum();
    Float_t newMax = 0.5+1.2*newmeWPullYvsBetaBarrelFP->GetMaximum();
    if refMax > newMax
    {
        meWPullYvsBetaBarrelFP->SetMaximum(refMax);
    }
    else
    {
        meWPullYvsBetaBarrelFP->SetMaximum(newMax);
    }      
    meWPullYvsBetaBarrelFP->SetName("Reference");
    newmeWPullYvsBetaBarrelFP->SetName("New Release");
    meWPullYvsBetaBarrelFP->Draw("e");
    newmeWPullYvsBetaBarrelFP->Draw("eSameS"); 
    myPV->PVCompute(meWPullYvsBetaBarrelFP, newmeWPullYvsBetaBarrelFP, te );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();    
    TPaveStats *s76 = (TPaveStats*)meWPullYvsBetaBarrelFP->GetListOfFunctions()->FindObject("stats");
    if (s76) {
      s76->SetX1NDC (0.55); //new x start position
      s76->SetX2NDC (0.75); //new x end position
    }
    can_WPullYvsBeta->cd(5);
    //gPad->SetLogy();
    SetUpProfileHistograms(meWPullYvsBetaZpPanel1, newmeWPullYvsBetaZpPanel1, "panel1, z>0, |beta| (deg)", "< | pull y | > ", lwpull, hwpull );
    Float_t refMax = 0.5+1.2*meWPullYvsBetaZpPanel1->GetMaximum();
    Float_t newMax = 0.5+1.2*newmeWPullYvsBetaZpPanel1->GetMaximum();
    if refMax > newMax
    {
        meWPullYvsBetaZpPanel1->SetMaximum(refMax);
    }
    else
    {
        meWPullYvsBetaZpPanel1->SetMaximum(newMax);
    }      
    meWPullYvsBetaZpPanel1->SetName("Reference");
    newmeWPullYvsBetaZpPanel1->SetName("New Release");
    meWPullYvsBetaZpPanel1->Draw("e");
    newmeWPullYvsBetaZpPanel1->Draw("esameS"); 
    myPV->PVCompute(meWPullYvsBetaZpPanel1, newmeWPullYvsBetaZpPanel1, te, 0.2, 0.2 );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();    
    TPaveStats *s77 = (TPaveStats*)meWPullYvsBetaZpPanel1->GetListOfFunctions()->FindObject("stats");
    if (s77) {
      s77->SetX1NDC (0.55); //new x start position
      s77->SetX2NDC (0.75); //new x end position
    }
    can_WPullYvsBeta->cd(6);
    //gPad->SetLogy();
    SetUpProfileHistograms(meWPullYvsBetaZpPanel2, newmeWPullYvsBetaZpPanel2, "panel2, z>0, |beta| (deg)", "< | pull y | > ", lwpull, hwpull );
    Float_t refMax = 0.5+1.2*meWPullYvsBetaZpPanel2->GetMaximum();
    Float_t newMax = 0.5+1.2*newmeWPullYvsBetaZpPanel2->GetMaximum();
    if refMax > newMax
    {
        meWPullYvsBetaZpPanel2->SetMaximum(refMax);
    }
    else
    {
        meWPullYvsBetaZpPanel2->SetMaximum(newMax);
    }      
    meWPullYvsBetaZpPanel2->SetName("Reference");
    newmeWPullYvsBetaZpPanel2->SetName("New Release");
    meWPullYvsBetaZpPanel2->Draw("e");
    newmeWPullYvsBetaZpPanel2->Draw("esameS"); 
    myPV->PVCompute(meWPullYvsBetaZpPanel2, newmeWPullYvsBetaZpPanel2, te, 0.2, 0.2 );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s78 = (TPaveStats*)meWPullYvsBetaZpPanel2->GetListOfFunctions()->FindObject("stats");
    if (s78) {
      s78->SetX1NDC (0.55); //new x start position
      s78->SetX2NDC (0.75); //new x end position
    }
    can_WPullYvsBeta->SaveAs("meWPullYvsBeta_compare.eps");
    can_WPullYvsBeta->SaveAs("meWPullYvsBeta_compare.gif");
  }



//if (   0   ) 
 if (1) 
   {
    TCanvas* can_PullYvsEta = new TCanvas("can_PullYvsEta", "can_PullYvsEta", 1200, 800);
    can_PullYvsEta->Divide(3,2);
    
    TProfile* mePullYvsEtaBarrel;
    TProfile* mePullYvsEtaZmPanel1;
    TProfile* mePullYvsEtaZmPanel2;
    TProfile* mePullYvsEtaZpPanel1;
    TProfile* mePullYvsEtaZpPanel2;
    
    TProfile* newmePullYvsEtaBarrel;
    TProfile* newmePullYvsEtaZmPanel1;
    TProfile* newmePullYvsEtaZmPanel2;
    TProfile* newmePullYvsEtaZpPanel1;
    TProfile* newmePullYvsEtaZpPanel2;

    rdir->GetObject("Histograms_all/mePullYvsEtaBarrel"  , mePullYvsEtaBarrel  );
    rdir->GetObject("Histograms_all/mePullYvsEtaZmPanel1", mePullYvsEtaZmPanel1);
    rdir->GetObject("Histograms_all/mePullYvsEtaZmPanel2", mePullYvsEtaZmPanel2);
    rdir->GetObject("Histograms_all/mePullYvsEtaZpPanel1", mePullYvsEtaZpPanel1);
    rdir->GetObject("Histograms_all/mePullYvsEtaZpPanel2", mePullYvsEtaZpPanel2);

    sdir->GetObject("Histograms_all/mePullYvsEtaBarrel"  , newmePullYvsEtaBarrel  ); 
    sdir->GetObject("Histograms_all/mePullYvsEtaZmPanel1", newmePullYvsEtaZmPanel1);
    sdir->GetObject("Histograms_all/mePullYvsEtaZmPanel2", newmePullYvsEtaZmPanel2);
    sdir->GetObject("Histograms_all/mePullYvsEtaZpPanel1", newmePullYvsEtaZpPanel1);
    sdir->GetObject("Histograms_all/mePullYvsEtaZpPanel2", newmePullYvsEtaZpPanel2);
  
    TLegend* leg16 = new TLegend(0.3, 0.2, 0.6, 0.4);
    can_PullYvsEta->cd(1);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsEtaBarrel, newmePullYvsEtaBarrel, "barrel, eta", "pull y", lpull, hpull, leg16 );
    Float_t refMax = 0.5+1.4*mePullYvsEtaBarrel->GetMaximum();
    Float_t newMax = 0.5+1.4*newmePullYvsEtaBarrel->GetMaximum();
    if refMax > newMax
    {
        mePullYvsEtaBarrel->SetMaximum(refMax);
    }
    else
    {
        mePullYvsEtaBarrel->SetMaximum(newMax);
    }      
    mePullYvsEtaBarrel->SetName("Reference");
    newmePullYvsEtaBarrel->SetName("New Release");
    mePullYvsEtaBarrel->Draw("e");
    newmePullYvsEtaBarrel->Draw("eSameS"); 
    myPV->PVCompute(mePullYvsEtaBarrel, newmePullYvsEtaBarrel, te, 0.3, 0.4 );
    leg16->Draw();
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s79 = (TPaveStats*)mePullYvsEtaBarrel->GetListOfFunctions()->FindObject("stats");
    if (s79) {
      s79->SetX1NDC (0.55); //new x start position
      s79->SetX2NDC (0.75); //new x end position
    }
    can_PullYvsEta->cd(2);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsEtaZmPanel1, newmePullYvsEtaZmPanel1, "panel1, z<0, eta", "pull y" , lpull, hpull);
    Float_t refMax = 0.5+1.4*mePullYvsEtaZmPanel1->GetMaximum();
    Float_t newMax = 0.5+1.4*newmePullYvsEtaZmPanel1->GetMaximum();
    if refMax > newMax
    {
        mePullYvsEtaZmPanel1->SetMaximum(refMax);
    }
    else
    {
        mePullYvsEtaZmPanel1->SetMaximum(newMax);
    }      
    mePullYvsEtaZmPanel1->SetName("Reference");
    newmePullYvsEtaZmPanel1->SetName("New Release");
    mePullYvsEtaZmPanel1->Draw("e");
    newmePullYvsEtaZmPanel1->Draw("esameS"); 
    myPV->PVCompute(mePullYvsEtaZmPanel1, newmePullYvsEtaZmPanel1, te, 0.2, 0.2 );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s80 = (TPaveStats*)mePullYvsEtaZmPanel1->GetListOfFunctions()->FindObject("stats");
    if (s80) {
      s80->SetX1NDC (0.55); //new x start position
      s80->SetX2NDC (0.75); //new x end position
    }
    can_PullYvsEta->cd(3);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsEtaZmPanel2, newmePullYvsEtaZmPanel2, "panel2, z<0, eta", "pull y", lpull, hpull );
    Float_t refMax = 0.5+1.2*mePullYvsEtaZmPanel2->GetMaximum();
    Float_t newMax = 0.5+1.2*newmePullYvsEtaZmPanel2->GetMaximum();
    if refMax > newMax
    {
        mePullYvsEtaZmPanel2->SetMaximum(refMax);
    }
    else
    {
        mePullYvsEtaZmPanel2->SetMaximum(newMax);
    }      
    mePullYvsEtaZmPanel2->SetName("Reference");
    newmePullYvsEtaZmPanel2->SetName("New Release");
    mePullYvsEtaZmPanel2->Draw("e");
    newmePullYvsEtaZmPanel2->Draw("esameS"); 
    myPV->PVCompute(mePullYvsEtaZmPanel2, newmePullYvsEtaZmPanel2, te, 0.2, 0.2 );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s81 = (TPaveStats*)mePullYvsEtaZmPanel2->GetListOfFunctions()->FindObject("stats");
    if (s81) {
      s81->SetX1NDC (0.55); //new x start position
      s81->SetX2NDC (0.75); //new x end position
    }
    can_PullYvsEta->cd(5);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsEtaZpPanel1, newmePullYvsEtaZpPanel1, "panel1, z>0, eta", "pull y", lpull, hpull );
    Float_t refMax = 0.5+1.4*mePullYvsEtaZpPanel1->GetMaximum();
    Float_t newMax = 0.5+1.4*newmePullYvsEtaZpPanel1->GetMaximum();
    if refMax > newMax
    {
        mePullYvsEtaZpPanel1->SetMaximum(refMax);
    }
    else
    {
        mePullYvsEtaZpPanel1->SetMaximum(newMax);
    }      
    mePullYvsEtaZpPanel1->SetName("Reference");
    newmePullYvsEtaZpPanel1->SetName("New Release");
    mePullYvsEtaZpPanel1->Draw("e");
    newmePullYvsEtaZpPanel1->Draw("esameS"); 
    myPV->PVCompute(mePullYvsEtaZpPanel1, newmePullYvsEtaZpPanel1, te, 0.2, 0.2 );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s82 = (TPaveStats*)mePullYvsEtaZpPanel1->GetListOfFunctions()->FindObject("stats");
    if (s82) {
      s82->SetX1NDC (0.55); //new x start position
      s82->SetX2NDC (0.75); //new x end position
    }
    can_PullYvsEta->cd(6);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsEtaZpPanel2, newmePullYvsEtaZpPanel2, "panel2, z>0, eta", "pull y", lpull, hpull );
    Float_t refMax = 0.5+1.4*mePullYvsEtaZpPanel2->GetMaximum();
    Float_t newMax = 0.5+1.4*newmePullYvsEtaZpPanel2->GetMaximum();
    if refMax > newMax
    {
        mePullYvsEtaZpPanel2->SetMaximum(refMax);
    }
    else
    {
        mePullYvsEtaZpPanel2->SetMaximum(newMax);
    }      
    mePullYvsEtaZpPanel2->SetName("Reference");
    newmePullYvsEtaZpPanel2->SetName("New Release");
    mePullYvsEtaZpPanel2->Draw("e");
    newmePullYvsEtaZpPanel2->Draw("esameS"); 
    myPV->PVCompute(mePullYvsEtaZpPanel2, newmePullYvsEtaZpPanel2, te, 0.2, 0.2 );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s83 = (TPaveStats*)mePullYvsEtaZpPanel2->GetListOfFunctions()->FindObject("stats");
    if (s83) {
      s83->SetX1NDC (0.55); //new x start position
      s83->SetX2NDC (0.75); //new x end position
    }
    can_PullYvsEta->SaveAs("mePullYvsEta_compare.eps");
    can_PullYvsEta->SaveAs("mePullYvsEta_compare.gif");
  }

 //if (   0   ) 
 if (1) 
 {
    TCanvas* can_PullYvsPhi = new TCanvas("can_PullYvsPhi", "can_PullYvsPhi", 1200, 800);
    can_PullYvsPhi->Divide(3,2);
    
    TProfile* mePullYvsPhiBarrel;
    TProfile* mePullYvsPhiZmPanel1;
    TProfile* mePullYvsPhiZmPanel2;
    TProfile* mePullYvsPhiZpPanel1;
    TProfile* mePullYvsPhiZpPanel2;
    
    TProfile* newmePullYvsPhiBarrel;
    TProfile* newmePullYvsPhiZmPanel1;
    TProfile* newmePullYvsPhiZmPanel2;
    TProfile* newmePullYvsPhiZpPanel1;
    TProfile* newmePullYvsPhiZpPanel2;

    rdir->GetObject("Histograms_all/mePullYvsPhiBarrel"  , mePullYvsPhiBarrel  );
    rdir->GetObject("Histograms_all/mePullYvsPhiZmPanel1", mePullYvsPhiZmPanel1);
    rdir->GetObject("Histograms_all/mePullYvsPhiZmPanel2", mePullYvsPhiZmPanel2);
    rdir->GetObject("Histograms_all/mePullYvsPhiZpPanel1", mePullYvsPhiZpPanel1);
    rdir->GetObject("Histograms_all/mePullYvsPhiZpPanel2", mePullYvsPhiZpPanel2);

    sdir->GetObject("Histograms_all/mePullYvsPhiBarrel"  , newmePullYvsPhiBarrel  ); 
    sdir->GetObject("Histograms_all/mePullYvsPhiZmPanel1", newmePullYvsPhiZmPanel1);
    sdir->GetObject("Histograms_all/mePullYvsPhiZmPanel2", newmePullYvsPhiZmPanel2);
    sdir->GetObject("Histograms_all/mePullYvsPhiZpPanel1", newmePullYvsPhiZpPanel1);
    sdir->GetObject("Histograms_all/mePullYvsPhiZpPanel2", newmePullYvsPhiZpPanel2);
  
    TLegend* leg17 = new TLegend(0.3, 0.2, 0.6, 0.4);
    can_PullYvsPhi->cd(1);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsPhiBarrel, newmePullYvsPhiBarrel, "barrel, phi (deg)", "pull y", lpull, hpull, leg17 );
    Float_t refMax = 0.5+1.2*mePullYvsPhiBarrel->GetMaximum();
    Float_t newMax = 0.5+1.2*newmePullYvsPhiBarrel->GetMaximum();
    if refMax > newMax
    {
        mePullYvsPhiBarrel->SetMaximum(refMax);
    }
    else
    {
        mePullYvsPhiBarrel->SetMaximum(newMax);
    }      
    mePullYvsPhiBarrel->SetName("Reference");
    newmePullYvsPhiBarrel->SetName("New Release");
    mePullYvsPhiBarrel->Draw("e");
    newmePullYvsPhiBarrel->Draw("eSameS"); 
    myPV->PVCompute(mePullYvsPhiBarrel, newmePullYvsPhiBarrel, te, 0.3, 0.4 );
    leg17->Draw();
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s84 = (TPaveStats*)mePullYvsPhiBarrel->GetListOfFunctions()->FindObject("stats");
    if (s84) {
      s84->SetX1NDC (0.55); //new x start position
      s84->SetX2NDC (0.75); //new x end position
    }
    can_PullYvsPhi->cd(2);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsPhiZmPanel1, newmePullYvsPhiZmPanel1, "panel1, z<0, phi (deg)", "pull y" , lpull, hpull);
    Float_t refMax = 0.5+1.2*mePullYvsPhiZmPanel1->GetMaximum();
    Float_t newMax = 0.5+1.2*newmePullYvsPhiZmPanel1->GetMaximum();
    if refMax > newMax
    {
        mePullYvsPhiZmPanel1->SetMaximum(refMax);
    }
    else
    {
        mePullYvsPhiZmPanel1->SetMaximum(newMax);
    }      
    mePullYvsPhiZmPanel1->SetName("Reference");
    newmePullYvsPhiZmPanel1->SetName("New Release");
    mePullYvsPhiZmPanel1->Draw("e");
    newmePullYvsPhiZmPanel1->Draw("esameS"); 
    myPV->PVCompute(mePullYvsPhiZmPanel1, newmePullYvsPhiZmPanel1, te, 0.2, 0.2);
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s85 = (TPaveStats*)mePullYvsPhiZmPanel1->GetListOfFunctions()->FindObject("stats");
    if (s85) {
      s85->SetX1NDC (0.55); //new x start position
      s85->SetX2NDC (0.75); //new x end position
    }
    can_PullYvsPhi->cd(3);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsPhiZmPanel2, newmePullYvsPhiZmPanel2, "panel2, z<0, phi (deg)", "pull y" , lpull, hpull);
    Float_t refMax = 0.5+1.2*mePullYvsPhiZmPanel2->GetMaximum();
    Float_t newMax = 0.5+1.2*newmePullYvsPhiZmPanel2->GetMaximum();
    if refMax > newMax
    {
        mePullYvsPhiZmPanel2->SetMaximum(refMax);
    }
    else
    {
        mePullYvsPhiZmPanel2->SetMaximum(newMax);
    }      
    mePullYvsPhiZmPanel2->SetName("Reference");
    newmePullYvsPhiZmPanel2->SetName("New Release");
    mePullYvsPhiZmPanel2->Draw("e");
    newmePullYvsPhiZmPanel2->Draw("esameS"); 
    myPV->PVCompute(mePullYvsPhiZmPanel2, newmePullYvsPhiZmPanel2, te, 0.2, 0.2);
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s86 = (TPaveStats*)mePullYvsPhiZmPanel2->GetListOfFunctions()->FindObject("stats");
    if (s86) {
      s86->SetX1NDC (0.55); //new x start position
      s86->SetX2NDC (0.75); //new x end position
    }
    can_PullYvsPhi->cd(5);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsPhiZpPanel1, newmePullYvsPhiZpPanel1, "panel1, z>0, phi (deg)", "pull y", lpull, hpull );
    Float_t refMax = 0.5+1.2*mePullYvsPhiZpPanel1->GetMaximum();
    Float_t newMax = 0.5+1.2*newmePullYvsPhiZpPanel1->GetMaximum();
    if refMax > newMax
    {
        mePullYvsPhiZpPanel1->SetMaximum(refMax);
    }
    else
    {
        mePullYvsPhiZpPanel1->SetMaximum(newMax);
    }      
    mePullYvsPhiZpPanel1->SetName("Reference");
    newmePullYvsPhiZpPanel1->SetName("New Release");
    mePullYvsPhiZpPanel1->Draw("e");
    newmePullYvsPhiZpPanel1->Draw("esameS"); 
    myPV->PVCompute(mePullYvsPhiZpPanel1, newmePullYvsPhiZpPanel1, te, 0.2, 0.2 );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s87 = (TPaveStats*)mePullYvsPhiZpPanel1->GetListOfFunctions()->FindObject("stats");
    if (s87) {
      s87->SetX1NDC (0.55); //new x start position
      s87->SetX2NDC (0.75); //new x end position
    }
    can_PullYvsPhi->cd(6);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsPhiZpPanel2, newmePullYvsPhiZpPanel2, "panel2, z>0, phi (deg)", "pull y", lpull, hpull );
    Float_t refMax = 0.5+1.2*mePullYvsPhiZpPanel2->GetMaximum();
    Float_t newMax = 0.5+1.2*newmePullYvsPhiZpPanel2->GetMaximum();
    if refMax > newMax
    {
        mePullYvsPhiZpPanel2->SetMaximum(refMax);
    }
    else
    {
        mePullYvsPhiZpPanel2->SetMaximum(newMax);
    }      
    mePullYvsPhiZpPanel2->SetName("Reference");
    newmePullYvsPhiZpPanel2->SetName("New Release");
    mePullYvsPhiZpPanel2->Draw("e");
    newmePullYvsPhiZpPanel2->Draw("esameS"); 
    myPV->PVCompute(mePullYvsPhiZpPanel2, newmePullYvsPhiZpPanel2, te, 0.2, 0.2 );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s88 = (TPaveStats*)mePullYvsPhiZpPanel2->GetListOfFunctions()->FindObject("stats");
    if (s88) {
      s88->SetX1NDC (0.55); //new x start position
      s88->SetX2NDC (0.75); //new x end position
    }
    can_PullYvsPhi->SaveAs("mePullYvsPhi_compare.eps");
    can_PullYvsPhi->SaveAs("mePullYvsPhi_compare.gif");
  }

if (1) 
  {
    TCanvas* can_mePullx = new TCanvas("can_mePullx", "can_mePullx", 1200, 800);
    can_mePullx->Divide(3,2);
    
    TH1F* mePullxBarrel;
    TH1F* mePullxZmPanel1;
    TH1F* mePullxZmPanel2;
    TH1F* mePullxZpPanel1;
    TH1F* mePullxZpPanel2;
    
    TH1F* newmePullxBarrel;
    TH1F* newmePullxZmPanel1;
    TH1F* newmePullxZmPanel2;
    TH1F* newmePullxZpPanel1;
    TH1F* newmePullxZpPanel2;

    rdir->GetObject("Histograms_all/mePullxBarrel"  , mePullxBarrel  );
    rdir->GetObject("Histograms_all/mePullxZmPanel1", mePullxZmPanel1);
    rdir->GetObject("Histograms_all/mePullxZmPanel2", mePullxZmPanel2);
    rdir->GetObject("Histograms_all/mePullxZpPanel1", mePullxZpPanel1);
    rdir->GetObject("Histograms_all/mePullxZpPanel2", mePullxZpPanel2);

    sdir->GetObject("Histograms_all/mePullxBarrel"  , newmePullxBarrel  ); 
    sdir->GetObject("Histograms_all/mePullxZmPanel1", newmePullxZmPanel1);
    sdir->GetObject("Histograms_all/mePullxZmPanel2", newmePullxZmPanel2);
    sdir->GetObject("Histograms_all/mePullxZpPanel1", newmePullxZpPanel1);
    sdir->GetObject("Histograms_all/mePullxZpPanel2", newmePullxZpPanel2);
  
    TLegend* leg18 = new TLegend(0.15, 0.67, 0.45, 0.87);
    can_mePullx->cd(1);
    //gPad->SetLogy();
    SetUpHistograms(mePullxBarrel, newmePullxBarrel, "barrel, pull x", leg18);
    Float_t refMax = 1.2*mePullxBarrel->GetMaximum();
    Float_t newMax = 1.2*newmePullxBarrel->GetMaximum();
    if refMax > newMax
    {
        mePullxBarrel->SetMaximum(refMax);
    }
    else
    {
        mePullxBarrel->SetMaximum(newMax);
    }      
    mePullxBarrel->SetName("Reference");
    newmePullxBarrel->SetName("New Release");
    mePullxBarrel->Draw("he");
    newmePullxBarrel->Draw("heSameS"); 
    myPV->PVCompute(mePullxBarrel, newmePullxBarrel, te);
    leg18->Draw();
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s89 = (TPaveStats*)mePullxBarrel->GetListOfFunctions()->FindObject("stats");
    if (s89) {
      s89->SetX1NDC (0.55); //new x start position
      s89->SetX2NDC (0.75); //new x end position
    }
    can_mePullx->cd(2);
    //gPad->SetLogy();
    SetUpHistograms(mePullxZmPanel1, newmePullxZmPanel1, "panel1, z<0, pull x" );
    Float_t refMax = 1.2*mePullxZmPanel1->GetMaximum();
    Float_t newMax = 1.2*newmePullxZmPanel1->GetMaximum();
    if refMax > newMax
    {
        mePullxZmPanel1->SetMaximum(refMax);
    }
    else
    {
        mePullxZmPanel1->SetMaximum(newMax);
    }      
    mePullxZmPanel1->SetName("Reference");
    newmePullxZmPanel1->SetName("New Release");
    mePullxZmPanel1->Draw("he");
    newmePullxZmPanel1->Draw("hesameS"); 
    myPV->PVCompute(mePullxZmPanel1, newmePullxZmPanel1, te );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s90 = (TPaveStats*)mePullxZmPanel1->GetListOfFunctions()->FindObject("stats");
    if (s90) {
      s90->SetX1NDC (0.55); //new x start position
      s90->SetX2NDC (0.75); //new x end position
    }
    can_mePullx->cd(3);
    //gPad->SetLogy();
    SetUpHistograms(mePullxZmPanel2, newmePullxZmPanel2, "panel2, z<0, pull x" );
    Float_t refMax = 1.2*mePullxZmPanel2->GetMaximum();
    Float_t newMax = 1.2*newmePullxZmPanel2->GetMaximum();
    if refMax > newMax
    {
        mePullxZmPanel2->SetMaximum(refMax);
    }
    else
    {
        mePullxZmPanel2->SetMaximum(newMax);
    }      
    mePullxZmPanel2->SetName("Reference");
    newmePullxZmPanel2->SetName("New Release");
    mePullxZmPanel2->Draw("he"); 
    newmePullxZmPanel2->Draw("hesameS"); 
    myPV->PVCompute(mePullxZmPanel2, newmePullxZmPanel2, te );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s91 = (TPaveStats*)mePullxZmPanel2->GetListOfFunctions()->FindObject("stats");
    if (s91) {
      s91->SetX1NDC (0.55); //new x start position
      s91->SetX2NDC (0.75); //new x end position
    }
    can_mePullx->cd(5);
    //gPad->SetLogy();
    SetUpHistograms(mePullxZpPanel1, newmePullxZpPanel1, "panel2, z>0, pull x" );
    Float_t refMax = 1.3*mePullxZpPanel1->GetMaximum();
    Float_t newMax = 1.3*newmePullxZpPanel1->GetMaximum();
    if refMax > newMax
    {
        mePullxZpPanel1->SetMaximum(refMax);
    }
    else
    {
        mePullxZpPanel1->SetMaximum(newMax);
    }      
    mePullxZpPanel1->SetName("Reference");
    newmePullxZpPanel1->SetName("New Release");
    mePullxZpPanel1->Draw("he");
    newmePullxZpPanel1->Draw("hesameS"); 
    myPV->PVCompute(mePullxZpPanel1, newmePullxZpPanel1, te);
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s92 = (TPaveStats*)mePullxZpPanel1->GetListOfFunctions()->FindObject("stats");
    if (s92) {
      s92->SetX1NDC (0.55); //new x start position
      s92->SetX2NDC (0.75); //new x end position
    }
    can_mePullx->cd(6);
    //gPad->SetLogy();
    SetUpHistograms(mePullxZpPanel2, newmePullxZpPanel2, "panel1, z>0, pull x" );
    Float_t refMax = 1.2*mePullxZpPanel2->GetMaximum();
    Float_t newMax = 1.2*newmePullxZpPanel2->GetMaximum();
    if refMax > newMax
    {
        mePullxZpPanel2->SetMaximum(refMax);
    }
    else
    {
        mePullxZpPanel2->SetMaximum(newMax);
    }      
    mePullxZpPanel2->SetName("Reference");
    newmePullxZpPanel2->SetName("New Release");
    mePullxZpPanel2->Draw("he");
    newmePullxZpPanel2->Draw("hesameS"); 
    myPV->PVCompute(mePullxZpPanel2, newmePullxZpPanel2, te );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s93 = (TPaveStats*)mePullxZpPanel2->GetListOfFunctions()->FindObject("stats");
    if (s93) {
      s93->SetX1NDC (0.55); //new x start position
      s93->SetX2NDC (0.75); //new x end position
    }
    can_mePullx->SaveAs("mePullx_compare.eps");
    can_mePullx->SaveAs("mePullx_compare.gif");
  }

if (1) 
  {
    TCanvas* can_mePully = new TCanvas("can_mePully", "can_mePully", 1200, 800);
    can_mePully->Divide(3,2);
    
    TH1F* mePullyBarrel;
    TH1F* mePullyZmPanel1;
    TH1F* mePullyZmPanel2;
    TH1F* mePullyZpPanel1;
    TH1F* mePullyZpPanel2;
    
    TH1F* newmePullyBarrel;
    TH1F* newmePullyZmPanel1;
    TH1F* newmePullyZmPanel2;
    TH1F* newmePullyZpPanel1;
    TH1F* newmePullyZpPanel2;

    rdir->GetObject("Histograms_all/mePullyBarrel"  , mePullyBarrel  );
    rdir->GetObject("Histograms_all/mePullyZmPanel1", mePullyZmPanel1);
    rdir->GetObject("Histograms_all/mePullyZmPanel2", mePullyZmPanel2);
    rdir->GetObject("Histograms_all/mePullyZpPanel1", mePullyZpPanel1);
    rdir->GetObject("Histograms_all/mePullyZpPanel2", mePullyZpPanel2);

    sdir->GetObject("Histograms_all/mePullyBarrel"  , newmePullyBarrel  ); 
    sdir->GetObject("Histograms_all/mePullyZmPanel1", newmePullyZmPanel1);
    sdir->GetObject("Histograms_all/mePullyZmPanel2", newmePullyZmPanel2);
    sdir->GetObject("Histograms_all/mePullyZpPanel1", newmePullyZpPanel1);
    sdir->GetObject("Histograms_all/mePullyZpPanel2", newmePullyZpPanel2);
  
    TLegend* leg19 = new TLegend(0.15, 0.67, 0.45, 0.87);
    can_mePully->cd(1);
    //gPad->SetLogy();
    SetUpHistograms(mePullyBarrel, newmePullyBarrel, "barrel, pull y", leg19 );
    Float_t refMax = 1.2*mePullyBarrel->GetMaximum();
    Float_t newMax = 1.2*newmePullyBarrel->GetMaximum();
    if refMax > newMax
    {
        mePullyBarrel->SetMaximum(refMax);
    }
    else
    {
        mePullyBarrel->SetMaximum(newMax);
    }      
    mePullyBarrel->SetName("Reference");
    newmePullyBarrel->SetName("New Release");
    mePullyBarrel->Draw("he");
    newmePullyBarrel->Draw("heSameS"); 
    myPV->PVCompute(mePullyBarrel, newmePullyBarrel, te);
    leg19->Draw();
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s94 = (TPaveStats*)mePullyBarrel->GetListOfFunctions()->FindObject("stats");
    if (s94) {
      s94->SetX1NDC (0.55); //new x start position
      s94->SetX2NDC (0.75); //new x end position
    }

    can_mePully->cd(2);
    //gPad->SetLogy();
    SetUpHistograms(mePullyZmPanel1, newmePullyZmPanel1, "panel1, z<0, pull y" );
    Float_t refMax = 1.2*mePullyZmPanel1->GetMaximum();
    Float_t newMax = 1.2*newmePullyZmPanel1->GetMaximum();
    if refMax > newMax
    {
        mePullyZmPanel1->SetMaximum(refMax);
    }
    else
    {
        mePullyZmPanel1->SetMaximum(newMax);
    }      
    mePullyZmPanel1->SetName("Reference");
    newmePullyZmPanel1->SetName("New Release");
    mePullyZmPanel1->Draw("he");
    newmePullyZmPanel1->Draw("hesameS"); 
    myPV->PVCompute(mePullyZmPanel1, newmePullyZmPanel1, te);
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s139 = (TPaveStats*)mePullyZmPanel1->GetListOfFunctions()->FindObject("stats");
    if (s139) {
    	s139->SetX1NDC (0.55); //new x start position
    	s139->SetX2NDC (0.75); //new x end position
    }
    can_mePully->cd(3);
    //gPad->SetLogy();
    SetUpHistograms(mePullyZmPanel2, newmePullyZmPanel2, "panel2, z<0, pull y" );
    Float_t refMax = 1.2*mePullyZmPanel2->GetMaximum();
    Float_t newMax = 1.2*newmePullyZmPanel2->GetMaximum();
    if refMax > newMax
    {
        mePullyZmPanel2->SetMaximum(refMax);
    }
    else
    {
        mePullyZmPanel2->SetMaximum(newMax);
    }      
    mePullyZmPanel2->SetName("Reference");
    newmePullyZmPanel2->SetName("New Release");
    mePullyZmPanel2->Draw("he");
    newmePullyZmPanel2->Draw("hesameS"); 
    myPV->PVCompute(mePullyZmPanel2, newmePullyZmPanel2, te);
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s95 = (TPaveStats*)mePullyZmPanel2->GetListOfFunctions()->FindObject("stats");
    if (s95) {
      s95->SetX1NDC (0.55); //new x start position
      s95->SetX2NDC (0.75); //new x end position
    }

    can_mePully->cd(5);
    //gPad->SetLogy();
    SetUpHistograms(mePullyZpPanel1, newmePullyZpPanel1, "panel1, z>0, pull y" );
    Float_t refMax = 1.2*mePullyZpPanel1->GetMaximum();
    Float_t newMax = 1.2*newmePullyZpPanel1->GetMaximum();
    if refMax > newMax
    {
        mePullyZpPanel1->SetMaximum(refMax);
    }
    else
    {
        mePullyZpPanel1->SetMaximum(newMax);
    }      
    mePullyZpPanel1->SetName("Reference");
    newmePullyZpPanel1->SetName("New Release");
    mePullyZpPanel1->Draw("he");
    newmePullyZpPanel1->Draw("hesameS"); 
    myPV->PVCompute(mePullyZpPanel1, newmePullyZpPanel1, te);
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s96 = (TPaveStats*)mePullyZpPanel1->GetListOfFunctions()->FindObject("stats");
    if (s96) {
      s96->SetX1NDC (0.55); //new x start position
      s96->SetX2NDC (0.75); //new x end position
    }

    can_mePully->cd(6);
    //gPad->SetLogy();
    SetUpHistograms(mePullyZpPanel2, newmePullyZpPanel2, "panel2, z>0, pull y" );
    Float_t refMax = 1.2*mePullyZpPanel2->GetMaximum();
    Float_t newMax = 1.2*newmePullyZpPanel2->GetMaximum();
    if refMax > newMax
    {
        mePullyZpPanel2->SetMaximum(refMax);
    }
    else
    {
        mePullyZpPanel2->SetMaximum(newMax);
    }      
    mePullyZpPanel2->SetName("Reference");
    newmePullyZpPanel2->SetName("New Release");
    mePullyZpPanel2->Draw("he");
    newmePullyZpPanel2->Draw("hesameS"); 
    myPV->PVCompute(mePullyZpPanel2, newmePullyZpPanel2, te);
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s97 = (TPaveStats*)mePullyZpPanel2->GetListOfFunctions()->FindObject("stats");
    if (s97) {
      s97->SetX1NDC (0.55); //new x start position
      s97->SetX2NDC (0.75); //new x end position
    }

    can_mePully->SaveAs("mePully_compare.eps");
    can_mePully->SaveAs("mePully_compare.gif");
  }

 double xmin = 0.0000;
 double xmax = 0.0020;

if (1) 
  {
    TCanvas* can_ResXvsAlpha = new TCanvas("can_ResXvsAlpha", "can_ResXvsAlpha", 1200, 800);
    can_ResXvsAlpha->Divide(3,2);
    
    TProfile* meResXvsAlphaBarrelFlippedLadders;
    TProfile* meResXvsAlphaBarrelNonFlippedLadders;
    TProfile* meResXvsAlphaZmPanel1;
    TProfile* meResXvsAlphaZmPanel2;
    TProfile* meResXvsAlphaZpPanel1;
    TProfile* meResXvsAlphaZpPanel2;
    
    TProfile* newmeResXvsAlphaBarrelFlippedLadders;
    TProfile* newmeResXvsAlphaBarrelNonFlippedLadders;
    TProfile* newmeResXvsAlphaZmPanel1;
    TProfile* newmeResXvsAlphaZmPanel2;
    TProfile* newmeResXvsAlphaZpPanel1;
    TProfile* newmeResXvsAlphaZpPanel2;

    rdir->GetObject("Histograms_all/meResXvsAlphaBarrelFlippedLadders"     , meResXvsAlphaBarrelFlippedLadders     );
    rdir->GetObject("Histograms_all/meResXvsAlphaBarrelNonFlippedLadders"  , meResXvsAlphaBarrelNonFlippedLadders  );

    rdir->GetObject("Histograms_all/meResXvsAlphaZmPanel1", meResXvsAlphaZmPanel1);
    rdir->GetObject("Histograms_all/meResXvsAlphaZmPanel2", meResXvsAlphaZmPanel2);
    rdir->GetObject("Histograms_all/meResXvsAlphaZpPanel1", meResXvsAlphaZpPanel1);
    rdir->GetObject("Histograms_all/meResXvsAlphaZpPanel2", meResXvsAlphaZpPanel2);

    sdir->GetObject("Histograms_all/meResXvsAlphaBarrelFlippedLadders"   , 
		     newmeResXvsAlphaBarrelFlippedLadders     );
    sdir->GetObject("Histograms_all/meResXvsAlphaBarrelNonFlippedLadders", 
		     newmeResXvsAlphaBarrelNonFlippedLadders  );
 
    sdir->GetObject("Histograms_all/meResXvsAlphaZmPanel1", newmeResXvsAlphaZmPanel1);
    sdir->GetObject("Histograms_all/meResXvsAlphaZmPanel2", newmeResXvsAlphaZmPanel2);
    sdir->GetObject("Histograms_all/meResXvsAlphaZpPanel1", newmeResXvsAlphaZpPanel1);
    sdir->GetObject("Histograms_all/meResXvsAlphaZpPanel2", newmeResXvsAlphaZpPanel2);
  
    TLegend* leg20 = new TLegend(0.3, 0.2, 0.6, 0.4);
    can_ResXvsAlpha->cd(1);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResXvsAlphaBarrelFlippedLadders, newmeResXvsAlphaBarrelFlippedLadders, "barrel, non-flipped ladders, |alpha| (deg)", "<|x residual|> (cm)", xmin, xmax, leg20 );
    Float_t refMax = 1.5*meResXvsAlphaBarrelFlippedLadders->GetMaximum();
    Float_t newMax = 1.5*newmeResXvsAlphaBarrelFlippedLadders->GetMaximum();
    if refMax > newMax
    {
        meResXvsAlphaBarrelFlippedLadders->SetMaximum(refMax);
    }
    else
    {
        meResXvsAlphaBarrelFlippedLadders->SetMaximum(newMax);
    }      
    meResXvsAlphaBarrelFlippedLadders->SetName("Reference");
    newmeResXvsAlphaBarrelFlippedLadders->SetName("New Release");
    meResXvsAlphaBarrelFlippedLadders->Draw("e");
    newmeResXvsAlphaBarrelFlippedLadders->Draw("eSameS"); 
    myPV->PVCompute(meResXvsAlphaBarrelFlippedLadders, newmeResXvsAlphaBarrelFlippedLadders, te, 0.3, 0.4 );
    leg20->Draw();
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s98 = (TPaveStats*)meResXvsAlphaBarrelFlippedLadders->GetListOfFunctions()->FindObject("stats");
    if (s98) {
      s98->SetX1NDC (0.55); //new x start position
      s98->SetX2NDC (0.75); //new x end position
    }
    can_ResXvsAlpha->cd(2);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResXvsAlphaZmPanel1, newmeResXvsAlphaZmPanel1, 
			   "panel1, z<0, |alpha| (deg)", "<|x residual|> (cm)", xmin, xmax );
    Float_t refMax = 1.2*meResXvsAlphaZmPanel1->GetMaximum();
    Float_t newMax = 1.2*newmeResXvsAlphaZmPanel1->GetMaximum();
    if refMax > newMax
    {
        meResXvsAlphaZmPanel1->SetMaximum(refMax);
    }
    else
    {
        meResXvsAlphaZmPanel1->SetMaximum(newMax);
    }      
    meResXvsAlphaZmPanel1->SetName("Reference");
    newmeResXvsAlphaZmPanel1->SetName("New Release");
    meResXvsAlphaZmPanel1->SetMinimum(xmin);
    meResXvsAlphaZmPanel1->SetMaximum(xmax);
    meResXvsAlphaZmPanel1->Draw("e");
    newmeResXvsAlphaZmPanel1->Draw("esameS"); 
    myPV->PVCompute(meResXvsAlphaZmPanel1, newmeResXvsAlphaZmPanel1, te );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s99 = (TPaveStats*)meResXvsAlphaZmPanel1->GetListOfFunctions()->FindObject("stats");
    if (s99) {
      s99->SetX1NDC (0.55); //new x start position
      s99->SetX2NDC (0.75); //new x end position
    }
    can_ResXvsAlpha->cd(3);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResXvsAlphaZmPanel2, newmeResXvsAlphaZmPanel2, "panel2, z<0, |alpha| (deg)", "<|x residual|> (cm)", xmin, xmax );
    Float_t refMax = 1.2*meResXvsAlphaZmPanel2->GetMaximum();
    Float_t newMax = 1.2*newmeResXvsAlphaZmPanel2->GetMaximum();
    if refMax > newMax
    {
        meResXvsAlphaZmPanel2->SetMaximum(refMax);
    }
    else
    {
        meResXvsAlphaZmPanel2->SetMaximum(newMax);
    }      
    meResXvsAlphaZmPanel2->SetName("Reference");
    newmeResXvsAlphaZmPanel2->SetName("New Release");
    meResXvsAlphaZmPanel2->SetMinimum(xmin);
    meResXvsAlphaZmPanel2->SetMaximum(xmax);
    meResXvsAlphaZmPanel2->Draw("e");
    newmeResXvsAlphaZmPanel2->Draw("esameS"); 
    myPV->PVCompute(meResXvsAlphaZmPanel2, newmeResXvsAlphaZmPanel2, te );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s100 = (TPaveStats*)meResXvsAlphaZmPanel2->GetListOfFunctions()->FindObject("stats");
    if (s100) {
      s100->SetX1NDC (0.55); //new x start position
      s100->SetX2NDC (0.75); //new x end position
    }
    can_ResXvsAlpha->cd(4);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResXvsAlphaBarrelNonFlippedLadders, newmeResXvsAlphaBarrelNonFlippedLadders, "barrel, flipped ladders, |alpha| (deg)", "<|x residual|> (cm)", xmin, xmax );
    Float_t refMax = 1.2*meResXvsAlphaBarrelNonFlippedLadders->GetMaximum();
    Float_t newMax = 1.2*newmeResXvsAlphaBarrelNonFlippedLadders->GetMaximum();
    if refMax > newMax
    {
        meResXvsAlphaBarrelNonFlippedLadders->SetMaximum(refMax);
    }
    else
    {
        meResXvsAlphaBarrelNonFlippedLadders->SetMaximum(newMax);
    }      
    meResXvsAlphaBarrelNonFlippedLadders->SetName("Reference");
    newmeResXvsAlphaBarrelNonFlippedLadders->SetName("New Release");
    meResXvsAlphaBarrelNonFlippedLadders->SetMinimum(xmin);
    meResXvsAlphaBarrelNonFlippedLadders->SetMaximum(xmax);
    meResXvsAlphaBarrelNonFlippedLadders->Draw("e");
    newmeResXvsAlphaBarrelNonFlippedLadders->Draw("eSameS"); 
    myPV->PVCompute(meResXvsAlphaBarrelNonFlippedLadders, newmeResXvsAlphaBarrelNonFlippedLadders, te );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s101 = (TPaveStats*)meResXvsAlphaBarrelNonFlippedLadders->GetListOfFunctions()->FindObject("stats");
    if (s101) {
      s101->SetX1NDC (0.55); //new x start position
      s101->SetX2NDC (0.75); //new x end position
    }
    can_ResXvsAlpha->cd(5);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResXvsAlphaZpPanel1, newmeResXvsAlphaZpPanel1, 
			   "panel1, z>0, |alpha| (deg)", "<|x residual|> (cm)", xmin, xmax );
    Float_t refMax = 1.2*meResXvsAlphaZpPanel1->GetMaximum();
    Float_t newMax = 1.2*newmeResXvsAlphaZpPanel1->GetMaximum();
    if refMax > newMax
    {
        meResXvsAlphaZpPanel1->SetMaximum(refMax);
    }
    else
    {
        meResXvsAlphaZpPanel1->SetMaximum(newMax);
    }      
    meResXvsAlphaZpPanel1->SetName("Reference");
    newmeResXvsAlphaZpPanel1->SetName("New Release");
    meResXvsAlphaZpPanel1->SetMinimum(xmin);
    meResXvsAlphaZpPanel1->SetMaximum(xmax);
    meResXvsAlphaZpPanel1->Draw("e");
    newmeResXvsAlphaZpPanel1->Draw("esameS"); 
    myPV->PVCompute(meResXvsAlphaZpPanel1, newmeResXvsAlphaZpPanel1, te, 0.2 , 0.2 );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s102 = (TPaveStats*)meResXvsAlphaZpPanel1->GetListOfFunctions()->FindObject("stats");
    if (s102) {
      s102->SetX1NDC (0.55); //new x start position
      s102->SetX2NDC (0.75); //new x end position
    }
    can_ResXvsAlpha->cd(6);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResXvsAlphaZpPanel2, newmeResXvsAlphaZpPanel2, "panel2, z>0, |alpha| (deg)", "<|x residual|> (cm)", xmin, xmax );
    Float_t refMax = 1.2*meResXvsAlphaZpPanel2->GetMaximum();
    Float_t newMax = 1.2*newmeResXvsAlphaZpPanel2->GetMaximum();
    if refMax > newMax
    {
        meResXvsAlphaZpPanel2->SetMaximum(refMax);
    }
    else
    {
        meResXvsAlphaZpPanel2->SetMaximum(newMax);
    }      
    meResXvsAlphaZpPanel2->SetName("Reference");
    newmeResXvsAlphaZpPanel2->SetName("New Release");
    meResXvsAlphaZpPanel2->SetMinimum(xmin);
    meResXvsAlphaZpPanel2->SetMaximum(xmax);
    meResXvsAlphaZpPanel2->Draw("e");
    newmeResXvsAlphaZpPanel2->Draw("esameS"); 
    myPV->PVCompute(meResXvsAlphaZpPanel2, newmeResXvsAlphaZpPanel2, te );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s103 = (TPaveStats*)meResXvsAlphaZpPanel2->GetListOfFunctions()->FindObject("stats");
    if (s103) {
      s103->SetX1NDC (0.55); //new x start position
      s103->SetX2NDC (0.75); //new x end position
    }
    can_ResXvsAlpha->SaveAs("meResXvsAlpha_compare.eps");
    can_ResXvsAlpha->SaveAs("meResXvsAlpha_compare.gif");
  }
    
if (1) 
  {
    TCanvas* can_ResXvsBeta = new TCanvas("can_ResXvsBeta", "can_ResXvsBeta", 1200, 800);
    can_ResXvsBeta->Divide(3,2);
    
    TProfile* meResXvsBetaBarrel;
    TProfile* meResXvsBetaZmPanel1;
    TProfile* meResXvsBetaZmPanel2;
    TProfile* meResXvsBetaZpPanel1;
    TProfile* meResXvsBetaZpPanel2;
    
    TProfile* newmeResXvsBetaBarrel;
    TProfile* newmeResXvsBetaZmPanel1;
    TProfile* newmeResXvsBetaZmPanel2;
    TProfile* newmeResXvsBetaZpPanel1;
    TProfile* newmeResXvsBetaZpPanel2;

    rdir->GetObject("Histograms_all/meResXvsBetaBarrel"  , meResXvsBetaBarrel  );
    rdir->GetObject("Histograms_all/meResXvsBetaZmPanel1", meResXvsBetaZmPanel1);
    rdir->GetObject("Histograms_all/meResXvsBetaZmPanel2", meResXvsBetaZmPanel2);
    rdir->GetObject("Histograms_all/meResXvsBetaZpPanel1", meResXvsBetaZpPanel1);
    rdir->GetObject("Histograms_all/meResXvsBetaZpPanel2", meResXvsBetaZpPanel2);

    sdir->GetObject("Histograms_all/meResXvsBetaBarrel"  , newmeResXvsBetaBarrel  ); 
    sdir->GetObject("Histograms_all/meResXvsBetaZmPanel1", newmeResXvsBetaZmPanel1);
    sdir->GetObject("Histograms_all/meResXvsBetaZmPanel2", newmeResXvsBetaZmPanel2);
    sdir->GetObject("Histograms_all/meResXvsBetaZpPanel1", newmeResXvsBetaZpPanel1);
    sdir->GetObject("Histograms_all/meResXvsBetaZpPanel2", newmeResXvsBetaZpPanel2);
  
    TLegend* leg21 = new TLegend(0.3, 0.2, 0.6, 0.4);
    can_ResXvsBeta->cd(1);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResXvsBetaBarrel, newmeResXvsBetaBarrel,"barrel, |beta| (deg)", "<|x residual|> (cm)", xmin, xmax, leg21 );
    Float_t refMax = 1.5*meResXvsBetaBarrel->GetMaximum();
    Float_t newMax = 1.5*newmeResXvsBetaBarrel->GetMaximum();
    if refMax > newMax
    {
        meResXvsBetaBarrel->SetMaximum(refMax);
    }
    else
    {
        meResXvsBetaBarrel->SetMaximum(newMax);
    }      
    meResXvsBetaBarrel->SetName("Reference");
    newmeResXvsBetaBarrel->SetName("New Release");
    meResXvsBetaBarrel->Draw("e");
    newmeResXvsBetaBarrel->Draw("eSameS"); 
    myPV->PVCompute(meResXvsBetaBarrel, newmeResXvsBetaBarrel, te, 0.3, 0.4 );
    leg21->Draw();
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s104 = (TPaveStats*)meResXvsBetaBarrel->GetListOfFunctions()->FindObject("stats");
    if (s104) {
      s104->SetX1NDC (0.55); //new x start position
      s104->SetX2NDC (0.75); //new x end position
    }
    can_ResXvsBeta->cd(2);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResXvsBetaZmPanel1, newmeResXvsBetaZmPanel1, "panel1, z<0, |beta| (deg)", "<|x residual|> (cm)", xmin, xmax );
    Float_t refMax = 1.5*meResXvsBetaZmPanel1->GetMaximum();
    Float_t newMax = 1.5*newmeResXvsBetaZmPanel1->GetMaximum();
    if refMax > newMax
    {
        meResXvsBetaZmPanel1->SetMaximum(refMax);
    }
    else
    {
        meResXvsBetaZmPanel1->SetMaximum(newMax);
    }      
    meResXvsBetaZmPanel1->SetName("Reference");
    newmeResXvsBetaZmPanel1->SetName("New Release");
    meResXvsBetaZmPanel1->Draw("e");
    newmeResXvsBetaZmPanel1->Draw("esameS"); 
    myPV->PVCompute(meResXvsBetaZmPanel1, newmeResXvsBetaZmPanel1, te, 0.2, 0.2 );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s105 = (TPaveStats*)meResXvsBetaZmPanel1->GetListOfFunctions()->FindObject("stats");
    if (s105) {
      s105->SetX1NDC (0.55); //new x start position
      s105->SetX2NDC (0.75); //new x end position
    }
    can_ResXvsBeta->cd(3);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResXvsBetaZmPanel2, newmeResXvsBetaZmPanel2, "panel2, z<0, |beta| (deg)", "<|x residual|> (cm)", xmin, xmax ); 
    Float_t refMax = 1.5*meResXvsBetaZmPanel2->GetMaximum();
    Float_t newMax = 1.5*newmeResXvsBetaZmPanel2->GetMaximum();
    if refMax > newMax
    {
        meResXvsBetaZmPanel2->SetMaximum(refMax);
    }
    else
    {
        meResXvsBetaZmPanel2->SetMaximum(newMax);
    }      
    meResXvsBetaZmPanel2->SetName("Reference");
    newmeResXvsBetaZmPanel2->SetName("New Release");
    meResXvsBetaZmPanel2->Draw("e");
    newmeResXvsBetaZmPanel2->Draw("esameS"); 
    myPV->PVCompute(meResXvsBetaZmPanel2, newmeResXvsBetaZmPanel2, te, 0.2, 0.2 );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s106 = (TPaveStats*)meResXvsBetaZmPanel2->GetListOfFunctions()->FindObject("stats");
    if (s106) {
      s106->SetX1NDC (0.55); //new x start position
      s106->SetX2NDC (0.75); //new x end position
    }
    can_ResXvsBeta->cd(5);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResXvsBetaZpPanel1, newmeResXvsBetaZpPanel1, "panel1, z>0, |beta| (deg)", "<|x residual|> (cm)", xmin, xmax );
    Float_t refMax = 1.5*meResXvsBetaZpPanel1->GetMaximum();
    Float_t newMax = 1.5*newmeResXvsBetaZpPanel1->GetMaximum();
    if refMax > newMax
    {
        meResXvsBetaZpPanel1->SetMaximum(refMax);
    }
    else
    {
        meResXvsBetaZpPanel1->SetMaximum(newMax);
    }      
    meResXvsBetaZpPanel1->SetName("Reference");
    newmeResXvsBetaZpPanel1->SetName("New Release");
    meResXvsBetaZpPanel1->Draw("e");
    newmeResXvsBetaZpPanel1->Draw("esameS"); 
    myPV->PVCompute(meResXvsBetaZpPanel1, newmeResXvsBetaZpPanel1, te, 0.2, 0.2 );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s107 = (TPaveStats*)meResXvsBetaZpPanel1->GetListOfFunctions()->FindObject("stats");
    if (s107) {
      s107->SetX1NDC (0.55); //new x start position
      s107->SetX2NDC (0.75); //new x end position
    }
    can_ResXvsBeta->cd(6);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResXvsBetaZpPanel2, newmeResXvsBetaZpPanel2, "panel2, z>0, |beta| (deg)", "<|x residual|> (cm)", xmin, xmax );
    Float_t refMax = 1.5*meResXvsBetaZpPanel2->GetMaximum();
    Float_t newMax = 1.5*newmeResXvsBetaZpPanel2->GetMaximum();
    if refMax > newMax
    {
        meResXvsBetaZpPanel2->SetMaximum(refMax);
    }
    else
    {
        meResXvsBetaZpPanel2->SetMaximum(newMax);
    }      
    meResXvsBetaZpPanel2->SetName("Reference");
    newmeResXvsBetaZpPanel2->SetName("New Release");
    meResXvsBetaZpPanel2->Draw("e");
    newmeResXvsBetaZpPanel2->Draw("esameS"); 
    myPV->PVCompute(meResXvsBetaZpPanel2, newmeResXvsBetaZpPanel2, te, 0.2, 0.2 );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s108 = (TPaveStats*)meResXvsBetaZpPanel2->GetListOfFunctions()->FindObject("stats");
    if (s108) {
      s108->SetX1NDC (0.55); //new x start position
      s108->SetX2NDC (0.75); //new x end position
    }
    can_ResXvsBeta->SaveAs("meResXvsBeta_compare.eps");
    can_ResXvsBeta->SaveAs("meResXvsBeta_compare.gif");
  }

 ymin = 0.0005;
 ymax = 0.0030;

if (1) 
  {
    TCanvas* can_ResYvsAlpha = new TCanvas("can_ResYvsAlpha", "can_ResYvsAlpha", 1200, 800);
    can_ResYvsAlpha->Divide(3,2);
    
    TProfile* meResYvsAlphaBarrel;
    TProfile* meResYvsAlphaZmPanel1;
    TProfile* meResYvsAlphaZmPanel2;
    TProfile* meResYvsAlphaZpPanel1;
    TProfile* meResYvsAlphaZpPanel2;
    
    TProfile* newmeResYvsAlphaBarrel;
    TProfile* newmeResYvsAlphaZmPanel1;
    TProfile* newmeResYvsAlphaZmPanel2;
    TProfile* newmeResYvsAlphaZpPanel1;
    TProfile* newmeResYvsAlphaZpPanel2;

    rdir->GetObject("Histograms_all/meResYvsAlphaBarrel"  , meResYvsAlphaBarrel  );
    rdir->GetObject("Histograms_all/meResYvsAlphaZmPanel1", meResYvsAlphaZmPanel1);
    rdir->GetObject("Histograms_all/meResYvsAlphaZmPanel2", meResYvsAlphaZmPanel2);
    rdir->GetObject("Histograms_all/meResYvsAlphaZpPanel1", meResYvsAlphaZpPanel1);
    rdir->GetObject("Histograms_all/meResYvsAlphaZpPanel2", meResYvsAlphaZpPanel2);

    sdir->GetObject("Histograms_all/meResYvsAlphaBarrel"  , newmeResYvsAlphaBarrel  ); 
    sdir->GetObject("Histograms_all/meResYvsAlphaZmPanel1", newmeResYvsAlphaZmPanel1);
    sdir->GetObject("Histograms_all/meResYvsAlphaZmPanel2", newmeResYvsAlphaZmPanel2);
    sdir->GetObject("Histograms_all/meResYvsAlphaZpPanel1", newmeResYvsAlphaZpPanel1);
    sdir->GetObject("Histograms_all/meResYvsAlphaZpPanel2", newmeResYvsAlphaZpPanel2);
  
    TLegend* leg22 = new TLegend(0.55, 0.15, 0.85, 0.35);
    can_ResYvsAlpha->cd(1);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResYvsAlphaBarrel, newmeResYvsAlphaBarrel, "barrel, |alpha| (deg)", "<|y residual|> (cm)", ymin+0.0010, ymax+0.0010, leg22 );
    Float_t refMax = 1.2*meResYvsAlphaBarrel->GetMaximum();
    Float_t newMax = 1.2*newmeResYvsAlphaBarrel->GetMaximum();
    if refMax > newMax
    {
        meResYvsAlphaBarrel->SetMaximum(refMax);
    }
    else
    {
        meResYvsAlphaBarrel->SetMaximum(newMax);
    }      
    meResYvsAlphaBarrel->SetName("Reference");
    newmeResYvsAlphaBarrel->SetName("New Release");
    meResYvsAlphaBarrel->Draw("e");
    newmeResYvsAlphaBarrel->Draw("eSameS"); 
    myPV->PVCompute(meResYvsAlphaBarrel, newmeResYvsAlphaBarrel, te, 0.2, 0.2 );
    leg22->Draw();
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s109 = (TPaveStats*)meResYvsAlphaBarrel->GetListOfFunctions()->FindObject("stats");
    if (s109) {
      s109->SetX1NDC (0.55); //new x start position
      s109->SetX2NDC (0.75); //new x end position
    }
    can_ResYvsAlpha->cd(2);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResYvsAlphaZmPanel1, newmeResYvsAlphaZmPanel1, "panel1, z<0, |alpha| (deg)", "<|y residual|> (cm)", ymin, ymax );
    Float_t refMax = 1.2*meResYvsAlphaZmPanel1->GetMaximum();
    Float_t newMax = 1.2*newmeResYvsAlphaZmPanel1->GetMaximum();
    if refMax > newMax
    {
        meResYvsAlphaZmPanel1->SetMaximum(refMax);
    }
    else
    {
        meResYvsAlphaZmPanel1->SetMaximum(newMax);
    }      
    meResYvsAlphaZmPanel1->SetName("Reference");
    newmeResYvsAlphaZmPanel1->SetName("New Release");
    meResYvsAlphaZmPanel1->Draw("e");
    newmeResYvsAlphaZmPanel1->Draw("esameS"); 
    myPV->PVCompute(meResYvsAlphaZmPanel1, newmeResYvsAlphaZmPanel1, te, 0.2, 0.2 );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s110 = (TPaveStats*)meResYvsAlphaZmPanel1->GetListOfFunctions()->FindObject("stats");
    if (s110) {
      s110->SetX1NDC (0.55); //new x start position
      s110->SetX2NDC (0.75); //new x end position
    }
    can_ResYvsAlpha->cd(3);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResYvsAlphaZmPanel2, newmeResYvsAlphaZmPanel2, "panel2, z<0, |alpha| (deg)", "<|y residual|> (cm)", ymin, ymax );
    Float_t refMax = 1.2*meResYvsAlphaZmPanel2->GetMaximum();
    Float_t newMax = 1.2*newmeResYvsAlphaZmPanel2->GetMaximum();
    if refMax > newMax
    {
        meResYvsAlphaZmPanel2->SetMaximum(refMax);
    }
    else
    {
        meResYvsAlphaZmPanel2->SetMaximum(newMax);
    }      
    meResYvsAlphaZmPanel2->SetName("Reference");
    newmeResYvsAlphaZmPanel2->SetName("New Release");
    meResYvsAlphaZmPanel2->Draw("e");
    newmeResYvsAlphaZmPanel2->Draw("esameS"); 
    myPV->PVCompute(meResYvsAlphaZmPanel2, newmeResYvsAlphaZmPanel2, te, 0.2, 0.2 );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s111 = (TPaveStats*)meResYvsAlphaZmPanel2->GetListOfFunctions()->FindObject("stats");
    if (s111) {
      s111->SetX1NDC (0.55); //new x start position
      s111->SetX2NDC (0.75); //new x end position
    }
    can_ResYvsAlpha->cd(5);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResYvsAlphaZpPanel1, newmeResYvsAlphaZpPanel1, "panel1, z>0, |alpha| (deg)", "<|y residual|> (cm)" , ymin, ymax);
    Float_t refMax = 1.2*meResYvsAlphaZpPanel1->GetMaximum();
    Float_t newMax = 1.2*newmeResYvsAlphaZpPanel1->GetMaximum();
    if refMax > newMax
    {
        meResYvsAlphaZpPanel1->SetMaximum(refMax);
    }
    else
    {
        meResYvsAlphaZpPanel1->SetMaximum(newMax);
    }      
    meResYvsAlphaZpPanel1->SetName("Reference");
    newmeResYvsAlphaZpPanel1->SetName("New Release");
    meResYvsAlphaZpPanel1->Draw("e");
    newmeResYvsAlphaZpPanel1->Draw("esameS"); 
    myPV->PVCompute(meResYvsAlphaZpPanel1, newmeResYvsAlphaZpPanel1, te, 0.2, 0.2 );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s112 = (TPaveStats*)meResYvsAlphaZpPanel1->GetListOfFunctions()->FindObject("stats");
    if (s112) {
      s112->SetX1NDC (0.55); //new x start position
      s112->SetX2NDC (0.75); //new x end position
    }
    can_ResYvsAlpha->cd(6);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResYvsAlphaZpPanel2, newmeResYvsAlphaZpPanel2, "panel2, z>0, |alpha| (deg)", "<|y residual|> (cm)", ymin, ymax );
    Float_t refMax = 1.2*meResYvsAlphaZpPanel2->GetMaximum();
    Float_t newMax = 1.2*newmeResYvsAlphaZpPanel2->GetMaximum();
    if refMax > newMax
    {
        meResYvsAlphaZpPanel2->SetMaximum(refMax);
    }
    else
    {
        meResYvsAlphaZpPanel2->SetMaximum(newMax);
    }      
    meResYvsAlphaZpPanel2->SetName("Reference");
    newmeResYvsAlphaZpPanel2->SetName("New Release");
    meResYvsAlphaZpPanel2->Draw("e");
    newmeResYvsAlphaZpPanel2->Draw("esameS"); 
    myPV->PVCompute(meResYvsAlphaZpPanel2, newmeResYvsAlphaZpPanel2, te, 0.2, 0.2 );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s113 = (TPaveStats*)meResYvsAlphaZpPanel2->GetListOfFunctions()->FindObject("stats");
    if (s113) {
      s113->SetX1NDC (0.55); //new x start position
      s113->SetX2NDC (0.75); //new x end position
    }
    can_ResYvsAlpha->SaveAs("meResYvsAlpha_compare.eps");
    can_ResYvsAlpha->SaveAs("meResYvsAlpha_compare.gif");
  }

if (1) 
  {
    TCanvas* can_ResYvsBeta = new TCanvas("can_ResYvsBeta", "can_ResYvsBeta", 1200, 800);
    can_ResYvsBeta->Divide(3,2);
    
    TProfile* meResYvsBetaBarrel;
    TProfile* meResYvsBetaZmPanel1;
    TProfile* meResYvsBetaZmPanel2;
    TProfile* meResYvsBetaZpPanel1;
    TProfile* meResYvsBetaZpPanel2;
    
    TProfile* newmeResYvsBetaBarrel;
    TProfile* newmeResYvsBetaZmPanel1;
    TProfile* newmeResYvsBetaZmPanel2;
    TProfile* newmeResYvsBetaZpPanel1;
    TProfile* newmeResYvsBetaZpPanel2;

    rdir->GetObject("Histograms_all/meResYvsBetaBarrel"  , meResYvsBetaBarrel  );
    rdir->GetObject("Histograms_all/meResYvsBetaZmPanel1", meResYvsBetaZmPanel1);
    rdir->GetObject("Histograms_all/meResYvsBetaZmPanel2", meResYvsBetaZmPanel2);
    rdir->GetObject("Histograms_all/meResYvsBetaZpPanel1", meResYvsBetaZpPanel1);
    rdir->GetObject("Histograms_all/meResYvsBetaZpPanel2", meResYvsBetaZpPanel2);

    sdir->GetObject("Histograms_all/meResYvsBetaBarrel"  , newmeResYvsBetaBarrel  ); 
    sdir->GetObject("Histograms_all/meResYvsBetaZmPanel1", newmeResYvsBetaZmPanel1);
    sdir->GetObject("Histograms_all/meResYvsBetaZmPanel2", newmeResYvsBetaZmPanel2);
    sdir->GetObject("Histograms_all/meResYvsBetaZpPanel1", newmeResYvsBetaZpPanel1);
    sdir->GetObject("Histograms_all/meResYvsBetaZpPanel2", newmeResYvsBetaZpPanel2);
  
    TLegend* leg23 = new TLegend(0.35, 0.15, 0.65, 0.3);
    can_ResYvsBeta->cd(1);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResYvsBetaBarrel, newmeResYvsBetaBarrel, "barrel, |beta| (deg)", "<|y residual|> (cm)", 0.0000, 0.0060, leg23 );
    Float_t refMax = 1.5*meResYvsBetaBarrel->GetMaximum();
    Float_t newMax = 1.5*newmeResYvsBetaBarrel->GetMaximum();
    if refMax > newMax
    {
        meResYvsBetaBarrel->SetMaximum(refMax);
    }
    else
    {
        meResYvsBetaBarrel->SetMaximum(newMax);
    }      
    meResYvsBetaBarrel->SetName("Reference");
    newmeResYvsBetaBarrel->SetName("New Release");
    meResYvsBetaBarrel->Draw("e");
    newmeResYvsBetaBarrel->Draw("eSameS"); 
    myPV->PVCompute(meResYvsBetaBarrel, newmeResYvsBetaBarrel, te );
    leg23->Draw();
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s114 = (TPaveStats*)meResYvsBetaBarrel->GetListOfFunctions()->FindObject("stats");
    if (s114) {
      s114->SetX1NDC (0.55); //new x start position
      s114->SetX2NDC (0.75); //new x end position
    }
    can_ResYvsBeta->cd(2);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResYvsBetaZmPanel1, newmeResYvsBetaZmPanel1, "panel1, z<0, |beta| (deg)", "<|y residual|> (cm)", ymin, ymax );
    Float_t refMax = 1.5*meResYvsBetaZmPanel1->GetMaximum();
    Float_t newMax = 1.5*newmeResYvsBetaZmPanel1->GetMaximum();
    if refMax > newMax
    {
        meResYvsBetaZmPanel1->SetMaximum(refMax);
    }
    else
    {
        meResYvsBetaZmPanel1->SetMaximum(newMax);
    }      
    meResYvsBetaZmPanel1->SetName("Reference");
    newmeResYvsBetaZmPanel1->SetName("New Release");
    meResYvsBetaZmPanel1->Draw("e");
    newmeResYvsBetaZmPanel1->Draw("esameS"); 
    myPV->PVCompute(meResYvsBetaZmPanel1, newmeResYvsBetaZmPanel1, te, 0.2, 0.2 );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s115 = (TPaveStats*)meResYvsBetaZmPanel1->GetListOfFunctions()->FindObject("stats");
    if (s115) {
      s115->SetX1NDC (0.55); //new x start position
      s115->SetX2NDC (0.75); //new x end position
    }
    can_ResYvsBeta->cd(3);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResYvsBetaZmPanel2, newmeResYvsBetaZmPanel2, "panel2, z<0, |beta| (deg)", "<|y residual|> (cm)", ymin, ymax );
    Float_t refMax = 1.5*meResYvsBetaZmPanel2->GetMaximum();
    Float_t newMax = 1.5*newmeResYvsBetaZmPanel2->GetMaximum();
    if refMax > newMax
    {
        meResYvsBetaZmPanel2->SetMaximum(refMax);
    }
    else
    {
        meResYvsBetaZmPanel2->SetMaximum(newMax);
    }      
    meResYvsBetaZmPanel2->SetName("Reference");
    newmeResYvsBetaZmPanel2->SetName("New Release");
    meResYvsBetaZmPanel2->Draw("e");
    newmeResYvsBetaZmPanel2->Draw("esameS"); 
    myPV->PVCompute(meResYvsBetaZmPanel2, newmeResYvsBetaZmPanel2, te, 0.2, 0.2 );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s116 = (TPaveStats*)meResYvsBetaZmPanel2->GetListOfFunctions()->FindObject("stats");
    if (s116) {
      s116->SetX1NDC (0.55); //new x start position
      s116->SetX2NDC (0.75); //new x end position
    }
    can_ResYvsBeta->cd(5);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResYvsBetaZpPanel1, newmeResYvsBetaZpPanel1, "panel1, z>0, |beta| (deg)", "<|y residual|> (cm)", ymin, ymax );
    Float_t refMax = 1.5*meResYvsBetaZpPanel1->GetMaximum();
    Float_t newMax = 1.5*newmeResYvsBetaZpPanel1->GetMaximum();
    if refMax > newMax
    {
        meResYvsBetaZpPanel1->SetMaximum(refMax);
    }
    else
    {
        meResYvsBetaZpPanel1->SetMaximum(newMax);
    }      
    meResYvsBetaZpPanel1->SetName("Reference");
    newmeResYvsBetaZpPanel1->SetName("New Release");
    meResYvsBetaZpPanel1->Draw("e");
    newmeResYvsBetaZpPanel1->Draw("esameS"); 
    myPV->PVCompute(meResYvsBetaZpPanel1, newmeResYvsBetaZpPanel1, te, 0.3, 0.7 );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s117 = (TPaveStats*)meResYvsBetaZpPanel1->GetListOfFunctions()->FindObject("stats");
    if (s117) {
      s117->SetX1NDC (0.55); //new x start position
      s117->SetX2NDC (0.75); //new x end position
    }
    can_ResYvsBeta->cd(6);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResYvsBetaZpPanel2, newmeResYvsBetaZpPanel2, "panel2, z>0, |beta| (deg)", "<|y residual|> (cm)", ymin, ymax );
    Float_t refMax = 1.5*meResYvsBetaZpPanel2->GetMaximum();
    Float_t newMax = 1.5*newmeResYvsBetaZpPanel2->GetMaximum();
    if refMax > newMax
    {
        meResYvsBetaZpPanel2->SetMaximum(refMax);
    }
    else
    {
        meResYvsBetaZpPanel2->SetMaximum(newMax);
    }      
    meResYvsBetaZpPanel2->SetName("Reference");
    newmeResYvsBetaZpPanel2->SetName("New Release");
    meResYvsBetaZpPanel2->Draw("e");
    newmeResYvsBetaZpPanel2->Draw("esameS"); 
    myPV->PVCompute(meResYvsBetaZpPanel2, newmeResYvsBetaZpPanel2, te, 0.2, 0.2 );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s118 = (TPaveStats*)meResYvsBetaZpPanel2->GetListOfFunctions()->FindObject("stats");
    if (s118) {
      s118->SetX1NDC (0.55); //new x start position
      s118->SetX2NDC (0.75); //new x end position
    }
    can_ResYvsBeta->SaveAs("meResYvsBeta_compare.eps");
    can_ResYvsBeta->SaveAs("meResYvsBeta_compare.gif");
  }

if (1) 
  {
    TCanvas* can_meResx = new TCanvas("can_meResx", "can_meResx", 1200, 800);
    can_meResx->Divide(3,2);
    
    TH1F* meResxBarrel;
    TH1F* meResxZmPanel1;
    TH1F* meResxZmPanel2;
    TH1F* meResxZpPanel1;
    TH1F* meResxZpPanel2;
    
    TH1F* newmeResxBarrel;
    TH1F* newmeResxZmPanel1;
    TH1F* newmeResxZmPanel2;
    TH1F* newmeResxZpPanel1;
    TH1F* newmeResxZpPanel2;

    rdir->GetObject("Histograms_all/meResxBarrel"  , meResxBarrel  );
    rdir->GetObject("Histograms_all/meResxZmPanel1", meResxZmPanel1);
    rdir->GetObject("Histograms_all/meResxZmPanel2", meResxZmPanel2);
    rdir->GetObject("Histograms_all/meResxZpPanel1", meResxZpPanel1);
    rdir->GetObject("Histograms_all/meResxZpPanel2", meResxZpPanel2);

    sdir->GetObject("Histograms_all/meResxBarrel"  , newmeResxBarrel  ); 
    sdir->GetObject("Histograms_all/meResxZmPanel1", newmeResxZmPanel1);
    sdir->GetObject("Histograms_all/meResxZmPanel2", newmeResxZmPanel2);
    sdir->GetObject("Histograms_all/meResxZpPanel1", newmeResxZpPanel1);
    sdir->GetObject("Histograms_all/meResxZpPanel2", newmeResxZpPanel2);
  
    TLegend* leg24 = new TLegend(0.15, 0.72, 0.42, 0.87);
    can_meResx->cd(1);
    gPad->SetLogy();
    SetUpHistograms(meResxBarrel, newmeResxBarrel, "barrel, x residual (cm)", leg24 );
    Float_t refMax = 1.2*meResxBarrel->GetMaximum();
    Float_t newMax = 1.2*newmeResxBarrel->GetMaximum();
    if refMax > newMax
    {
        meResxBarrel->SetMaximum(refMax);
    }
    else
    {
        meResxBarrel->SetMaximum(newMax);
    }      
    meResxBarrel->SetName("Reference");
    newmeResxBarrel->SetName("New Release");
    meResxBarrel->Draw("he");
    newmeResxBarrel->Draw("heSameS"); 
    myPV->PVCompute(meResxBarrel, newmeResxBarrel, te );
    leg24->Draw();
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s119 = (TPaveStats*)meResxBarrel->GetListOfFunctions()->FindObject("stats");
    if (s119) {
      s119->SetX1NDC (0.55); //new x start position
      s119->SetX2NDC (0.75); //new x end position
    }
    can_meResx->cd(2);
    gPad->SetLogy();
    SetUpHistograms(meResxZmPanel1, newmeResxZmPanel1, "panel1, z<0, x residual (cm)" );
    Float_t refMax = 1.2*meResxZmPanel1->GetMaximum();
    Float_t newMax = 1.2*newmeResxZmPanel1->GetMaximum();
    if refMax > newMax
    {
        meResxZmPanel1->SetMaximum(refMax);
    }
    else
    {
        meResxZmPanel1->SetMaximum(newMax);
    }      
    meResxZmPanel1->SetName("Reference");
    newmeResxZmPanel1->SetName("New Release");
    meResxZmPanel1->Draw("he");
    newmeResxZmPanel1->Draw("hesameS"); 
    myPV->PVCompute(meResxZmPanel1, newmeResxZmPanel1, te );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s120 = (TPaveStats*)meResxZmPanel1->GetListOfFunctions()->FindObject("stats");
    if (s120) {
      s120->SetX1NDC (0.55); //new x start position
      s120->SetX2NDC (0.75); //new x end position
    }
    can_meResx->cd(3);
    gPad->SetLogy();
    SetUpHistograms(meResxZmPanel2,  newmeResxZmPanel2, "panel2, z<0, x residual (cm)");
    Float_t refMax = 1.2*meResxZmPanel2->GetMaximum();
    Float_t newMax = 1.2*newmeResxZmPanel2->GetMaximum();
    if refMax > newMax
    {
        meResxZmPanel2->SetMaximum(refMax);
    }
    else
    {
        meResxZmPanel2->SetMaximum(newMax);
    }      
    meResxZmPanel2->SetName("Reference");
    newmeResxZmPanel2->SetName("New Release");
    meResxZmPanel2->Draw("he");
    newmeResxZmPanel2->Draw("hesameS"); 
    myPV->PVCompute(meResxZmPanel2, newmeResxZmPanel2, te );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s121 = (TPaveStats*)meResxZmPanel2->GetListOfFunctions()->FindObject("stats");
    if (s121) {
      s121->SetX1NDC (0.55); //new x start position
      s121->SetX2NDC (0.75); //new x end position
    }
    can_meResx->cd(5);
    gPad->SetLogy();
    SetUpHistograms(meResxZpPanel1, newmeResxZpPanel1, "panel1, z>0, x residual (cm)" );
    Float_t refMax = 1.2*meResxZpPanel1->GetMaximum();
    Float_t newMax = 1.2*newmeResxZpPanel1->GetMaximum();
    if refMax > newMax
    {
        meResxZpPanel1->SetMaximum(refMax);
    }
    else
    {
        meResxZpPanel1->SetMaximum(newMax);
    }      
    meResxZpPanel1->SetName("Reference");
    newmeResxZpPanel1->SetName("New Release");
    meResxZpPanel1->Draw("he");
    newmeResxZpPanel1->Draw("hesameS"); 
    myPV->PVCompute(meResxZpPanel1, newmeResxZpPanel1, te );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s122 = (TPaveStats*)meResxZpPanel1->GetListOfFunctions()->FindObject("stats");
    if (s122) {
      s122->SetX1NDC (0.55); //new x start position
      s122->SetX2NDC (0.75); //new x end position
    }
    can_meResx->cd(6);
    gPad->SetLogy();
    SetUpHistograms(meResxZpPanel2, newmeResxZpPanel2, "panel2, z>0, x residual (cm)" );
    Float_t refMax = 1.2*meResxZpPanel2->GetMaximum();
    Float_t newMax = 1.2*newmeResxZpPanel2->GetMaximum();
    if refMax > newMax
    {
        meResxZpPanel2->SetMaximum(refMax);
    }
    else
    {
        meResxZpPanel2->SetMaximum(newMax);
    }      
    meResxZpPanel2->SetName("Reference");
    newmeResxZpPanel2->SetName("New Release");
    meResxZpPanel2->Draw("he");
    newmeResxZpPanel2->Draw("hesameS"); 
    myPV->PVCompute(meResxZpPanel2, newmeResxZpPanel2, te );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s123 = (TPaveStats*)meResxZpPanel2->GetListOfFunctions()->FindObject("stats");
    if (s123) {
      s123->SetX1NDC (0.55); //new x start position
      s123->SetX2NDC (0.75); //new x end position
    }
    can_meResx->SaveAs("meResx_compare.eps");
    can_meResx->SaveAs("meResx_compare.gif");
  }

 if (1) 
  {
    TCanvas* can_meResy = new TCanvas("can_meResy", "can_meResy", 1200, 800);
    can_meResy->Divide(3,2);
    
    TH1F* meResyBarrel;
    TH1F* meResyZmPanel1;
    TH1F* meResyZmPanel2;
    TH1F* meResyZpPanel1;
    TH1F* meResyZpPanel2;
    
    TH1F* newmeResyBarrel;
    TH1F* newmeResyZmPanel1;
    TH1F* newmeResyZmPanel2;
    TH1F* newmeResyZpPanel1;
    TH1F* newmeResyZpPanel2;

    rdir->GetObject("Histograms_all/meResyBarrel"  , meResyBarrel  );
    rdir->GetObject("Histograms_all/meResyZmPanel1", meResyZmPanel1);
    rdir->GetObject("Histograms_all/meResyZmPanel2", meResyZmPanel2);
    rdir->GetObject("Histograms_all/meResyZpPanel1", meResyZpPanel1);
    rdir->GetObject("Histograms_all/meResyZpPanel2", meResyZpPanel2);

    sdir->GetObject("Histograms_all/meResyBarrel"  , newmeResyBarrel  ); 
    sdir->GetObject("Histograms_all/meResyZmPanel1", newmeResyZmPanel1);
    sdir->GetObject("Histograms_all/meResyZmPanel2", newmeResyZmPanel2);
    sdir->GetObject("Histograms_all/meResyZpPanel1", newmeResyZpPanel1);
    sdir->GetObject("Histograms_all/meResyZpPanel2", newmeResyZpPanel2);
  
    TLegend* leg25 = new TLegend(0.15, 0.72, 0.42, 0.87);
    can_meResy->cd(1);
    gPad->SetLogy();
    SetUpHistograms(meResyBarrel, newmeResyBarrel, "barrel, y residual (cm)", leg25 );
    Float_t refMax = 1.2*meResyBarrel->GetMaximum();
    Float_t newMax = 1.2*newmeResyBarrel->GetMaximum();
    if refMax > newMax
    {
        meResyBarrel->SetMaximum(refMax);
    }
    else
    {
        meResyBarrel->SetMaximum(newMax);
    }      
    meResyBarrel->SetName("Reference");
    newmeResyBarrel->SetName("New Release");
    meResyBarrel->Draw("he");
    newmeResyBarrel->Draw("heSameS"); 
    myPV->PVCompute(meResyBarrel, newmeResyBarrel, te );
    leg25->Draw();
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s124 = (TPaveStats*)meResyBarrel->GetListOfFunctions()->FindObject("stats");
    if (s124) {
      s124->SetX1NDC (0.55); //new x start position
      s124->SetX2NDC (0.75); //new x end position
    }
    can_meResy->cd(2);
    gPad->SetLogy();
    SetUpHistograms(meResyZmPanel1, newmeResyZmPanel1, "panel1, z<0, y residual (cm)" );
    Float_t refMax = 1.2*meResyZmPanel1->GetMaximum();
    Float_t newMax = 1.2*newmeResyZmPanel1->GetMaximum();
    if refMax > newMax
    {
        meResyZmPanel1->SetMaximum(refMax);
    }
    else
    {
        meResyZmPanel1->SetMaximum(newMax);
    }      
    meResyZmPanel1->SetName("Reference");
    newmeResyZmPanel1->SetName("New Release");
    meResyZmPanel1->Draw("he");
    newmeResyZmPanel1->Draw("hesameS"); 
    myPV->PVCompute(meResyZmPanel1, newmeResyZmPanel1, te );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s125 = (TPaveStats*)meResyZmPanel1->GetListOfFunctions()->FindObject("stats");
    if (s125) {
      s125->SetX1NDC (0.55); //new x start position
      s125->SetX2NDC (0.75); //new x end position
    }
    can_meResy->cd(3);
    gPad->SetLogy();
    SetUpHistograms(meResyZmPanel2, newmeResyZmPanel2, "panel2, z<0, y residual (cm) " );
    Float_t refMax = 1.2*meResyZmPanel2->GetMaximum();
    Float_t newMax = 1.2*newmeResyZmPanel2->GetMaximum();
    if refMax > newMax
    {
        meResyZmPanel2->SetMaximum(refMax);
    }
    else
    {
        meResyZmPanel2->SetMaximum(newMax);
    }      
    meResyZmPanel2->SetName("Reference");
    newmeResyZmPanel2->SetName("New Release");
    meResyZmPanel2->Draw("he");
    newmeResyZmPanel2->Draw("hesameS"); 
    myPV->PVCompute(meResyZmPanel2, newmeResyZmPanel2, te );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s126 = (TPaveStats*)meResyZmPanel2->GetListOfFunctions()->FindObject("stats");
    if (s126) {
      s126->SetX1NDC (0.55); //new x start position
      s126->SetX2NDC (0.75); //new x end position
    }
    can_meResy->cd(5);
    gPad->SetLogy();
    SetUpHistograms(meResyZpPanel1, newmeResyZpPanel1, "panel1, z>0, y residual (cm)" );
    Float_t refMax = 1.2*meResyZpPanel1->GetMaximum();
    Float_t newMax = 1.2*newmeResyZpPanel1->GetMaximum();
    if refMax > newMax
    {
        meResyZpPanel1->SetMaximum(refMax);
    }
    else
    {
        meResyZpPanel1->SetMaximum(newMax);
    }      
    meResyZpPanel1->SetName("Reference");
    newmeResyZpPanel1->SetName("New Release");
    meResyZpPanel1->Draw("he");
    newmeResyZpPanel1->Draw("hesameS"); 
    myPV->PVCompute(meResyZpPanel1, newmeResyZpPanel1, te );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s127 = (TPaveStats*)meResyZpPanel1->GetListOfFunctions()->FindObject("stats");
    if (s127) {
      s127->SetX1NDC (0.55); //new x start position
      s127->SetX2NDC (0.75); //new x end position
    }
    can_meResy->cd(6);
    gPad->SetLogy();
    SetUpHistograms(meResyZpPanel2, newmeResyZpPanel2, "panel2, z>0, y residual (cm)" );
    Float_t refMax = 1.2*meResyZpPanel2->GetMaximum();
    Float_t newMax = 1.2*newmeResyZpPanel2->GetMaximum();
    if refMax > newMax
    {
        meResyZpPanel2->SetMaximum(refMax);
    }
    else
    {
        meResyZpPanel2->SetMaximum(newMax);
    }      
    meResyZpPanel2->SetName("Reference");
    newmeResyZpPanel2->SetName("New Release");
    meResyZpPanel2->Draw("he");
    newmeResyZpPanel2->Draw("hesameS"); 
    myPV->PVCompute(meResyZpPanel2, newmeResyZpPanel2, te );
    h_pv->SetBinContent(++bin, myPV->getPV());
    gPad->Update();
    TPaveStats *s128 = (TPaveStats*)meResyZpPanel2->GetListOfFunctions()->FindObject("stats");
    if (s128) {
      s128->SetX1NDC (0.55); //new x start position
      s128->SetX2NDC (0.75); //new x end position
    }
    can_meResy->SaveAs("meResy_compare.eps");
    can_meResy->SaveAs("meResy_compare.gif");
  }

 // Look at the charge distribution on each module 

 rfile->cd("Histograms_per_ring-layer_or_disk-plaquette");
 sfile->cd("Histograms_per_ring-layer_or_disk-plaquette");
 
 Char_t histo[200];

 TCanvas* can_meChargeRingLayer = new TCanvas("can_meChargeRingLayer", "can_meChargeRingLayer", 1200, 800);
 can_meChargeRingLayer->Divide(8,3);
 
 TH1F* meChargeLayerModule[3][8]; 
 TH1F* newmeChargeLayerModule[3][8];
 
 for (Int_t i=0; i<3; i++) // loop ovel layers
   for (Int_t j=0; j<8; j++) // loop ovel rings
     {
       sprintf(histo, "Histograms_per_ring-layer_or_disk-plaquette/meChargeBarrelLayerModule_%d_%d", 
	       i+1, j+1);
       rdir->GetObject(histo, meChargeLayerModule[i][j]);

       sprintf(histo, "Histograms_per_ring-layer_or_disk-plaquette/meChargeBarrelLayerModule_%d_%d", 
	       i+1, j+1);
       sdir->GetObject(histo, newmeChargeLayerModule[i][j]); 
       SetUpHistograms(meChargeLayerModule[i][j], newmeChargeLayerModule[i][j], "barrel, charge (elec)" );
       can_meChargeRingLayer->cd(8*i + j + 1);
       //gPad->SetLogy();
       Float_t refMax = 1.2*meChargeLayerModule[i][j]->GetMaximum();
       Float_t newMax = 1.2*newmeChargeLayerModule[i][j]->GetMaximum();
       if refMax > newMax
       {
           meChargeLayerModule[i][j]->SetMaximum(refMax);
       }
       else
       {
           meChargeLayerModule[i][j]->SetMaximum(newMax);
       }      
       meChargeLayerModule[i][j]->SetName("Reference");
       newmeChargeLayerModule[i][j]->SetName("New Release");
       meChargeLayerModule[i][j]->Draw("he");
       newmeChargeLayerModule[i][j]->Draw("hesameS"); 
       myPV->PVCompute(meChargeLayerModule[i][j], newmeChargeLayerModule[i][j], te );
       h_pv->SetBinContent(++bin, myPV->getPV());
       gPad->Update();
       TPaveStats *s129 = (TPaveStats*)meChargeLayerModule[i][j]->GetListOfFunctions()->FindObject("stats");
       if (s129) {
	 s129->SetX1NDC (0.55); //new x start position
	 s129->SetX2NDC (0.75); //new x end position
       }
     }
 TLegend* leg26 = new TLegend(0.45, 0.45, 0.75, 0.65);
 leg26->SetBorderSize(0);
 leg26->AddEntry(   meChargeLayerModule[0][0], "reference  ", "l");
 leg26->AddEntry(newmeChargeLayerModule[0][0], "new release", "l");
 leg26->Draw();

 can_meChargeRingLayer->SaveAs("meChargeBarrelLayerModule_compare.eps");
 can_meChargeRingLayer->SaveAs("meChargeBarrelLayerModule_compare.gif");


 TCanvas* can_meChargeZmPanel1DiskPlaq = new TCanvas("can_meChargeZmPanel1DiskPlaq", "can_meChargeZmPanel1DiskPlaq", 600, 800);
 can_meChargeZmPanel1DiskPlaq->Divide(2,4);
 
 TH1F* meChargeZmPanel1DiskPlaq[2][4];
 TH1F* newmeChargeZmPanel1DiskPlaq[2][4];

 for (Int_t i=0; i<2; i++) // loop over disks
   for (Int_t j=0; j<4; j++) // loop over plaquetes
     {
       sprintf(histo, "Histograms_per_ring-layer_or_disk-plaquette/meChargeZmPanel1DiskPlaq_%d_%d", i+1, j+1);
       rdir->GetObject(histo, meChargeZmPanel1DiskPlaq[i][j]);
       sprintf(histo, "Histograms_per_ring-layer_or_disk-plaquette/meChargeZmPanel1DiskPlaq_%d_%d", i+1, j+1);
       sdir->GetObject(histo, newmeChargeZmPanel1DiskPlaq[i][j]); 
       
       can_meChargeZmPanel1DiskPlaq->cd(4*i + j + 1);
       //gPad->SetLogy();
       SetUpHistograms(meChargeZmPanel1DiskPlaq[i][j], newmeChargeZmPanel1DiskPlaq[i][j], "panel1, z<0, charge (elec)" );
       Float_t refMax = 1.2*meChargeZmPanel1DiskPlaq[i][j]->GetMaximum();
       Float_t newMax = 1.2*newmeChargeZmPanel1DiskPlaq[i][j]->GetMaximum();
       if refMax > newMax
       {
           meChargeZmPanel1DiskPlaq[i][j]->SetMaximum(refMax);
       }
       else
       {
           meChargeZmPanel1DiskPlaq[i][j]->SetMaximum(newMax);
       }      
       meChargeZmPanel1DiskPlaq[i][j]->SetName("Reference");
       newmeChargeZmPanel1DiskPlaq[i][j]->SetName("New Release");
       meChargeZmPanel1DiskPlaq[i][j]->Draw("he");
       newmeChargeZmPanel1DiskPlaq[i][j]->Draw("hesameS"); 
       myPV->PVCompute(meChargeZmPanel1DiskPlaq[i][j], newmeChargeZmPanel1DiskPlaq[i][j], te );
       h_pv->SetBinContent(++bin, myPV->getPV());
       gPad->Update();
       TPaveStats *s130 = (TPaveStats*)meChargeZmPanel1DiskPlaq[i][j]->GetListOfFunctions()->FindObject("stats");
       if (s130) {
	 s130->SetX1NDC (0.55); //new x start position
	 s130->SetX2NDC (0.75); //new x end position
       }
     }
 TLegend* leg27 = new TLegend(0.5, 0.4, 0.8, 0.6);
 leg27->SetBorderSize(0);
 leg27->AddEntry(   meChargeZmPanel1DiskPlaq[0][0], "reference  ", "l");
 leg27->AddEntry(newmeChargeZmPanel1DiskPlaq[0][0], "new release", "l");
 leg27->Draw();
 
 can_meChargeZmPanel1DiskPlaq->SaveAs("meChargeZmPanel1DiskPlaq_compare.eps");
 can_meChargeZmPanel1DiskPlaq->SaveAs("meChargeZmPanel1DiskPlaq_compare.gif");


 TCanvas* can_meChargeZmPanel2DiskPlaq = new TCanvas("can_meChargeZmPanel2DiskPlaq", "can_meChargeZmPanel2DiskPlaq", 600, 800);
 can_meChargeZmPanel2DiskPlaq->Divide(2,3);
 
 TH1F* meChargeZmPanel2DiskPlaq[2][3];
 TH1F* newmeChargeZmPanel2DiskPlaq[2][3];
 
 for (Int_t i=0; i<2; i++) // loop ovel disks
   for (Int_t j=0; j<3; j++) // loop ovel plaguetes
     {
       sprintf(histo, "Histograms_per_ring-layer_or_disk-plaquette/meChargeZmPanel2DiskPlaq_%d_%d", i+1, j+1);
       rdir->GetObject(histo, meChargeZmPanel2DiskPlaq[i][j]);
       sprintf(histo, "Histograms_per_ring-layer_or_disk-plaquette/meChargeZmPanel2DiskPlaq_%d_%d", i+1, j+1);
       sdir->GetObject(histo, newmeChargeZmPanel2DiskPlaq[i][j]); 
       
       can_meChargeZmPanel2DiskPlaq->cd(3*i + j + 1);
       //gPad->SetLogy();
       SetUpHistograms(meChargeZmPanel2DiskPlaq[i][j], newmeChargeZmPanel2DiskPlaq[i][j], "panel2, z<0, charge (elec)" );
       Float_t refMax = 1.2*meChargeZmPanel2DiskPlaq[i][j]->GetMaximum();
       Float_t newMax = 1.2*newmeChargeZmPanel2DiskPlaq[i][j]->GetMaximum();
       if refMax > newMax
       {
           meChargeZmPanel2DiskPlaq[i][j]->SetMaximum(refMax);
       }
       else
       {
           meChargeZmPanel2DiskPlaq[i][j]->SetMaximum(newMax);
       }      
       meChargeZmPanel2DiskPlaq[i][j]->SetName("Reference");
       newmeChargeZmPanel2DiskPlaq[i][j]->SetName("New Release");
       meChargeZmPanel2DiskPlaq[i][j]->Draw("he");
       newmeChargeZmPanel2DiskPlaq[i][j]->Draw("hesameS"); 
       myPV->PVCompute(meChargeZmPanel2DiskPlaq[i][j], newmeChargeZmPanel2DiskPlaq[i][j], te );
       h_pv->SetBinContent(++bin, myPV->getPV());
       gPad->Update();
       TPaveStats *s131 = (TPaveStats*)meChargeZmPanel2DiskPlaq[i][j]->GetListOfFunctions()->FindObject("stats");
       if (s131) {
	 s131->SetX1NDC (0.55); //new x start position
	 s131->SetX2NDC (0.75); //new x end position
       }
     }
 TLegend* leg28 = new TLegend(0.5, 0.4, 0.8, 0.6);
 leg28->SetBorderSize(0);
 leg28->AddEntry(   meChargeZmPanel2DiskPlaq[0][0], "reference  ", "l");
 leg28->AddEntry(newmeChargeZmPanel2DiskPlaq[0][0], "new release", "l");
 leg28->Draw();

 can_meChargeZmPanel2DiskPlaq->SaveAs("meChargeZmPanel2DiskPlaq_compare.eps");
 can_meChargeZmPanel2DiskPlaq->SaveAs("meChargeZmPanel2DiskPlaq_compare.gif");


 TCanvas* can_meChargeZpPanel1DiskPlaq = new TCanvas("can_meChargeZpPanel1DiskPlaq", "can_meChargeZpPanel1DiskPlaq", 600, 800);
 can_meChargeZpPanel1DiskPlaq->Divide(2,4);
 
 TH1F* meChargeZpPanel1DiskPlaq[2][4];
 TH1F* newmeChargeZpPanel1DiskPlaq[2][4];
 
 for (Int_t i=0; i<2; i++) // loop ovel disks
   for (Int_t j=0; j<4; j++) // loop ovel plaguetes
     {
       sprintf(histo, "Histograms_per_ring-layer_or_disk-plaquette/meChargeZpPanel1DiskPlaq_%d_%d", i+1, j+1);
       rdir->GetObject(histo, meChargeZpPanel1DiskPlaq[i][j]);
       sprintf(histo, "Histograms_per_ring-layer_or_disk-plaquette/meChargeZpPanel1DiskPlaq_%d_%d", i+1, j+1);
       sdir->GetObject(histo, newmeChargeZpPanel1DiskPlaq[i][j]); 
       
       can_meChargeZpPanel1DiskPlaq->cd(4*i + j + 1);
       //gPad->SetLogy();
       SetUpHistograms(meChargeZpPanel1DiskPlaq[i][j], newmeChargeZpPanel1DiskPlaq[i][j], "panel1, z>0, charge (elec)");
       Float_t refMax = 1.2*meChargeZpPanel1DiskPlaq[i][j]->GetMaximum();
       Float_t newMax = 1.2*newmeChargeZpPanel1DiskPlaq[i][j]->GetMaximum();
       if refMax > newMax
       {
           meChargeZpPanel1DiskPlaq[i][j]->SetMaximum(refMax);
       }
       else
       {
           meChargeZpPanel1DiskPlaq[i][j]->SetMaximum(newMax);
       }      
       meChargeZpPanel1DiskPlaq[i][j]->SetName("Reference");
       newmeChargeZpPanel1DiskPlaq[i][j]->SetName("New Release");
       meChargeZpPanel1DiskPlaq[i][j]->Draw("he");
       newmeChargeZpPanel1DiskPlaq[i][j]->Draw("hesameS"); 
       myPV->PVCompute(meChargeZpPanel1DiskPlaq[i][j], newmeChargeZpPanel1DiskPlaq[i][j], te );
       h_pv->SetBinContent(++bin, myPV->getPV());
       gPad->Update();
       TPaveStats *s132 = (TPaveStats*)meChargeZpPanel1DiskPlaq[i][j]->GetListOfFunctions()->FindObject("stats");
       if (s132) {
	 s132->SetX1NDC (0.55); //new x start position
	 s132->SetX2NDC (0.75); //new x end position
       }
     }
 TLegend* leg29 = new TLegend(0.5, 0.4, 0.8, 0.6);
 leg29->SetBorderSize(0);
 leg29->AddEntry(   meChargeZmPanel1DiskPlaq[0][0], "reference  ", "l");
 leg29->AddEntry(newmeChargeZmPanel1DiskPlaq[0][0], "new release", "l");
 leg29->Draw();
 
 can_meChargeZpPanel1DiskPlaq->SaveAs("meChargeZpPanel1DiskPlaq_compare.eps");
 can_meChargeZpPanel1DiskPlaq->SaveAs("meChargeZpPanel1DiskPlaq_compare.gif");


 TCanvas* can_meChargeZpPanel2DiskPlaq = new TCanvas("can_meChargeZpPanel2DiskPlaq", "can_meChargeZpPanel2DiskPlaq", 600, 800);
 can_meChargeZpPanel2DiskPlaq->Divide(2,3);
 
 TH1F* meChargeZpPanel2DiskPlaq[2][3];
 TH1F* newmeChargeZpPanel2DiskPlaq[2][3];
  
 for (Int_t i=0; i<2; i++) // loop ovel disks
   for (Int_t j=0; j<3; j++) // loop ovel plaguetes
     {
       sprintf(histo, "Histograms_per_ring-layer_or_disk-plaquette/meChargeZpPanel2DiskPlaq_%d_%d", i+1, j+1);
       rdir->GetObject(histo, meChargeZpPanel2DiskPlaq[i][j]);
       sprintf(histo, "Histograms_per_ring-layer_or_disk-plaquette/meChargeZpPanel2DiskPlaq_%d_%d", i+1, j+1);
       sdir->GetObject(histo, newmeChargeZpPanel2DiskPlaq[i][j]); 
       
       can_meChargeZpPanel2DiskPlaq->cd(3*i + j + 1);
       //gPad->SetLogy();
       SetUpHistograms(meChargeZpPanel2DiskPlaq[i][j], newmeChargeZpPanel2DiskPlaq[i][j], "panel2, z>0, charge (elec)" );
       Float_t refMax = 1.2*meChargeZpPanel2DiskPlaq[i][j]->GetMaximum();
       Float_t newMax = 1.2*newmeChargeZpPanel2DiskPlaq[i][j]->GetMaximum();
       if refMax > newMax
       {
           meChargeZpPanel2DiskPlaq[i][j]->SetMaximum(refMax);
       }
       else
       {
           meChargeZpPanel2DiskPlaq[i][j]->SetMaximum(newMax);
       }      
       meChargeZpPanel2DiskPlaq[i][j]->SetName("Reference");
       newmeChargeZpPanel2DiskPlaq[i][j]->SetName("New Release");
       meChargeZpPanel2DiskPlaq[i][j]->Draw("he");
       newmeChargeZpPanel2DiskPlaq[i][j]->Draw("hesameS"); 
       myPV->PVCompute(meChargeZpPanel2DiskPlaq[i][j], newmeChargeZpPanel2DiskPlaq[i][j], te );
       h_pv->SetBinContent(++bin, myPV->getPV());
       gPad->Update();
       TPaveStats *s133 = (TPaveStats*)meChargeZpPanel2DiskPlaq[i][j]->GetListOfFunctions()->FindObject("stats");
       if (s133) {
	 s133->SetX1NDC (0.55); //new x start position
	 s133->SetX2NDC (0.75); //new x end position
       }
     }
 TLegend* leg30 = new TLegend(0.5, 0.4, 0.8, 0.6);
 leg30->SetBorderSize(0);
 leg30->AddEntry(   meChargeZmPanel2DiskPlaq[0][0], "reference  ", "l");
 leg30->AddEntry(newmeChargeZmPanel2DiskPlaq[0][0], "new release", "l");
 leg30->Draw();

 can_meChargeZpPanel2DiskPlaq->SaveAs("meChargeZpPanel2DiskPlaq_compare.eps");
 can_meChargeZpPanel2DiskPlaq->SaveAs("meChargeZpPanel2DiskPlaq_compare.gif");


 TCanvas* can_meResLayers = new TCanvas("can_meResLayers", "can_meResLayers", 1200, 800);
 can_meResLayers->Divide(3,2);
 
 TH1F* meResxBarrelLayer[3];
 TH1F* newmeResxBarrelLayer[3];
  
 Char_t xtitle[100];

 for (Int_t i=0; i<3; i++) // loop layers
   {
     sprintf(histo, "Histograms_per_ring-layer_or_disk-plaquette/meResxBarrelLayer_%d", i+1);
     rdir->GetObject(histo, meResxBarrelLayer[i]);
     sprintf(histo, "Histograms_per_ring-layer_or_disk-plaquette/meResxBarrelLayer_%d", i+1);
     sdir->GetObject(histo, newmeResxBarrelLayer[i]); 
     
     can_meResLayers->cd(i+1);
     gPad->SetLogy();
     sprintf(xtitle, "barrel, layer %d, res x", i+1);
     SetUpHistograms(meResxBarrelLayer[i], newmeResxBarrelLayer[i], xtitle );
     Float_t refMax = 1.2*meResxBarrelLayer[i]->GetMaximum();
     Float_t newMax = 1.2*newmeResxBarrelLayer[i]->GetMaximum();
     if refMax > newMax
     {
         meResxBarrelLayer[i]->SetMaximum(refMax);
     }
     else
     {
         meResxBarrelLayer[i]->SetMaximum(newMax);
     }      
     meResxBarrelLayer[i]->SetName("Reference");
     newmeResxBarrelLayer[i]->SetName("New Release");
     meResxBarrelLayer[i]->Draw("he");
     newmeResxBarrelLayer[i]->Draw("hesameS"); 
     myPV->PVCompute(meResxBarrelLayer[i], newmeResxBarrelLayer[i], te );
     h_pv->SetBinContent(++bin, myPV->getPV());
     gPad->Update();
     TPaveStats *s134 = (TPaveStats*)meResxBarrelLayer[i]->GetListOfFunctions()->FindObject("stats");
     if (s134) {
       s134->SetX1NDC (0.55); //new x start position
       s134->SetX2NDC (0.75); //new x end position
     }
   }
 TLegend* leg31 = new TLegend(0.15, 0.65, 0.45, 0.85);
 leg31->SetBorderSize(0);
 leg31->AddEntry(   meResxBarrelLayer[0], "reference  ", "l");
 leg31->AddEntry(newmeResxBarrelLayer[0], "new release", "l");
 leg31->Draw();

 TH1F* meResyBarrelLayer[3];
 TH1F* newmeResyBarrelLayer[3];
  
 for (Int_t i=0; i<3; i++) // loop layers
   {
     sprintf(histo, "Histograms_per_ring-layer_or_disk-plaquette/meResyBarrelLayer_%d", i+1);
     rdir->GetObject(histo, meResyBarrelLayer[i]);
     sprintf(histo, "Histograms_per_ring-layer_or_disk-plaquette/meResyBarrelLayer_%d", i+1);
     sdir->GetObject(histo, newmeResyBarrelLayer[i]); 
       
     can_meResLayers->cd(3+i+1);
     gPad->SetLogy();
     sprintf(xtitle, "barrel, layer %d, res y", i+1);
     SetUpHistograms(meResyBarrelLayer[i], newmeResyBarrelLayer[i], xtitle );
     Float_t refMax = 1.2*meResyBarrelLayer[i]->GetMaximum();
     Float_t newMax = 1.2*newmeResyBarrelLayer[i]->GetMaximum();
     if refMax > newMax
     {
         meResyBarrelLayer[i]->SetMaximum(refMax);
     }
     else
     {
         meResyBarrelLayer[i]->SetMaximum(newMax);
     }      
     meResyBarrelLayer[i]->SetName("Reference");
     newmeResyBarrelLayer[i]->SetName("New Release");
     meResyBarrelLayer[i]->Draw("he");
     newmeResyBarrelLayer[i]->Draw("hesameS"); 
     myPV->PVCompute(meResyBarrelLayer[i], newmeResyBarrelLayer[i], te );
     h_pv->SetBinContent(++bin, myPV->getPV());
     gPad->Update();
     TPaveStats *s135 = (TPaveStats*)meResyBarrelLayer[i]->GetListOfFunctions()->FindObject("stats");
     if (s135) {
       s135->SetX1NDC (0.55); //new x start position
       s135->SetX2NDC (0.75); //new x end position
     }
   }
 
 can_meResLayers->SaveAs("meResBarrelLayers_compare.eps");
 can_meResLayers->SaveAs("meResBarrelLayers_compare.gif");



 TCanvas* can_mePullLayers = new TCanvas("can_mePullLayers", "can_mePullLayers", 1200, 800);
 can_mePullLayers->Divide(3,2);
 
 TH1F* mePullxBarrelLayer[3];
 TH1F* newmePullxBarrelLayer[3];
  
 Char_t xtitle[100];

 for (Int_t i=0; i<3; i++) // loop layers
   {
     sprintf(histo, "Histograms_per_ring-layer_or_disk-plaquette/mePullxBarrelLayer_%d", i+1);
     rdir->GetObject(histo, mePullxBarrelLayer[i]);
     sprintf(histo, "Histograms_per_ring-layer_or_disk-plaquette/mePullxBarrelLayer_%d", i+1);
     sdir->GetObject(histo, newmePullxBarrelLayer[i]); 
     
     can_mePullLayers->cd(i+1);
     //gPad->SetLogy();
     sprintf(xtitle, "barrel, layer %d, pull x", i+1);
     SetUpHistograms(mePullxBarrelLayer[i], newmePullxBarrelLayer[i], xtitle );
     Float_t refMax = 1.2*mePullxBarrelLayer[i]->GetMaximum();
     Float_t newMax = 1.2*newmePullxBarrelLayer[i]->GetMaximum();
     if refMax > newMax
     {
         mePullxBarrelLayer[i]->SetMaximum(refMax);
     }
     else
     {
         mePullxBarrelLayer[i]->SetMaximum(newMax);
     }      
     mePullxBarrelLayer[i]->SetName("Reference");
     newmePullxBarrelLayer[i]->SetName("New Release");
     mePullxBarrelLayer[i]->Draw("he");
     newmePullxBarrelLayer[i]->Draw("hesameS"); 
     myPV->PVCompute(mePullxBarrelLayer[i], newmePullxBarrelLayer[i], te );
     h_pv->SetBinContent(++bin, myPV->getPV());
     gPad->Update();
     TPaveStats *s136 = (TPaveStats*)mePullxBarrelLayer[i]->GetListOfFunctions()->FindObject("stats");
     if (s136) {
       s136->SetX1NDC (0.55); //new x start position
       s136->SetX2NDC (0.75); //new x end position
     }
   }
 TLegend* leg32 = new TLegend(0.15, 0.65, 0.45, 0.85);
 leg32->SetBorderSize(0);
 leg32->AddEntry(   mePullxBarrelLayer[0], "reference  ", "l");
 leg32->AddEntry(newmePullxBarrelLayer[0], "new release", "l");
 leg32->Draw();

 TH1F* mePullyBarrelLayer[3];
 TH1F* newmePullyBarrelLayer[3];
  
 for (Int_t i=0; i<3; i++) // loop layers
   {
     sprintf(histo, "Histograms_per_ring-layer_or_disk-plaquette/mePullyBarrelLayer_%d", i+1);
     rdir->GetObject(histo, mePullyBarrelLayer[i]);
     sprintf(histo, "Histograms_per_ring-layer_or_disk-plaquette/mePullyBarrelLayer_%d", i+1);
     sdir->GetObject(histo, newmePullyBarrelLayer[i]); 
       
     can_mePullLayers->cd(3+i+1);
     //gPad->SetLogy();
     sprintf(xtitle, "barrel, layer %d, pull y", i+1);
     SetUpHistograms(mePullyBarrelLayer[i], newmePullyBarrelLayer[i], xtitle );
     Float_t refMax = 1.2*mePullyBarrelLayer[i]->GetMaximum();
     Float_t newMax = 1.2*newmePullyBarrelLayer[i]->GetMaximum();
     if refMax > newMax
     {
         mePullyBarrelLayer[i]->SetMaximum(refMax);
     }
     else
     {
         mePullyBarrelLayer[i]->SetMaximum(newMax);
     }      
     mePullyBarrelLayer[i]->SetName("Reference");
     newmePullyBarrelLayer[i]->SetName("New Release");
     mePullyBarrelLayer[i]->Draw("he");
     newmePullyBarrelLayer[i]->Draw("hesameS"); 
     myPV->PVCompute(mePullyBarrelLayer[i], newmePullyBarrelLayer[i], te );
     h_pv->SetBinContent(++bin, myPV->getPV());
     gPad->Update();
     TPaveStats *s137 = (TPaveStats*)mePullyBarrelLayer[i]->GetListOfFunctions()->FindObject("stats");
     if (s137) {
       s137->SetX1NDC (0.55); //new x start position
       s137->SetX2NDC (0.75); //new x end position
     }
   }
 
 can_mePullLayers->SaveAs("mePullBarrelLayers_compare.eps");
 can_mePullLayers->SaveAs("mePullBarrelLayers_compare.gif");


 
 TCanvas* can_pv = new TCanvas("can_pv", "can_mepv", 1200, 500);
 gPad->SetLogy();
 h_pv->SetXTitle("histogram number");
 h_pv->SetYTitle("Probability");
 h_pv->SetTitleOffset(0.7, "Y");
 h_pv->Draw();

 can_pv->SaveAs("summary_pv.eps");
 can_pv->SaveAs("summary_pv.gif");

 if ( n_bins != bin )
   cout << "   We have " << bin << " histograms but " << n_bins << " bins in the probability summary plots. " << endl 
	<< "   Please update n_bins to equal " << bin << "." << " Thank you !" << endl; 
 
 delete myPV;

}

