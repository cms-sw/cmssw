
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

void SiPixelRecoCompare()
{
  gROOT ->Reset();
    
  char*  sfilename = "./pixeltrackingrechitshist.root"; // file to be checked
  char*  rfilename = "../pixeltrackingrechitshist.root"; // reference file 
  
  delete gROOT->GetListOfFiles()->FindObject(rfilename);
  delete gROOT->GetListOfFiles()->FindObject(sfilename);
  
  TText* te = new TText();
  TFile* rfile = new TFile(rfilename);
  TDirectory * rdir=gDirectory; 
  TFile * sfile = new TFile(sfilename);
  TDirectory * sdir=gDirectory; 
  

 if(rfile->cd("DQMData/Run 1/RecoTrackV"))rfile->cd("DQMData/Run 1/RecoTrackV/Run summary/TrackingRecHits/Pixel");
 else if(rfile->cd("DQMData/RecoTrackV/TrackingRecHits/Pixel"))rfile->cd("DQMData/RecoTrackV/TrackingRecHits/Pixel");
 else if(rfile->cd("DQMData/Run 1/Tracking"))rfile->cd("DQMData/Run 1/Tracking/Run summary/TrackingRecHits/Pixel");
 else if(rfile->cd("DQMData/Tracking/TrackingRecHits/Pixel"))rfile->cd("DQMData/Tracking/TrackingRecHits/Pixel");
 rdir=gDirectory;

 if(sfile->cd("DQMData/Run 1/RecoTrackV"))sfile->cd("DQMData/Run 1/RecoTrackV/Run summary/TrackingRecHits/Pixel");
 else if(sfile->cd("DQMData/RecoTrackV/TrackingRecHits/Pixel"))sfile->cd("DQMData/RecoTrackV/TrackingRecHits/Pixel");
 else if(sfile->cd("DQMData/Run 1/Tracking"))sfile->cd("DQMData/Run 1/Tracking/Run summary/TrackingRecHits/Pixel");
 else if(sfile->cd("DQMData/Tracking/TrackingRecHits/Pixel"))sfile->cd("DQMData/Tracking/TrackingRecHits/Pixel");
 sdir=gDirectory; 


  Char_t histo[200];
    
  gROOT->ProcessLine(".x HistoCompare_Pixels.C");
  HistoCompare_Pixels* myPV = new HistoCompare_Pixels("RecoTrack_SiPixelRecoCompare.txt");
  //myPV->setName("RecoTrack_SiPixelRecoCompare");

  int n_bins = 194;
  double low = 0.5;
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
      
      rdir->GetObject("Histograms_all/meTracksPerEvent", meTracksPerEvent );
      rdir->GetObject("Histograms_all/mePixRecHitsPerTrack", mePixRecHitsPerTrack );
      
      sdir->GetObject("Histograms_all/meTracksPerEvent", newmeTracksPerEvent );
      sdir->GetObject("Histograms_all/mePixRecHitsPerTrack", newmePixRecHitsPerTrack );

      TLegend* leg1 = new TLegend(0.3, 0.7, 0.6, 0.9);
      can_meControl->cd(1);
      //gPad->SetLogy();
      SetUpHistograms(meTracksPerEvent, newmeTracksPerEvent, "tracks per event", leg1 );
      meTracksPerEvent->Draw("he");
      newmeTracksPerEvent->Draw("samehe"); 
      myPV->PVCompute(meTracksPerEvent, newmeTracksPerEvent, te );
      leg1->Draw();
      h_pv->SetBinContent(++bin, myPV->getPV());
    
      can_meControl->cd(2);
      //gPad->SetLogy();
      SetUpHistograms(mePixRecHitsPerTrack, newmePixRecHitsPerTrack, "pixel hits per track" );
      mePixRecHitsPerTrack->Draw("he");
      newmePixRecHitsPerTrack->Draw("samehe"); 
      myPV->PVCompute(mePixRecHitsPerTrack, newmePixRecHitsPerTrack, te );
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
      TLegend* leg2 = new TLegend(0.3, 0.7, 0.6, 0.9);
      can_meCharge->cd(1);
      //gPad->SetLogy();
      SetUpHistograms(meChargeBarrel, newmeChargeBarrel, "barrel, cluster charge (elec) ", leg2 );
      meChargeBarrel->Draw("he");
      newmeChargeBarrel->Draw("samehe"); 
      myPV->PVCompute(meChargeBarrel, newmeChargeBarrel, te );
      h_pv->SetBinContent(++bin, myPV->getPV());
      leg2->Draw();
      
      can_meCharge->cd(2);
      //gPad->SetLogy();
      SetUpHistograms(meChargeZmPanel1, newmeChargeZmPanel1, "panel1, z<0, cluster charge (elec)" );
      meChargeZmPanel1->Draw("he");
      newmeChargeZmPanel1->Draw("samehe"); 
      myPV->PVCompute(meChargeZmPanel1, newmeChargeZmPanel1, te );
      h_pv->SetBinContent(++bin, myPV->getPV());

      can_meCharge->cd(3);
      //gPad->SetLogy();
      SetUpHistograms(meChargeZmPanel2, newmeChargeZmPanel2, "panel2, z<0, cluster charge (elec)" );
      meChargeZmPanel2->Draw("he");
      newmeChargeZmPanel2->Draw("samehe"); 
      myPV->PVCompute(meChargeZmPanel2, newmeChargeZmPanel2, te );
      h_pv->SetBinContent(++bin, myPV->getPV());

      can_meCharge->cd(5);
      //gPad->SetLogy();
      SetUpHistograms(meChargeZpPanel1, newmeChargeZpPanel1, "panel1, z>0, cluster charge (elec)" );
      meChargeZpPanel1->Draw("he");
      newmeChargeZpPanel1->Draw("samehe"); 
      myPV->PVCompute(meChargeZpPanel1, newmeChargeZpPanel1, te );
      h_pv->SetBinContent(++bin, myPV->getPV());

      can_meCharge->cd(6);
      //gPad->SetLogy();
      SetUpHistograms(meChargeZpPanel2, newmeChargeZpPanel2, "panel2, z>0, cluster charge (elec)" );  
      meChargeZpPanel2->Draw("he");
      newmeChargeZpPanel2->Draw("samehe"); 
      myPV->PVCompute(meChargeZpPanel2, newmeChargeZpPanel2, te );
      h_pv->SetBinContent(++bin, myPV->getPV());

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
      
      TLegend* leg3 = new TLegend(0.3, 0.7, 0.6, 0.9);
      can_Errx->cd(1);
      //gPad->SetLogy();
      SetUpHistograms(meErrxBarrel, newmeErrxBarrel, "barrel, x position error (cm)", leg3 );
      meErrxBarrel->Draw("he");
      newmeErrxBarrel->Draw("Samehe"); 
      myPV->PVCompute(meErrxBarrel, newmeErrxBarrel, te );
      leg3->Draw();
      h_pv->SetBinContent(++bin, myPV->getPV());

      can_Errx->cd(2);
      //gPad->SetLogy();
      SetUpHistograms(meErrxZmPanel1, newmeErrxZmPanel1, "panel1, z<0, x position error (cm)" );
      meErrxZmPanel1->Draw("he");
      newmeErrxZmPanel1->Draw("samehe"); 
      myPV->PVCompute(meErrxZmPanel1, newmeErrxZmPanel1, te );
      h_pv->SetBinContent(++bin, myPV->getPV());
      
      can_Errx->cd(3);
      //gPad->SetLogy();
      SetUpHistograms(meErrxZmPanel2, newmeErrxZmPanel2, "panel2, z<0, x position error (cm)" );
      meErrxZmPanel2->Draw("he");
      newmeErrxZmPanel2->Draw("samehe"); 
      myPV->PVCompute(meErrxZmPanel2, newmeErrxZmPanel2, te );
      h_pv->SetBinContent(++bin, myPV->getPV());

      can_Errx->cd(5);
      //gPad->SetLogy();
      SetUpHistograms(meErrxZpPanel1, newmeErrxZpPanel1, "panel1, z>0, x position error (cm)" );
      meErrxZpPanel1->Draw("he");
      newmeErrxZpPanel1->Draw("samehe"); 
      myPV->PVCompute(meErrxZpPanel1, newmeErrxZpPanel1, te );
      h_pv->SetBinContent(++bin, myPV->getPV());

      can_Errx->cd(6);
      //gPad->SetLogy();
      SetUpHistograms(meErrxZpPanel2, newmeErrxZpPanel2, "panel2, z>0, x position error (cm)" );
      meErrxZpPanel2->Draw("he");
      newmeErrxZpPanel2->Draw("samehe"); 
      myPV->PVCompute(meErrxZpPanel2, newmeErrxZpPanel2, te );
      h_pv->SetBinContent(++bin, myPV->getPV());

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
      
      TLegend* leg4 = new TLegend(0.3, 0.7, 0.6, 0.9);
      can_Erry->cd(1);
      //gPad->SetLogy();
      SetUpHistograms(meErryBarrel, newmeErryBarrel, "barrel, y position error (cm)", leg4 );
      meErryBarrel->Draw("he");
      newmeErryBarrel->Draw("Samehe"); 
      myPV->PVCompute(meErryBarrel, newmeErryBarrel, te );
      leg4->Draw();
      h_pv->SetBinContent(++bin, myPV->getPV());

      can_Erry->cd(2);
      //gPad->SetLogy();
      SetUpHistograms(meErryZmPanel1, newmeErryZmPanel1, "panel1, z<0, y position error (cm)"  );
      meErryZmPanel1->Draw("he");
      newmeErryZmPanel1->Draw("samehe"); 
      myPV->PVCompute(meErryZmPanel1, newmeErryZmPanel1, te );
      h_pv->SetBinContent(++bin, myPV->getPV());

      can_Erry->cd(3);
      //gPad->SetLogy();
      SetUpHistograms(meErryZmPanel2, newmeErryZmPanel2, "panel2, z<0, y position error (cm)" );
      meErryZmPanel2->Draw("he");
      newmeErryZmPanel2->Draw("samehe"); 
      myPV->PVCompute(meErryZmPanel2, newmeErryZmPanel2, te );
      h_pv->SetBinContent(++bin, myPV->getPV());

      can_Erry->cd(5);
      //gPad->SetLogy();
      SetUpHistograms(meErryZpPanel1, newmeErryZpPanel1, "panel1, z>0, y position error (cm)" );
      meErryZpPanel1->Draw("he");
      newmeErryZpPanel1->Draw("samehe"); 
      myPV->PVCompute(meErryZpPanel1, newmeErryZpPanel1, te );
      h_pv->SetBinContent(++bin, myPV->getPV());

      can_Erry->cd(6);
      //gPad->SetLogy();
      SetUpHistograms(meErryZpPanel2, newmeErryZpPanel2, "panel2, z>0, y position error (cm)" );
      meErryZpPanel2->Draw("he");
      newmeErryZpPanel2->Draw("samehe"); 
      myPV->PVCompute(meErryZpPanel2, newmeErryZpPanel2, te );
       h_pv->SetBinContent(++bin, myPV->getPV());

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
      
      TLegend* leg5 = new TLegend(0.3, 0.7, 0.6, 0.9);
      can_Npix->cd(1);
      //gPad->SetLogy();
      SetUpHistograms(meNpixBarrel, newmeNpixBarrel, "barrel, cluster size (pixels)", leg5 );
      meNpixBarrel->Draw("he");
      newmeNpixBarrel->Draw("Samehe"); 
      myPV->PVCompute(meNpixBarrel, newmeNpixBarrel, te );
      leg5->Draw();
      h_pv->SetBinContent(++bin, myPV->getPV());

      can_Npix->cd(2);
      //gPad->SetLogy();
      SetUpHistograms(meNpixZmPanel1, newmeNpixZmPanel1, "panel1, z<0, cluster size (pixels)"  );
      meNpixZmPanel1->Draw("he");
      newmeNpixZmPanel1->Draw("samehe"); 
      myPV->PVCompute(meNpixZmPanel1, newmeNpixZmPanel1, te );
      h_pv->SetBinContent(++bin, myPV->getPV());

      can_Npix->cd(3);
      //gPad->SetLogy();
      SetUpHistograms(meNpixZmPanel2, newmeNpixZmPanel2, "panel2, z<0, cluster size (pixels)" );
      meNpixZmPanel2->Draw("he");
      newmeNpixZmPanel2->Draw("samehe"); 
      myPV->PVCompute(meNpixZmPanel2, newmeNpixZmPanel2, te );
      h_pv->SetBinContent(++bin, myPV->getPV());

      can_Npix->cd(5);
      //gPad->SetLogy();
      SetUpHistograms(meNpixZpPanel1, newmeNpixZpPanel1, "panel1, z>0, cluster size (pixels)" );
      meNpixZpPanel1->Draw("he");
      newmeNpixZpPanel1->Draw("samehe"); 
      myPV->PVCompute(meNpixZpPanel1, newmeNpixZpPanel1, te );
      h_pv->SetBinContent(++bin, myPV->getPV());

      can_Npix->cd(6);
      //gPad->SetLogy();
      SetUpHistograms(meNpixZpPanel2, newmeNpixZpPanel2, "panel2, z>0, cluster size (pixels)" );
      meNpixZpPanel2->Draw("he");
      newmeNpixZpPanel2->Draw("samehe"); 
      myPV->PVCompute(meNpixZpPanel2, newmeNpixZpPanel2, te );
      h_pv->SetBinContent(++bin, myPV->getPV());
      
      //can_Npix->SaveAs("meNpix_compare.eps");
      //can_Npix->SaveAs("meNpix_compare.gif");
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
      
      TLegend* leg6 = new TLegend(0.3, 0.7, 0.6, 0.9);
      can_Nxpix->cd(1);
      //gPad->SetLogy();
      SetUpHistograms(meNxpixBarrel, newmeNxpixBarrel, "barrel, cluster x size (pixels)", leg6 );
      meNxpixBarrel->Draw("he");
      newmeNxpixBarrel->Draw("Samehe"); 
      myPV->PVCompute(meNxpixBarrel, newmeNxpixBarrel, te );
      leg6->Draw();
      h_pv->SetBinContent(++bin, myPV->getPV());

      can_Nxpix->cd(2);
      //gPad->SetLogy();
      SetUpHistograms(meNxpixZmPanel1, newmeNxpixZmPanel1, "panel1, z<0, cluster x size (pixels)" );
      meNxpixZmPanel1->Draw("he");
      newmeNxpixZmPanel1->Draw("samehe"); 
      myPV->PVCompute(meNxpixZmPanel1, newmeNxpixZmPanel1, te );
      h_pv->SetBinContent(++bin, myPV->getPV());

      can_Nxpix->cd(3);
      //gPad->SetLogy();
      SetUpHistograms(meNxpixZmPanel2, newmeNxpixZmPanel2, "panel2, z<0, cluster x size (pixels)" );
      meNxpixZmPanel2->Draw("he");
      newmeNxpixZmPanel2->Draw("samehe"); 
      myPV->PVCompute(meNxpixZmPanel2, newmeNxpixZmPanel2, te );
      h_pv->SetBinContent(++bin, myPV->getPV());

      can_Nxpix->cd(5);
      //gPad->SetLogy();
      SetUpHistograms(meNxpixZpPanel1, newmeNxpixZpPanel1, "panel1, z>0, cluster x size (pixels)" );
      meNxpixZpPanel1->Draw("he");
      newmeNxpixZpPanel1->Draw("samehe"); 
      myPV->PVCompute(meNxpixZpPanel1, newmeNxpixZpPanel1, te );
      h_pv->SetBinContent(++bin, myPV->getPV());

      can_Nxpix->cd(6);
      //gPad->SetLogy();
      SetUpHistograms(meNxpixZpPanel2, newmeNxpixZpPanel2, "panel2, z>0, cluster x size (pixels)" );
      meNxpixZpPanel2->Draw("he");
      newmeNxpixZpPanel2->Draw("samehe"); 
      myPV->PVCompute(meNxpixZpPanel2, newmeNxpixZpPanel2, te );
      h_pv->SetBinContent(++bin, myPV->getPV());

      //can_Nxpix->SaveAs("meNxpix_compare.eps");
      //can_Nxpix->SaveAs("meNxpix_compare.gif");
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
      
      TLegend* leg7 = new TLegend(0.3, 0.7, 0.6, 0.9);
      can_Nypix->cd(1);
      //gPad->SetLogy();
      SetUpHistograms(meNypixBarrel, newmeNypixBarrel, "barrel, cluster y size (pixels)", leg7 );
      meNypixBarrel->Draw("he");
      newmeNypixBarrel->Draw("Samehe"); 
      myPV->PVCompute(meNypixBarrel, newmeNypixBarrel, te );
      leg7->Draw();
      h_pv->SetBinContent(++bin, myPV->getPV());

      can_Nypix->cd(2);
      //gPad->SetLogy();
      SetUpHistograms(meNypixZmPanel1, newmeNypixZmPanel1, "panel1, z<0, cluster y size (pixels)" );
      meNypixZmPanel1->Draw("he");
      newmeNypixZmPanel1->Draw("samehe"); 
      myPV->PVCompute(meNypixZmPanel1, newmeNypixZmPanel1, te );
      h_pv->SetBinContent(++bin, myPV->getPV());

      can_Nypix->cd(3);
      //gPad->SetLogy();
      SetUpHistograms(meNypixZmPanel2, newmeNypixZmPanel2, "panel2, z<0, cluster y size (pixels)" );
      meNypixZmPanel2->Draw("he");
      newmeNypixZmPanel2->Draw("samehe"); 
      myPV->PVCompute(meNypixZmPanel2, newmeNypixZmPanel2, te );
      h_pv->SetBinContent(++bin, myPV->getPV());

      can_Nypix->cd(5);
      //gPad->SetLogy();
      SetUpHistograms(meNypixZpPanel1, newmeNypixZpPanel1, "panel1, z>0, cluster y size (pixels)" );
      meNypixZpPanel1->Draw("he");
      newmeNypixZpPanel1->Draw("samehe"); 
      myPV->PVCompute(meNypixZpPanel1, newmeNypixZpPanel1, te );
      h_pv->SetBinContent(++bin, myPV->getPV());

      can_Nypix->cd(6);
      //gPad->SetLogy();
      SetUpHistograms(meNypixZpPanel2, newmeNypixZpPanel2, "panel2, z>0, cluster y size (pixels)" );
      meNypixZpPanel2->Draw("he");
      newmeNypixZpPanel2->Draw("samehe"); 
      myPV->PVCompute(meNypixZpPanel2, newmeNypixZpPanel2, te );
      h_pv->SetBinContent(++bin, myPV->getPV());

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
      
      TLegend* leg8 = new TLegend(0.3, 0.7, 0.6, 0.9);
      can_Posx->cd(1);
      //gPad->SetLogy();
      SetUpHistograms(mePosxBarrel, newmePosxBarrel, "barrel, x (cm)", leg8 );
      mePosxBarrel->Draw("he");
      newmePosxBarrel->Draw("Samehe"); 
      myPV->PVCompute(mePosxBarrel, newmePosxBarrel, te );
      leg8->Draw();
      h_pv->SetBinContent(++bin, myPV->getPV());
      
      can_Posx->cd(2);
      //gPad->SetLogy();
      SetUpHistograms(mePosxZmPanel1, newmePosxZmPanel1, "panel1, z<0, x (cm)" );
      mePosxZmPanel1->Draw("he");
      newmePosxZmPanel1->Draw("samehe"); 
      myPV->PVCompute(mePosxZmPanel1, newmePosxZmPanel1, te );
      h_pv->SetBinContent(++bin, myPV->getPV());

      can_Posx->cd(3);
      //gPad->SetLogy();
      SetUpHistograms(mePosxZmPanel2, newmePosxZmPanel2, "panel2, z>0, x (cm)" );
      mePosxZmPanel2->Draw("he");
      newmePosxZmPanel2->Draw("samehe"); 
      myPV->PVCompute(mePosxZmPanel2, newmePosxZmPanel2, te );
      h_pv->SetBinContent(++bin, myPV->getPV());

      can_Posx->cd(5);
      //gPad->SetLogy();
      SetUpHistograms(mePosxZpPanel1, newmePosxZpPanel1, "panel1, z<0, x (cm)" );
      mePosxZpPanel1->Draw("he");
      newmePosxZpPanel1->Draw("samehe"); 
      myPV->PVCompute(mePosxZpPanel1, newmePosxZpPanel1, te );
      h_pv->SetBinContent(++bin, myPV->getPV());

      can_Posx->cd(6);
      //gPad->SetLogy();
      SetUpHistograms(mePosxZpPanel2, newmePosxZpPanel2, "panel2, z>0, x (cm)" );
      mePosxZpPanel2->Draw("he");
      newmePosxZpPanel2->Draw("samehe"); 
      myPV->PVCompute(mePosxZpPanel2, newmePosxZpPanel2, te );
      h_pv->SetBinContent(++bin, myPV->getPV());

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
      
      TLegend* leg9 = new TLegend(0.3, 0.7, 0.6, 0.9);
      can_Posy->cd(1);
      //gPad->SetLogy();
      SetUpHistograms(mePosyBarrel, newmePosyBarrel, "barrel, y (cm)", leg9 );
      mePosyBarrel->Draw("he");
      newmePosyBarrel->Draw("Samehe"); 
      myPV->PVCompute(mePosyBarrel, newmePosyBarrel, te );
      leg9->Draw();
      h_pv->SetBinContent(++bin, myPV->getPV());

      can_Posy->cd(2);
      //gPad->SetLogy();
      SetUpHistograms(mePosyZmPanel1,  newmePosyZmPanel1, "panel1, z<0, y (cm)" );
      mePosyZmPanel1->Draw("he");
      newmePosyZmPanel1->Draw("samehe"); 
      myPV->PVCompute(mePosyZmPanel1, newmePosyZmPanel1, te );
      h_pv->SetBinContent(++bin, myPV->getPV());

      can_Posy->cd(3);
      //gPad->SetLogy();
      SetUpHistograms(mePosyZmPanel2, newmePosyZmPanel2, "panel2, z<0, y (cm)" );
      mePosyZmPanel2->Draw("he");
      newmePosyZmPanel2->Draw("samehe"); 
      myPV->PVCompute(mePosyZmPanel2, newmePosyZmPanel2, te );
      h_pv->SetBinContent(++bin, myPV->getPV());

      can_Posy->cd(5);
      //gPad->SetLogy();
      SetUpHistograms(mePosyZpPanel1, newmePosyZpPanel1, "panel1, z>0, y (cm)" );
      mePosyZpPanel1->Draw("he");
      newmePosyZpPanel1->Draw("samehe"); 
      myPV->PVCompute(mePosyZpPanel1, newmePosyZpPanel1, te );
      h_pv->SetBinContent(++bin, myPV->getPV());

      can_Posy->cd(6);
      //gPad->SetLogy();
      SetUpHistograms(mePosyZpPanel2, newmePosyZpPanel2, "panel2, z>0, y (cm)" );
      mePosyZpPanel2->Draw("he");
      newmePosyZpPanel2->Draw("samehe"); 
      myPV->PVCompute(mePosyZpPanel2, newmePosyZpPanel2, te );
      h_pv->SetBinContent(++bin, myPV->getPV());

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
      
      TLegend* leg10 = new TLegend(0.3, 0.7, 0.6, 0.9);
      can_PullXvsAlpha->cd(1);
      //gPad->SetLogy();
      SetUpProfileHistograms(mePullXvsAlphaBarrel, newmePullXvsAlphaBarrel, "barrel, |alpha| (deg)", "pull x", lpull, hpull, leg10 );
      mePullXvsAlphaBarrel->Draw("e");
      newmePullXvsAlphaBarrel->Draw("Samee"); 
      myPV->PVCompute(mePullXvsAlphaBarrel, newmePullXvsAlphaBarrel, te );
      leg10->Draw();
      h_pv->SetBinContent(++bin, myPV->getPV());

      can_PullXvsAlpha->cd(2);
      //gPad->SetLogy();
      SetUpProfileHistograms(mePullXvsAlphaZmPanel1, newmePullXvsAlphaZmPanel1, "panel1, z<0, |alpha| (deg)", "pull x", lpull, hpull );
      mePullXvsAlphaZmPanel1->Draw("e");
      newmePullXvsAlphaZmPanel1->Draw("samee"); 
      myPV->PVCompute(mePullXvsAlphaZmPanel1, newmePullXvsAlphaZmPanel1, te );
      h_pv->SetBinContent(++bin, myPV->getPV());

      can_PullXvsAlpha->cd(3);
      //gPad->SetLogy();
      SetUpProfileHistograms(mePullXvsAlphaZmPanel2, newmePullXvsAlphaZmPanel2, "panel2, z<0, |alpha| (deg)", "pull x", lpull, hpull );
      mePullXvsAlphaZmPanel2->Draw("e");
      newmePullXvsAlphaZmPanel2->Draw("samee"); 
      myPV->PVCompute(mePullXvsAlphaZmPanel2, newmePullXvsAlphaZmPanel2, te );
      h_pv->SetBinContent(++bin, myPV->getPV());

      can_PullXvsAlpha->cd(5);
      //gPad->SetLogy();
      SetUpProfileHistograms(mePullXvsAlphaZpPanel1, newmePullXvsAlphaZpPanel1, "panel1, z>0, |alpha| (deg)", "pull x", lpull, hpull );
      mePullXvsAlphaZpPanel1->Draw("e");
      newmePullXvsAlphaZpPanel1->Draw("samee"); 
      myPV->PVCompute(mePullXvsAlphaZpPanel1, newmePullXvsAlphaZpPanel1, te );
      h_pv->SetBinContent(++bin, myPV->getPV());

      can_PullXvsAlpha->cd(6);
      //gPad->SetLogy();
      SetUpProfileHistograms(mePullXvsAlphaZpPanel2, newmePullXvsAlphaZpPanel2, "panel2, z>0, |alpha| (deg)", "pull x", lpull, hpull );
      mePullXvsAlphaZpPanel2->Draw("e");
      newmePullXvsAlphaZpPanel2->Draw("samee"); 
      myPV->PVCompute(mePullXvsAlphaZpPanel2, newmePullXvsAlphaZpPanel2, te );
      h_pv->SetBinContent(++bin, myPV->getPV());

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
      
      TLegend* leg11 = new TLegend(0.3, 0.7, 0.6, 0.9);
      can_PullXvsBeta->cd(1);
      //gPad->SetLogy();
      SetUpProfileHistograms(mePullXvsBetaBarrel, newmePullXvsBetaBarrel, "barrel, |beta| (deg)", "pull x", lpull, hpull, leg11 );
      mePullXvsBetaBarrel->Draw("e");
      newmePullXvsBetaBarrel->Draw("Samee"); 
      myPV->PVCompute(mePullXvsBetaBarrel, newmePullXvsBetaBarrel, te );
      leg11->Draw();
      h_pv->SetBinContent(++bin, myPV->getPV());

      can_PullXvsBeta->cd(2);
      //gPad->SetLogy();
      SetUpProfileHistograms(mePullXvsBetaZmPanel1, newmePullXvsBetaZmPanel1, "panel1, z<0, |beta| (deg)", "pull x", lpull, hpull );
      mePullXvsBetaZmPanel1->Draw("e");
      newmePullXvsBetaZmPanel1->Draw("samee"); 
      myPV->PVCompute(mePullXvsBetaZmPanel1, newmePullXvsBetaZmPanel1, te );
      h_pv->SetBinContent(++bin, myPV->getPV());

      can_PullXvsBeta->cd(3);
      //gPad->SetLogy();
      SetUpProfileHistograms(mePullXvsBetaZmPanel2, newmePullXvsBetaZmPanel2, "panel2, z<0, |beta| (deg)", "pull x", lpull, hpull );
      mePullXvsBetaZmPanel2->Draw("e");
      newmePullXvsBetaZmPanel2->Draw("samee"); 
      myPV->PVCompute(mePullXvsBetaZmPanel2, newmePullXvsBetaZmPanel2, te );
      h_pv->SetBinContent(++bin, myPV->getPV());

      can_PullXvsBeta->cd(5);
      //gPad->SetLogy();
      SetUpProfileHistograms(mePullXvsBetaZpPanel1, newmePullXvsBetaZpPanel1, "panel1, z>0, |beta| (deg)", "pull x", lpull, hpull );
      mePullXvsBetaZpPanel1->Draw("e");
      newmePullXvsBetaZpPanel1->Draw("samee"); 
      myPV->PVCompute(mePullXvsBetaZpPanel1, newmePullXvsBetaZpPanel1, te );
      h_pv->SetBinContent(++bin, myPV->getPV());

      can_PullXvsBeta->cd(6);
      //gPad->SetLogy();
      SetUpProfileHistograms(mePullXvsBetaZpPanel2, newmePullXvsBetaZpPanel2, "panel2, z>0, |beta| (deg)", "pull x", lpull, hpull );
      mePullXvsBetaZpPanel2->Draw("e");
      newmePullXvsBetaZpPanel2->Draw("samee"); 
      myPV->PVCompute(mePullXvsBetaZpPanel2, newmePullXvsBetaZpPanel2, te );
      h_pv->SetBinContent(++bin, myPV->getPV());

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
      TProfile* newmeWPullXvsAlphaBarrel;
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
      
      sdir->GetObject("Histograms_all/meWPullXvsAlphaBarrelNonFlippedLadders", 
		       newmeWPullXvsAlphaBarrelNFP  );
      sdir->GetObject("Histograms_all/meWPullXvsAlphaBarrelFlippedLadders"   , 
		       newmeWPullXvsAlphaBarrelFP   );
      sdir->GetObject("Histograms_all/meWPullXvsAlphaZmPanel1", newmeWPullXvsAlphaZmPanel1);
      sdir->GetObject("Histograms_all/meWPullXvsAlphaZmPanel2", newmeWPullXvsAlphaZmPanel2);
      sdir->GetObject("Histograms_all/meWPullXvsAlphaZpPanel1", newmeWPullXvsAlphaZpPanel1);
      sdir->GetObject("Histograms_all/meWPullXvsAlphaZpPanel2", newmeWPullXvsAlphaZpPanel2);
      
      TLegend* leg10 = new TLegend(0.3, 0.7, 0.6, 0.9);
      can_WPullXvsAlpha->cd(1);
      //gPad->SetLogy();
      SetUpProfileHistograms(meWPullXvsAlphaBarrelNFP, newmeWPullXvsAlphaBarrelNFP, 
			     "non-flipped  ladders, barrel, |alpha| (deg)", "< | pull x | >", lwpull, hwpull, leg10 );
      meWPullXvsAlphaBarrelNFP->Draw("e");
      newmeWPullXvsAlphaBarrelNFP->Draw("Samee"); 
      myPV->PVCompute(meWPullXvsAlphaBarrelNFP, newmeWPullXvsAlphaBarrelNFP, te );
      leg10->Draw();
      h_pv->SetBinContent(++bin, myPV->getPV());

      can_WPullXvsAlpha->cd(2);
      //gPad->SetLogy();
      SetUpProfileHistograms(meWPullXvsAlphaZmPanel1, newmeWPullXvsAlphaZmPanel1, 
			     "panel1, z<0, |alpha| (deg)", "< | pull x | >", lwpull, hwpull );
      meWPullXvsAlphaZmPanel1->Draw("e");
      newmeWPullXvsAlphaZmPanel1->Draw("samee"); 
      myPV->PVCompute(meWPullXvsAlphaZmPanel1, newmeWPullXvsAlphaZmPanel1, te );
      h_pv->SetBinContent(++bin, myPV->getPV());

      can_WPullXvsAlpha->cd(3);
      //gPad->SetLogy();
      SetUpProfileHistograms(meWPullXvsAlphaZmPanel2, newmeWPullXvsAlphaZmPanel2, 
			     "panel2, z<0, |alpha| (deg)", "< | pull x | >", lwpull, hwpull );
      meWPullXvsAlphaZmPanel2->Draw("e");
      newmeWPullXvsAlphaZmPanel2->Draw("samee"); 
      myPV->PVCompute(meWPullXvsAlphaZmPanel2, newmeWPullXvsAlphaZmPanel2, te );
      h_pv->SetBinContent(++bin, myPV->getPV());

      can_WPullXvsAlpha->cd(4);
      //gPad->SetLogy();
      SetUpProfileHistograms(meWPullXvsAlphaBarrelFP, newmeWPullXvsAlphaBarrelFP, 
			     "flipped ladders, barrel, |alpha| (deg)", "< | pull x | >", lwpull, hwpull);
      meWPullXvsAlphaBarrelFP->Draw("e");
      newmeWPullXvsAlphaBarrelFP->Draw("Samee"); 
      myPV->PVCompute(meWPullXvsAlphaBarrelFP, newmeWPullXvsAlphaBarrelFP, te );
      h_pv->SetBinContent(++bin, myPV->getPV());

      can_WPullXvsAlpha->cd(5);
      //gPad->SetLogy();
      SetUpProfileHistograms(meWPullXvsAlphaZpPanel1, newmeWPullXvsAlphaZpPanel1, 
			     "panel1, z>0, |alpha| (deg)", "< | pull x | >", lwpull, hwpull );
      meWPullXvsAlphaZpPanel1->Draw("e");
      newmeWPullXvsAlphaZpPanel1->Draw("samee"); 
      myPV->PVCompute(meWPullXvsAlphaZpPanel1, newmeWPullXvsAlphaZpPanel1, te );
      h_pv->SetBinContent(++bin, myPV->getPV());

      can_WPullXvsAlpha->cd(6);
      //gPad->SetLogy();
      SetUpProfileHistograms(meWPullXvsAlphaZpPanel2, newmeWPullXvsAlphaZpPanel2, 
			     "panel2, z>0, |alpha| (deg)", "< | pull x | >", lwpull, hwpull );
      meWPullXvsAlphaZpPanel2->Draw("e");
      newmeWPullXvsAlphaZpPanel2->Draw("samee"); 
      myPV->PVCompute(meWPullXvsAlphaZpPanel2, newmeWPullXvsAlphaZpPanel2, te );
      h_pv->SetBinContent(++bin, myPV->getPV());

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
  
    TLegend* leg13 = new TLegend(0.3, 0.7, 0.6, 0.9);
    can_PullXvsPhi->cd(1);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullXvsPhiBarrel, newmePullXvsPhiBarrel, "barrel, phi (deg)", "pull x", lpull, hpull, leg13 );
    mePullXvsPhiBarrel->Draw("e");
    newmePullXvsPhiBarrel->Draw("Samee"); 
    myPV->PVCompute(mePullXvsPhiBarrel, newmePullXvsPhiBarrel, te );
    leg13->Draw();
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_PullXvsPhi->cd(2);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullXvsPhiZmPanel1, newmePullXvsPhiZmPanel1, "panel1, z<0, phi (deg)", "pull x", lpull, hpull );
    mePullXvsPhiZmPanel1->Draw("e");
    newmePullXvsPhiZmPanel1->Draw("samee"); 
    myPV->PVCompute(mePullXvsPhiZmPanel1, newmePullXvsPhiZmPanel1, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_PullXvsPhi->cd(3);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullXvsPhiZmPanel2, newmePullXvsPhiZmPanel2, "panel2, z<0, phi (deg)", "pull x", lpull, hpull );
    mePullXvsPhiZmPanel2->Draw("e");
    newmePullXvsPhiZmPanel2->Draw("samee"); 
    myPV->PVCompute(mePullXvsPhiZmPanel2, newmePullXvsPhiZmPanel2, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_PullXvsPhi->cd(5);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullXvsPhiZpPanel1, newmePullXvsPhiZpPanel1, "panel1, z>0, phi (deg)", "pull x", lpull, hpull );
    mePullXvsPhiZpPanel1->Draw("e");
    newmePullXvsPhiZpPanel1->Draw("samee"); 
    myPV->PVCompute(mePullXvsPhiZpPanel1, newmePullXvsPhiZpPanel1, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_PullXvsPhi->cd(6);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullXvsPhiZpPanel2, newmePullXvsPhiZpPanel2, "panel2, z>0, phi (deg)", "pull x" , lpull, hpull);
    mePullXvsPhiZpPanel2->Draw("e");
    newmePullXvsPhiZpPanel2->Draw("samee"); 
    myPV->PVCompute(mePullXvsPhiZpPanel2, newmePullXvsPhiZpPanel2, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

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
  
    TLegend* leg14 = new TLegend(0.3, 0.7, 0.6, 0.9);
    can_PullYvsAlpha->cd(1);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsAlphaBarrel, newmePullYvsAlphaBarrel, "barrel, |alpha| (deg)", "pull y", lpull, hpull, leg14 );
    mePullYvsAlphaBarrel->Draw("e");
    newmePullYvsAlphaBarrel->Draw("Samee"); 
    myPV->PVCompute(mePullYvsAlphaBarrel, newmePullYvsAlphaBarrel, te );
    leg14->Draw();
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_PullYvsAlpha->cd(2);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsAlphaZmPanel1, newmePullYvsAlphaZmPanel1, "panel1, z<0, |alpha| (deg)", "pull y", lpull, hpull );
    mePullYvsAlphaZmPanel1->Draw("e");
    newmePullYvsAlphaZmPanel1->Draw("samee"); 
    myPV->PVCompute(mePullYvsAlphaZmPanel1, newmePullYvsAlphaZmPanel1, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_PullYvsAlpha->cd(3);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsAlphaZmPanel2, newmePullYvsAlphaZmPanel2, "panel2, z<0, |alpha| (deg)", "pull y", lpull, hpull );
    mePullYvsAlphaZmPanel2->Draw("e");
    newmePullYvsAlphaZmPanel2->Draw("samee"); 
    myPV->PVCompute(mePullYvsAlphaZmPanel2, newmePullYvsAlphaZmPanel2, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_PullYvsAlpha->cd(5);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsAlphaZpPanel1, newmePullYvsAlphaZpPanel1, "panel1, z>0, |alpha| (deg)", "pull y", lpull, hpull );
    mePullYvsAlphaZpPanel1->Draw("e");
    newmePullYvsAlphaZpPanel1->Draw("samee"); 
    myPV->PVCompute(mePullYvsAlphaZpPanel1, newmePullYvsAlphaZpPanel1, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_PullYvsAlpha->cd(6);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsAlphaZpPanel2, newmePullYvsAlphaZpPanel2, "panel2, z>0, |alpha| (deg)", "pull y" , lpull, hpull);
    mePullYvsAlphaZpPanel2->Draw("e");
    newmePullYvsAlphaZpPanel2->Draw("samee"); 
    myPV->PVCompute(mePullYvsAlphaZpPanel2, newmePullYvsAlphaZpPanel2, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

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
  
    TLegend* leg15 = new TLegend(0.3, 0.7, 0.6, 0.9);
    can_PullYvsBeta->cd(1);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsBetaBarrel, newmePullYvsBetaBarrel, "barrel, |beta| (deg)", "pull y", lpull, hpull, leg15 );
    mePullYvsBetaBarrel->Draw("e");
    newmePullYvsBetaBarrel->Draw("Samee"); 
    myPV->PVCompute(mePullYvsBetaBarrel, newmePullYvsBetaBarrel, te );
    leg15->Draw();
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_PullYvsBeta->cd(2);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsBetaZmPanel1, newmePullYvsBetaZmPanel1, "panel1, z<0, |beta| (deg)", "pull y", lpull, hpull );
    mePullYvsBetaZmPanel1->Draw("e");
    newmePullYvsBetaZmPanel1->Draw("samee"); 
    myPV->PVCompute(mePullYvsBetaZmPanel1, newmePullYvsBetaZmPanel1, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_PullYvsBeta->cd(3);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsBetaZmPanel2, newmePullYvsBetaZmPanel2, "panel2, z<0, |beta| (deg)", "pull y", lpull, hpull );
    mePullYvsBetaZmPanel2->Draw("e");
    newmePullYvsBetaZmPanel2->Draw("samee"); 
    myPV->PVCompute(mePullYvsBetaZmPanel2, newmePullYvsBetaZmPanel2, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_PullYvsBeta->cd(5);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsBetaZpPanel1, newmePullYvsBetaZpPanel1, "panel1, z>0, |beta| (deg)", "pull y", lpull, hpull );
    mePullYvsBetaZpPanel1->Draw("e");
    newmePullYvsBetaZpPanel1->Draw("samee"); 
    myPV->PVCompute(mePullYvsBetaZpPanel1, newmePullYvsBetaZpPanel1, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_PullYvsBeta->cd(6);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsBetaZpPanel2, newmePullYvsBetaZpPanel2, "panel2, z>0, |beta| (deg)", "pull y", lpull, hpull );
    mePullYvsBetaZpPanel2->Draw("e");
    newmePullYvsBetaZpPanel2->Draw("samee"); 
    myPV->PVCompute(mePullYvsBetaZpPanel2, newmePullYvsBetaZpPanel2, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

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
  
    TLegend* leg15 = new TLegend(0.3, 0.7, 0.6, 0.9);
    can_WPullYvsBeta->cd(1);
    //gPad->SetLogy();
    SetUpProfileHistograms(meWPullYvsBetaBarrelNFP, newmeWPullYvsBetaBarrelNFP, 
			   "non-flipped ladders, barrel, |beta| (deg)", "< | pull y | > ", lwpull, hwpull, leg15 );
    meWPullYvsBetaBarrelNFP->Draw("e");
    newmeWPullYvsBetaBarrelNFP->Draw("Samee"); 
    myPV->PVCompute(meWPullYvsBetaBarrelNFP, newmeWPullYvsBetaBarrelNFP, te );
    leg15->Draw();
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_WPullYvsBeta->cd(2);
    //gPad->SetLogy();
    SetUpProfileHistograms(meWPullYvsBetaZmPanel1, newmeWPullYvsBetaZmPanel1, 
			   "panel1, z<0, |beta| (deg)", "< | pull y | > ", lwpull, hwpull );
    meWPullYvsBetaZmPanel1->Draw("e");
    newmeWPullYvsBetaZmPanel1->Draw("samee"); 
    myPV->PVCompute(meWPullYvsBetaZmPanel1, newmeWPullYvsBetaZmPanel1, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_WPullYvsBeta->cd(3);
    //gPad->SetLogy();
    SetUpProfileHistograms(meWPullYvsBetaZmPanel2, newmeWPullYvsBetaZmPanel2, 
			   "panel2, z<0, |beta| (deg)", "< | pull y | > ", lwpull, hwpull );
    meWPullYvsBetaZmPanel2->Draw("e");
    newmeWPullYvsBetaZmPanel2->Draw("samee"); 
    myPV->PVCompute(meWPullYvsBetaZmPanel2, newmeWPullYvsBetaZmPanel2, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_WPullYvsBeta->cd(4);
    //gPad->SetLogy();
    SetUpProfileHistograms(meWPullYvsBetaBarrelFP, newmeWPullYvsBetaBarrelFP, 
			   "flipped ladders, barrel, |beta| (deg)", "< | pull y | > ", lwpull, hwpull);
    meWPullYvsBetaBarrelFP->Draw("e");
    newmeWPullYvsBetaBarrelFP->Draw("Samee"); 
    myPV->PVCompute(meWPullYvsBetaBarrelFP, newmeWPullYvsBetaBarrelFP, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_WPullYvsBeta->cd(5);
    //gPad->SetLogy();
    SetUpProfileHistograms(meWPullYvsBetaZpPanel1, newmeWPullYvsBetaZpPanel1, 
			   "panel1, z>0, |beta| (deg)", "< | pull y | > ", lwpull, hwpull );
    meWPullYvsBetaZpPanel1->Draw("e");
    newmeWPullYvsBetaZpPanel1->Draw("samee"); 
    myPV->PVCompute(meWPullYvsBetaZpPanel1, newmeWPullYvsBetaZpPanel1, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_WPullYvsBeta->cd(6);
    //gPad->SetLogy();
    SetUpProfileHistograms(meWPullYvsBetaZpPanel2, newmeWPullYvsBetaZpPanel2, 
			   "panel2, z>0, |beta| (deg)", "< | pull y | > ", lwpull, hwpull );
    meWPullYvsBetaZpPanel2->Draw("e");
    newmeWPullYvsBetaZpPanel2->Draw("samee"); 
    myPV->PVCompute(meWPullYvsBetaZpPanel2, newmeWPullYvsBetaZpPanel2, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

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
  
    TLegend* leg16 = new TLegend(0.3, 0.7, 0.6, 0.9);
    can_PullYvsEta->cd(1);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsEtaBarrel, newmePullYvsEtaBarrel, "barrel, eta", "pull y", lpull, hpull, leg16 );
    mePullYvsEtaBarrel->Draw("e");
    newmePullYvsEtaBarrel->Draw("Samee"); 
    myPV->PVCompute(mePullYvsEtaBarrel, newmePullYvsEtaBarrel, te );
    leg16->Draw();
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_PullYvsEta->cd(2);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsEtaZmPanel1, newmePullYvsEtaZmPanel1, "panel1, z<0, eta", "pull y" , lpull, hpull);
    mePullYvsEtaZmPanel1->Draw("e");
    newmePullYvsEtaZmPanel1->Draw("samee"); 
    myPV->PVCompute(mePullYvsEtaZmPanel1, newmePullYvsEtaZmPanel1, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_PullYvsEta->cd(3);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsEtaZmPanel2, newmePullYvsEtaZmPanel2, "panel2, z<0, eta", "pull y", lpull, hpull );
    mePullYvsEtaZmPanel2->Draw("e");
    newmePullYvsEtaZmPanel2->Draw("samee"); 
    myPV->PVCompute(mePullYvsEtaZmPanel2, newmePullYvsEtaZmPanel2, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_PullYvsEta->cd(5);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsEtaZpPanel1, newmePullYvsEtaZpPanel1, "panel1, z>0, eta", "pull y", lpull, hpull );
    mePullYvsEtaZpPanel1->Draw("e");
    newmePullYvsEtaZpPanel1->Draw("samee"); 
    myPV->PVCompute(mePullYvsEtaZpPanel1, newmePullYvsEtaZpPanel1, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_PullYvsEta->cd(6);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsEtaZpPanel2, newmePullYvsEtaZpPanel2, "panel2, z>0, eta", "pull y", lpull, hpull );
    mePullYvsEtaZpPanel2->Draw("e");
    newmePullYvsEtaZpPanel2->Draw("samee"); 
    myPV->PVCompute(mePullYvsEtaZpPanel2, newmePullYvsEtaZpPanel2, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

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
  
    TLegend* leg17 = new TLegend(0.3, 0.7, 0.6, 0.9);
    can_PullYvsPhi->cd(1);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsPhiBarrel, newmePullYvsPhiBarrel, "barrel, phi (deg)", "pull y", lpull, hpull, leg17 );
    mePullYvsPhiBarrel->Draw("e");
    newmePullYvsPhiBarrel->Draw("Samee"); 
    myPV->PVCompute(mePullYvsPhiBarrel, newmePullYvsPhiBarrel, te );
    leg17->Draw();
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_PullYvsPhi->cd(2);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsPhiZmPanel1, newmePullYvsPhiZmPanel1, "panel1, z<0, phi (deg)", "pull y" , lpull, hpull);
    mePullYvsPhiZmPanel1->Draw("e");
    newmePullYvsPhiZmPanel1->Draw("samee"); 
    myPV->PVCompute(mePullYvsPhiZmPanel1, newmePullYvsPhiZmPanel1, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_PullYvsPhi->cd(3);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsPhiZmPanel2, newmePullYvsPhiZmPanel2, "panel2, z<0, phi (deg)", "pull y" , lpull, hpull);
    mePullYvsPhiZmPanel2->Draw("e");
    newmePullYvsPhiZmPanel2->Draw("samee"); 
    myPV->PVCompute(mePullYvsPhiZmPanel2, newmePullYvsPhiZmPanel2, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_PullYvsPhi->cd(5);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsPhiZpPanel1, newmePullYvsPhiZpPanel1, "panel1, z>0, phi (deg)", "pull y", lpull, hpull );
    mePullYvsPhiZpPanel1->Draw("e");
    newmePullYvsPhiZpPanel1->Draw("samee"); 
    myPV->PVCompute(mePullYvsPhiZpPanel1, newmePullYvsPhiZpPanel1, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_PullYvsPhi->cd(6);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsPhiZpPanel2, newmePullYvsPhiZpPanel2, "panel2, z>0, phi (deg)", "pull y", lpull, hpull );
    mePullYvsPhiZpPanel2->Draw("e");
    newmePullYvsPhiZpPanel2->Draw("samee"); 
    myPV->PVCompute(mePullYvsPhiZpPanel2, newmePullYvsPhiZpPanel2, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

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
  
    TLegend* leg18 = new TLegend(0.3, 0.7, 0.6, 0.9);
    can_mePullx->cd(1);
    //gPad->SetLogy();
    SetUpHistograms(mePullxBarrel, newmePullxBarrel, "barrel, pull x", leg18);
    mePullxBarrel->Draw("he");
    newmePullxBarrel->Draw("Samehe"); 
    myPV->PVCompute(mePullxBarrel, newmePullxBarrel, te );
    leg18->Draw();
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_mePullx->cd(2);
    //gPad->SetLogy();
    SetUpHistograms(mePullxZmPanel1, newmePullxZmPanel1, "panel1, z<0, pull x" );
    mePullxZmPanel1->Draw("he");
    newmePullxZmPanel1->Draw("samehe"); 
    myPV->PVCompute(mePullxZmPanel1, newmePullxZmPanel1, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_mePullx->cd(3);
    //gPad->SetLogy();
    SetUpHistograms(mePullxZmPanel2, newmePullxZmPanel2, "panel2, z<0, pull x" );
    mePullxZmPanel2->Draw("he");
    newmePullxZmPanel2->Draw("samehe"); 
    myPV->PVCompute(mePullxZmPanel2, newmePullxZmPanel2, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_mePullx->cd(5);
    //gPad->SetLogy();
    SetUpHistograms(mePullxZpPanel1, newmePullxZpPanel1, "panel2, z>0, pull x" );
    mePullxZpPanel1->Draw("he");
    newmePullxZpPanel1->Draw("samehe"); 
    myPV->PVCompute(mePullxZpPanel1, newmePullxZpPanel1, te );
    h_pv->SetBinContent(++bin, myPV->getPV());
    
    can_mePullx->cd(6);
    //gPad->SetLogy();
    SetUpHistograms(mePullxZpPanel2, newmePullxZpPanel2, "panel1, z>0, pull x" );
    mePullxZpPanel2->Draw("he");
    newmePullxZpPanel2->Draw("samehe"); 
    myPV->PVCompute(mePullxZpPanel2, newmePullxZpPanel2, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

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
  
    TLegend* leg19 = new TLegend(0.3, 0.7, 0.6, 0.9);
    can_mePully->cd(1);
    //gPad->SetLogy();
    SetUpHistograms(mePullyBarrel, newmePullyBarrel, "barrel, pull y", leg19 );
    mePullyBarrel->Draw("he");
    newmePullyBarrel->Draw("Samehe"); 
    myPV->PVCompute(mePullyBarrel, newmePullyBarrel, te );
    leg19->Draw();
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_mePully->cd(2);
    //gPad->SetLogy();
    SetUpHistograms(mePullyZmPanel1, newmePullyZmPanel1, "panel1, z<0, pull y" );
    mePullyZmPanel1->Draw("he");
    newmePullyZmPanel1->Draw("samehe"); 
    myPV->PVCompute(mePullyZmPanel1, newmePullyZmPanel1, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_mePully->cd(3);
    //gPad->SetLogy();
    SetUpHistograms(mePullyZmPanel2, newmePullyZmPanel2, "panel2, z<0, pull y" );
    mePullyZmPanel2->Draw("he");
    newmePullyZmPanel2->Draw("samehe"); 
    myPV->PVCompute(mePullyZmPanel2, newmePullyZmPanel2, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_mePully->cd(5);
    //gPad->SetLogy();
    SetUpHistograms(mePullyZpPanel1, newmePullyZpPanel1, "panel1, z>0, pull y" );
    mePullyZpPanel1->Draw("he");
    newmePullyZpPanel1->Draw("samehe"); 
    myPV->PVCompute(mePullyZpPanel1, newmePullyZpPanel1, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_mePully->cd(6);
    //gPad->SetLogy();
    SetUpHistograms(mePullyZpPanel2, newmePullyZpPanel2, "panel2, z>0, pull y" );
    mePullyZpPanel2->Draw("he");
    newmePullyZpPanel2->Draw("samehe"); 
    myPV->PVCompute(mePullyZpPanel2, newmePullyZpPanel2, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

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
  
    TLegend* leg20 = new TLegend(0.3, 0.7, 0.6, 0.9);
    can_ResXvsAlpha->cd(1);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResXvsAlphaBarrelFlippedLadders, newmeResXvsAlphaBarrelFlippedLadders, 
			   "barrel, non-flipped ladders, |alpha| (deg)", "<|x residual|> (cm)", xmin, xmax, leg20 );
    meResXvsAlphaBarrelFlippedLadders->Draw("e");
    newmeResXvsAlphaBarrelFlippedLadders->Draw("Samee"); 
    myPV->PVCompute(meResXvsAlphaBarrelFlippedLadders, newmeResXvsAlphaBarrelFlippedLadders, te );
    leg20->Draw();
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_ResXvsAlpha->cd(2);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResXvsAlphaZmPanel1, newmeResXvsAlphaZmPanel1, 
			   "panel1, z<0, |alpha| (deg)", "<|x residual|> (cm)", xmin, xmax );
    meResXvsAlphaZmPanel1->SetMinimum(xmin);
    meResXvsAlphaZmPanel1->SetMaximum(xmax);
    meResXvsAlphaZmPanel1->Draw("e");
    newmeResXvsAlphaZmPanel1->Draw("samee"); 
    myPV->PVCompute(meResXvsAlphaZmPanel1, newmeResXvsAlphaZmPanel1, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_ResXvsAlpha->cd(3);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResXvsAlphaZmPanel2, newmeResXvsAlphaZmPanel2, 
			   "panel2, z<0, |alpha| (deg)", "<|x residual|> (cm)", xmin, xmax );
    meResXvsAlphaZmPanel2->SetMinimum(xmin);
    meResXvsAlphaZmPanel2->SetMaximum(xmax);
    meResXvsAlphaZmPanel2->Draw("e");
    newmeResXvsAlphaZmPanel2->Draw("samee"); 
    myPV->PVCompute(meResXvsAlphaZmPanel2, newmeResXvsAlphaZmPanel2, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_ResXvsAlpha->cd(4);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResXvsAlphaBarrelNonFlippedLadders, newmeResXvsAlphaBarrelNonFlippedLadders, 
			   "barrel, flipped ladders, |alpha| (deg)", "<|x residual|> (cm)", xmin, xmax );
    meResXvsAlphaBarrelNonFlippedLadders->SetMinimum(xmin);
    meResXvsAlphaBarrelNonFlippedLadders->SetMaximum(xmax);
    meResXvsAlphaBarrelNonFlippedLadders->Draw("e");
    newmeResXvsAlphaBarrelNonFlippedLadders->Draw("Samee"); 
    myPV->PVCompute(meResXvsAlphaBarrelNonFlippedLadders, newmeResXvsAlphaBarrelNonFlippedLadders, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_ResXvsAlpha->cd(5);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResXvsAlphaZpPanel1, newmeResXvsAlphaZpPanel1, 
			   "panel1, z>0, |alpha| (deg)", "<|x residual|> (cm)", xmin, xmax );
    meResXvsAlphaZpPanel1->SetMinimum(xmin);
    meResXvsAlphaZpPanel1->SetMaximum(xmax);
    meResXvsAlphaZpPanel1->Draw("e");
    newmeResXvsAlphaZpPanel1->Draw("samee"); 
    myPV->PVCompute(meResXvsAlphaZpPanel1, newmeResXvsAlphaZpPanel1, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_ResXvsAlpha->cd(6);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResXvsAlphaZpPanel2, newmeResXvsAlphaZpPanel2,
			   "panel2, z>0, |alpha| (deg)", "<|x residual|> (cm)", xmin, xmax );
    meResXvsAlphaZpPanel2->SetMinimum(xmin);
    meResXvsAlphaZpPanel2->SetMaximum(xmax);
    meResXvsAlphaZpPanel2->Draw("e");
    newmeResXvsAlphaZpPanel2->Draw("samee"); 
    myPV->PVCompute(meResXvsAlphaZpPanel2, newmeResXvsAlphaZpPanel2, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

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
  
    TLegend* leg21 = new TLegend(0.3, 0.7, 0.6, 0.9);
    can_ResXvsBeta->cd(1);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResXvsBetaBarrel, newmeResXvsBetaBarrel, 
			   "barrel, |beta| (deg)", "<|x residual|> (cm)", xmin, xmax, leg21 );
    meResXvsBetaBarrel->Draw("e");
    newmeResXvsBetaBarrel->Draw("Samee"); 
    myPV->PVCompute(meResXvsBetaBarrel, newmeResXvsBetaBarrel, te );
    leg21->Draw();
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_ResXvsBeta->cd(2);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResXvsBetaZmPanel1, newmeResXvsBetaZmPanel1, 
			   "panel1, z<0, |beta| (deg)", "<|x residual|> (cm)", xmin, xmax );
    meResXvsBetaZmPanel1->Draw("e");
    newmeResXvsBetaZmPanel1->Draw("samee"); 
    myPV->PVCompute(meResXvsBetaZmPanel1, newmeResXvsBetaZmPanel1, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_ResXvsBeta->cd(3);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResXvsBetaZmPanel2, newmeResXvsBetaZmPanel2, 
			   "panel2, z<0, |beta| (deg)", "<|x residual|> (cm)", xmin, xmax ); 
    meResXvsBetaZmPanel2->Draw("e");
    newmeResXvsBetaZmPanel2->Draw("samee"); 
    myPV->PVCompute(meResXvsBetaZmPanel2, newmeResXvsBetaZmPanel2, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_ResXvsBeta->cd(5);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResXvsBetaZpPanel1, newmeResXvsBetaZpPanel1, "panel1, z>0, |beta| (deg)", "<|x residual|> (cm)", xmin, xmax );
    meResXvsBetaZpPanel1->Draw("e");
    newmeResXvsBetaZpPanel1->Draw("samee"); 
    myPV->PVCompute(meResXvsBetaZpPanel1, newmeResXvsBetaZpPanel1, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_ResXvsBeta->cd(6);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResXvsBetaZpPanel2, newmeResXvsBetaZpPanel2, "panel2, z>0, |beta| (deg)", "<|x residual|> (cm)", xmin, xmax );
    meResXvsBetaZpPanel2->Draw("e");
    newmeResXvsBetaZpPanel2->Draw("samee"); 
    myPV->PVCompute(meResXvsBetaZpPanel2, newmeResXvsBetaZpPanel2, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

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
  
    TLegend* leg22 = new TLegend(0.3, 0.7, 0.6, 0.9);
    can_ResYvsAlpha->cd(1);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResYvsAlphaBarrel, newmeResYvsAlphaBarrel, 
			   "barrel, |alpha| (deg)", "<|y residual|> (cm)", ymin+0.0010, ymax+0.0010, leg22 );
    meResYvsAlphaBarrel->Draw("e");
    newmeResYvsAlphaBarrel->Draw("Samee"); 
    myPV->PVCompute(meResYvsAlphaBarrel, newmeResYvsAlphaBarrel, te );
    leg22->Draw();
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_ResYvsAlpha->cd(2);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResYvsAlphaZmPanel1, newmeResYvsAlphaZmPanel1, 
			   "panel1, z<0, |alpha| (deg)", "<|y residual|> (cm)", ymin, ymax );
    meResYvsAlphaZmPanel1->Draw("e");
    newmeResYvsAlphaZmPanel1->Draw("samee"); 
    myPV->PVCompute(meResYvsAlphaZmPanel1, newmeResYvsAlphaZmPanel1, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_ResYvsAlpha->cd(3);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResYvsAlphaZmPanel2, newmeResYvsAlphaZmPanel2, 
			   "panel2, z<0, |alpha| (deg)", "<|y residual|> (cm)", ymin, ymax );
    meResYvsAlphaZmPanel2->Draw("e");
    newmeResYvsAlphaZmPanel2->Draw("samee"); 
    myPV->PVCompute(meResYvsAlphaZmPanel2, newmeResYvsAlphaZmPanel2, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_ResYvsAlpha->cd(5);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResYvsAlphaZpPanel1, newmeResYvsAlphaZpPanel1, 
			   "panel1, z>0, |alpha| (deg)", "<|y residual|> (cm)" , ymin, ymax);
    meResYvsAlphaZpPanel1->Draw("e");
    newmeResYvsAlphaZpPanel1->Draw("samee"); 
    myPV->PVCompute(meResYvsAlphaZpPanel1, newmeResYvsAlphaZpPanel1, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_ResYvsAlpha->cd(6);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResYvsAlphaZpPanel2, newmeResYvsAlphaZpPanel2, 
			   "panel2, z>0, |alpha| (deg)", "<|y residual|> (cm)", ymin, ymax );
    meResYvsAlphaZpPanel2->Draw("e");
    newmeResYvsAlphaZpPanel2->Draw("samee"); 
    myPV->PVCompute(meResYvsAlphaZpPanel2, newmeResYvsAlphaZpPanel2, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

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
  
    TLegend* leg23 = new TLegend(0.3, 0.7, 0.6, 0.9);
    can_ResYvsBeta->cd(1);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResYvsBetaBarrel, newmeResYvsBetaBarrel, 
			   "barrel, |beta| (deg)", "<|y residual|> (cm)", 0.0000, 0.0060, leg23 );
    meResYvsBetaBarrel->Draw("e");
    newmeResYvsBetaBarrel->Draw("Samee"); 
    myPV->PVCompute(meResYvsBetaBarrel, newmeResYvsBetaBarrel, te );
    leg23->Draw();
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_ResYvsBeta->cd(2);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResYvsBetaZmPanel1, newmeResYvsBetaZmPanel1, "panel1, z<0, |beta| (deg)", "<|y residual|> (cm)", ymin, ymax );
    meResYvsBetaZmPanel1->Draw("e");
    newmeResYvsBetaZmPanel1->Draw("samee"); 
    myPV->PVCompute(meResYvsBetaZmPanel1, newmeResYvsBetaZmPanel1, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_ResYvsBeta->cd(3);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResYvsBetaZmPanel2, newmeResYvsBetaZmPanel2, "panel2, z<0, |beta| (deg)", "<|y residual|> (cm)", ymin, ymax );
    meResYvsBetaZmPanel2->Draw("e");
    newmeResYvsBetaZmPanel2->Draw("samee"); 
    myPV->PVCompute(meResYvsBetaZmPanel2, newmeResYvsBetaZmPanel2, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_ResYvsBeta->cd(5);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResYvsBetaZpPanel1, newmeResYvsBetaZpPanel1, "panel1, z>0, |beta| (deg)", "<|y residual|> (cm)", ymin, ymax );
    meResYvsBetaZpPanel1->Draw("e");
    newmeResYvsBetaZpPanel1->Draw("samee"); 
    myPV->PVCompute(meResYvsBetaZpPanel1, newmeResYvsBetaZpPanel1, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_ResYvsBeta->cd(6);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResYvsBetaZpPanel2, newmeResYvsBetaZpPanel2, "panel2, z>0, |beta| (deg)", "<|y residual|> (cm)", ymin, ymax );
    meResYvsBetaZpPanel2->Draw("e");
    newmeResYvsBetaZpPanel2->Draw("samee"); 
    myPV->PVCompute(meResYvsBetaZpPanel2, newmeResYvsBetaZpPanel2, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

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
  
    TLegend* leg24 = new TLegend(0.3, 0.7, 0.6, 0.9);
    can_meResx->cd(1);
    gPad->SetLogy();
    SetUpHistograms(meResxBarrel, newmeResxBarrel, "barrel, x residual (cm)", leg24 );
    meResxBarrel->Draw("he");
    newmeResxBarrel->Draw("Samehe"); 
    myPV->PVCompute(meResxBarrel, newmeResxBarrel, te );
    leg24->Draw();
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_meResx->cd(2);
    gPad->SetLogy();
    SetUpHistograms(meResxZmPanel1, newmeResxZmPanel1, "panel1, z<0, x residual (cm)" );
    meResxZmPanel1->Draw("he");
    newmeResxZmPanel1->Draw("samehe"); 
    myPV->PVCompute(meResxZmPanel1, newmeResxZmPanel1, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_meResx->cd(3);
    gPad->SetLogy();
    SetUpHistograms(meResxZmPanel2,  newmeResxZmPanel2, "panel2, z<0, x residual (cm)");
    meResxZmPanel2->Draw("he");
    newmeResxZmPanel2->Draw("samehe"); 
    myPV->PVCompute(meResxZmPanel2, newmeResxZmPanel2, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_meResx->cd(5);
    gPad->SetLogy();
    SetUpHistograms(meResxZpPanel1, newmeResxZpPanel1, "panel1, z>0, x residual (cm)" );
    meResxZpPanel1->Draw("he");
    newmeResxZpPanel1->Draw("samehe"); 
    myPV->PVCompute(meResxZpPanel1, newmeResxZpPanel1, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_meResx->cd(6);
    gPad->SetLogy();
    SetUpHistograms(meResxZpPanel2, newmeResxZpPanel2, "panel2, z>0, x residual (cm)" );
    meResxZpPanel2->Draw("he");
    newmeResxZpPanel2->Draw("samehe"); 
    myPV->PVCompute(meResxZpPanel2, newmeResxZpPanel2, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

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
  
    TLegend* leg25 = new TLegend(0.3, 0.7, 0.6, 0.9);
    can_meResy->cd(1);
    gPad->SetLogy();
    SetUpHistograms(meResyBarrel, newmeResyBarrel, "barrel, y residual (cm)", leg25 );
    meResyBarrel->Draw("he");
    newmeResyBarrel->Draw("Samehe"); 
    myPV->PVCompute(meResyBarrel, newmeResyBarrel, te );
    leg25->Draw();
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_meResy->cd(2);
    gPad->SetLogy();
    SetUpHistograms(meResyZmPanel1, newmeResyZmPanel1, "panel1, z<0, y residual (cm)" );
    meResyZmPanel1->Draw("he");
    newmeResyZmPanel1->Draw("samehe"); 
    myPV->PVCompute(meResyZmPanel1, newmeResyZmPanel1, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_meResy->cd(3);
    gPad->SetLogy();
    SetUpHistograms(meResyZmPanel2, newmeResyZmPanel2, "panel2, z<0, y residual (cm) " );
    meResyZmPanel2->Draw("he");
    newmeResyZmPanel2->Draw("samehe"); 
    myPV->PVCompute(meResyZmPanel2, newmeResyZmPanel2, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_meResy->cd(5);
    gPad->SetLogy();
    SetUpHistograms(meResyZpPanel1, newmeResyZpPanel1, "panel1, z>0, y residual (cm)" );
    meResyZpPanel1->Draw("he");
    newmeResyZpPanel1->Draw("samehe"); 
    myPV->PVCompute(meResyZpPanel1, newmeResyZpPanel1, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

    can_meResy->cd(6);
    gPad->SetLogy();
    SetUpHistograms(meResyZpPanel2, newmeResyZpPanel2, "panel2, z>0, y residual (cm)" );
    meResyZpPanel2->Draw("he");
    newmeResyZpPanel2->Draw("samehe"); 
    myPV->PVCompute(meResyZpPanel2, newmeResyZpPanel2, te );
    h_pv->SetBinContent(++bin, myPV->getPV());

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
    
       meChargeLayerModule[i][j]->Draw("he");
       newmeChargeLayerModule[i][j]->Draw("samehe"); 
       myPV->PVCompute(meChargeLayerModule[i][j], newmeChargeLayerModule[i][j], te );
       h_pv->SetBinContent(++bin, myPV->getPV());
     }
 TLegend* leg26 = new TLegend(0.3, 0.7, 0.6, 0.9);
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
       meChargeZmPanel1DiskPlaq[i][j]->Draw("he");
       newmeChargeZmPanel1DiskPlaq[i][j]->Draw("samehe"); 
       myPV->PVCompute(meChargeZmPanel1DiskPlaq[i][j], newmeChargeZmPanel1DiskPlaq[i][j], te );
       h_pv->SetBinContent(++bin, myPV->getPV());
     }
 TLegend* leg27 = new TLegend(0.3, 0.7, 0.6, 0.9);
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
       meChargeZmPanel2DiskPlaq[i][j]->Draw("he");
       newmeChargeZmPanel2DiskPlaq[i][j]->Draw("samehe"); 
       myPV->PVCompute(meChargeZmPanel2DiskPlaq[i][j], newmeChargeZmPanel2DiskPlaq[i][j], te );
       h_pv->SetBinContent(++bin, myPV->getPV());
     }
 TLegend* leg28 = new TLegend(0.3, 0.7, 0.6, 0.9);
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
       meChargeZpPanel1DiskPlaq[i][j]->Draw("he");
       newmeChargeZpPanel1DiskPlaq[i][j]->Draw("samehe"); 
       myPV->PVCompute(meChargeZpPanel1DiskPlaq[i][j], newmeChargeZpPanel1DiskPlaq[i][j], te );
       h_pv->SetBinContent(++bin, myPV->getPV());
     }
 TLegend* leg29 = new TLegend(0.3, 0.7, 0.6, 0.9);
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
       meChargeZpPanel2DiskPlaq[i][j]->Draw("he");
       newmeChargeZpPanel2DiskPlaq[i][j]->Draw("samehe"); 
       myPV->PVCompute(meChargeZpPanel2DiskPlaq[i][j], newmeChargeZpPanel2DiskPlaq[i][j], te );
       h_pv->SetBinContent(++bin, myPV->getPV());
     }
 TLegend* leg30 = new TLegend(0.3, 0.7, 0.6, 0.9);
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
     meResxBarrelLayer[i]->Draw("he");
     newmeResxBarrelLayer[i]->Draw("samehe"); 
     myPV->PVCompute(meResxBarrelLayer[i], newmeResxBarrelLayer[i], te );
     h_pv->SetBinContent(++bin, myPV->getPV());
   }
 TLegend* leg31 = new TLegend(0.3, 0.7, 0.6, 0.9);
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
     meResyBarrelLayer[i]->Draw("he");
     newmeResyBarrelLayer[i]->Draw("samehe"); 
     myPV->PVCompute(meResyBarrelLayer[i], newmeResyBarrelLayer[i], te );
     h_pv->SetBinContent(++bin, myPV->getPV());
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
     mePullxBarrelLayer[i]->Draw("he");
     newmePullxBarrelLayer[i]->Draw("samehe"); 
     myPV->PVCompute(mePullxBarrelLayer[i], newmePullxBarrelLayer[i], te );
     h_pv->SetBinContent(++bin, myPV->getPV());
   }
 TLegend* leg32 = new TLegend(0.3, 0.7, 0.6, 0.9);
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
     mePullyBarrelLayer[i]->Draw("he");
     newmePullyBarrelLayer[i]->Draw("samehe"); 
     myPV->PVCompute(mePullyBarrelLayer[i], newmePullyBarrelLayer[i], te );
     h_pv->SetBinContent(++bin, myPV->getPV());
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
