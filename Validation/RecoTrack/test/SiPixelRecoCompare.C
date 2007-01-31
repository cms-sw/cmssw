
void SetUpHistograms(TH1F* h1, TH1F* h2, const char* xtitle, TLegend* leg = 0)
{
  float scale = -9999.9;
  h1->Sumw2();
  h2->Sumw2();
  scale = 1.0/h1->Integral();
  h1->Scale(scale);
  scale = 1.0/h2->Integral();
  h2->Scale(scale);
  
  h1->SetLineWidth(2);
  h2->SetLineWidth(2);
  h1->SetLineColor(2);
  h2->SetLineColor(4);
  h2->SetLineStyle(2);  

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
  //char*  sfilename = "./ROOT_FILES/pixeltrackingrechitshist34.root";
  //char*  rfilename = "./ROOT_FILES/pixeltrackingrechitshist56.root"; 
  
  char*  sfilename = "./pixeltrackingrechitshist.root"; // file to be checked
  char*  rfilename = "../data/pixeltrackingrechitshist.root"; // reference file 
  
  delete gROOT->GetListOfFiles()->FindObject(rfilename);
  delete gROOT->GetListOfFiles()->FindObject(sfilename);
  
  TText* te = new TText();
  TFile* rfile = new TFile(rfilename);
  TFile* sfile = new TFile(sfilename);
  Char_t histo[200];
  
  rfile->cd("DQMData/Histograms_all");
  //gDirectory->ls();
  
  sfile->cd("DQMData/Histograms_all");
  //gDirectory->ls();
  
  gROOT->ProcessLine(".x HistoCompare.C");
  HistoCompare* myPV = new HistoCompare();
  
  if (1) 
    {
      TCanvas* can_meControl = new TCanvas("can_meControl", "can_meControl", 1000, 500);
      can_meControl->Divide(2,1);
      
      TH1F* meTracksPerEvent;
      TH1F* mePixRecHitsPerTrack;
      
      TH1F* newmeTracksPerEvent;
      TH1F* newmePixRecHitsPerTrack;
      
      rfile->GetObject("DQMData/Histograms_all/meTracksPerEvent", meTracksPerEvent );
      rfile->GetObject("DQMData/Histograms_all/mePixRecHitsPerTrack", mePixRecHitsPerTrack );
      
      sfile->GetObject("DQMData/Histograms_all/meTracksPerEvent", newmeTracksPerEvent );
      sfile->GetObject("DQMData/Histograms_all/mePixRecHitsPerTrack", newmePixRecHitsPerTrack );
      
      TLegend* leg1 = new TLegend(0.3, 0.7, 0.6, 0.9);
      can_meControl->cd(1);
      //gPad->SetLogy();
      SetUpHistograms(meTracksPerEvent, newmeTracksPerEvent, "tracks per event", leg1 );
      meTracksPerEvent->Draw("he");
      newmeTracksPerEvent->Draw("samehe"); 
      myPV->PVCompute(meTracksPerEvent, newmeTracksPerEvent, te );
      leg1->Draw();
      
      can_meControl->cd(2);
      //gPad->SetLogy();
      SetUpHistograms(mePixRecHitsPerTrack, newmePixRecHitsPerTrack, "pixel hits per track" );
      mePixRecHitsPerTrack->Draw("he");
      newmePixRecHitsPerTrack->Draw("samehe"); 
      myPV->PVCompute(mePixRecHitsPerTrack, newmePixRecHitsPerTrack, te );
      
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
      
      rfile->GetObject("DQMData/Histograms_all/meChargeBarrel"  , meChargeBarrel  );
      rfile->GetObject("DQMData/Histograms_all/meChargeZmPanel1", meChargeZmPanel1);
      rfile->GetObject("DQMData/Histograms_all/meChargeZmPanel2", meChargeZmPanel2);
      rfile->GetObject("DQMData/Histograms_all/meChargeZpPanel1", meChargeZpPanel1);
      rfile->GetObject("DQMData/Histograms_all/meChargeZpPanel2", meChargeZpPanel2);
      
      sfile->GetObject("DQMData/Histograms_all/meChargeBarrel"  , newmeChargeBarrel  ); 
      sfile->GetObject("DQMData/Histograms_all/meChargeZmPanel1", newmeChargeZmPanel1);
      sfile->GetObject("DQMData/Histograms_all/meChargeZmPanel2", newmeChargeZmPanel2);
      sfile->GetObject("DQMData/Histograms_all/meChargeZpPanel1", newmeChargeZpPanel1);
      sfile->GetObject("DQMData/Histograms_all/meChargeZpPanel2", newmeChargeZpPanel2);
      
      TLegend* leg2 = new TLegend(0.3, 0.7, 0.6, 0.9);
      can_meCharge->cd(1);
      //gPad->SetLogy();
      SetUpHistograms(meChargeBarrel, newmeChargeBarrel, "barrel, cluster charge (elec) ", leg2 );
      meChargeBarrel->Draw("he");
      newmeChargeBarrel->Draw("samehe"); 
      myPV->PVCompute(meChargeBarrel, newmeChargeBarrel, te );
      leg2->Draw();
      
      can_meCharge->cd(2);
      //gPad->SetLogy();
      SetUpHistograms(meChargeZmPanel1, newmeChargeZmPanel1, "panel1, z<0, cluster charge (elec)" );
      meChargeZmPanel1->Draw("he");
      newmeChargeZmPanel1->Draw("samehe"); 
      myPV->PVCompute(meChargeZmPanel1, newmeChargeZmPanel1, te );
      
      can_meCharge->cd(3);
      //gPad->SetLogy();
      SetUpHistograms(meChargeZmPanel2, newmeChargeZmPanel2, "panel2, z<0, cluster charge (elec)" );
      meChargeZmPanel2->Draw("he");
      newmeChargeZmPanel2->Draw("samehe"); 
      myPV->PVCompute(meChargeZmPanel2, newmeChargeZmPanel2, te );
      
      can_meCharge->cd(5);
      //gPad->SetLogy();
      SetUpHistograms(meChargeZpPanel1, newmeChargeZpPanel1, "panel1, z>0, cluster charge (elec)" );
      meChargeZpPanel1->Draw("he");
      newmeChargeZpPanel1->Draw("samehe"); 
      myPV->PVCompute(meChargeZpPanel1, newmeChargeZpPanel1, te );
      
      can_meCharge->cd(6);
      //gPad->SetLogy();
      SetUpHistograms(meChargeZpPanel2, newmeChargeZpPanel2, "panel2, z>0, cluster charge (elec)" );  
      meChargeZpPanel2->Draw("he");
      newmeChargeZpPanel2->Draw("samehe"); 
      myPV->PVCompute(meChargeZpPanel2, newmeChargeZpPanel2, te );
      
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
      
      rfile->GetObject("DQMData/Histograms_all/meErrxBarrel"  , meErrxBarrel  );
      rfile->GetObject("DQMData/Histograms_all/meErrxZmPanel1", meErrxZmPanel1);
      rfile->GetObject("DQMData/Histograms_all/meErrxZmPanel2", meErrxZmPanel2);
      rfile->GetObject("DQMData/Histograms_all/meErrxZpPanel1", meErrxZpPanel1);
      rfile->GetObject("DQMData/Histograms_all/meErrxZpPanel2", meErrxZpPanel2);
      
      sfile->GetObject("DQMData/Histograms_all/meErrxBarrel"  , newmeErrxBarrel  ); 
      sfile->GetObject("DQMData/Histograms_all/meErrxZmPanel1", newmeErrxZmPanel1);
      sfile->GetObject("DQMData/Histograms_all/meErrxZmPanel2", newmeErrxZmPanel2);
      sfile->GetObject("DQMData/Histograms_all/meErrxZpPanel1", newmeErrxZpPanel1);
      sfile->GetObject("DQMData/Histograms_all/meErrxZpPanel2", newmeErrxZpPanel2);
      
      TLegend* leg3 = new TLegend(0.3, 0.7, 0.6, 0.9);
      can_Errx->cd(1);
      //gPad->SetLogy();
      SetUpHistograms(meErrxBarrel, newmeErrxBarrel, "barrel, x position error (cm)", leg3 );
      meErrxBarrel->Draw("he");
      newmeErrxBarrel->Draw("Samehe"); 
      myPV->PVCompute(meErrxBarrel, newmeErrxBarrel, te );
      leg3->Draw();
      
      can_Errx->cd(2);
      //gPad->SetLogy();
      SetUpHistograms(meErrxZmPanel1, newmeErrxZmPanel1, "panel1, z<0, x position error (cm)" );
      meErrxZmPanel1->Draw("he");
      newmeErrxZmPanel1->Draw("samehe"); 
      myPV->PVCompute(meErrxZmPanel1, newmeErrxZmPanel1, te );
      
      can_Errx->cd(3);
      //gPad->SetLogy();
      SetUpHistograms(meErrxZmPanel2, newmeErrxZmPanel2, "panel2, z<0, x position error (cm)" );
      meErrxZmPanel2->Draw("he");
      newmeErrxZmPanel2->Draw("samehe"); 
      myPV->PVCompute(meErrxZmPanel2, newmeErrxZmPanel2, te );
      
      can_Errx->cd(5);
      //gPad->SetLogy();
      SetUpHistograms(meErrxZpPanel1, newmeErrxZpPanel1, "panel1, z>0, x position error (cm)" );
      meErrxZpPanel1->Draw("he");
      newmeErrxZpPanel1->Draw("samehe"); 
      myPV->PVCompute(meErrxZpPanel1, newmeErrxZpPanel1, te );
      
      can_Errx->cd(6);
      //gPad->SetLogy();
      SetUpHistograms(meErrxZpPanel2, newmeErrxZpPanel2, "panel2, z>0, x position error (cm)" );
      meErrxZpPanel2->Draw("he");
      newmeErrxZpPanel2->Draw("samehe"); 
      myPV->PVCompute(meErrxZpPanel2, newmeErrxZpPanel2, te );
      
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
      
      rfile->GetObject("DQMData/Histograms_all/meErryBarrel"  , meErryBarrel  );
      rfile->GetObject("DQMData/Histograms_all/meErryZmPanel1", meErryZmPanel1);
      rfile->GetObject("DQMData/Histograms_all/meErryZmPanel2", meErryZmPanel2);
      rfile->GetObject("DQMData/Histograms_all/meErryZpPanel1", meErryZpPanel1);
      rfile->GetObject("DQMData/Histograms_all/meErryZpPanel2", meErryZpPanel2);
      
      sfile->GetObject("DQMData/Histograms_all/meErryBarrel"  , newmeErryBarrel  ); 
      sfile->GetObject("DQMData/Histograms_all/meErryZmPanel1", newmeErryZmPanel1);
      sfile->GetObject("DQMData/Histograms_all/meErryZmPanel2", newmeErryZmPanel2);
      sfile->GetObject("DQMData/Histograms_all/meErryZpPanel1", newmeErryZpPanel1);
      sfile->GetObject("DQMData/Histograms_all/meErryZpPanel2", newmeErryZpPanel2);
      
      TLegend* leg4 = new TLegend(0.3, 0.7, 0.6, 0.9);
      can_Erry->cd(1);
      //gPad->SetLogy();
      SetUpHistograms(meErryBarrel, newmeErryBarrel, "barrel, y position error (cm)", leg4 );
      meErryBarrel->Draw("he");
      newmeErryBarrel->Draw("Samehe"); 
      myPV->PVCompute(meErryBarrel, newmeErryBarrel, te );
      leg4->Draw();
      
      can_Erry->cd(2);
      //gPad->SetLogy();
      SetUpHistograms(meErryZmPanel1, newmeErryZmPanel1, "panel1, z<0, y position error (cm)"  );
      meErryZmPanel1->Draw("he");
      newmeErryZmPanel1->Draw("samehe"); 
      myPV->PVCompute(meErryZmPanel1, newmeErryZmPanel1, te );
      
      can_Erry->cd(3);
      //gPad->SetLogy();
      SetUpHistograms(meErryZmPanel2, newmeErryZmPanel2, "panel2, z<0, y position error (cm)" );
      meErryZmPanel2->Draw("he");
      newmeErryZmPanel2->Draw("samehe"); 
      myPV->PVCompute(meErryZmPanel2, newmeErryZmPanel2, te );
      
      can_Erry->cd(5);
      //gPad->SetLogy();
      SetUpHistograms(meErryZpPanel1, newmeErryZpPanel1, "panel1, z>0, y position error (cm)" );
      meErryZpPanel1->Draw("he");
      newmeErryZpPanel1->Draw("samehe"); 
      myPV->PVCompute(meErryZpPanel1, newmeErryZpPanel1, te );
      
      can_Erry->cd(6);
      //gPad->SetLogy();
      SetUpHistograms(meErryZpPanel2, newmeErryZpPanel2, "panel2, z>0, y position error (cm)" );
      meErryZpPanel2->Draw("he");
      newmeErryZpPanel2->Draw("samehe"); 
      myPV->PVCompute(meErryZpPanel2, newmeErryZpPanel2, te );
      
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
      
      rfile->GetObject("DQMData/Histograms_all/meNpixBarrel"  , meNpixBarrel  );
      rfile->GetObject("DQMData/Histograms_all/meNpixZmPanel1", meNpixZmPanel1);
      rfile->GetObject("DQMData/Histograms_all/meNpixZmPanel2", meNpixZmPanel2);
      rfile->GetObject("DQMData/Histograms_all/meNpixZpPanel1", meNpixZpPanel1);
      rfile->GetObject("DQMData/Histograms_all/meNpixZpPanel2", meNpixZpPanel2);
      
      sfile->GetObject("DQMData/Histograms_all/meNpixBarrel"  , newmeNpixBarrel  ); 
      sfile->GetObject("DQMData/Histograms_all/meNpixZmPanel1", newmeNpixZmPanel1);
      sfile->GetObject("DQMData/Histograms_all/meNpixZmPanel2", newmeNpixZmPanel2);
      sfile->GetObject("DQMData/Histograms_all/meNpixZpPanel1", newmeNpixZpPanel1);
      sfile->GetObject("DQMData/Histograms_all/meNpixZpPanel2", newmeNpixZpPanel2);
      
      TLegend* leg5 = new TLegend(0.3, 0.7, 0.6, 0.9);
      can_Npix->cd(1);
      //gPad->SetLogy();
      SetUpHistograms(meNpixBarrel, newmeNpixBarrel, "barrel, cluster size (pixels)", leg5 );
      meNpixBarrel->Draw("he");
      newmeNpixBarrel->Draw("Samehe"); 
      myPV->PVCompute(meNpixBarrel, newmeNpixBarrel, te );
      leg5->Draw();
      
      can_Npix->cd(2);
      //gPad->SetLogy();
      SetUpHistograms(meNpixZmPanel1, newmeNpixZmPanel1, "panel1, z<0, cluster size (pixels)"  );
      meNpixZmPanel1->Draw("he");
      newmeNpixZmPanel1->Draw("samehe"); 
      myPV->PVCompute(meNpixZmPanel1, newmeNpixZmPanel1, te );
      
      can_Npix->cd(3);
      //gPad->SetLogy();
      SetUpHistograms(meNpixZmPanel2, newmeNpixZmPanel2, "panel2, z<0, cluster size (pixels)" );
      meNpixZmPanel2->Draw("he");
      newmeNpixZmPanel2->Draw("samehe"); 
      myPV->PVCompute(meNpixZmPanel2, newmeNpixZmPanel2, te );
      
      can_Npix->cd(5);
      //gPad->SetLogy();
      SetUpHistograms(meNpixZpPanel1, newmeNpixZpPanel1, "panel1, z>0, cluster size (pixels)" );
      meNpixZpPanel1->Draw("he");
      newmeNpixZpPanel1->Draw("samehe"); 
      myPV->PVCompute(meNpixZpPanel1, newmeNpixZpPanel1, te );
      
      can_Npix->cd(6);
      //gPad->SetLogy();
      SetUpHistograms(meNpixZpPanel2, newmeNpixZpPanel2, "panel2, z>0, cluster size (pixels)" );
      meNpixZpPanel2->Draw("he");
      newmeNpixZpPanel2->Draw("samehe"); 
      myPV->PVCompute(meNpixZpPanel2, newmeNpixZpPanel2, te );
      
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
      
      rfile->GetObject("DQMData/Histograms_all/meNxpixBarrel"  , meNxpixBarrel  );
      rfile->GetObject("DQMData/Histograms_all/meNxpixZmPanel1", meNxpixZmPanel1);
      rfile->GetObject("DQMData/Histograms_all/meNxpixZmPanel2", meNxpixZmPanel2);
      rfile->GetObject("DQMData/Histograms_all/meNxpixZpPanel1", meNxpixZpPanel1);
      rfile->GetObject("DQMData/Histograms_all/meNxpixZpPanel2", meNxpixZpPanel2);
      
      sfile->GetObject("DQMData/Histograms_all/meNxpixBarrel"  , newmeNxpixBarrel  ); 
      sfile->GetObject("DQMData/Histograms_all/meNxpixZmPanel1", newmeNxpixZmPanel1);
      sfile->GetObject("DQMData/Histograms_all/meNxpixZmPanel2", newmeNxpixZmPanel2);
      sfile->GetObject("DQMData/Histograms_all/meNxpixZpPanel1", newmeNxpixZpPanel1);
      sfile->GetObject("DQMData/Histograms_all/meNxpixZpPanel2", newmeNxpixZpPanel2);
      
      TLegend* leg6 = new TLegend(0.3, 0.7, 0.6, 0.9);
      can_Nxpix->cd(1);
      //gPad->SetLogy();
      SetUpHistograms(meNxpixBarrel, newmeNxpixBarrel, "barrel, cluster x size (pixels)", leg6 );
      meNxpixBarrel->Draw("he");
      newmeNxpixBarrel->Draw("Samehe"); 
      myPV->PVCompute(meNxpixBarrel, newmeNxpixBarrel, te );
      leg6->Draw();
      
      can_Nxpix->cd(2);
      //gPad->SetLogy();
      SetUpHistograms(meNxpixZmPanel1, newmeNxpixZmPanel1, "panel1, z<0, cluster x size (pixels)" );
      meNxpixZmPanel1->Draw("he");
      newmeNxpixZmPanel1->Draw("samehe"); 
      myPV->PVCompute(meNxpixZmPanel1, newmeNxpixZmPanel1, te );
      
      can_Nxpix->cd(3);
      //gPad->SetLogy();
      SetUpHistograms(meNxpixZmPanel2, newmeNxpixZmPanel2, "panel2, z<0, cluster x size (pixels)" );
      meNxpixZmPanel2->Draw("he");
      newmeNxpixZmPanel2->Draw("samehe"); 
      myPV->PVCompute(meNxpixZmPanel2, newmeNxpixZmPanel2, te );
      
      can_Nxpix->cd(5);
      //gPad->SetLogy();
      SetUpHistograms(meNxpixZpPanel1, newmeNxpixZpPanel1, "panel1, z>0, cluster x size (pixels)" );
      meNxpixZpPanel1->Draw("he");
      newmeNxpixZpPanel1->Draw("samehe"); 
      myPV->PVCompute(meNxpixZpPanel1, newmeNxpixZpPanel1, te );
      
      can_Nxpix->cd(6);
      //gPad->SetLogy();
      SetUpHistograms(meNxpixZpPanel2, newmeNxpixZpPanel2, "panel2, z>0, cluster x size (pixels)" );
      meNxpixZpPanel2->Draw("he");
      newmeNxpixZpPanel2->Draw("samehe"); 
      myPV->PVCompute(meNxpixZpPanel2, newmeNxpixZpPanel2, te );
      
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
      
      rfile->GetObject("DQMData/Histograms_all/meNypixBarrel"  , meNypixBarrel  );
      rfile->GetObject("DQMData/Histograms_all/meNypixZmPanel1", meNypixZmPanel1);
      rfile->GetObject("DQMData/Histograms_all/meNypixZmPanel2", meNypixZmPanel2);
      rfile->GetObject("DQMData/Histograms_all/meNypixZpPanel1", meNypixZpPanel1);
      rfile->GetObject("DQMData/Histograms_all/meNypixZpPanel2", meNypixZpPanel2);
      
      sfile->GetObject("DQMData/Histograms_all/meNypixBarrel"  , newmeNypixBarrel  ); 
      sfile->GetObject("DQMData/Histograms_all/meNypixZmPanel1", newmeNypixZmPanel1);
      sfile->GetObject("DQMData/Histograms_all/meNypixZmPanel2", newmeNypixZmPanel2);
      sfile->GetObject("DQMData/Histograms_all/meNypixZpPanel1", newmeNypixZpPanel1);
      sfile->GetObject("DQMData/Histograms_all/meNypixZpPanel2", newmeNypixZpPanel2);
      
      TLegend* leg7 = new TLegend(0.3, 0.7, 0.6, 0.9);
      can_Nypix->cd(1);
      //gPad->SetLogy();
      SetUpHistograms(meNypixBarrel, newmeNypixBarrel, "barrel, cluster y size (pixels)", leg7 );
      meNypixBarrel->Draw("he");
      newmeNypixBarrel->Draw("Samehe"); 
      myPV->PVCompute(meNypixBarrel, newmeNypixBarrel, te );
      leg7->Draw();
      
      can_Nypix->cd(2);
      //gPad->SetLogy();
      SetUpHistograms(meNypixZmPanel1, newmeNypixZmPanel1, "panel1, z<0, cluster y size (pixels)" );
      meNypixZmPanel1->Draw("he");
      newmeNypixZmPanel1->Draw("samehe"); 
      myPV->PVCompute(meNypixZmPanel1, newmeNypixZmPanel1, te );
      
      can_Nypix->cd(3);
      //gPad->SetLogy();
      SetUpHistograms(meNypixZmPanel2, newmeNypixZmPanel2, "panel2, z<0, cluster y size (pixels)" );
      meNypixZmPanel2->Draw("he");
      newmeNypixZmPanel2->Draw("samehe"); 
      myPV->PVCompute(meNypixZmPanel2, newmeNypixZmPanel2, te );
      
      can_Nypix->cd(5);
      //gPad->SetLogy();
      SetUpHistograms(meNypixZpPanel1, newmeNypixZpPanel1, "panel1, z>0, cluster y size (pixels)" );
      meNypixZpPanel1->Draw("he");
      newmeNypixZpPanel1->Draw("samehe"); 
      myPV->PVCompute(meNypixZpPanel1, newmeNypixZpPanel1, te );
      
      can_Nypix->cd(6);
      //gPad->SetLogy();
      SetUpHistograms(meNypixZpPanel2, newmeNypixZpPanel2, "panel2, z>0, cluster y size (pixels)" );
      meNypixZpPanel2->Draw("he");
      newmeNypixZpPanel2->Draw("samehe"); 
      myPV->PVCompute(meNypixZpPanel2, newmeNypixZpPanel2, te );
      
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
      
      rfile->GetObject("DQMData/Histograms_all/mePosxBarrel"  , mePosxBarrel  );
      rfile->GetObject("DQMData/Histograms_all/mePosxZmPanel1", mePosxZmPanel1);
      rfile->GetObject("DQMData/Histograms_all/mePosxZmPanel2", mePosxZmPanel2);
      rfile->GetObject("DQMData/Histograms_all/mePosxZpPanel1", mePosxZpPanel1);
      rfile->GetObject("DQMData/Histograms_all/mePosxZpPanel2", mePosxZpPanel2);
      
      sfile->GetObject("DQMData/Histograms_all/mePosxBarrel"  , newmePosxBarrel  ); 
      sfile->GetObject("DQMData/Histograms_all/mePosxZmPanel1", newmePosxZmPanel1);
      sfile->GetObject("DQMData/Histograms_all/mePosxZmPanel2", newmePosxZmPanel2);
      sfile->GetObject("DQMData/Histograms_all/mePosxZpPanel1", newmePosxZpPanel1);
      sfile->GetObject("DQMData/Histograms_all/mePosxZpPanel2", newmePosxZpPanel2);
      
      TLegend* leg8 = new TLegend(0.3, 0.7, 0.6, 0.9);
      can_Posx->cd(1);
      //gPad->SetLogy();
      SetUpHistograms(mePosxBarrel, newmePosxBarrel, "barrel, x (cm)", leg8 );
      mePosxBarrel->Draw("he");
      newmePosxBarrel->Draw("Samehe"); 
      myPV->PVCompute(mePosxBarrel, newmePosxBarrel, te );
      leg8->Draw();
      
      can_Posx->cd(2);
      //gPad->SetLogy();
      SetUpHistograms(mePosxZmPanel1, newmePosxZmPanel1, "panel1, z<0, x (cm)" );
      mePosxZmPanel1->Draw("he");
      newmePosxZmPanel1->Draw("samehe"); 
      myPV->PVCompute(mePosxZmPanel1, newmePosxZmPanel1, te );
      
      can_Posx->cd(3);
      //gPad->SetLogy();
      SetUpHistograms(mePosxZmPanel2, newmePosxZmPanel2, "panel2, z>0, x (cm)" );
      mePosxZmPanel2->Draw("he");
      newmePosxZmPanel2->Draw("samehe"); 
      myPV->PVCompute(mePosxZmPanel2, newmePosxZmPanel2, te );
      
      can_Posx->cd(5);
      //gPad->SetLogy();
      SetUpHistograms(mePosxZpPanel1, newmePosxZpPanel1, "panel1, z<0, x (cm)" );
      mePosxZpPanel1->Draw("he");
      newmePosxZpPanel1->Draw("samehe"); 
      myPV->PVCompute(mePosxZpPanel1, newmePosxZpPanel1, te );
      
      can_Posx->cd(6);
      //gPad->SetLogy();
      SetUpHistograms(mePosxZpPanel2, newmePosxZpPanel2, "panel2, z>0, x (cm)" );
      mePosxZpPanel2->Draw("he");
      newmePosxZpPanel2->Draw("samehe"); 
      myPV->PVCompute(mePosxZpPanel2, newmePosxZpPanel2, te );
      
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
      
      rfile->GetObject("DQMData/Histograms_all/mePosyBarrel"  , mePosyBarrel  );
      rfile->GetObject("DQMData/Histograms_all/mePosyZmPanel1", mePosyZmPanel1);
      rfile->GetObject("DQMData/Histograms_all/mePosyZmPanel2", mePosyZmPanel2);
      rfile->GetObject("DQMData/Histograms_all/mePosyZpPanel1", mePosyZpPanel1);
      rfile->GetObject("DQMData/Histograms_all/mePosyZpPanel2", mePosyZpPanel2);
      
      sfile->GetObject("DQMData/Histograms_all/mePosyBarrel"  , newmePosyBarrel  ); 
      sfile->GetObject("DQMData/Histograms_all/mePosyZmPanel1", newmePosyZmPanel1);
      sfile->GetObject("DQMData/Histograms_all/mePosyZmPanel2", newmePosyZmPanel2);
      sfile->GetObject("DQMData/Histograms_all/mePosyZpPanel1", newmePosyZpPanel1);
      sfile->GetObject("DQMData/Histograms_all/mePosyZpPanel2", newmePosyZpPanel2);
      
      TLegend* leg9 = new TLegend(0.3, 0.7, 0.6, 0.9);
      can_Posy->cd(1);
      //gPad->SetLogy();
      SetUpHistograms(mePosyBarrel, newmePosyBarrel, "barrel, y (cm)", leg9 );
      mePosyBarrel->Draw("he");
      newmePosyBarrel->Draw("Samehe"); 
      myPV->PVCompute(mePosyBarrel, newmePosyBarrel, te );
      leg9->Draw();
      
      can_Posy->cd(2);
      //gPad->SetLogy();
      SetUpHistograms(mePosyZmPanel1,  newmePosyZmPanel1, "panel1, z<0, y (cm)" );
      mePosyZmPanel1->Draw("he");
      newmePosyZmPanel1->Draw("samehe"); 
      myPV->PVCompute(mePosyZmPanel1, newmePosyZmPanel1, te );
      
      can_Posy->cd(3);
      //gPad->SetLogy();
      SetUpHistograms(mePosyZmPanel2, newmePosyZmPanel2, "panel2, z<0, y (cm)" );
      mePosyZmPanel2->Draw("he");
      newmePosyZmPanel2->Draw("samehe"); 
      myPV->PVCompute(mePosyZmPanel2, newmePosyZmPanel2, te );
      
      can_Posy->cd(5);
      //gPad->SetLogy();
      SetUpHistograms(mePosyZpPanel1, newmePosyZpPanel1, "panel1, z>0, y (cm)" );
      mePosyZpPanel1->Draw("he");
      newmePosyZpPanel1->Draw("samehe"); 
      myPV->PVCompute(mePosyZpPanel1, newmePosyZpPanel1, te );
      
      can_Posy->cd(6);
      //gPad->SetLogy();
      SetUpHistograms(mePosyZpPanel2, newmePosyZpPanel2, "panel2, z>0, y (cm)" );
      mePosyZpPanel2->Draw("he");
      newmePosyZpPanel2->Draw("samehe"); 
      myPV->PVCompute(mePosyZpPanel2, newmePosyZpPanel2, te );
      
      can_Posy->SaveAs("mePosy_compare.eps");
      can_Posy->SaveAs("mePosy_compare.gif");
    }
  
  double lpull = -0.030;
  double hpull =  0.030;

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
      
      rfile->GetObject("DQMData/Histograms_all/mePullXvsAlphaBarrel"  , mePullXvsAlphaBarrel  );
      rfile->GetObject("DQMData/Histograms_all/mePullXvsAlphaZmPanel1", mePullXvsAlphaZmPanel1);
      rfile->GetObject("DQMData/Histograms_all/mePullXvsAlphaZmPanel2", mePullXvsAlphaZmPanel2);
      rfile->GetObject("DQMData/Histograms_all/mePullXvsAlphaZpPanel1", mePullXvsAlphaZpPanel1);
      rfile->GetObject("DQMData/Histograms_all/mePullXvsAlphaZpPanel2", mePullXvsAlphaZpPanel2);
      
      sfile->GetObject("DQMData/Histograms_all/mePullXvsAlphaBarrel"  , newmePullXvsAlphaBarrel  ); 
      sfile->GetObject("DQMData/Histograms_all/mePullXvsAlphaZmPanel1", newmePullXvsAlphaZmPanel1);
      sfile->GetObject("DQMData/Histograms_all/mePullXvsAlphaZmPanel2", newmePullXvsAlphaZmPanel2);
      sfile->GetObject("DQMData/Histograms_all/mePullXvsAlphaZpPanel1", newmePullXvsAlphaZpPanel1);
      sfile->GetObject("DQMData/Histograms_all/mePullXvsAlphaZpPanel2", newmePullXvsAlphaZpPanel2);
      
      TLegend* leg10 = new TLegend(0.3, 0.7, 0.6, 0.9);
      can_PullXvsAlpha->cd(1);
      //gPad->SetLogy();
      SetUpProfileHistograms(mePullXvsAlphaBarrel, newmePullXvsAlphaBarrel, "barrel, |alpha| (deg)", "pull x", lpull, hpull, leg10 );
      mePullXvsAlphaBarrel->Draw("e");
      newmePullXvsAlphaBarrel->Draw("Samee"); 
      myPV->PVCompute(mePullXvsAlphaBarrel, newmePullXvsAlphaBarrel, te );
      leg10->Draw();
      
      can_PullXvsAlpha->cd(2);
      //gPad->SetLogy();
      SetUpProfileHistograms(mePullXvsAlphaZmPanel1, newmePullXvsAlphaZmPanel1, "panel1, z<0, |alpha| (deg)", "pull x", lpull, hpull );
      mePullXvsAlphaZmPanel1->Draw("e");
      newmePullXvsAlphaZmPanel1->Draw("samee"); 
      myPV->PVCompute(mePullXvsAlphaZmPanel1, newmePullXvsAlphaZmPanel1, te );
      
      can_PullXvsAlpha->cd(3);
      //gPad->SetLogy();
      SetUpProfileHistograms(mePullXvsAlphaZmPanel2, newmePullXvsAlphaZmPanel2, "panel2, z<0, |alpha| (deg)", "pull x", lpull, hpull );
      mePullXvsAlphaZmPanel2->Draw("e");
      newmePullXvsAlphaZmPanel2->Draw("samee"); 
      myPV->PVCompute(mePullXvsAlphaZmPanel2, newmePullXvsAlphaZmPanel2, te );
      
      can_PullXvsAlpha->cd(5);
      //gPad->SetLogy();
      SetUpProfileHistograms(mePullXvsAlphaZpPanel1, newmePullXvsAlphaZpPanel1, "panel1, z>0, |alpha| (deg)", "pull x", lpull, hpull );
      mePullXvsAlphaZpPanel1->Draw("e");
      newmePullXvsAlphaZpPanel1->Draw("samee"); 
      myPV->PVCompute(mePullXvsAlphaZpPanel1, newmePullXvsAlphaZpPanel1, te );
      
      can_PullXvsAlpha->cd(6);
      //gPad->SetLogy();
      SetUpProfileHistograms(mePullXvsAlphaZpPanel2, newmePullXvsAlphaZpPanel2, "panel2, z>0, |alpha| (deg)", "pull x", lpull, hpull );
      mePullXvsAlphaZpPanel2->Draw("e");
      newmePullXvsAlphaZpPanel2->Draw("samee"); 
      myPV->PVCompute(mePullXvsAlphaZpPanel2, newmePullXvsAlphaZpPanel2, te );
      
      can_PullXvsAlpha->SaveAs("mePullXvsAlpha_compare.eps");
      can_PullXvsAlpha->SaveAs("mePullXvsAlpha_compare.gif");
    }
  
  if (0) 
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
      
      rfile->GetObject("DQMData/Histograms_all/mePullXvsBetaBarrel"  , mePullXvsBetaBarrel  );
      rfile->GetObject("DQMData/Histograms_all/mePullXvsBetaZmPanel1", mePullXvsBetaZmPanel1);
      rfile->GetObject("DQMData/Histograms_all/mePullXvsBetaZmPanel2", mePullXvsBetaZmPanel2);
      rfile->GetObject("DQMData/Histograms_all/mePullXvsBetaZpPanel1", mePullXvsBetaZpPanel1);
      rfile->GetObject("DQMData/Histograms_all/mePullXvsBetaZpPanel2", mePullXvsBetaZpPanel2);
      
      sfile->GetObject("DQMData/Histograms_all/mePullXvsBetaBarrel"  , newmePullXvsBetaBarrel  ); 
      sfile->GetObject("DQMData/Histograms_all/mePullXvsBetaZmPanel1", newmePullXvsBetaZmPanel1);
      sfile->GetObject("DQMData/Histograms_all/mePullXvsBetaZmPanel2", newmePullXvsBetaZmPanel2);
      sfile->GetObject("DQMData/Histograms_all/mePullXvsBetaZpPanel1", newmePullXvsBetaZpPanel1);
      sfile->GetObject("DQMData/Histograms_all/mePullXvsBetaZpPanel2", newmePullXvsBetaZpPanel2);
      
      TLegend* leg11 = new TLegend(0.3, 0.7, 0.6, 0.9);
      can_PullXvsBeta->cd(1);
      //gPad->SetLogy();
      SetUpProfileHistograms(mePullXvsBetaBarrel, newmePullXvsBetaBarrel, "barrel, |beta| (deg)", "pull x", lpull, hpull, leg11 );
      mePullXvsBetaBarrel->Draw("e");
      newmePullXvsBetaBarrel->Draw("Samee"); 
      myPV->PVCompute(mePullXvsBetaBarrel, newmePullXvsBetaBarrel, te );
      leg11->Draw();

      can_PullXvsBeta->cd(2);
      //gPad->SetLogy();
      SetUpProfileHistograms(mePullXvsBetaZmPanel1, newmePullXvsBetaZmPanel1, "panel1, z<0, |beta| (deg)", "pull x", lpull, hpull );
      mePullXvsBetaZmPanel1->Draw("e");
      newmePullXvsBetaZmPanel1->Draw("samee"); 
      myPV->PVCompute(mePullXvsBetaZmPanel1, newmePullXvsBetaZmPanel1, te );
      
      can_PullXvsBeta->cd(3);
      //gPad->SetLogy();
      SetUpProfileHistograms(mePullXvsBetaZmPanel2, newmePullXvsBetaZmPanel2, "panel2, z<0, |beta| (deg)", "pull x", lpull, hpull );
      mePullXvsBetaZmPanel2->Draw("e");
      newmePullXvsBetaZmPanel2->Draw("samee"); 
      myPV->PVCompute(mePullXvsBetaZmPanel2, newmePullXvsBetaZmPanel2, te );
      
      can_PullXvsBeta->cd(5);
      //gPad->SetLogy();
      SetUpProfileHistograms(mePullXvsBetaZpPanel1, newmePullXvsBetaZpPanel1, "panel1, z>0, |beta| (deg)", "pull x", lpull, hpull );
      mePullXvsBetaZpPanel1->Draw("e");
      newmePullXvsBetaZpPanel1->Draw("samee"); 
      myPV->PVCompute(mePullXvsBetaZpPanel1, newmePullXvsBetaZpPanel1, te );
      
      can_PullXvsBeta->cd(6);
      //gPad->SetLogy();
      SetUpProfileHistograms(mePullXvsBetaZpPanel2, newmePullXvsBetaZpPanel2, "panel2, z>0, |beta| (deg)", "pull x", lpull, hpull );
      mePullXvsBetaZpPanel2->Draw("e");
      newmePullXvsBetaZpPanel2->Draw("samee"); 
      myPV->PVCompute(mePullXvsBetaZpPanel2, newmePullXvsBetaZpPanel2, te );

      can_PullXvsBeta->SaveAs("mePullXvsBeta_compare.eps");
      can_PullXvsBeta->SaveAs("mePullXvsBeta_compare.gif");
    }

if (0) 
  {
    TCanvas* can_PullXvsEta = new TCanvas("can_PullXvsEta", "can_PullXvsEta", 1200, 800);
    can_PullXvsEta->Divide(3,2);
    
    TProfile* mePullXvsEtaBarrel;
    TProfile* mePullXvsEtaZmPanel1;
    TProfile* mePullXvsEtaZmPanel2;
    TProfile* mePullXvsEtaZpPanel1;
    TProfile* mePullXvsEtaZpPanel2;
    
    TProfile* newmePullXvsEtaBarrel;
    TProfile* newmePullXvsEtaZmPanel1;
    TProfile* newmePullXvsEtaZmPanel2;
    TProfile* newmePullXvsEtaZpPanel1;
    TProfile* newmePullXvsEtaZpPanel2;

    rfile->GetObject("DQMData/Histograms_all/mePullXvsEtaBarrel"  , mePullXvsEtaBarrel  );
    rfile->GetObject("DQMData/Histograms_all/mePullXvsEtaZmPanel1", mePullXvsEtaZmPanel1);
    rfile->GetObject("DQMData/Histograms_all/mePullXvsEtaZmPanel2", mePullXvsEtaZmPanel2);
    rfile->GetObject("DQMData/Histograms_all/mePullXvsEtaZpPanel1", mePullXvsEtaZpPanel1);
    rfile->GetObject("DQMData/Histograms_all/mePullXvsEtaZpPanel2", mePullXvsEtaZpPanel2);

    sfile->GetObject("DQMData/Histograms_all/mePullXvsEtaBarrel"  , newmePullXvsEtaBarrel  ); 
    sfile->GetObject("DQMData/Histograms_all/mePullXvsEtaZmPanel1", newmePullXvsEtaZmPanel1);
    sfile->GetObject("DQMData/Histograms_all/mePullXvsEtaZmPanel2", newmePullXvsEtaZmPanel2);
    sfile->GetObject("DQMData/Histograms_all/mePullXvsEtaZpPanel1", newmePullXvsEtaZpPanel1);
    sfile->GetObject("DQMData/Histograms_all/mePullXvsEtaZpPanel2", newmePullXvsEtaZpPanel2);
  
    TLegend* leg12 = new TLegend(0.3, 0.7, 0.6, 0.9);
    can_PullXvsEta->cd(1);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullXvsEtaBarrel, newmePullXvsEtaBarrel, "barrel, eta", "pull x", lpull, hpull, leg12 );
    mePullXvsEtaBarrel->Draw("e");
    newmePullXvsEtaBarrel->Draw("Samee"); 
    myPV->PVCompute(mePullXvsEtaBarrel, newmePullXvsEtaBarrel, te );
    leg12->Draw();

    can_PullXvsEta->cd(2);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullXvsEtaZmPanel1, newmePullXvsEtaZmPanel1, "panel1, z<0, eta", "pull x", lpull, hpull );
    mePullXvsEtaZmPanel1->Draw("e");
    newmePullXvsEtaZmPanel1->Draw("samee"); 
    myPV->PVCompute(mePullXvsEtaZmPanel1, newmePullXvsEtaZmPanel1, te );
  
    can_PullXvsEta->cd(3);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullXvsEtaZmPanel2, newmePullXvsEtaZmPanel2, "panel2, z<0, eta", "pull x", lpull, hpull );
    mePullXvsEtaZmPanel2->Draw("e");
    newmePullXvsEtaZmPanel2->Draw("samee"); 
    myPV->PVCompute(mePullXvsEtaZmPanel2, newmePullXvsEtaZmPanel2, te );

    can_PullXvsEta->cd(5);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullXvsEtaZpPanel1, newmePullXvsEtaZpPanel1, "panel1, z>0, eta", "pull x", lpull, hpull );
    mePullXvsEtaZpPanel1->Draw("e");
    newmePullXvsEtaZpPanel1->Draw("samee"); 
    myPV->PVCompute(mePullXvsEtaZpPanel1, newmePullXvsEtaZpPanel1, te );

    can_PullXvsEta->cd(6);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullXvsEtaZpPanel2, newmePullXvsEtaZpPanel2, "panel2, z>0, eta", "pull x", lpull, hpull );
    mePullXvsEtaZpPanel2->Draw("e");
    newmePullXvsEtaZpPanel2->Draw("samee"); 
    myPV->PVCompute(mePullXvsEtaZpPanel2, newmePullXvsEtaZpPanel2, te );

    can_PullXvsEta->SaveAs("mePullXvsEta_compare.eps");
    can_PullXvsEta->SaveAs("mePullXvsEta_compare.gif");
  }

if (0) 
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

    rfile->GetObject("DQMData/Histograms_all/mePullXvsPhiBarrel"  , mePullXvsPhiBarrel  );
    rfile->GetObject("DQMData/Histograms_all/mePullXvsPhiZmPanel1", mePullXvsPhiZmPanel1);
    rfile->GetObject("DQMData/Histograms_all/mePullXvsPhiZmPanel2", mePullXvsPhiZmPanel2);
    rfile->GetObject("DQMData/Histograms_all/mePullXvsPhiZpPanel1", mePullXvsPhiZpPanel1);
    rfile->GetObject("DQMData/Histograms_all/mePullXvsPhiZpPanel2", mePullXvsPhiZpPanel2);

    sfile->GetObject("DQMData/Histograms_all/mePullXvsPhiBarrel"  , newmePullXvsPhiBarrel  ); 
    sfile->GetObject("DQMData/Histograms_all/mePullXvsPhiZmPanel1", newmePullXvsPhiZmPanel1);
    sfile->GetObject("DQMData/Histograms_all/mePullXvsPhiZmPanel2", newmePullXvsPhiZmPanel2);
    sfile->GetObject("DQMData/Histograms_all/mePullXvsPhiZpPanel1", newmePullXvsPhiZpPanel1);
    sfile->GetObject("DQMData/Histograms_all/mePullXvsPhiZpPanel2", newmePullXvsPhiZpPanel2);
  
    TLegend* leg13 = new TLegend(0.3, 0.7, 0.6, 0.9);
    can_PullXvsPhi->cd(1);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullXvsPhiBarrel, newmePullXvsPhiBarrel, "barrel, phi (deg)", "pull x", lpull, hpull, leg13 );
    mePullXvsPhiBarrel->Draw("e");
    newmePullXvsPhiBarrel->Draw("Samee"); 
    myPV->PVCompute(mePullXvsPhiBarrel, newmePullXvsPhiBarrel, te );
    leg13->Draw();

    can_PullXvsPhi->cd(2);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullXvsPhiZmPanel1, newmePullXvsPhiZmPanel1, "panel1, z<0, phi (deg)", "pull x", lpull, hpull );
    mePullXvsPhiZmPanel1->Draw("e");
    newmePullXvsPhiZmPanel1->Draw("samee"); 
    myPV->PVCompute(mePullXvsPhiZmPanel1, newmePullXvsPhiZmPanel1, te );
  
    can_PullXvsPhi->cd(3);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullXvsPhiZmPanel2, newmePullXvsPhiZmPanel2, "panel2, z<0, phi (deg)", "pull x", lpull, hpull );
    mePullXvsPhiZmPanel2->Draw("e");
    newmePullXvsPhiZmPanel2->Draw("samee"); 
    myPV->PVCompute(mePullXvsPhiZmPanel2, newmePullXvsPhiZmPanel2, te );

    can_PullXvsPhi->cd(5);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullXvsPhiZpPanel1, newmePullXvsPhiZpPanel1, "panel1, z>0, phi (deg)", "pull x", lpull, hpull );
    mePullXvsPhiZpPanel1->Draw("e");
    newmePullXvsPhiZpPanel1->Draw("samee"); 
    myPV->PVCompute(mePullXvsPhiZpPanel1, newmePullXvsPhiZpPanel1, te );

    can_PullXvsPhi->cd(6);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullXvsPhiZpPanel2, newmePullXvsPhiZpPanel2, "panel2, z>0, phi (deg)", "pull x" , lpull, hpull);
    mePullXvsPhiZpPanel2->Draw("e");
    newmePullXvsPhiZpPanel2->Draw("samee"); 
    myPV->PVCompute(mePullXvsPhiZpPanel2, newmePullXvsPhiZpPanel2, te );

    can_PullXvsPhi->SaveAs("mePullXvsPhi_compare.eps");
    can_PullXvsPhi->SaveAs("mePullXvsPhi_compare.gif");
  }

if (0) 
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

    rfile->GetObject("DQMData/Histograms_all/mePullYvsAlphaBarrel"  , mePullYvsAlphaBarrel  );
    rfile->GetObject("DQMData/Histograms_all/mePullYvsAlphaZmPanel1", mePullYvsAlphaZmPanel1);
    rfile->GetObject("DQMData/Histograms_all/mePullYvsAlphaZmPanel2", mePullYvsAlphaZmPanel2);
    rfile->GetObject("DQMData/Histograms_all/mePullYvsAlphaZpPanel1", mePullYvsAlphaZpPanel1);
    rfile->GetObject("DQMData/Histograms_all/mePullYvsAlphaZpPanel2", mePullYvsAlphaZpPanel2);

    sfile->GetObject("DQMData/Histograms_all/mePullYvsAlphaBarrel"  , newmePullYvsAlphaBarrel  ); 
    sfile->GetObject("DQMData/Histograms_all/mePullYvsAlphaZmPanel1", newmePullYvsAlphaZmPanel1);
    sfile->GetObject("DQMData/Histograms_all/mePullYvsAlphaZmPanel2", newmePullYvsAlphaZmPanel2);
    sfile->GetObject("DQMData/Histograms_all/mePullYvsAlphaZpPanel1", newmePullYvsAlphaZpPanel1);
    sfile->GetObject("DQMData/Histograms_all/mePullYvsAlphaZpPanel2", newmePullYvsAlphaZpPanel2);
  
    TLegend* leg14 = new TLegend(0.3, 0.7, 0.6, 0.9);
    can_PullYvsAlpha->cd(1);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsAlphaBarrel, newmePullYvsAlphaBarrel, "barrel, |alpha| (deg)", "pull y", lpull, hpull, leg14 );
    mePullYvsAlphaBarrel->Draw("e");
    newmePullYvsAlphaBarrel->Draw("Samee"); 
    myPV->PVCompute(mePullYvsAlphaBarrel, newmePullYvsAlphaBarrel, te );
    leg14->Draw();

    can_PullYvsAlpha->cd(2);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsAlphaZmPanel1, newmePullYvsAlphaZmPanel1, "panel1, z<0, |alpha| (deg)", "pull y", lpull, hpull );
    mePullYvsAlphaZmPanel1->Draw("e");
    newmePullYvsAlphaZmPanel1->Draw("samee"); 
    myPV->PVCompute(mePullYvsAlphaZmPanel1, newmePullYvsAlphaZmPanel1, te );
  
    can_PullYvsAlpha->cd(3);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsAlphaZmPanel2, newmePullYvsAlphaZmPanel2, "panel2, z<0, |alpha| (deg)", "pull y", lpull, hpull );
    mePullYvsAlphaZmPanel2->Draw("e");
    newmePullYvsAlphaZmPanel2->Draw("samee"); 
    myPV->PVCompute(mePullYvsAlphaZmPanel2, newmePullYvsAlphaZmPanel2, te );

    can_PullYvsAlpha->cd(5);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsAlphaZpPanel1, newmePullYvsAlphaZpPanel1, "panel1, z>0, |alpha| (deg)", "pull y", lpull, hpull );
    mePullYvsAlphaZpPanel1->Draw("e");
    newmePullYvsAlphaZpPanel1->Draw("samee"); 
    myPV->PVCompute(mePullYvsAlphaZpPanel1, newmePullYvsAlphaZpPanel1, te );

    can_PullYvsAlpha->cd(6);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsAlphaZpPanel2, newmePullYvsAlphaZpPanel2, "panel2, z>0, |alpha| (deg)", "pull y" , lpull, hpull);
    mePullYvsAlphaZpPanel2->Draw("e");
    newmePullYvsAlphaZpPanel2->Draw("samee"); 
    myPV->PVCompute(mePullYvsAlphaZpPanel2, newmePullYvsAlphaZpPanel2, te );

    can_PullYvsAlpha->SaveAs("mePullYvsAlpha_compare.eps");
    can_PullYvsAlpha->SaveAs("mePullYvsAlpha_compare.gif");
  }

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

    rfile->GetObject("DQMData/Histograms_all/mePullYvsBetaBarrel"  , mePullYvsBetaBarrel  );
    rfile->GetObject("DQMData/Histograms_all/mePullYvsBetaZmPanel1", mePullYvsBetaZmPanel1);
    rfile->GetObject("DQMData/Histograms_all/mePullYvsBetaZmPanel2", mePullYvsBetaZmPanel2);
    rfile->GetObject("DQMData/Histograms_all/mePullYvsBetaZpPanel1", mePullYvsBetaZpPanel1);
    rfile->GetObject("DQMData/Histograms_all/mePullYvsBetaZpPanel2", mePullYvsBetaZpPanel2);

    sfile->GetObject("DQMData/Histograms_all/mePullYvsBetaBarrel"  , newmePullYvsBetaBarrel  ); 
    sfile->GetObject("DQMData/Histograms_all/mePullYvsBetaZmPanel1", newmePullYvsBetaZmPanel1);
    sfile->GetObject("DQMData/Histograms_all/mePullYvsBetaZmPanel2", newmePullYvsBetaZmPanel2);
    sfile->GetObject("DQMData/Histograms_all/mePullYvsBetaZpPanel1", newmePullYvsBetaZpPanel1);
    sfile->GetObject("DQMData/Histograms_all/mePullYvsBetaZpPanel2", newmePullYvsBetaZpPanel2);
  
    TLegend* leg15 = new TLegend(0.3, 0.7, 0.6, 0.9);
    can_PullYvsBeta->cd(1);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsBetaBarrel, newmePullYvsBetaBarrel, "barrel, |beta| (deg)", "pull y", lpull, hpull, leg15 );
    mePullYvsBetaBarrel->Draw("e");
    newmePullYvsBetaBarrel->Draw("Samee"); 
    myPV->PVCompute(mePullYvsBetaBarrel, newmePullYvsBetaBarrel, te );
    leg15->Draw();

    can_PullYvsBeta->cd(2);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsBetaZmPanel1, newmePullYvsBetaZmPanel1, "panel1, z<0, |beta| (deg)", "pull y", lpull, hpull );
    mePullYvsBetaZmPanel1->Draw("e");
    newmePullYvsBetaZmPanel1->Draw("samee"); 
    myPV->PVCompute(mePullYvsBetaZmPanel1, newmePullYvsBetaZmPanel1, te );
  
    can_PullYvsBeta->cd(3);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsBetaZmPanel2, newmePullYvsBetaZmPanel2, "panel2, z<0, |beta| (deg)", "pull y", lpull, hpull );
    mePullYvsBetaZmPanel2->Draw("e");
    newmePullYvsBetaZmPanel2->Draw("samee"); 
    myPV->PVCompute(mePullYvsBetaZmPanel2, newmePullYvsBetaZmPanel2, te );

    can_PullYvsBeta->cd(5);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsBetaZpPanel1, newmePullYvsBetaZpPanel1, "panel1, z>0, |beta| (deg)", "pull y", lpull, hpull );
    mePullYvsBetaZpPanel1->Draw("e");
    newmePullYvsBetaZpPanel1->Draw("samee"); 
    myPV->PVCompute(mePullYvsBetaZpPanel1, newmePullYvsBetaZpPanel1, te );

    can_PullYvsBeta->cd(6);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsBetaZpPanel2, newmePullYvsBetaZpPanel2, "panel2, z>0, |beta| (deg)", "pull y", lpull, hpull );
    mePullYvsBetaZpPanel2->Draw("e");
    newmePullYvsBetaZpPanel2->Draw("samee"); 
    myPV->PVCompute(mePullYvsBetaZpPanel2, newmePullYvsBetaZpPanel2, te );

    can_PullYvsBeta->SaveAs("mePullYvsBeta_compare.eps");
    can_PullYvsBeta->SaveAs("mePullYvsBeta_compare.gif");
  }

if (0) 
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

    rfile->GetObject("DQMData/Histograms_all/mePullYvsEtaBarrel"  , mePullYvsEtaBarrel  );
    rfile->GetObject("DQMData/Histograms_all/mePullYvsEtaZmPanel1", mePullYvsEtaZmPanel1);
    rfile->GetObject("DQMData/Histograms_all/mePullYvsEtaZmPanel2", mePullYvsEtaZmPanel2);
    rfile->GetObject("DQMData/Histograms_all/mePullYvsEtaZpPanel1", mePullYvsEtaZpPanel1);
    rfile->GetObject("DQMData/Histograms_all/mePullYvsEtaZpPanel2", mePullYvsEtaZpPanel2);

    sfile->GetObject("DQMData/Histograms_all/mePullYvsEtaBarrel"  , newmePullYvsEtaBarrel  ); 
    sfile->GetObject("DQMData/Histograms_all/mePullYvsEtaZmPanel1", newmePullYvsEtaZmPanel1);
    sfile->GetObject("DQMData/Histograms_all/mePullYvsEtaZmPanel2", newmePullYvsEtaZmPanel2);
    sfile->GetObject("DQMData/Histograms_all/mePullYvsEtaZpPanel1", newmePullYvsEtaZpPanel1);
    sfile->GetObject("DQMData/Histograms_all/mePullYvsEtaZpPanel2", newmePullYvsEtaZpPanel2);
  
    TLegend* leg16 = new TLegend(0.3, 0.7, 0.6, 0.9);
    can_PullYvsEta->cd(1);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsEtaBarrel, newmePullYvsEtaBarrel, "barrel, eta", "pull y", lpull, hpull, leg16 );
    mePullYvsEtaBarrel->Draw("e");
    newmePullYvsEtaBarrel->Draw("Samee"); 
    myPV->PVCompute(mePullYvsEtaBarrel, newmePullYvsEtaBarrel, te );
    leg16->Draw();

    can_PullYvsEta->cd(2);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsEtaZmPanel1, newmePullYvsEtaZmPanel1, "panel1, z<0, eta", "pull y" , lpull, hpull);
    mePullYvsEtaZmPanel1->Draw("e");
    newmePullYvsEtaZmPanel1->Draw("samee"); 
    myPV->PVCompute(mePullYvsEtaZmPanel1, newmePullYvsEtaZmPanel1, te );
  
    can_PullYvsEta->cd(3);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsEtaZmPanel2, newmePullYvsEtaZmPanel2, "panel2, z<0, eta", "pull y", lpull, hpull );
    mePullYvsEtaZmPanel2->Draw("e");
    newmePullYvsEtaZmPanel2->Draw("samee"); 
    myPV->PVCompute(mePullYvsEtaZmPanel2, newmePullYvsEtaZmPanel2, te );

    can_PullYvsEta->cd(5);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsEtaZpPanel1, newmePullYvsEtaZpPanel1, "panel1, z>0, eta", "pull y", lpull, hpull );
    mePullYvsEtaZpPanel1->Draw("e");
    newmePullYvsEtaZpPanel1->Draw("samee"); 
    myPV->PVCompute(mePullYvsEtaZpPanel1, newmePullYvsEtaZpPanel1, te );

    can_PullYvsEta->cd(6);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsEtaZpPanel2, newmePullYvsEtaZpPanel2, "panel2, z>0, eta", "pull y", lpull, hpull );
    mePullYvsEtaZpPanel2->Draw("e");
    newmePullYvsEtaZpPanel2->Draw("samee"); 
    myPV->PVCompute(mePullYvsEtaZpPanel2, newmePullYvsEtaZpPanel2, te );

    can_PullYvsEta->SaveAs("mePullYvsEta_compare.eps");
    can_PullYvsEta->SaveAs("mePullYvsEta_compare.gif");
  }

if (0) 
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

    rfile->GetObject("DQMData/Histograms_all/mePullYvsPhiBarrel"  , mePullYvsPhiBarrel  );
    rfile->GetObject("DQMData/Histograms_all/mePullYvsPhiZmPanel1", mePullYvsPhiZmPanel1);
    rfile->GetObject("DQMData/Histograms_all/mePullYvsPhiZmPanel2", mePullYvsPhiZmPanel2);
    rfile->GetObject("DQMData/Histograms_all/mePullYvsPhiZpPanel1", mePullYvsPhiZpPanel1);
    rfile->GetObject("DQMData/Histograms_all/mePullYvsPhiZpPanel2", mePullYvsPhiZpPanel2);

    sfile->GetObject("DQMData/Histograms_all/mePullYvsPhiBarrel"  , newmePullYvsPhiBarrel  ); 
    sfile->GetObject("DQMData/Histograms_all/mePullYvsPhiZmPanel1", newmePullYvsPhiZmPanel1);
    sfile->GetObject("DQMData/Histograms_all/mePullYvsPhiZmPanel2", newmePullYvsPhiZmPanel2);
    sfile->GetObject("DQMData/Histograms_all/mePullYvsPhiZpPanel1", newmePullYvsPhiZpPanel1);
    sfile->GetObject("DQMData/Histograms_all/mePullYvsPhiZpPanel2", newmePullYvsPhiZpPanel2);
  
    TLegend* leg17 = new TLegend(0.3, 0.7, 0.6, 0.9);
    can_PullYvsPhi->cd(1);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsPhiBarrel, newmePullYvsPhiBarrel, "barrel, phi (deg)", "pull y", lpull, hpull, leg17 );
    mePullYvsPhiBarrel->Draw("e");
    newmePullYvsPhiBarrel->Draw("Samee"); 
    myPV->PVCompute(mePullYvsPhiBarrel, newmePullYvsPhiBarrel, te );
    leg17->Draw();

    can_PullYvsPhi->cd(2);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsPhiZmPanel1, newmePullYvsPhiZmPanel1, "panel1, z<0, phi (deg)", "pull y" , lpull, hpull);
    mePullYvsPhiZmPanel1->Draw("e");
    newmePullYvsPhiZmPanel1->Draw("samee"); 
    myPV->PVCompute(mePullYvsPhiZmPanel1, newmePullYvsPhiZmPanel1, te );
  
    can_PullYvsPhi->cd(3);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsPhiZmPanel2, newmePullYvsPhiZmPanel2, "panel2, z<0, phi (deg)", "pull y" , lpull, hpull);
    mePullYvsPhiZmPanel2->Draw("e");
    newmePullYvsPhiZmPanel2->Draw("samee"); 
    myPV->PVCompute(mePullYvsPhiZmPanel2, newmePullYvsPhiZmPanel2, te );

    can_PullYvsPhi->cd(5);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsPhiZpPanel1, newmePullYvsPhiZpPanel1, "panel1, z>0, phi (deg)", "pull y", lpull, hpull );
    mePullYvsPhiZpPanel1->Draw("e");
    newmePullYvsPhiZpPanel1->Draw("samee"); 
    myPV->PVCompute(mePullYvsPhiZpPanel1, newmePullYvsPhiZpPanel1, te );

    can_PullYvsPhi->cd(6);
    //gPad->SetLogy();
    SetUpProfileHistograms(mePullYvsPhiZpPanel2, newmePullYvsPhiZpPanel2, "panel2, z>0, phi (deg)", "pull y", lpull, hpull );
    mePullYvsPhiZpPanel2->Draw("e");
    newmePullYvsPhiZpPanel2->Draw("samee"); 
    myPV->PVCompute(mePullYvsPhiZpPanel2, newmePullYvsPhiZpPanel2, te );

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

    rfile->GetObject("DQMData/Histograms_all/mePullxBarrel"  , mePullxBarrel  );
    rfile->GetObject("DQMData/Histograms_all/mePullxZmPanel1", mePullxZmPanel1);
    rfile->GetObject("DQMData/Histograms_all/mePullxZmPanel2", mePullxZmPanel2);
    rfile->GetObject("DQMData/Histograms_all/mePullxZpPanel1", mePullxZpPanel1);
    rfile->GetObject("DQMData/Histograms_all/mePullxZpPanel2", mePullxZpPanel2);

    sfile->GetObject("DQMData/Histograms_all/mePullxBarrel"  , newmePullxBarrel  ); 
    sfile->GetObject("DQMData/Histograms_all/mePullxZmPanel1", newmePullxZmPanel1);
    sfile->GetObject("DQMData/Histograms_all/mePullxZmPanel2", newmePullxZmPanel2);
    sfile->GetObject("DQMData/Histograms_all/mePullxZpPanel1", newmePullxZpPanel1);
    sfile->GetObject("DQMData/Histograms_all/mePullxZpPanel2", newmePullxZpPanel2);
  
    TLegend* leg18 = new TLegend(0.3, 0.7, 0.6, 0.9);
    can_mePullx->cd(1);
    //gPad->SetLogy();
    SetUpHistograms(mePullxBarrel, newmePullxBarrel, "barrel, pull x", leg18);
    mePullxBarrel->Draw("he");
    newmePullxBarrel->Draw("Samehe"); 
    myPV->PVCompute(mePullxBarrel, newmePullxBarrel, te );
    leg18->Draw();

    can_mePullx->cd(2);
    //gPad->SetLogy();
    SetUpHistograms(mePullxZmPanel1, newmePullxZmPanel1, "panel1, z<0, pull x" );
    mePullxZmPanel1->Draw("he");
    newmePullxZmPanel1->Draw("samehe"); 
    myPV->PVCompute(mePullxZmPanel1, newmePullxZmPanel1, te );
  
    can_mePullx->cd(3);
    //gPad->SetLogy();
    SetUpHistograms(mePullxZmPanel2, newmePullxZmPanel2, "panel2, z<0, pull x" );
    mePullxZmPanel2->Draw("he");
    newmePullxZmPanel2->Draw("samehe"); 
    myPV->PVCompute(mePullxZmPanel2, newmePullxZmPanel2, te );

    can_mePullx->cd(5);
    //gPad->SetLogy();
    SetUpHistograms(mePullxZpPanel1, newmePullxZpPanel1, "panel2, z>0, pull x" );
    mePullxZpPanel1->Draw("he");
    newmePullxZpPanel1->Draw("samehe"); 
    myPV->PVCompute(mePullxZpPanel1, newmePullxZpPanel1, te );

    can_mePullx->cd(6);
    //gPad->SetLogy();
    SetUpHistograms(mePullxZpPanel2, newmePullxZpPanel2, "panel1, z>0, pull x" );
    mePullxZpPanel2->Draw("he");
    newmePullxZpPanel2->Draw("samehe"); 
    myPV->PVCompute(mePullxZpPanel2, newmePullxZpPanel2, te );

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

    rfile->GetObject("DQMData/Histograms_all/mePullyBarrel"  , mePullyBarrel  );
    rfile->GetObject("DQMData/Histograms_all/mePullyZmPanel1", mePullyZmPanel1);
    rfile->GetObject("DQMData/Histograms_all/mePullyZmPanel2", mePullyZmPanel2);
    rfile->GetObject("DQMData/Histograms_all/mePullyZpPanel1", mePullyZpPanel1);
    rfile->GetObject("DQMData/Histograms_all/mePullyZpPanel2", mePullyZpPanel2);

    sfile->GetObject("DQMData/Histograms_all/mePullyBarrel"  , newmePullyBarrel  ); 
    sfile->GetObject("DQMData/Histograms_all/mePullyZmPanel1", newmePullyZmPanel1);
    sfile->GetObject("DQMData/Histograms_all/mePullyZmPanel2", newmePullyZmPanel2);
    sfile->GetObject("DQMData/Histograms_all/mePullyZpPanel1", newmePullyZpPanel1);
    sfile->GetObject("DQMData/Histograms_all/mePullyZpPanel2", newmePullyZpPanel2);
  
    TLegend* leg19 = new TLegend(0.3, 0.7, 0.6, 0.9);
    can_mePully->cd(1);
    //gPad->SetLogy();
    SetUpHistograms(mePullyBarrel, newmePullyBarrel, "barrel, pull y", leg19 );
    mePullyBarrel->Draw("he");
    newmePullyBarrel->Draw("Samehe"); 
    myPV->PVCompute(mePullyBarrel, newmePullyBarrel, te );
    leg19->Draw();
    
    can_mePully->cd(2);
    //gPad->SetLogy();
    SetUpHistograms(mePullyZmPanel1, newmePullyZmPanel1, "panel1, z<0, pull y" );
    mePullyZmPanel1->Draw("he");
    newmePullyZmPanel1->Draw("samehe"); 
    myPV->PVCompute(mePullyZmPanel1, newmePullyZmPanel1, te );
  
    can_mePully->cd(3);
    //gPad->SetLogy();
    SetUpHistograms(mePullyZmPanel2, newmePullyZmPanel2, "panel2, z<0, pull y" );
    mePullyZmPanel2->Draw("he");
    newmePullyZmPanel2->Draw("samehe"); 
    myPV->PVCompute(mePullyZmPanel2, newmePullyZmPanel2, te );

    can_mePully->cd(5);
    //gPad->SetLogy();
    SetUpHistograms(mePullyZpPanel1, newmePullyZpPanel1, "panel1, z>0, pull y" );
    mePullyZpPanel1->Draw("he");
    newmePullyZpPanel1->Draw("samehe"); 
    myPV->PVCompute(mePullyZpPanel1, newmePullyZpPanel1, te );

    can_mePully->cd(6);
    //gPad->SetLogy();
    SetUpHistograms(mePullyZpPanel2, newmePullyZpPanel2, "panel2, z>0, pull y" );
    mePullyZpPanel2->Draw("he");
    newmePullyZpPanel2->Draw("samehe"); 
    myPV->PVCompute(mePullyZpPanel2, newmePullyZpPanel2, te );

    can_mePully->SaveAs("mePully_compare.eps");
    can_mePully->SaveAs("mePully_compare.gif");
  }

 double ymin = 0.0005;
 double ymax = 0.0015;

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

    rfile->GetObject("DQMData/Histograms_all/meResXvsAlphaBarrelFlippedLadders"     , meResXvsAlphaBarrelFlippedLadders     );
    rfile->GetObject("DQMData/Histograms_all/meResXvsAlphaBarrelNonFlippedLadders"  , meResXvsAlphaBarrelNonFlippedLadders  );

    rfile->GetObject("DQMData/Histograms_all/meResXvsAlphaZmPanel1", meResXvsAlphaZmPanel1);
    rfile->GetObject("DQMData/Histograms_all/meResXvsAlphaZmPanel2", meResXvsAlphaZmPanel2);
    rfile->GetObject("DQMData/Histograms_all/meResXvsAlphaZpPanel1", meResXvsAlphaZpPanel1);
    rfile->GetObject("DQMData/Histograms_all/meResXvsAlphaZpPanel2", meResXvsAlphaZpPanel2);

    sfile->GetObject("DQMData/Histograms_all/meResXvsAlphaBarrelFlippedLadders"   , newmeResXvsAlphaBarrelFlippedLadders     );
    sfile->GetObject("DQMData/Histograms_all/meResXvsAlphaBarrelNonFlippedLadders", newmeResXvsAlphaBarrelNonFlippedLadders  );
 
    sfile->GetObject("DQMData/Histograms_all/meResXvsAlphaZmPanel1", newmeResXvsAlphaZmPanel1);
    sfile->GetObject("DQMData/Histograms_all/meResXvsAlphaZmPanel2", newmeResXvsAlphaZmPanel2);
    sfile->GetObject("DQMData/Histograms_all/meResXvsAlphaZpPanel1", newmeResXvsAlphaZpPanel1);
    sfile->GetObject("DQMData/Histograms_all/meResXvsAlphaZpPanel2", newmeResXvsAlphaZpPanel2);
  
    TLegend* leg20 = new TLegend(0.3, 0.7, 0.6, 0.9);
    can_ResXvsAlpha->cd(1);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResXvsAlphaBarrelFlippedLadders, newmeResXvsAlphaBarrelFlippedLadders, 
			   "barrel, non-flipped ladders, |alpha| (deg)", "<|x residual|> (cm)", ymin, ymax, leg20 );
    //meResXvsAlphaBarrelFlippedLadders->SetTitleOffset(2.5, "Y");
    //meResXvsAlphaBarrelFlippedLadders->SetMinimum(0.0005);
    //meResXvsAlphaBarrelFlippedLadders->SetMaximum(0.0015);
    meResXvsAlphaBarrelFlippedLadders->Draw("e");
    newmeResXvsAlphaBarrelFlippedLadders->Draw("Samee"); 
    myPV->PVCompute(meResXvsAlphaBarrelFlippedLadders, newmeResXvsAlphaBarrelFlippedLadders, te );
    leg20->Draw();

    can_ResXvsAlpha->cd(2);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResXvsAlphaZmPanel1, newmeResXvsAlphaZmPanel1, 
			   "panel1, z<0, |alpha| (deg)", "<|x residual|> (cm)", ymin, ymax );
    meResXvsAlphaZmPanel1->SetMinimum(0.0005);
    meResXvsAlphaZmPanel1->SetMaximum(0.0015);
    meResXvsAlphaZmPanel1->Draw("e");
    newmeResXvsAlphaZmPanel1->Draw("samee"); 
    myPV->PVCompute(meResXvsAlphaZmPanel1, newmeResXvsAlphaZmPanel1, te );
  
    can_ResXvsAlpha->cd(3);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResXvsAlphaZmPanel2, newmeResXvsAlphaZmPanel2, 
			   "panel2, z<0, |alpha| (deg)", "<|x residual|> (cm)", ymin, ymax );
    meResXvsAlphaZmPanel2->SetMinimum(0.0005);
    meResXvsAlphaZmPanel2->SetMaximum(0.0015);
    meResXvsAlphaZmPanel2->Draw("e");
    newmeResXvsAlphaZmPanel2->Draw("samee"); 
    myPV->PVCompute(meResXvsAlphaZmPanel2, newmeResXvsAlphaZmPanel2, te );

    can_ResXvsAlpha->cd(4);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResXvsAlphaBarrelNonFlippedLadders, newmeResXvsAlphaBarrelNonFlippedLadders, 
			   "barrel, flipped ladders, |alpha| (deg)", "<|x residual|> (cm)", ymin, ymax );
    meResXvsAlphaBarrelNonFlippedLadders->SetMinimum(0.0005);
    meResXvsAlphaBarrelNonFlippedLadders->SetMaximum(0.0015);
    meResXvsAlphaBarrelNonFlippedLadders->Draw("e");
    newmeResXvsAlphaBarrelNonFlippedLadders->Draw("Samee"); 
    myPV->PVCompute(meResXvsAlphaBarrelNonFlippedLadders, newmeResXvsAlphaBarrelNonFlippedLadders, te );

    can_ResXvsAlpha->cd(5);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResXvsAlphaZpPanel1, newmeResXvsAlphaZpPanel1, 
			   "panel1, z>0, |alpha| (deg)", "<|x residual|> (cm)", ymin, ymax );
    meResXvsAlphaZpPanel1->SetMinimum(0.0005);
    meResXvsAlphaZpPanel1->SetMaximum(0.0015);
    meResXvsAlphaZpPanel1->Draw("e");
    newmeResXvsAlphaZpPanel1->Draw("samee"); 
    myPV->PVCompute(meResXvsAlphaZpPanel1, newmeResXvsAlphaZpPanel1, te );

    can_ResXvsAlpha->cd(6);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResXvsAlphaZpPanel2, newmeResXvsAlphaZpPanel2,
			   "panel2, z>0, |alpha| (deg)", "<|x residual|> (cm)", ymin, ymax );
    meResXvsAlphaZpPanel2->SetMinimum(0.0005);
    meResXvsAlphaZpPanel2->SetMaximum(0.0015);
    meResXvsAlphaZpPanel2->Draw("e");
    newmeResXvsAlphaZpPanel2->Draw("samee"); 
    myPV->PVCompute(meResXvsAlphaZpPanel2, newmeResXvsAlphaZpPanel2, te );

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

    rfile->GetObject("DQMData/Histograms_all/meResXvsBetaBarrel"  , meResXvsBetaBarrel  );
    rfile->GetObject("DQMData/Histograms_all/meResXvsBetaZmPanel1", meResXvsBetaZmPanel1);
    rfile->GetObject("DQMData/Histograms_all/meResXvsBetaZmPanel2", meResXvsBetaZmPanel2);
    rfile->GetObject("DQMData/Histograms_all/meResXvsBetaZpPanel1", meResXvsBetaZpPanel1);
    rfile->GetObject("DQMData/Histograms_all/meResXvsBetaZpPanel2", meResXvsBetaZpPanel2);

    sfile->GetObject("DQMData/Histograms_all/meResXvsBetaBarrel"  , newmeResXvsBetaBarrel  ); 
    sfile->GetObject("DQMData/Histograms_all/meResXvsBetaZmPanel1", newmeResXvsBetaZmPanel1);
    sfile->GetObject("DQMData/Histograms_all/meResXvsBetaZmPanel2", newmeResXvsBetaZmPanel2);
    sfile->GetObject("DQMData/Histograms_all/meResXvsBetaZpPanel1", newmeResXvsBetaZpPanel1);
    sfile->GetObject("DQMData/Histograms_all/meResXvsBetaZpPanel2", newmeResXvsBetaZpPanel2);
  
    TLegend* leg21 = new TLegend(0.3, 0.7, 0.6, 0.9);
    can_ResXvsBeta->cd(1);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResXvsBetaBarrel, newmeResXvsBetaBarrel, 
			   "barrel, |beta| (deg)", "<|x residual|> (cm)", ymin, ymax, leg21 );
    meResXvsBetaBarrel->Draw("e");
    newmeResXvsBetaBarrel->Draw("Samee"); 
    myPV->PVCompute(meResXvsBetaBarrel, newmeResXvsBetaBarrel, te );
    leg21->Draw();

    can_ResXvsBeta->cd(2);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResXvsBetaZmPanel1, newmeResXvsBetaZmPanel1, 
			   "panel1, z<0, |beta| (deg)", "<|x residual|> (cm)", ymin, ymax );
    meResXvsBetaZmPanel1->Draw("e");
    newmeResXvsBetaZmPanel1->Draw("samee"); 
    myPV->PVCompute(meResXvsBetaZmPanel1, newmeResXvsBetaZmPanel1, te );
  
    can_ResXvsBeta->cd(3);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResXvsBetaZmPanel2, newmeResXvsBetaZmPanel2, 
			   "panel2, z<0, |beta| (deg)", "<|x residual|> (cm)", ymin, ymax ); 
    meResXvsBetaZmPanel2->Draw("e");
    newmeResXvsBetaZmPanel2->Draw("samee"); 
    myPV->PVCompute(meResXvsBetaZmPanel2, newmeResXvsBetaZmPanel2, te );

    can_ResXvsBeta->cd(5);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResXvsBetaZpPanel1, newmeResXvsBetaZpPanel1, "panel1, z>0, |beta| (deg)", "<|x residual|> (cm)", ymin, ymax );
    meResXvsBetaZpPanel1->Draw("e");
    newmeResXvsBetaZpPanel1->Draw("samee"); 
    myPV->PVCompute(meResXvsBetaZpPanel1, newmeResXvsBetaZpPanel1, te );

    can_ResXvsBeta->cd(6);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResXvsBetaZpPanel2, newmeResXvsBetaZpPanel2, "panel2, z>0, |beta| (deg)", "<|x residual|> (cm)", ymin, ymax );
    meResXvsBetaZpPanel2->Draw("e");
    newmeResXvsBetaZpPanel2->Draw("samee"); 
    myPV->PVCompute(meResXvsBetaZpPanel2, newmeResXvsBetaZpPanel2, te );

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

    rfile->GetObject("DQMData/Histograms_all/meResYvsAlphaBarrel"  , meResYvsAlphaBarrel  );
    rfile->GetObject("DQMData/Histograms_all/meResYvsAlphaZmPanel1", meResYvsAlphaZmPanel1);
    rfile->GetObject("DQMData/Histograms_all/meResYvsAlphaZmPanel2", meResYvsAlphaZmPanel2);
    rfile->GetObject("DQMData/Histograms_all/meResYvsAlphaZpPanel1", meResYvsAlphaZpPanel1);
    rfile->GetObject("DQMData/Histograms_all/meResYvsAlphaZpPanel2", meResYvsAlphaZpPanel2);

    sfile->GetObject("DQMData/Histograms_all/meResYvsAlphaBarrel"  , newmeResYvsAlphaBarrel  ); 
    sfile->GetObject("DQMData/Histograms_all/meResYvsAlphaZmPanel1", newmeResYvsAlphaZmPanel1);
    sfile->GetObject("DQMData/Histograms_all/meResYvsAlphaZmPanel2", newmeResYvsAlphaZmPanel2);
    sfile->GetObject("DQMData/Histograms_all/meResYvsAlphaZpPanel1", newmeResYvsAlphaZpPanel1);
    sfile->GetObject("DQMData/Histograms_all/meResYvsAlphaZpPanel2", newmeResYvsAlphaZpPanel2);
  
    TLegend* leg22 = new TLegend(0.3, 0.7, 0.6, 0.9);
    can_ResYvsAlpha->cd(1);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResYvsAlphaBarrel, newmeResYvsAlphaBarrel, 
			   "barrel, |alpha| (deg)", "<|y residual|> (cm)", ymin+0.0010, ymax+0.0010, leg22 );
    meResYvsAlphaBarrel->Draw("e");
    newmeResYvsAlphaBarrel->Draw("Samee"); 
    myPV->PVCompute(meResYvsAlphaBarrel, newmeResYvsAlphaBarrel, te );
    leg22->Draw();

    can_ResYvsAlpha->cd(2);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResYvsAlphaZmPanel1, newmeResYvsAlphaZmPanel1, "panel1, z<0, |alpha| (deg)", "<|y residual|> (cm)", ymin, ymax );
    meResYvsAlphaZmPanel1->Draw("e");
    newmeResYvsAlphaZmPanel1->Draw("samee"); 
    myPV->PVCompute(meResYvsAlphaZmPanel1, newmeResYvsAlphaZmPanel1, te );
  
    can_ResYvsAlpha->cd(3);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResYvsAlphaZmPanel2, newmeResYvsAlphaZmPanel2, "panel2, z<0, |alpha| (deg)", "<|y residual|> (cm)", ymin, ymax );
    meResYvsAlphaZmPanel2->Draw("e");
    newmeResYvsAlphaZmPanel2->Draw("samee"); 
    myPV->PVCompute(meResYvsAlphaZmPanel2, newmeResYvsAlphaZmPanel2, te );

    can_ResYvsAlpha->cd(5);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResYvsAlphaZpPanel1, newmeResYvsAlphaZpPanel1, "panel1, z>0, |alpha| (deg)", "<|y residual|> (cm)" , ymin, ymax);
    meResYvsAlphaZpPanel1->Draw("e");
    newmeResYvsAlphaZpPanel1->Draw("samee"); 
    myPV->PVCompute(meResYvsAlphaZpPanel1, newmeResYvsAlphaZpPanel1, te );

    can_ResYvsAlpha->cd(6);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResYvsAlphaZpPanel2, newmeResYvsAlphaZpPanel2, "panel2, z>0, |alpha| (deg)", "<|y residual|> (cm)", ymin, ymax );
    meResYvsAlphaZpPanel2->Draw("e");
    newmeResYvsAlphaZpPanel2->Draw("samee"); 
    myPV->PVCompute(meResYvsAlphaZpPanel2, newmeResYvsAlphaZpPanel2, te );

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

    rfile->GetObject("DQMData/Histograms_all/meResYvsBetaBarrel"  , meResYvsBetaBarrel  );
    rfile->GetObject("DQMData/Histograms_all/meResYvsBetaZmPanel1", meResYvsBetaZmPanel1);
    rfile->GetObject("DQMData/Histograms_all/meResYvsBetaZmPanel2", meResYvsBetaZmPanel2);
    rfile->GetObject("DQMData/Histograms_all/meResYvsBetaZpPanel1", meResYvsBetaZpPanel1);
    rfile->GetObject("DQMData/Histograms_all/meResYvsBetaZpPanel2", meResYvsBetaZpPanel2);

    sfile->GetObject("DQMData/Histograms_all/meResYvsBetaBarrel"  , newmeResYvsBetaBarrel  ); 
    sfile->GetObject("DQMData/Histograms_all/meResYvsBetaZmPanel1", newmeResYvsBetaZmPanel1);
    sfile->GetObject("DQMData/Histograms_all/meResYvsBetaZmPanel2", newmeResYvsBetaZmPanel2);
    sfile->GetObject("DQMData/Histograms_all/meResYvsBetaZpPanel1", newmeResYvsBetaZpPanel1);
    sfile->GetObject("DQMData/Histograms_all/meResYvsBetaZpPanel2", newmeResYvsBetaZpPanel2);
  
    TLegend* leg23 = new TLegend(0.3, 0.7, 0.6, 0.9);
    can_ResYvsBeta->cd(1);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResYvsBetaBarrel, newmeResYvsBetaBarrel, 
			   "barrel, |beta| (deg)", "<|y residual|> (cm)", 0.0000, 0.0060, leg23 );
    meResYvsBetaBarrel->Draw("e");
    newmeResYvsBetaBarrel->Draw("Samee"); 
    myPV->PVCompute(meResYvsBetaBarrel, newmeResYvsBetaBarrel, te );
    leg23->Draw();

    can_ResYvsBeta->cd(2);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResYvsBetaZmPanel1, newmeResYvsBetaZmPanel1, "panel1, z<0, |beta| (deg)", "<|y residual|> (cm)", ymin, ymax );
    meResYvsBetaZmPanel1->Draw("e");
    newmeResYvsBetaZmPanel1->Draw("samee"); 
    myPV->PVCompute(meResYvsBetaZmPanel1, newmeResYvsBetaZmPanel1, te );
  
    can_ResYvsBeta->cd(3);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResYvsBetaZmPanel2, newmeResYvsBetaZmPanel2, "panel2, z<0, |beta| (deg)", "<|y residual|> (cm)", ymin, ymax );
    meResYvsBetaZmPanel2->Draw("e");
    newmeResYvsBetaZmPanel2->Draw("samee"); 
    myPV->PVCompute(meResYvsBetaZmPanel2, newmeResYvsBetaZmPanel2, te );

    can_ResYvsBeta->cd(5);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResYvsBetaZpPanel1, newmeResYvsBetaZpPanel1, "panel1, z>0, |beta| (deg)", "<|y residual|> (cm)", ymin, ymax );
    meResYvsBetaZpPanel1->Draw("e");
    newmeResYvsBetaZpPanel1->Draw("samee"); 
    myPV->PVCompute(meResYvsBetaZpPanel1, newmeResYvsBetaZpPanel1, te );

    can_ResYvsBeta->cd(6);
    //gPad->SetLogy();
    SetUpProfileHistograms(meResYvsBetaZpPanel2, newmeResYvsBetaZpPanel2, "panel2, z>0, |beta| (deg)", "<|y residual|> (cm)", ymin, ymax );
    meResYvsBetaZpPanel2->Draw("e");
    newmeResYvsBetaZpPanel2->Draw("samee"); 
    myPV->PVCompute(meResYvsBetaZpPanel2, newmeResYvsBetaZpPanel2, te );

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

    rfile->GetObject("DQMData/Histograms_all/meResxBarrel"  , meResxBarrel  );
    rfile->GetObject("DQMData/Histograms_all/meResxZmPanel1", meResxZmPanel1);
    rfile->GetObject("DQMData/Histograms_all/meResxZmPanel2", meResxZmPanel2);
    rfile->GetObject("DQMData/Histograms_all/meResxZpPanel1", meResxZpPanel1);
    rfile->GetObject("DQMData/Histograms_all/meResxZpPanel2", meResxZpPanel2);

    sfile->GetObject("DQMData/Histograms_all/meResxBarrel"  , newmeResxBarrel  ); 
    sfile->GetObject("DQMData/Histograms_all/meResxZmPanel1", newmeResxZmPanel1);
    sfile->GetObject("DQMData/Histograms_all/meResxZmPanel2", newmeResxZmPanel2);
    sfile->GetObject("DQMData/Histograms_all/meResxZpPanel1", newmeResxZpPanel1);
    sfile->GetObject("DQMData/Histograms_all/meResxZpPanel2", newmeResxZpPanel2);
  
    TLegend* leg24 = new TLegend(0.3, 0.7, 0.6, 0.9);
    can_meResx->cd(1);
    gPad->SetLogy();
    SetUpHistograms(meResxBarrel, newmeResxBarrel, "barrel, x residual (cm)", leg24 );
    meResxBarrel->Draw("he");
    newmeResxBarrel->Draw("Samehe"); 
    myPV->PVCompute(meResxBarrel, newmeResxBarrel, te );
    leg24->Draw();

    can_meResx->cd(2);
    gPad->SetLogy();
    SetUpHistograms(meResxZmPanel1, newmeResxZmPanel1, "panel1, z<0, x residual (cm)" );
    meResxZmPanel1->Draw("he");
    newmeResxZmPanel1->Draw("samehe"); 
    myPV->PVCompute(meResxZmPanel1, newmeResxZmPanel1, te );
  
    can_meResx->cd(3);
    gPad->SetLogy();
    SetUpHistograms(meResxZmPanel2,  newmeResxZmPanel2, "panel2, z<0, x residual (cm)");
    meResxZmPanel2->Draw("he");
    newmeResxZmPanel2->Draw("samehe"); 
    myPV->PVCompute(meResxZmPanel2, newmeResxZmPanel2, te );

    can_meResx->cd(5);
    gPad->SetLogy();
    SetUpHistograms(meResxZpPanel1, newmeResxZpPanel1, "panel1, z>0, x residual (cm)" );
    meResxZpPanel1->Draw("he");
    newmeResxZpPanel1->Draw("samehe"); 
    myPV->PVCompute(meResxZpPanel1, newmeResxZpPanel1, te );

    can_meResx->cd(6);
    gPad->SetLogy();
    SetUpHistograms(meResxZpPanel2, newmeResxZpPanel2, "panel2, z>0, x residual (cm)" );
    meResxZpPanel2->Draw("he");
    newmeResxZpPanel2->Draw("samehe"); 
    myPV->PVCompute(meResxZpPanel2, newmeResxZpPanel2, te );
    
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

    rfile->GetObject("DQMData/Histograms_all/meResyBarrel"  , meResyBarrel  );
    rfile->GetObject("DQMData/Histograms_all/meResyZmPanel1", meResyZmPanel1);
    rfile->GetObject("DQMData/Histograms_all/meResyZmPanel2", meResyZmPanel2);
    rfile->GetObject("DQMData/Histograms_all/meResyZpPanel1", meResyZpPanel1);
    rfile->GetObject("DQMData/Histograms_all/meResyZpPanel2", meResyZpPanel2);

    sfile->GetObject("DQMData/Histograms_all/meResyBarrel"  , newmeResyBarrel  ); 
    sfile->GetObject("DQMData/Histograms_all/meResyZmPanel1", newmeResyZmPanel1);
    sfile->GetObject("DQMData/Histograms_all/meResyZmPanel2", newmeResyZmPanel2);
    sfile->GetObject("DQMData/Histograms_all/meResyZpPanel1", newmeResyZpPanel1);
    sfile->GetObject("DQMData/Histograms_all/meResyZpPanel2", newmeResyZpPanel2);
  
    TLegend* leg25 = new TLegend(0.3, 0.7, 0.6, 0.9);
    can_meResy->cd(1);
    gPad->SetLogy();
    SetUpHistograms(meResyBarrel, newmeResyBarrel, "barrel, y residual (cm)", leg25 );
    meResyBarrel->Draw("he");
    newmeResyBarrel->Draw("Samehe"); 
    myPV->PVCompute(meResyBarrel, newmeResyBarrel, te );
    leg25->Draw();

    can_meResy->cd(2);
    gPad->SetLogy();
    SetUpHistograms(meResyZmPanel1, newmeResyZmPanel1, "panel1, z<0, y residual (cm)" );
    meResyZmPanel1->Draw("he");
    newmeResyZmPanel1->Draw("samehe"); 
    myPV->PVCompute(meResyZmPanel1, newmeResyZmPanel1, te );
  
    can_meResy->cd(3);
    gPad->SetLogy();
    SetUpHistograms(meResyZmPanel2, newmeResyZmPanel2, "panel2, z<0, y residual (cm) " );
    meResyZmPanel2->Draw("he");
    newmeResyZmPanel2->Draw("samehe"); 
    myPV->PVCompute(meResyZmPanel2, newmeResyZmPanel2, te );

    can_meResy->cd(5);
    gPad->SetLogy();
    SetUpHistograms(meResyZpPanel1, newmeResyZpPanel1, "panel1, z>0, y residual (cm)" );
    meResyZpPanel1->Draw("he");
    newmeResyZpPanel1->Draw("samehe"); 
    myPV->PVCompute(meResyZpPanel1, newmeResyZpPanel1, te );

    can_meResy->cd(6);
    gPad->SetLogy();
    SetUpHistograms(meResyZpPanel2, newmeResyZpPanel2, "panel2, z>0, y residual (cm)" );
    meResyZpPanel2->Draw("he");
    newmeResyZpPanel2->Draw("samehe"); 
    myPV->PVCompute(meResyZpPanel2, newmeResyZpPanel2, te );

    can_meResy->SaveAs("meResy_compare.eps");
    can_meResy->SaveAs("meResy_compare.gif");
  }

 // Look at the charge distribution on each module 

 rfile->cd("DQMData/Histograms_per_ring-layer_or_disk-plaquette");
 sfile->cd("DQMData/Histograms_per_ring-layer_or_disk-plaquette");
 
 Char_t histo[200];

 TCanvas* can_meChargeRingLayer = new TCanvas("can_meChargeRingLayer", "can_meChargeRingLayer", 1200, 800);
 can_meChargeRingLayer->Divide(8,3);
 
 TH1F* meChargeLayerModule[3][8]; 
 TH1F* newmeChargeLayerModule[3][8];
 
 for (Int_t i=0; i<3; i++) // loop ovel layers
   for (Int_t j=0; j<8; j++) // loop ovel rings
     {
       sprintf(histo, "DQMData/Histograms_per_ring-layer_or_disk-plaquette/meChargeBarrelLayerModule_%d_%d", i+1, j+1);
       rfile->GetObject(histo, meChargeLayerModule[i][j]);
       sfile->GetObject(histo, newmeChargeLayerModule[i][j]); 
       SetUpHistograms(meChargeLayerModule[i][j], newmeChargeLayerModule[i][j], "barrel, charge (elec)" );
       can_meChargeRingLayer->cd(8*i + j + 1);
       //gPad->SetLogy();
    
       meChargeLayerModule[i][j]->Draw("he");
       newmeChargeLayerModule[i][j]->Draw("samehe"); 
       myPV->PVCompute(meChargeLayerModule[i][j], newmeChargeLayerModule[i][j], te );
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

 for (Int_t i=0; i<2; i++) // loop ovel disks
   for (Int_t j=0; j<4; j++) // loop ovel plaguetes
     {
       sprintf(histo, "DQMData/Histograms_per_ring-layer_or_disk-plaquette/meChargeZmPanel1DiskPlaq_%d_%d", i+1, j+1);
       rfile->GetObject(histo, meChargeZmPanel1DiskPlaq[i][j]);
       sfile->GetObject(histo, newmeChargeZmPanel1DiskPlaq[i][j]); 
       
       can_meChargeZmPanel1DiskPlaq->cd(4*i + j + 1);
       //gPad->SetLogy();
       SetUpHistograms(meChargeZmPanel1DiskPlaq[i][j], newmeChargeZmPanel1DiskPlaq[i][j], "panel1, z<0, charge (elec)" );
       meChargeZmPanel1DiskPlaq[i][j]->Draw("he");
       newmeChargeZmPanel1DiskPlaq[i][j]->Draw("samehe"); 
       myPV->PVCompute(meChargeZmPanel1DiskPlaq[i][j], newmeChargeZmPanel1DiskPlaq[i][j], te );
     }
 TLegend* leg27 = new TLegend(0.3, 0.7, 0.6, 0.9);
 leg27->SetBorderSize(0);
 leg27->AddEntry(   meChargeZmPanel1DiskPlaq[0][0], "reference  ", "l");
 leg27->AddEntry(newmeChargeZmPanel1DiskPlaq[0][0], "new release", "l");
 leg27->Draw();
 
 can_meChargeZmPanel1DiskPlaq->SaveAs("meChargeBarrelZmPanel1DiskPlaq_compare.eps");
 can_meChargeZmPanel1DiskPlaq->SaveAs("meChargeBarrelZmPanel1DiskPlaq_compare.gif");


 TCanvas* can_meChargeZmPanel2DiskPlaq = new TCanvas("can_meChargeZmPanel2DiskPlaq", "can_meChargeZmPanel2DiskPlaq", 600, 800);
 can_meChargeZmPanel2DiskPlaq->Divide(2,3);
 
 TH1F* meChargeZmPanel2DiskPlaq[2][3];
 TH1F* newmeChargeZmPanel2DiskPlaq[2][3];
 
 for (Int_t i=0; i<2; i++) // loop ovel disks
   for (Int_t j=0; j<3; j++) // loop ovel plaguetes
     {
       sprintf(histo, "DQMData/Histograms_per_ring-layer_or_disk-plaquette/meChargeZmPanel2DiskPlaq_%d_%d", i+1, j+1);
       rfile->GetObject(histo, meChargeZmPanel2DiskPlaq[i][j]);
       sfile->GetObject(histo, newmeChargeZmPanel2DiskPlaq[i][j]); 
       
       can_meChargeZmPanel2DiskPlaq->cd(3*i + j + 1);
       //gPad->SetLogy();
       SetUpHistograms(meChargeZmPanel2DiskPlaq[i][j], newmeChargeZmPanel2DiskPlaq[i][j], "panel2, z<0, charge (elec)" );
       meChargeZmPanel2DiskPlaq[i][j]->Draw("he");
       newmeChargeZmPanel2DiskPlaq[i][j]->Draw("samehe"); 
       myPV->PVCompute(meChargeZmPanel2DiskPlaq[i][j], newmeChargeZmPanel2DiskPlaq[i][j], te );
     }
 TLegend* leg28 = new TLegend(0.3, 0.7, 0.6, 0.9);
 leg28->SetBorderSize(0);
 leg28->AddEntry(   meChargeZmPanel2DiskPlaq[0][0], "reference  ", "l");
 leg28->AddEntry(newmeChargeZmPanel2DiskPlaq[0][0], "new release", "l");
 leg28->Draw();

 can_meChargeZmPanel2DiskPlaq->SaveAs("meChargeBarrelZmPanel2DiskPlaq_compare.eps");
 can_meChargeZmPanel2DiskPlaq->SaveAs("meChargeBarrelZmPanel2DiskPlaq_compare.gif");


 TCanvas* can_meChargeZpPanel1DiskPlaq = new TCanvas("can_meChargeZpPanel1DiskPlaq", "can_meChargeZpPanel1DiskPlaq", 600, 800);
 can_meChargeZpPanel1DiskPlaq->Divide(2,4);
 
 TH1F* meChargeZpPanel1DiskPlaq[2][4];
 TH1F* newmeChargeZpPanel1DiskPlaq[2][4];
 
 for (Int_t i=0; i<2; i++) // loop ovel disks
   for (Int_t j=0; j<4; j++) // loop ovel plaguetes
     {
       sprintf(histo, "DQMData/Histograms_per_ring-layer_or_disk-plaquette/meChargeZpPanel1DiskPlaq_%d_%d", i+1, j+1);
       rfile->GetObject(histo, meChargeZpPanel1DiskPlaq[i][j]);
       sfile->GetObject(histo, newmeChargeZpPanel1DiskPlaq[i][j]); 
       
       can_meChargeZpPanel1DiskPlaq->cd(4*i + j + 1);
       //gPad->SetLogy();
       SetUpHistograms(meChargeZpPanel1DiskPlaq[i][j], newmeChargeZpPanel1DiskPlaq[i][j], "panel1, z>0, charge (elec)");
       meChargeZpPanel1DiskPlaq[i][j]->Draw("he");
       newmeChargeZpPanel1DiskPlaq[i][j]->Draw("samehe"); 
       myPV->PVCompute(meChargeZpPanel1DiskPlaq[i][j], newmeChargeZpPanel1DiskPlaq[i][j], te );
     }
 TLegend* leg29 = new TLegend(0.3, 0.7, 0.6, 0.9);
 leg29->SetBorderSize(0);
 leg29->AddEntry(   meChargeZmPanel1DiskPlaq[0][0], "reference  ", "l");
 leg29->AddEntry(newmeChargeZmPanel1DiskPlaq[0][0], "new release", "l");
 leg29->Draw();
 
 can_meChargeZpPanel1DiskPlaq->SaveAs("meChargeBarrelZpPanel1DiskPlaq_compare.eps");
 can_meChargeZpPanel1DiskPlaq->SaveAs("meChargeBarrelZpPanel1DiskPlaq_compare.gif");


 TCanvas* can_meChargeZpPanel2DiskPlaq = new TCanvas("can_meChargeZpPanel2DiskPlaq", "can_meChargeZpPanel2DiskPlaq", 600, 800);
 can_meChargeZpPanel2DiskPlaq->Divide(2,3);
 
 TH1F* meChargeZpPanel2DiskPlaq[2][3];
 TH1F* newmeChargeZpPanel2DiskPlaq[2][3];
  
 for (Int_t i=0; i<2; i++) // loop ovel disks
   for (Int_t j=0; j<3; j++) // loop ovel plaguetes
     {
       sprintf(histo, "DQMData/Histograms_per_ring-layer_or_disk-plaquette/meChargeZpPanel2DiskPlaq_%d_%d", i+1, j+1);
       rfile->GetObject(histo, meChargeZpPanel2DiskPlaq[i][j]);
       sfile->GetObject(histo, newmeChargeZpPanel2DiskPlaq[i][j]); 
       
       can_meChargeZpPanel2DiskPlaq->cd(3*i + j + 1);
       //gPad->SetLogy();
       SetUpHistograms(meChargeZpPanel2DiskPlaq[i][j], newmeChargeZpPanel2DiskPlaq[i][j], "panel2, z>0, charge (elec)" );
       meChargeZpPanel2DiskPlaq[i][j]->Draw("he");
       newmeChargeZpPanel2DiskPlaq[i][j]->Draw("samehe"); 
       myPV->PVCompute(meChargeZpPanel2DiskPlaq[i][j], newmeChargeZpPanel2DiskPlaq[i][j], te );
     }
 TLegend* leg30 = new TLegend(0.3, 0.7, 0.6, 0.9);
 leg30->SetBorderSize(0);
 leg30->AddEntry(   meChargeZmPanel2DiskPlaq[0][0], "reference  ", "l");
 leg30->AddEntry(newmeChargeZmPanel2DiskPlaq[0][0], "new release", "l");
 leg30->Draw();

}
