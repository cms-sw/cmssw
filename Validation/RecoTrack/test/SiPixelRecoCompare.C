void SiPixelRecoCompare()
{
 gROOT ->Reset();

 char* rfilename = "pixeltrackingrechitshist.root";
 char* sfilename = "../data/pixeltrackingrechitshist.root";

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
  
    can_meCharge->cd(1);
    //gPad->SetLogy();
    meChargeBarrel->SetLineColor(2);
    meChargeBarrel->Draw();
    newmeChargeBarrel->SetLineColor(4);
    newmeChargeBarrel->SetLineStyle(2);
    newmeChargeBarrel->Draw("Sames"); 
    myPV->PVCompute(meChargeBarrel, newmeChargeBarrel, te );
  
    can_meCharge->cd(2);
    //gPad->SetLogy();
    meChargeZmPanel1->SetLineColor(2);
    meChargeZmPanel1->Draw();
    newmeChargeZmPanel1->SetLineColor(4);
    newmeChargeZmPanel1->SetLineStyle(2);
    newmeChargeZmPanel1->Draw("same"); 
    myPV->PVCompute(meChargeZmPanel1, newmeChargeZmPanel1, te );
  
    can_meCharge->cd(3);
    //gPad->SetLogy();
    meChargeZmPanel2->SetLineColor(2);
    meChargeZmPanel2->Draw();
    newmeChargeZmPanel2->SetLineColor(4);
    newmeChargeZmPanel2->SetLineStyle(2);
    newmeChargeZmPanel2->Draw("same"); 
    myPV->PVCompute(meChargeZmPanel2, newmeChargeZmPanel2, te );

    can_meCharge->cd(5);
    //gPad->SetLogy();
    meChargeZpPanel1->SetLineColor(2);
    meChargeZpPanel1->Draw();
    newmeChargeZpPanel1->SetLineColor(4);
    newmeChargeZpPanel1->SetLineStyle(2);
    newmeChargeZpPanel1->Draw("same"); 
    myPV->PVCompute(meChargeZpPanel1, newmeChargeZpPanel1, te );

    can_meCharge->cd(6);
    //gPad->SetLogy();
    meChargeZpPanel2->SetLineColor(2);
    meChargeZpPanel2->Draw();
    newmeChargeZpPanel2->SetLineColor(4);
    newmeChargeZpPanel2->SetLineStyle(2);
    newmeChargeZpPanel2->Draw("same"); 
    myPV->PVCompute(meChargeZpPanel2, newmeChargeZpPanel2, te );

    //can_meCharge->SaveAs("meCharge_compare.eps");
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
  
    can_Errx->cd(1);
    //gPad->SetLogy();
    meErrxBarrel->SetLineColor(2);
    meErrxBarrel->Draw();
    newmeErrxBarrel->SetLineColor(4);
    newmeErrxBarrel->SetLineStyle(2);
    newmeErrxBarrel->Draw("Sames"); 
    myPV->PVCompute(meErrxBarrel, newmeErrxBarrel, te );
  
    can_Errx->cd(2);
    //gPad->SetLogy();
    meErrxZmPanel1->SetLineColor(2);
    meErrxZmPanel1->Draw();
    newmeErrxZmPanel1->SetLineColor(4);
    newmeErrxZmPanel1->SetLineStyle(2);
    newmeErrxZmPanel1->Draw("same"); 
    myPV->PVCompute(meErrxZmPanel1, newmeErrxZmPanel1, te );
  
    can_Errx->cd(3);
    //gPad->SetLogy();
    meErrxZmPanel2->SetLineColor(2);
    meErrxZmPanel2->Draw();
    newmeErrxZmPanel2->SetLineColor(4);
    newmeErrxZmPanel2->SetLineStyle(2);
    newmeErrxZmPanel2->Draw("same"); 
    myPV->PVCompute(meErrxZmPanel2, newmeErrxZmPanel2, te );

    can_Errx->cd(5);
    //gPad->SetLogy();
    meErrxZpPanel1->SetLineColor(2);
    meErrxZpPanel1->Draw();
    newmeErrxZpPanel1->SetLineColor(4);
    newmeErrxZpPanel1->SetLineStyle(2);
    newmeErrxZpPanel1->Draw("same"); 
    myPV->PVCompute(meErrxZpPanel1, newmeErrxZpPanel1, te );

    can_Errx->cd(6);
    //gPad->SetLogy();
    meErrxZpPanel2->SetLineColor(2);
    meErrxZpPanel2->Draw();
    newmeErrxZpPanel2->SetLineColor(4);
    newmeErrxZpPanel2->SetLineStyle(2);
    newmeErrxZpPanel2->Draw("same"); 
    myPV->PVCompute(meErrxZpPanel2, newmeErrxZpPanel2, te );

    //can_Errx->SaveAs("meErrx_compare.eps");
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
  
    can_Erry->cd(1);
    //gPad->SetLogy();
    meErryBarrel->SetLineColor(2);
    meErryBarrel->Draw();
    newmeErryBarrel->SetLineColor(4);
    newmeErryBarrel->SetLineStyle(2);
    newmeErryBarrel->Draw("Sames"); 
    myPV->PVCompute(meErryBarrel, newmeErryBarrel, te );
  
    can_Erry->cd(2);
    //gPad->SetLogy();
    meErryZmPanel1->SetLineColor(2);
    meErryZmPanel1->Draw();
    newmeErryZmPanel1->SetLineColor(4);
    newmeErryZmPanel1->SetLineStyle(2);
    newmeErryZmPanel1->Draw("same"); 
    myPV->PVCompute(meErryZmPanel1, newmeErryZmPanel1, te );
  
    can_Erry->cd(3);
    //gPad->SetLogy();
    meErryZmPanel2->SetLineColor(2);
    meErryZmPanel2->Draw();
    newmeErryZmPanel2->SetLineColor(4);
    newmeErryZmPanel2->SetLineStyle(2);
    newmeErryZmPanel2->Draw("same"); 
    myPV->PVCompute(meErryZmPanel2, newmeErryZmPanel2, te );

    can_Erry->cd(5);
    //gPad->SetLogy();
    meErryZpPanel1->SetLineColor(2);
    meErryZpPanel1->Draw();
    newmeErryZpPanel1->SetLineColor(4);
    newmeErryZpPanel1->SetLineStyle(2);
    newmeErryZpPanel1->Draw("same"); 
    myPV->PVCompute(meErryZpPanel1, newmeErryZpPanel1, te );

    can_Erry->cd(6);
    //gPad->SetLogy();
    meErryZpPanel2->SetLineColor(2);
    meErryZpPanel2->Draw();
    newmeErryZpPanel2->SetLineColor(4);
    newmeErryZpPanel2->SetLineStyle(2);
    newmeErryZpPanel2->Draw("same"); 
    myPV->PVCompute(meErryZpPanel2, newmeErryZpPanel2, te );

    //can_Erry->SaveAs("meErry_compare.eps");
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
  
    can_Npix->cd(1);
    //gPad->SetLogy();
    meNpixBarrel->SetLineColor(2);
    meNpixBarrel->Draw();
    newmeNpixBarrel->SetLineColor(4);
    newmeNpixBarrel->SetLineStyle(2);
    newmeNpixBarrel->Draw("Sames"); 
    myPV->PVCompute(meNpixBarrel, newmeNpixBarrel, te );
  
    can_Npix->cd(2);
    //gPad->SetLogy();
    meNpixZmPanel1->SetLineColor(2);
    meNpixZmPanel1->Draw();
    newmeNpixZmPanel1->SetLineColor(4);
    newmeNpixZmPanel1->SetLineStyle(2);
    newmeNpixZmPanel1->Draw("same"); 
    myPV->PVCompute(meNpixZmPanel1, newmeNpixZmPanel1, te );
  
    can_Npix->cd(3);
    //gPad->SetLogy();
    meNpixZmPanel2->SetLineColor(2);
    meNpixZmPanel2->Draw();
    newmeNpixZmPanel2->SetLineColor(4);
    newmeNpixZmPanel2->SetLineStyle(2);
    newmeNpixZmPanel2->Draw("same"); 
    myPV->PVCompute(meNpixZmPanel2, newmeNpixZmPanel2, te );

    can_Npix->cd(5);
    //gPad->SetLogy();
    meNpixZpPanel1->SetLineColor(2);
    meNpixZpPanel1->Draw();
    newmeNpixZpPanel1->SetLineColor(4);
    newmeNpixZpPanel1->SetLineStyle(2);
    newmeNpixZpPanel1->Draw("same"); 
    myPV->PVCompute(meNpixZpPanel1, newmeNpixZpPanel1, te );

    can_Npix->cd(6);
    //gPad->SetLogy();
    meNpixZpPanel2->SetLineColor(2);
    meNpixZpPanel2->Draw();
    newmeNpixZpPanel2->SetLineColor(4);
    newmeNpixZpPanel2->SetLineStyle(2);
    newmeNpixZpPanel2->Draw("same"); 
    myPV->PVCompute(meNpixZpPanel2, newmeNpixZpPanel2, te );

    //can_Npix->SaveAs("meNpix_compare.eps");
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
  
    can_Nxpix->cd(1);
    //gPad->SetLogy();
    meNxpixBarrel->SetLineColor(2);
    meNxpixBarrel->Draw();
    newmeNxpixBarrel->SetLineColor(4);
    newmeNxpixBarrel->SetLineStyle(2);
    newmeNxpixBarrel->Draw("Sames"); 
    myPV->PVCompute(meNxpixBarrel, newmeNxpixBarrel, te );
  
    can_Nxpix->cd(2);
    //gPad->SetLogy();
    meNxpixZmPanel1->SetLineColor(2);
    meNxpixZmPanel1->Draw();
    newmeNxpixZmPanel1->SetLineColor(4);
    newmeNxpixZmPanel1->SetLineStyle(2);
    newmeNxpixZmPanel1->Draw("same"); 
    myPV->PVCompute(meNxpixZmPanel1, newmeNxpixZmPanel1, te );
  
    can_Nxpix->cd(3);
    //gPad->SetLogy();
    meNxpixZmPanel2->SetLineColor(2);
    meNxpixZmPanel2->Draw();
    newmeNxpixZmPanel2->SetLineColor(4);
    newmeNxpixZmPanel2->SetLineStyle(2);
    newmeNxpixZmPanel2->Draw("same"); 
    myPV->PVCompute(meNxpixZmPanel2, newmeNxpixZmPanel2, te );

    can_Nxpix->cd(5);
    //gPad->SetLogy();
    meNxpixZpPanel1->SetLineColor(2);
    meNxpixZpPanel1->Draw();
    newmeNxpixZpPanel1->SetLineColor(4);
    newmeNxpixZpPanel1->SetLineStyle(2);
    newmeNxpixZpPanel1->Draw("same"); 
    myPV->PVCompute(meNxpixZpPanel1, newmeNxpixZpPanel1, te );

    can_Nxpix->cd(6);
    //gPad->SetLogy();
    meNxpixZpPanel2->SetLineColor(2);
    meNxpixZpPanel2->Draw();
    newmeNxpixZpPanel2->SetLineColor(4);
    newmeNxpixZpPanel2->SetLineStyle(2);
    newmeNxpixZpPanel2->Draw("same"); 
    myPV->PVCompute(meNxpixZpPanel2, newmeNxpixZpPanel2, te );

    //can_Nxpix->SaveAs("meNxpix_compare.eps");
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
  
    can_Nypix->cd(1);
    //gPad->SetLogy();
    meNypixBarrel->SetLineColor(2);
    meNypixBarrel->Draw();
    newmeNypixBarrel->SetLineColor(4);
    newmeNypixBarrel->SetLineStyle(2);
    newmeNypixBarrel->Draw("Sames"); 
    myPV->PVCompute(meNypixBarrel, newmeNypixBarrel, te );
  
    can_Nypix->cd(2);
    //gPad->SetLogy();
    meNypixZmPanel1->SetLineColor(2);
    meNypixZmPanel1->Draw();
    newmeNypixZmPanel1->SetLineColor(4);
    newmeNypixZmPanel1->SetLineStyle(2);
    newmeNypixZmPanel1->Draw("same"); 
    myPV->PVCompute(meNypixZmPanel1, newmeNypixZmPanel1, te );
  
    can_Nypix->cd(3);
    //gPad->SetLogy();
    meNypixZmPanel2->SetLineColor(2);
    meNypixZmPanel2->Draw();
    newmeNypixZmPanel2->SetLineColor(4);
    newmeNypixZmPanel2->SetLineStyle(2);
    newmeNypixZmPanel2->Draw("same"); 
    myPV->PVCompute(meNypixZmPanel2, newmeNypixZmPanel2, te );

    can_Nypix->cd(5);
    //gPad->SetLogy();
    meNypixZpPanel1->SetLineColor(2);
    meNypixZpPanel1->Draw();
    newmeNypixZpPanel1->SetLineColor(4);
    newmeNypixZpPanel1->SetLineStyle(2);
    newmeNypixZpPanel1->Draw("same"); 
    myPV->PVCompute(meNypixZpPanel1, newmeNypixZpPanel1, te );

    can_Nypix->cd(6);
    //gPad->SetLogy();
    meNypixZpPanel2->SetLineColor(2);
    meNypixZpPanel2->Draw();
    newmeNypixZpPanel2->SetLineColor(4);
    newmeNypixZpPanel2->SetLineStyle(2);
    newmeNypixZpPanel2->Draw("same"); 
    myPV->PVCompute(meNypixZpPanel2, newmeNypixZpPanel2, te );

    //can_Nypix->SaveAs("meNypix_compare.eps");
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
  
    can_Posx->cd(1);
    //gPad->SetLogy();
    mePosxBarrel->SetLineColor(2);
    mePosxBarrel->Draw();
    newmePosxBarrel->SetLineColor(4);
    newmePosxBarrel->SetLineStyle(2);
    newmePosxBarrel->Draw("Sames"); 
    myPV->PVCompute(mePosxBarrel, newmePosxBarrel, te );
  
    can_Posx->cd(2);
    //gPad->SetLogy();
    mePosxZmPanel1->SetLineColor(2);
    mePosxZmPanel1->Draw();
    newmePosxZmPanel1->SetLineColor(4);
    newmePosxZmPanel1->SetLineStyle(2);
    newmePosxZmPanel1->Draw("same"); 
    myPV->PVCompute(mePosxZmPanel1, newmePosxZmPanel1, te );
  
    can_Posx->cd(3);
    //gPad->SetLogy();
    mePosxZmPanel2->SetLineColor(2);
    mePosxZmPanel2->Draw();
    newmePosxZmPanel2->SetLineColor(4);
    newmePosxZmPanel2->SetLineStyle(2);
    newmePosxZmPanel2->Draw("same"); 
    myPV->PVCompute(mePosxZmPanel2, newmePosxZmPanel2, te );

    can_Posx->cd(5);
    //gPad->SetLogy();
    mePosxZpPanel1->SetLineColor(2);
    mePosxZpPanel1->Draw();
    newmePosxZpPanel1->SetLineColor(4);
    newmePosxZpPanel1->SetLineStyle(2);
    newmePosxZpPanel1->Draw("same"); 
    myPV->PVCompute(mePosxZpPanel1, newmePosxZpPanel1, te );

    can_Posx->cd(6);
    //gPad->SetLogy();
    mePosxZpPanel2->SetLineColor(2);
    mePosxZpPanel2->Draw();
    newmePosxZpPanel2->SetLineColor(4);
    newmePosxZpPanel2->SetLineStyle(2);
    newmePosxZpPanel2->Draw("same"); 
    myPV->PVCompute(mePosxZpPanel2, newmePosxZpPanel2, te );

    //can_Posx->SaveAs("mePosx_compare.eps");
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
  
    can_Posy->cd(1);
    //gPad->SetLogy();
    mePosyBarrel->SetLineColor(2);
    mePosyBarrel->Draw();
    newmePosyBarrel->SetLineColor(4);
    newmePosyBarrel->SetLineStyle(2);
    newmePosyBarrel->Draw("Sames"); 
    myPV->PVCompute(mePosyBarrel, newmePosyBarrel, te );
  
    can_Posy->cd(2);
    //gPad->SetLogy();
    mePosyZmPanel1->SetLineColor(2);
    mePosyZmPanel1->Draw();
    newmePosyZmPanel1->SetLineColor(4);
    newmePosyZmPanel1->SetLineStyle(2);
    newmePosyZmPanel1->Draw("same"); 
    myPV->PVCompute(mePosyZmPanel1, newmePosyZmPanel1, te );
  
    can_Posy->cd(3);
    //gPad->SetLogy();
    mePosyZmPanel2->SetLineColor(2);
    mePosyZmPanel2->Draw();
    newmePosyZmPanel2->SetLineColor(4);
    newmePosyZmPanel2->SetLineStyle(2);
    newmePosyZmPanel2->Draw("same"); 
    myPV->PVCompute(mePosyZmPanel2, newmePosyZmPanel2, te );

    can_Posy->cd(5);
    //gPad->SetLogy();
    mePosyZpPanel1->SetLineColor(2);
    mePosyZpPanel1->Draw();
    newmePosyZpPanel1->SetLineColor(4);
    newmePosyZpPanel1->SetLineStyle(2);
    newmePosyZpPanel1->Draw("same"); 
    myPV->PVCompute(mePosyZpPanel1, newmePosyZpPanel1, te );

    can_Posy->cd(6);
    //gPad->SetLogy();
    mePosyZpPanel2->SetLineColor(2);
    mePosyZpPanel2->Draw();
    newmePosyZpPanel2->SetLineColor(4);
    newmePosyZpPanel2->SetLineStyle(2);
    newmePosyZpPanel2->Draw("same"); 
    myPV->PVCompute(mePosyZpPanel2, newmePosyZpPanel2, te );

    //can_Posy->SaveAs("mePosy_compare.eps");
  }

if (1) 
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
  
    can_PullXvsAlpha->cd(1);
    //gPad->SetLogy();
    mePullXvsAlphaBarrel->SetLineColor(2);
    mePullXvsAlphaBarrel->Draw();
    newmePullXvsAlphaBarrel->SetLineColor(4);
    newmePullXvsAlphaBarrel->SetLineStyle(2);
    newmePullXvsAlphaBarrel->Draw("Sames"); 
    myPV->PVCompute(mePullXvsAlphaBarrel, newmePullXvsAlphaBarrel, te );
  
    can_PullXvsAlpha->cd(2);
    //gPad->SetLogy();
    mePullXvsAlphaZmPanel1->SetLineColor(2);
    mePullXvsAlphaZmPanel1->Draw();
    newmePullXvsAlphaZmPanel1->SetLineColor(4);
    newmePullXvsAlphaZmPanel1->SetLineStyle(2);
    newmePullXvsAlphaZmPanel1->Draw("same"); 
    myPV->PVCompute(mePullXvsAlphaZmPanel1, newmePullXvsAlphaZmPanel1, te );
  
    can_PullXvsAlpha->cd(3);
    //gPad->SetLogy();
    mePullXvsAlphaZmPanel2->SetLineColor(2);
    mePullXvsAlphaZmPanel2->Draw();
    newmePullXvsAlphaZmPanel2->SetLineColor(4);
    newmePullXvsAlphaZmPanel2->SetLineStyle(2);
    newmePullXvsAlphaZmPanel2->Draw("same"); 
    myPV->PVCompute(mePullXvsAlphaZmPanel2, newmePullXvsAlphaZmPanel2, te );

    can_PullXvsAlpha->cd(5);
    //gPad->SetLogy();
    mePullXvsAlphaZpPanel1->SetLineColor(2);
    mePullXvsAlphaZpPanel1->Draw();
    newmePullXvsAlphaZpPanel1->SetLineColor(4);
    newmePullXvsAlphaZpPanel1->SetLineStyle(2);
    newmePullXvsAlphaZpPanel1->Draw("same"); 
    myPV->PVCompute(mePullXvsAlphaZpPanel1, newmePullXvsAlphaZpPanel1, te );

    can_PullXvsAlpha->cd(6);
    //gPad->SetLogy();
    mePullXvsAlphaZpPanel2->SetLineColor(2);
    mePullXvsAlphaZpPanel2->Draw();
    newmePullXvsAlphaZpPanel2->SetLineColor(4);
    newmePullXvsAlphaZpPanel2->SetLineStyle(2);
    newmePullXvsAlphaZpPanel2->Draw("same"); 
    myPV->PVCompute(mePullXvsAlphaZpPanel2, newmePullXvsAlphaZpPanel2, te );

    //can_PullXvsAlpha->SaveAs("mePullXvsAlpha_compare.eps");
  }

if (1) 
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
  
    can_PullXvsBeta->cd(1);
    //gPad->SetLogy();
    mePullXvsBetaBarrel->SetLineColor(2);
    mePullXvsBetaBarrel->Draw();
    newmePullXvsBetaBarrel->SetLineColor(4);
    newmePullXvsBetaBarrel->SetLineStyle(2);
    newmePullXvsBetaBarrel->Draw("Sames"); 
    myPV->PVCompute(mePullXvsBetaBarrel, newmePullXvsBetaBarrel, te );
  
    can_PullXvsBeta->cd(2);
    //gPad->SetLogy();
    mePullXvsBetaZmPanel1->SetLineColor(2);
    mePullXvsBetaZmPanel1->Draw();
    newmePullXvsBetaZmPanel1->SetLineColor(4);
    newmePullXvsBetaZmPanel1->SetLineStyle(2);
    newmePullXvsBetaZmPanel1->Draw("same"); 
    myPV->PVCompute(mePullXvsBetaZmPanel1, newmePullXvsBetaZmPanel1, te );
  
    can_PullXvsBeta->cd(3);
    //gPad->SetLogy();
    mePullXvsBetaZmPanel2->SetLineColor(2);
    mePullXvsBetaZmPanel2->Draw();
    newmePullXvsBetaZmPanel2->SetLineColor(4);
    newmePullXvsBetaZmPanel2->SetLineStyle(2);
    newmePullXvsBetaZmPanel2->Draw("same"); 
    myPV->PVCompute(mePullXvsBetaZmPanel2, newmePullXvsBetaZmPanel2, te );

    can_PullXvsBeta->cd(5);
    //gPad->SetLogy();
    mePullXvsBetaZpPanel1->SetLineColor(2);
    mePullXvsBetaZpPanel1->Draw();
    newmePullXvsBetaZpPanel1->SetLineColor(4);
    newmePullXvsBetaZpPanel1->SetLineStyle(2);
    newmePullXvsBetaZpPanel1->Draw("same"); 
    myPV->PVCompute(mePullXvsBetaZpPanel1, newmePullXvsBetaZpPanel1, te );

    can_PullXvsBeta->cd(6);
    //gPad->SetLogy();
    mePullXvsBetaZpPanel2->SetLineColor(2);
    mePullXvsBetaZpPanel2->Draw();
    newmePullXvsBetaZpPanel2->SetLineColor(4);
    newmePullXvsBetaZpPanel2->SetLineStyle(2);
    newmePullXvsBetaZpPanel2->Draw("same"); 
    myPV->PVCompute(mePullXvsBetaZpPanel2, newmePullXvsBetaZpPanel2, te );

    //can_PullXvsBeta->SaveAs("mePullXvsBeta_compare.eps");
  }

if (1) 
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
  
    can_PullXvsEta->cd(1);
    //gPad->SetLogy();
    mePullXvsEtaBarrel->SetLineColor(2);
    mePullXvsEtaBarrel->Draw();
    newmePullXvsEtaBarrel->SetLineColor(4);
    newmePullXvsEtaBarrel->SetLineStyle(2);
    newmePullXvsEtaBarrel->Draw("Sames"); 
    myPV->PVCompute(mePullXvsEtaBarrel, newmePullXvsEtaBarrel, te );
  
    can_PullXvsEta->cd(2);
    //gPad->SetLogy();
    mePullXvsEtaZmPanel1->SetLineColor(2);
    mePullXvsEtaZmPanel1->Draw();
    newmePullXvsEtaZmPanel1->SetLineColor(4);
    newmePullXvsEtaZmPanel1->SetLineStyle(2);
    newmePullXvsEtaZmPanel1->Draw("same"); 
    myPV->PVCompute(mePullXvsEtaZmPanel1, newmePullXvsEtaZmPanel1, te );
  
    can_PullXvsEta->cd(3);
    //gPad->SetLogy();
    mePullXvsEtaZmPanel2->SetLineColor(2);
    mePullXvsEtaZmPanel2->Draw();
    newmePullXvsEtaZmPanel2->SetLineColor(4);
    newmePullXvsEtaZmPanel2->SetLineStyle(2);
    newmePullXvsEtaZmPanel2->Draw("same"); 
    myPV->PVCompute(mePullXvsEtaZmPanel2, newmePullXvsEtaZmPanel2, te );

    can_PullXvsEta->cd(5);
    //gPad->SetLogy();
    mePullXvsEtaZpPanel1->SetLineColor(2);
    mePullXvsEtaZpPanel1->Draw();
    newmePullXvsEtaZpPanel1->SetLineColor(4);
    newmePullXvsEtaZpPanel1->SetLineStyle(2);
    newmePullXvsEtaZpPanel1->Draw("same"); 
    myPV->PVCompute(mePullXvsEtaZpPanel1, newmePullXvsEtaZpPanel1, te );

    can_PullXvsEta->cd(6);
    //gPad->SetLogy();
    mePullXvsEtaZpPanel2->SetLineColor(2);
    mePullXvsEtaZpPanel2->Draw();
    newmePullXvsEtaZpPanel2->SetLineColor(4);
    newmePullXvsEtaZpPanel2->SetLineStyle(2);
    newmePullXvsEtaZpPanel2->Draw("same"); 
    myPV->PVCompute(mePullXvsEtaZpPanel2, newmePullXvsEtaZpPanel2, te );

    //can_PullXvsEta->SaveAs("mePullXvsEta_compare.eps");
  }

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
  
    can_PullXvsPhi->cd(1);
    //gPad->SetLogy();
    mePullXvsPhiBarrel->SetLineColor(2);
    mePullXvsPhiBarrel->Draw();
    newmePullXvsPhiBarrel->SetLineColor(4);
    newmePullXvsPhiBarrel->SetLineStyle(2);
    newmePullXvsPhiBarrel->Draw("Sames"); 
    myPV->PVCompute(mePullXvsPhiBarrel, newmePullXvsPhiBarrel, te );
  
    can_PullXvsPhi->cd(2);
    //gPad->SetLogy();
    mePullXvsPhiZmPanel1->SetLineColor(2);
    mePullXvsPhiZmPanel1->Draw();
    newmePullXvsPhiZmPanel1->SetLineColor(4);
    newmePullXvsPhiZmPanel1->SetLineStyle(2);
    newmePullXvsPhiZmPanel1->Draw("same"); 
    myPV->PVCompute(mePullXvsPhiZmPanel1, newmePullXvsPhiZmPanel1, te );
  
    can_PullXvsPhi->cd(3);
    //gPad->SetLogy();
    mePullXvsPhiZmPanel2->SetLineColor(2);
    mePullXvsPhiZmPanel2->Draw();
    newmePullXvsPhiZmPanel2->SetLineColor(4);
    newmePullXvsPhiZmPanel2->SetLineStyle(2);
    newmePullXvsPhiZmPanel2->Draw("same"); 
    myPV->PVCompute(mePullXvsPhiZmPanel2, newmePullXvsPhiZmPanel2, te );

    can_PullXvsPhi->cd(5);
    //gPad->SetLogy();
    mePullXvsPhiZpPanel1->SetLineColor(2);
    mePullXvsPhiZpPanel1->Draw();
    newmePullXvsPhiZpPanel1->SetLineColor(4);
    newmePullXvsPhiZpPanel1->SetLineStyle(2);
    newmePullXvsPhiZpPanel1->Draw("same"); 
    myPV->PVCompute(mePullXvsPhiZpPanel1, newmePullXvsPhiZpPanel1, te );

    can_PullXvsPhi->cd(6);
    //gPad->SetLogy();
    mePullXvsPhiZpPanel2->SetLineColor(2);
    mePullXvsPhiZpPanel2->Draw();
    newmePullXvsPhiZpPanel2->SetLineColor(4);
    newmePullXvsPhiZpPanel2->SetLineStyle(2);
    newmePullXvsPhiZpPanel2->Draw("same"); 
    myPV->PVCompute(mePullXvsPhiZpPanel2, newmePullXvsPhiZpPanel2, te );

    //can_PullXvsPhi->SaveAs("mePullXvsPhi_compare.eps");
  }

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
  
    can_PullYvsAlpha->cd(1);
    //gPad->SetLogy();
    mePullYvsAlphaBarrel->SetLineColor(2);
    mePullYvsAlphaBarrel->Draw();
    newmePullYvsAlphaBarrel->SetLineColor(4);
    newmePullYvsAlphaBarrel->SetLineStyle(2);
    newmePullYvsAlphaBarrel->Draw("Sames"); 
    myPV->PVCompute(mePullYvsAlphaBarrel, newmePullYvsAlphaBarrel, te );
  
    can_PullYvsAlpha->cd(2);
    //gPad->SetLogy();
    mePullYvsAlphaZmPanel1->SetLineColor(2);
    mePullYvsAlphaZmPanel1->Draw();
    newmePullYvsAlphaZmPanel1->SetLineColor(4);
    newmePullYvsAlphaZmPanel1->SetLineStyle(2);
    newmePullYvsAlphaZmPanel1->Draw("same"); 
    myPV->PVCompute(mePullYvsAlphaZmPanel1, newmePullYvsAlphaZmPanel1, te );
  
    can_PullYvsAlpha->cd(3);
    //gPad->SetLogy();
    mePullYvsAlphaZmPanel2->SetLineColor(2);
    mePullYvsAlphaZmPanel2->Draw();
    newmePullYvsAlphaZmPanel2->SetLineColor(4);
    newmePullYvsAlphaZmPanel2->SetLineStyle(2);
    newmePullYvsAlphaZmPanel2->Draw("same"); 
    myPV->PVCompute(mePullYvsAlphaZmPanel2, newmePullYvsAlphaZmPanel2, te );

    can_PullYvsAlpha->cd(5);
    //gPad->SetLogy();
    mePullYvsAlphaZpPanel1->SetLineColor(2);
    mePullYvsAlphaZpPanel1->Draw();
    newmePullYvsAlphaZpPanel1->SetLineColor(4);
    newmePullYvsAlphaZpPanel1->SetLineStyle(2);
    newmePullYvsAlphaZpPanel1->Draw("same"); 
    myPV->PVCompute(mePullYvsAlphaZpPanel1, newmePullYvsAlphaZpPanel1, te );

    can_PullYvsAlpha->cd(6);
    //gPad->SetLogy();
    mePullYvsAlphaZpPanel2->SetLineColor(2);
    mePullYvsAlphaZpPanel2->Draw();
    newmePullYvsAlphaZpPanel2->SetLineColor(4);
    newmePullYvsAlphaZpPanel2->SetLineStyle(2);
    newmePullYvsAlphaZpPanel2->Draw("same"); 
    myPV->PVCompute(mePullYvsAlphaZpPanel2, newmePullYvsAlphaZpPanel2, te );

    //can_PullYvsAlpha->SaveAs("mePullYvsAlpha_compare.eps");
  }

if (1) 
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
  
    can_PullYvsBeta->cd(1);
    //gPad->SetLogy();
    mePullYvsBetaBarrel->SetLineColor(2);
    mePullYvsBetaBarrel->Draw();
    newmePullYvsBetaBarrel->SetLineColor(4);
    newmePullYvsBetaBarrel->SetLineStyle(2);
    newmePullYvsBetaBarrel->Draw("Sames"); 
    myPV->PVCompute(mePullYvsBetaBarrel, newmePullYvsBetaBarrel, te );
  
    can_PullYvsBeta->cd(2);
    //gPad->SetLogy();
    mePullYvsBetaZmPanel1->SetLineColor(2);
    mePullYvsBetaZmPanel1->Draw();
    newmePullYvsBetaZmPanel1->SetLineColor(4);
    newmePullYvsBetaZmPanel1->SetLineStyle(2);
    newmePullYvsBetaZmPanel1->Draw("same"); 
    myPV->PVCompute(mePullYvsBetaZmPanel1, newmePullYvsBetaZmPanel1, te );
  
    can_PullYvsBeta->cd(3);
    //gPad->SetLogy();
    mePullYvsBetaZmPanel2->SetLineColor(2);
    mePullYvsBetaZmPanel2->Draw();
    newmePullYvsBetaZmPanel2->SetLineColor(4);
    newmePullYvsBetaZmPanel2->SetLineStyle(2);
    newmePullYvsBetaZmPanel2->Draw("same"); 
    myPV->PVCompute(mePullYvsBetaZmPanel2, newmePullYvsBetaZmPanel2, te );

    can_PullYvsBeta->cd(5);
    //gPad->SetLogy();
    mePullYvsBetaZpPanel1->SetLineColor(2);
    mePullYvsBetaZpPanel1->Draw();
    newmePullYvsBetaZpPanel1->SetLineColor(4);
    newmePullYvsBetaZpPanel1->SetLineStyle(2);
    newmePullYvsBetaZpPanel1->Draw("same"); 
    myPV->PVCompute(mePullYvsBetaZpPanel1, newmePullYvsBetaZpPanel1, te );

    can_PullYvsBeta->cd(6);
    //gPad->SetLogy();
    mePullYvsBetaZpPanel2->SetLineColor(2);
    mePullYvsBetaZpPanel2->Draw();
    newmePullYvsBetaZpPanel2->SetLineColor(4);
    newmePullYvsBetaZpPanel2->SetLineStyle(2);
    newmePullYvsBetaZpPanel2->Draw("same"); 
    myPV->PVCompute(mePullYvsBetaZpPanel2, newmePullYvsBetaZpPanel2, te );

    //can_PullYvsBeta->SaveAs("mePullYvsBeta_compare.eps");
  }

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
  
    can_PullYvsEta->cd(1);
    //gPad->SetLogy();
    mePullYvsEtaBarrel->SetLineColor(2);
    mePullYvsEtaBarrel->Draw();
    newmePullYvsEtaBarrel->SetLineColor(4);
    newmePullYvsEtaBarrel->SetLineStyle(2);
    newmePullYvsEtaBarrel->Draw("Sames"); 
    myPV->PVCompute(mePullYvsEtaBarrel, newmePullYvsEtaBarrel, te );
  
    can_PullYvsEta->cd(2);
    //gPad->SetLogy();
    mePullYvsEtaZmPanel1->SetLineColor(2);
    mePullYvsEtaZmPanel1->Draw();
    newmePullYvsEtaZmPanel1->SetLineColor(4);
    newmePullYvsEtaZmPanel1->SetLineStyle(2);
    newmePullYvsEtaZmPanel1->Draw("same"); 
    myPV->PVCompute(mePullYvsEtaZmPanel1, newmePullYvsEtaZmPanel1, te );
  
    can_PullYvsEta->cd(3);
    //gPad->SetLogy();
    mePullYvsEtaZmPanel2->SetLineColor(2);
    mePullYvsEtaZmPanel2->Draw();
    newmePullYvsEtaZmPanel2->SetLineColor(4);
    newmePullYvsEtaZmPanel2->SetLineStyle(2);
    newmePullYvsEtaZmPanel2->Draw("same"); 
    myPV->PVCompute(mePullYvsEtaZmPanel2, newmePullYvsEtaZmPanel2, te );

    can_PullYvsEta->cd(5);
    //gPad->SetLogy();
    mePullYvsEtaZpPanel1->SetLineColor(2);
    mePullYvsEtaZpPanel1->Draw();
    newmePullYvsEtaZpPanel1->SetLineColor(4);
    newmePullYvsEtaZpPanel1->SetLineStyle(2);
    newmePullYvsEtaZpPanel1->Draw("same"); 
    myPV->PVCompute(mePullYvsEtaZpPanel1, newmePullYvsEtaZpPanel1, te );

    can_PullYvsEta->cd(6);
    //gPad->SetLogy();
    mePullYvsEtaZpPanel2->SetLineColor(2);
    mePullYvsEtaZpPanel2->Draw();
    newmePullYvsEtaZpPanel2->SetLineColor(4);
    newmePullYvsEtaZpPanel2->SetLineStyle(2);
    newmePullYvsEtaZpPanel2->Draw("same"); 
    myPV->PVCompute(mePullYvsEtaZpPanel2, newmePullYvsEtaZpPanel2, te );

    //can_PullYvsEta->SaveAs("mePullYvsEta_compare.eps");
  }

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
  
    can_PullYvsPhi->cd(1);
    //gPad->SetLogy();
    mePullYvsPhiBarrel->SetLineColor(2);
    mePullYvsPhiBarrel->Draw();
    newmePullYvsPhiBarrel->SetLineColor(4);
    newmePullYvsPhiBarrel->SetLineStyle(2);
    newmePullYvsPhiBarrel->Draw("Sames"); 
    myPV->PVCompute(mePullYvsPhiBarrel, newmePullYvsPhiBarrel, te );
  
    can_PullYvsPhi->cd(2);
    //gPad->SetLogy();
    mePullYvsPhiZmPanel1->SetLineColor(2);
    mePullYvsPhiZmPanel1->Draw();
    newmePullYvsPhiZmPanel1->SetLineColor(4);
    newmePullYvsPhiZmPanel1->SetLineStyle(2);
    newmePullYvsPhiZmPanel1->Draw("same"); 
    myPV->PVCompute(mePullYvsPhiZmPanel1, newmePullYvsPhiZmPanel1, te );
  
    can_PullYvsPhi->cd(3);
    //gPad->SetLogy();
    mePullYvsPhiZmPanel2->SetLineColor(2);
    mePullYvsPhiZmPanel2->Draw();
    newmePullYvsPhiZmPanel2->SetLineColor(4);
    newmePullYvsPhiZmPanel2->SetLineStyle(2);
    newmePullYvsPhiZmPanel2->Draw("same"); 
    myPV->PVCompute(mePullYvsPhiZmPanel2, newmePullYvsPhiZmPanel2, te );

    can_PullYvsPhi->cd(5);
    //gPad->SetLogy();
    mePullYvsPhiZpPanel1->SetLineColor(2);
    mePullYvsPhiZpPanel1->Draw();
    newmePullYvsPhiZpPanel1->SetLineColor(4);
    newmePullYvsPhiZpPanel1->SetLineStyle(2);
    newmePullYvsPhiZpPanel1->Draw("same"); 
    myPV->PVCompute(mePullYvsPhiZpPanel1, newmePullYvsPhiZpPanel1, te );

    can_PullYvsPhi->cd(6);
    //gPad->SetLogy();
    mePullYvsPhiZpPanel2->SetLineColor(2);
    mePullYvsPhiZpPanel2->Draw();
    newmePullYvsPhiZpPanel2->SetLineColor(4);
    newmePullYvsPhiZpPanel2->SetLineStyle(2);
    newmePullYvsPhiZpPanel2->Draw("same"); 
    myPV->PVCompute(mePullYvsPhiZpPanel2, newmePullYvsPhiZpPanel2, te );

    //can_PullYvsPhi->SaveAs("mePullYvsPhi_compare.eps");
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
  
    can_mePullx->cd(1);
    //gPad->SetLogy();
    mePullxBarrel->SetLineColor(2);
    mePullxBarrel->Draw();
    newmePullxBarrel->SetLineColor(4);
    newmePullxBarrel->SetLineStyle(2);
    newmePullxBarrel->Draw("Sames"); 
    myPV->PVCompute(mePullxBarrel, newmePullxBarrel, te );
  
    can_mePullx->cd(2);
    //gPad->SetLogy();
    mePullxZmPanel1->SetLineColor(2);
    mePullxZmPanel1->Draw();
    newmePullxZmPanel1->SetLineColor(4);
    newmePullxZmPanel1->SetLineStyle(2);
    newmePullxZmPanel1->Draw("same"); 
    myPV->PVCompute(mePullxZmPanel1, newmePullxZmPanel1, te );
  
    can_mePullx->cd(3);
    //gPad->SetLogy();
    mePullxZmPanel2->SetLineColor(2);
    mePullxZmPanel2->Draw();
    newmePullxZmPanel2->SetLineColor(4);
    newmePullxZmPanel2->SetLineStyle(2);
    newmePullxZmPanel2->Draw("same"); 
    myPV->PVCompute(mePullxZmPanel2, newmePullxZmPanel2, te );

    can_mePullx->cd(5);
    //gPad->SetLogy();
    mePullxZpPanel1->SetLineColor(2);
    mePullxZpPanel1->Draw();
    newmePullxZpPanel1->SetLineColor(4);
    newmePullxZpPanel1->SetLineStyle(2);
    newmePullxZpPanel1->Draw("same"); 
    myPV->PVCompute(mePullxZpPanel1, newmePullxZpPanel1, te );

    can_mePullx->cd(6);
    //gPad->SetLogy();
    mePullxZpPanel2->SetLineColor(2);
    mePullxZpPanel2->Draw();
    newmePullxZpPanel2->SetLineColor(4);
    newmePullxZpPanel2->SetLineStyle(2);
    newmePullxZpPanel2->Draw("same"); 
    myPV->PVCompute(mePullxZpPanel2, newmePullxZpPanel2, te );

    //can_mePullx->SaveAs("mePullx_compare.eps");
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
  
    can_mePully->cd(1);
    //gPad->SetLogy();
    mePullyBarrel->SetLineColor(2);
    mePullyBarrel->Draw();
    newmePullyBarrel->SetLineColor(4);
    newmePullyBarrel->SetLineStyle(2);
    newmePullyBarrel->Draw("Sames"); 
    myPV->PVCompute(mePullyBarrel, newmePullyBarrel, te );
  
    can_mePully->cd(2);
    //gPad->SetLogy();
    mePullyZmPanel1->SetLineColor(2);
    mePullyZmPanel1->Draw();
    newmePullyZmPanel1->SetLineColor(4);
    newmePullyZmPanel1->SetLineStyle(2);
    newmePullyZmPanel1->Draw("same"); 
    myPV->PVCompute(mePullyZmPanel1, newmePullyZmPanel1, te );
  
    can_mePully->cd(3);
    //gPad->SetLogy();
    mePullyZmPanel2->SetLineColor(2);
    mePullyZmPanel2->Draw();
    newmePullyZmPanel2->SetLineColor(4);
    newmePullyZmPanel2->SetLineStyle(2);
    newmePullyZmPanel2->Draw("same"); 
    myPV->PVCompute(mePullyZmPanel2, newmePullyZmPanel2, te );

    can_mePully->cd(5);
    //gPad->SetLogy();
    mePullyZpPanel1->SetLineColor(2);
    mePullyZpPanel1->Draw();
    newmePullyZpPanel1->SetLineColor(4);
    newmePullyZpPanel1->SetLineStyle(2);
    newmePullyZpPanel1->Draw("same"); 
    myPV->PVCompute(mePullyZpPanel1, newmePullyZpPanel1, te );

    can_mePully->cd(6);
    //gPad->SetLogy();
    mePullyZpPanel2->SetLineColor(2);
    mePullyZpPanel2->Draw();
    newmePullyZpPanel2->SetLineColor(4);
    newmePullyZpPanel2->SetLineStyle(2);
    newmePullyZpPanel2->Draw("same"); 
    myPV->PVCompute(mePullyZpPanel2, newmePullyZpPanel2, te );

    //can_mePully->SaveAs("mePully_compare.eps");
  }

if (1) 
  {
    TCanvas* can_ResXvsAlpha = new TCanvas("can_ResXvsAlpha", "can_ResXvsAlpha", 1200, 800);
    can_ResXvsAlpha->Divide(3,2);
    
    TProfile* meResXvsAlphaBarrel;
    TProfile* meResXvsAlphaZmPanel1;
    TProfile* meResXvsAlphaZmPanel2;
    TProfile* meResXvsAlphaZpPanel1;
    TProfile* meResXvsAlphaZpPanel2;
    
    TProfile* newmeResXvsAlphaBarrel;
    TProfile* newmeResXvsAlphaZmPanel1;
    TProfile* newmeResXvsAlphaZmPanel2;
    TProfile* newmeResXvsAlphaZpPanel1;
    TProfile* newmeResXvsAlphaZpPanel2;

    rfile->GetObject("DQMData/Histograms_all/meResXvsAlphaBarrel"  , meResXvsAlphaBarrel  );
    rfile->GetObject("DQMData/Histograms_all/meResXvsAlphaZmPanel1", meResXvsAlphaZmPanel1);
    rfile->GetObject("DQMData/Histograms_all/meResXvsAlphaZmPanel2", meResXvsAlphaZmPanel2);
    rfile->GetObject("DQMData/Histograms_all/meResXvsAlphaZpPanel1", meResXvsAlphaZpPanel1);
    rfile->GetObject("DQMData/Histograms_all/meResXvsAlphaZpPanel2", meResXvsAlphaZpPanel2);

    sfile->GetObject("DQMData/Histograms_all/meResXvsAlphaBarrel"  , newmeResXvsAlphaBarrel  ); 
    sfile->GetObject("DQMData/Histograms_all/meResXvsAlphaZmPanel1", newmeResXvsAlphaZmPanel1);
    sfile->GetObject("DQMData/Histograms_all/meResXvsAlphaZmPanel2", newmeResXvsAlphaZmPanel2);
    sfile->GetObject("DQMData/Histograms_all/meResXvsAlphaZpPanel1", newmeResXvsAlphaZpPanel1);
    sfile->GetObject("DQMData/Histograms_all/meResXvsAlphaZpPanel2", newmeResXvsAlphaZpPanel2);
  
    can_ResXvsAlpha->cd(1);
    //gPad->SetLogy();
    meResXvsAlphaBarrel->SetLineColor(2);
    meResXvsAlphaBarrel->Draw();
    newmeResXvsAlphaBarrel->SetLineColor(4);
    newmeResXvsAlphaBarrel->SetLineStyle(2);
    newmeResXvsAlphaBarrel->Draw("Sames"); 
    myPV->PVCompute(meResXvsAlphaBarrel, newmeResXvsAlphaBarrel, te );
  
    can_ResXvsAlpha->cd(2);
    //gPad->SetLogy();
    meResXvsAlphaZmPanel1->SetLineColor(2);
    meResXvsAlphaZmPanel1->Draw();
    newmeResXvsAlphaZmPanel1->SetLineColor(4);
    newmeResXvsAlphaZmPanel1->SetLineStyle(2);
    newmeResXvsAlphaZmPanel1->Draw("same"); 
    myPV->PVCompute(meResXvsAlphaZmPanel1, newmeResXvsAlphaZmPanel1, te );
  
    can_ResXvsAlpha->cd(3);
    //gPad->SetLogy();
    meResXvsAlphaZmPanel2->SetLineColor(2);
    meResXvsAlphaZmPanel2->Draw();
    newmeResXvsAlphaZmPanel2->SetLineColor(4);
    newmeResXvsAlphaZmPanel2->SetLineStyle(2);
    newmeResXvsAlphaZmPanel2->Draw("same"); 
    myPV->PVCompute(meResXvsAlphaZmPanel2, newmeResXvsAlphaZmPanel2, te );

    can_ResXvsAlpha->cd(5);
    //gPad->SetLogy();
    meResXvsAlphaZpPanel1->SetLineColor(2);
    meResXvsAlphaZpPanel1->Draw();
    newmeResXvsAlphaZpPanel1->SetLineColor(4);
    newmeResXvsAlphaZpPanel1->SetLineStyle(2);
    newmeResXvsAlphaZpPanel1->Draw("same"); 
    myPV->PVCompute(meResXvsAlphaZpPanel1, newmeResXvsAlphaZpPanel1, te );

    can_ResXvsAlpha->cd(6);
    //gPad->SetLogy();
    meResXvsAlphaZpPanel2->SetLineColor(2);
    meResXvsAlphaZpPanel2->Draw();
    newmeResXvsAlphaZpPanel2->SetLineColor(4);
    newmeResXvsAlphaZpPanel2->SetLineStyle(2);
    newmeResXvsAlphaZpPanel2->Draw("same"); 
    myPV->PVCompute(meResXvsAlphaZpPanel2, newmeResXvsAlphaZpPanel2, te );

    //can_ResXvsAlpha->SaveAs("meResXvsAlpha_compare.eps");
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
  
    can_ResXvsBeta->cd(1);
    //gPad->SetLogy();
    meResXvsBetaBarrel->SetLineColor(2);
    meResXvsBetaBarrel->Draw();
    newmeResXvsBetaBarrel->SetLineColor(4);
    newmeResXvsBetaBarrel->SetLineStyle(2);
    newmeResXvsBetaBarrel->Draw("Sames"); 
    myPV->PVCompute(meResXvsBetaBarrel, newmeResXvsBetaBarrel, te );
  
    can_ResXvsBeta->cd(2);
    //gPad->SetLogy();
    meResXvsBetaZmPanel1->SetLineColor(2);
    meResXvsBetaZmPanel1->Draw();
    newmeResXvsBetaZmPanel1->SetLineColor(4);
    newmeResXvsBetaZmPanel1->SetLineStyle(2);
    newmeResXvsBetaZmPanel1->Draw("same"); 
    myPV->PVCompute(meResXvsBetaZmPanel1, newmeResXvsBetaZmPanel1, te );
  
    can_ResXvsBeta->cd(3);
    //gPad->SetLogy();
    meResXvsBetaZmPanel2->SetLineColor(2);
    meResXvsBetaZmPanel2->Draw();
    newmeResXvsBetaZmPanel2->SetLineColor(4);
    newmeResXvsBetaZmPanel2->SetLineStyle(2);
    newmeResXvsBetaZmPanel2->Draw("same"); 
    myPV->PVCompute(meResXvsBetaZmPanel2, newmeResXvsBetaZmPanel2, te );

    can_ResXvsBeta->cd(5);
    //gPad->SetLogy();
    meResXvsBetaZpPanel1->SetLineColor(2);
    meResXvsBetaZpPanel1->Draw();
    newmeResXvsBetaZpPanel1->SetLineColor(4);
    newmeResXvsBetaZpPanel1->SetLineStyle(2);
    newmeResXvsBetaZpPanel1->Draw("same"); 
    myPV->PVCompute(meResXvsBetaZpPanel1, newmeResXvsBetaZpPanel1, te );

    can_ResXvsBeta->cd(6);
    //gPad->SetLogy();
    meResXvsBetaZpPanel2->SetLineColor(2);
    meResXvsBetaZpPanel2->Draw();
    newmeResXvsBetaZpPanel2->SetLineColor(4);
    newmeResXvsBetaZpPanel2->SetLineStyle(2);
    newmeResXvsBetaZpPanel2->Draw("same"); 
    myPV->PVCompute(meResXvsBetaZpPanel2, newmeResXvsBetaZpPanel2, te );

    //can_ResXvsBeta->SaveAs("meResXvsBeta_compare.eps");
  }

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
  
    can_ResYvsAlpha->cd(1);
    //gPad->SetLogy();
    meResYvsAlphaBarrel->SetLineColor(2);
    meResYvsAlphaBarrel->Draw();
    newmeResYvsAlphaBarrel->SetLineColor(4);
    newmeResYvsAlphaBarrel->SetLineStyle(2);
    newmeResYvsAlphaBarrel->Draw("Sames"); 
    myPV->PVCompute(meResYvsAlphaBarrel, newmeResYvsAlphaBarrel, te );
  
    can_ResYvsAlpha->cd(2);
    //gPad->SetLogy();
    meResYvsAlphaZmPanel1->SetLineColor(2);
    meResYvsAlphaZmPanel1->Draw();
    newmeResYvsAlphaZmPanel1->SetLineColor(4);
    newmeResYvsAlphaZmPanel1->SetLineStyle(2);
    newmeResYvsAlphaZmPanel1->Draw("same"); 
    myPV->PVCompute(meResYvsAlphaZmPanel1, newmeResYvsAlphaZmPanel1, te );
  
    can_ResYvsAlpha->cd(3);
    //gPad->SetLogy();
    meResYvsAlphaZmPanel2->SetLineColor(2);
    meResYvsAlphaZmPanel2->Draw();
    newmeResYvsAlphaZmPanel2->SetLineColor(4);
    newmeResYvsAlphaZmPanel2->SetLineStyle(2);
    newmeResYvsAlphaZmPanel2->Draw("same"); 
    myPV->PVCompute(meResYvsAlphaZmPanel2, newmeResYvsAlphaZmPanel2, te );

    can_ResYvsAlpha->cd(5);
    //gPad->SetLogy();
    meResYvsAlphaZpPanel1->SetLineColor(2);
    meResYvsAlphaZpPanel1->Draw();
    newmeResYvsAlphaZpPanel1->SetLineColor(4);
    newmeResYvsAlphaZpPanel1->SetLineStyle(2);
    newmeResYvsAlphaZpPanel1->Draw("same"); 
    myPV->PVCompute(meResYvsAlphaZpPanel1, newmeResYvsAlphaZpPanel1, te );

    can_ResYvsAlpha->cd(6);
    //gPad->SetLogy();
    meResYvsAlphaZpPanel2->SetLineColor(2);
    meResYvsAlphaZpPanel2->Draw();
    newmeResYvsAlphaZpPanel2->SetLineColor(4);
    newmeResYvsAlphaZpPanel2->SetLineStyle(2);
    newmeResYvsAlphaZpPanel2->Draw("same"); 
    myPV->PVCompute(meResYvsAlphaZpPanel2, newmeResYvsAlphaZpPanel2, te );

    //can_ResYvsAlpha->SaveAs("meResYvsAlpha_compare.eps");
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
  
    can_ResYvsBeta->cd(1);
    //gPad->SetLogy();
    meResYvsBetaBarrel->SetLineColor(2);
    meResYvsBetaBarrel->Draw();
    newmeResYvsBetaBarrel->SetLineColor(4);
    newmeResYvsBetaBarrel->SetLineStyle(2);
    newmeResYvsBetaBarrel->Draw("Sames"); 
    myPV->PVCompute(meResYvsBetaBarrel, newmeResYvsBetaBarrel, te );
  
    can_ResYvsBeta->cd(2);
    //gPad->SetLogy();
    meResYvsBetaZmPanel1->SetLineColor(2);
    meResYvsBetaZmPanel1->Draw();
    newmeResYvsBetaZmPanel1->SetLineColor(4);
    newmeResYvsBetaZmPanel1->SetLineStyle(2);
    newmeResYvsBetaZmPanel1->Draw("same"); 
    myPV->PVCompute(meResYvsBetaZmPanel1, newmeResYvsBetaZmPanel1, te );
  
    can_ResYvsBeta->cd(3);
    //gPad->SetLogy();
    meResYvsBetaZmPanel2->SetLineColor(2);
    meResYvsBetaZmPanel2->Draw();
    newmeResYvsBetaZmPanel2->SetLineColor(4);
    newmeResYvsBetaZmPanel2->SetLineStyle(2);
    newmeResYvsBetaZmPanel2->Draw("same"); 
    myPV->PVCompute(meResYvsBetaZmPanel2, newmeResYvsBetaZmPanel2, te );

    can_ResYvsBeta->cd(5);
    //gPad->SetLogy();
    meResYvsBetaZpPanel1->SetLineColor(2);
    meResYvsBetaZpPanel1->Draw();
    newmeResYvsBetaZpPanel1->SetLineColor(4);
    newmeResYvsBetaZpPanel1->SetLineStyle(2);
    newmeResYvsBetaZpPanel1->Draw("same"); 
    myPV->PVCompute(meResYvsBetaZpPanel1, newmeResYvsBetaZpPanel1, te );

    can_ResYvsBeta->cd(6);
    //gPad->SetLogy();
    meResYvsBetaZpPanel2->SetLineColor(2);
    meResYvsBetaZpPanel2->Draw();
    newmeResYvsBetaZpPanel2->SetLineColor(4);
    newmeResYvsBetaZpPanel2->SetLineStyle(2);
    newmeResYvsBetaZpPanel2->Draw("same"); 
    myPV->PVCompute(meResYvsBetaZpPanel2, newmeResYvsBetaZpPanel2, te );

    //can_ResYvsBeta->SaveAs("meResYvsBeta_compare.eps");
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
  
    can_meResx->cd(1);
    //gPad->SetLogy();
    meResxBarrel->SetLineColor(2);
    meResxBarrel->Draw();
    newmeResxBarrel->SetLineColor(4);
    newmeResxBarrel->SetLineStyle(2);
    newmeResxBarrel->Draw("Sames"); 
    myPV->PVCompute(meResxBarrel, newmeResxBarrel, te );
  
    can_meResx->cd(2);
    //gPad->SetLogy();
    meResxZmPanel1->SetLineColor(2);
    meResxZmPanel1->Draw();
    newmeResxZmPanel1->SetLineColor(4);
    newmeResxZmPanel1->SetLineStyle(2);
    newmeResxZmPanel1->Draw("same"); 
    myPV->PVCompute(meResxZmPanel1, newmeResxZmPanel1, te );
  
    can_meResx->cd(3);
    //gPad->SetLogy();
    meResxZmPanel2->SetLineColor(2);
    meResxZmPanel2->Draw();
    newmeResxZmPanel2->SetLineColor(4);
    newmeResxZmPanel2->SetLineStyle(2);
    newmeResxZmPanel2->Draw("same"); 
    myPV->PVCompute(meResxZmPanel2, newmeResxZmPanel2, te );

    can_meResx->cd(5);
    //gPad->SetLogy();
    meResxZpPanel1->SetLineColor(2);
    meResxZpPanel1->Draw();
    newmeResxZpPanel1->SetLineColor(4);
    newmeResxZpPanel1->SetLineStyle(2);
    newmeResxZpPanel1->Draw("same"); 
    myPV->PVCompute(meResxZpPanel1, newmeResxZpPanel1, te );

    can_meResx->cd(6);
    //gPad->SetLogy();
    meResxZpPanel2->SetLineColor(2);
    meResxZpPanel2->Draw();
    newmeResxZpPanel2->SetLineColor(4);
    newmeResxZpPanel2->SetLineStyle(2);
    newmeResxZpPanel2->Draw("same"); 
    myPV->PVCompute(meResxZpPanel2, newmeResxZpPanel2, te );

    //can_meResx->SaveAs("meResx_compare.eps");
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
  
    can_meResy->cd(1);
    //gPad->SetLogy();
    meResyBarrel->SetLineColor(2);
    meResyBarrel->Draw();
    newmeResyBarrel->SetLineColor(4);
    newmeResyBarrel->SetLineStyle(2);
    newmeResyBarrel->Draw("Sames"); 
    myPV->PVCompute(meResyBarrel, newmeResyBarrel, te );
  
    can_meResy->cd(2);
    //gPad->SetLogy();
    meResyZmPanel1->SetLineColor(2);
    meResyZmPanel1->Draw();
    newmeResyZmPanel1->SetLineColor(4);
    newmeResyZmPanel1->SetLineStyle(2);
    newmeResyZmPanel1->Draw("same"); 
    myPV->PVCompute(meResyZmPanel1, newmeResyZmPanel1, te );
  
    can_meResy->cd(3);
    //gPad->SetLogy();
    meResyZmPanel2->SetLineColor(2);
    meResyZmPanel2->Draw();
    newmeResyZmPanel2->SetLineColor(4);
    newmeResyZmPanel2->SetLineStyle(2);
    newmeResyZmPanel2->Draw("same"); 
    myPV->PVCompute(meResyZmPanel2, newmeResyZmPanel2, te );

    can_meResy->cd(5);
    //gPad->SetLogy();
    meResyZpPanel1->SetLineColor(2);
    meResyZpPanel1->Draw();
    newmeResyZpPanel1->SetLineColor(4);
    newmeResyZpPanel1->SetLineStyle(2);
    newmeResyZpPanel1->Draw("same"); 
    myPV->PVCompute(meResyZpPanel1, newmeResyZpPanel1, te );

    can_meResy->cd(6);
    //gPad->SetLogy();
    meResyZpPanel2->SetLineColor(2);
    meResyZpPanel2->Draw();
    newmeResyZpPanel2->SetLineColor(4);
    newmeResyZpPanel2->SetLineStyle(2);
    newmeResyZpPanel2->Draw("same"); 
    myPV->PVCompute(meResyZpPanel2, newmeResyZpPanel2, te );

    //can_meResy->SaveAs("meResy_compare.eps");
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
       
       can_meChargeRingLayer->cd(8*i + j + 1);
       //gPad->SetLogy();
       meChargeLayerModule[i][j]->SetLineColor(2);
       meChargeLayerModule[i][j]->Draw();
       newmeChargeLayerModule[i][j]->SetLineColor(4);
       newmeChargeLayerModule[i][j]->SetLineStyle(2);
       newmeChargeLayerModule[i][j]->Draw("same"); 
       myPV->PVCompute(meChargeLayerModule[i][j], newmeChargeLayerModule[i][j], te );
     }
 
 //can_meChargeRingLayer->SaveAs("meChargeBarrelLayerModule_compare.eps");




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
       meChargeZmPanel1DiskPlaq[i][j]->SetLineColor(2);
       meChargeZmPanel1DiskPlaq[i][j]->Draw();
       newmeChargeZmPanel1DiskPlaq[i][j]->SetLineColor(4);
       newmeChargeZmPanel1DiskPlaq[i][j]->SetLineStyle(2);
       newmeChargeZmPanel1DiskPlaq[i][j]->Draw("same"); 
       myPV->PVCompute(meChargeZmPanel1DiskPlaq[i][j], newmeChargeZmPanel1DiskPlaq[i][j], te );
     }
 
 //can_meChargeZmPanel1DiskPlaq->SaveAs("meChargeBarrelZmPanel1DiskPlaq_compare.eps");



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
       meChargeZmPanel2DiskPlaq[i][j]->SetLineColor(2);
       meChargeZmPanel2DiskPlaq[i][j]->Draw();
       newmeChargeZmPanel2DiskPlaq[i][j]->SetLineColor(4);
       newmeChargeZmPanel2DiskPlaq[i][j]->SetLineStyle(2);
       newmeChargeZmPanel2DiskPlaq[i][j]->Draw("same"); 
       myPV->PVCompute(meChargeZmPanel2DiskPlaq[i][j], newmeChargeZmPanel2DiskPlaq[i][j], te );
     }
 
 //can_meChargeZmPanel2DiskPlaq->SaveAs("meChargeBarrelZmPanel2DiskPlaq_compare.eps");



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
       meChargeZpPanel1DiskPlaq[i][j]->SetLineColor(2);
       meChargeZpPanel1DiskPlaq[i][j]->Draw();
       newmeChargeZpPanel1DiskPlaq[i][j]->SetLineColor(4);
       newmeChargeZpPanel1DiskPlaq[i][j]->SetLineStyle(2);
       newmeChargeZpPanel1DiskPlaq[i][j]->Draw("same"); 
       myPV->PVCompute(meChargeZpPanel1DiskPlaq[i][j], newmeChargeZpPanel1DiskPlaq[i][j], te );
     }
 
 //can_meChargeZpPanel1DiskPlaq->SaveAs("meChargeBarrelZpPanel1DiskPlaq_compare.eps");



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
       meChargeZpPanel2DiskPlaq[i][j]->SetLineColor(2);
       meChargeZpPanel2DiskPlaq[i][j]->Draw();
       newmeChargeZpPanel2DiskPlaq[i][j]->SetLineColor(4);
       newmeChargeZpPanel2DiskPlaq[i][j]->SetLineStyle(2);
       newmeChargeZpPanel2DiskPlaq[i][j]->Draw("same"); 
       myPV->PVCompute(meChargeZpPanel2DiskPlaq[i][j], newmeChargeZpPanel2DiskPlaq[i][j], te );
     }
 
 //can_meChargeZpPanel2DiskPlaq->SaveAs("meChargeBarrelZpPanel2DiskPlaq_compare.eps");





}
