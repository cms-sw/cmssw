#include "TFile.h"
#include "TTree.h"
#include "TText.h"

void EcalRecHitsPlotCompare( TString currentfile = "EcalRecHitsValidation_new.root",
                             TString referencefile = "EcalRecHitsValidation_old.root")
{

 gROOT ->Reset();
 char*  rfilename = referencefile;
 char*  sfilename = currentfile;

 int rcolor = 2;
 int scolor = 4;

 delete gROOT->GetListOfFiles()->FindObject(rfilename);
 delete gROOT->GetListOfFiles()->FindObject(sfilename);

 TText* te = new TText();
 te->SetTextSize(0.1);
 TFile * rfile = new TFile(rfilename);
 TFile * sfile = new TFile(sfilename);

 rfile->cd("DQMData/EcalRecHitsTask");
 gDirectory->ls();

 sfile->cd("DQMData/EcalRecHitsTask");
 gDirectory->ls();

 Char_t histo[200];

 gStyle->SetOptStat("n");

 gROOT->ProcessLine(".x HistoCompare.C");
 HistoCompare * myPV = new HistoCompare();

 //////////////////////////////////////////////////////////////
 
 // General class: Particle Gun 
 if (1) 
   {   
     TH1* meGunEnergy_;
     rfile->GetObject("DQMData/EcalRecHitsTask/EcalRecHitsTask, Gun Momentum;1",meGunEnergy_);
     meGunEnergy_;
     meGunEnergy_->SetLineColor(rcolor);
     
     TH1* meGunEta_;
     rfile->GetObject("DQMData/EcalRecHitsTask/EcalRecHitsTask, Gun Eta;1",meGunEta_);
     meGunEta_;
     meGunEta_->SetLineColor(rcolor);
     
     TH1* meGunPhi_; 
     rfile->GetObject("DQMData/EcalRecHitsTask/EcalRecHitsTask, Gun Phi;1",meGunPhi_);
     meGunPhi_; 
     meGunPhi_->SetLineColor(rcolor);
     
     TH1* newmeGunEnergy_;
     sfile->GetObject("DQMData/EcalRecHitsTask/EcalRecHitsTask, Gun Momentum;1",newmeGunEnergy_);
     newmeGunEnergy_;
     newmeGunEnergy_->SetLineColor(scolor);
     
     TH1* newmeGunEta_;
     sfile->GetObject("DQMData/EcalRecHitsTask/EcalRecHitsTask, Gun Eta;1",newmeGunEta_);
     newmeGunEta_;
     newmeGunEta_->SetLineColor(scolor);
     
     TH1* newmeGunPhi_; 
     sfile->GetObject("DQMData/EcalRecHitsTask/EcalRecHitsTask, Gun Phi;1",newmeGunPhi_);
     newmeGunPhi_; 
     newmeGunPhi_->SetLineColor(scolor);
     
    
     // --------------------------------
     TCanvas * Ecal = new TCanvas("Ecal","Ecal",800,1000);
     Ecal->Divide(1,3);
     
     Ecal->cd(1); 
     if ( meGunEnergy_ && newmeGunEnergy_ ) {
       meGunEnergy_->Draw(); 
       newmeGunEnergy_->Draw("same"); 
       myPV->PVCompute( meGunEnergy_ , newmeGunEnergy_ , te );
     }

     Ecal->cd(2); 
     if ( meGunEta_ && newmeGunEta_ ) {
       meGunEta_->Draw(); 
       newmeGunEta_->Draw("same"); 
       myPV->PVCompute( meGunEta_ , newmeGunEta_ , te );
     }

     Ecal->cd(3); 
     if ( meGunPhi_ && newmeGunPhi_ ) {
       meGunPhi_->Draw(); 
       newmeGunPhi_->Draw("same"); 
       myPV->PVCompute( meGunPhi_ , newmeGunPhi_ , te );
     }

     Ecal->Print("ParticleGun_compare.eps"); 
     delete Ecal;
   }
 


 // General class: sim/rec hit ratios
 if (1) 
   {
     TH1 *meEBRecHitSimHitRatio_;
     rfile->GetObject("DQMData/EcalRecHitsTask/EcalRecHitsTask, Barrel RecSimHit Ratio;1",meEBRecHitSimHitRatio_);
     meEBRecHitSimHitRatio_;
     meEBRecHitSimHitRatio_->SetLineColor(rcolor);
     
     TH1 *meEERecHitSimHitRatio_;
     rfile->GetObject("DQMData/EcalRecHitsTask/EcalRecHitsTask, Endcap RecSimHit Ratio;1",meEERecHitSimHitRatio_);
     meEERecHitSimHitRatio_;
     meEERecHitSimHitRatio_->SetLineColor(rcolor);
     
     TH1 *meESRecHitSimHitRatio_;
     rfile->GetObject("DQMData/EcalRecHitsTask/EcalRecHitsTask, Preshower RecSimHit Ratio;1",meESRecHitSimHitRatio_);
     meESRecHitSimHitRatio_;
     meESRecHitSimHitRatio_->SetLineColor(rcolor);

     TH1 *newmeEBRecHitSimHitRatio_;
     sfile->GetObject("DQMData/EcalRecHitsTask/EcalRecHitsTask, Barrel RecSimHit Ratio;1",newmeEBRecHitSimHitRatio_);
     newmeEBRecHitSimHitRatio_;
     newmeEBRecHitSimHitRatio_->SetLineColor(scolor);
     
     TH1 *newmeEERecHitSimHitRatio_;
     sfile->GetObject("DQMData/EcalRecHitsTask/EcalRecHitsTask, Endcap RecSimHit Ratio;1",newmeEERecHitSimHitRatio_);
     newmeEERecHitSimHitRatio_;
     newmeEERecHitSimHitRatio_->SetLineColor(scolor);
     
     TH1 *newmeESRecHitSimHitRatio_;
     sfile->GetObject("DQMData/EcalRecHitsTask/EcalRecHitsTask, Preshower RecSimHit Ratio;1",newmeESRecHitSimHitRatio_);
     newmeESRecHitSimHitRatio_;
     newmeESRecHitSimHitRatio_->SetLineColor(scolor);

     TH1 *meEBRecHitSimHitRatioGt35_;
     rfile->GetObject("DQMData/EcalRecHitsTask/EcalRecHitsTask, Barrel RecSimHit Ratio gt 3.5 GeV;1",meEBRecHitSimHitRatioGt35_);
     meEBRecHitSimHitRatioGt35_;
     meEBRecHitSimHitRatioGt35_->SetLineColor(rcolor);
     
     TH1 *meEERecHitSimHitRatioGt35_;
     rfile->GetObject("DQMData/EcalRecHitsTask/EcalRecHitsTask, Endcap RecSimHit Ratio gt 3.5 GeV;1",meEERecHitSimHitRatioGt35_);
     meEERecHitSimHitRatioGt35_;
     meEERecHitSimHitRatioGt35_->SetLineColor(rcolor);
     
     TH1 *newmeEBRecHitSimHitRatioGt35_;
     sfile->GetObject("DQMData/EcalRecHitsTask/EcalRecHitsTask, Barrel RecSimHit Ratio gt 3.5 GeV;1",newmeEBRecHitSimHitRatioGt35_);
     newmeEBRecHitSimHitRatioGt35_;
     newmeEBRecHitSimHitRatioGt35_->SetLineColor(scolor);
     
     TH1 *newmeEERecHitSimHitRatioGt35_;
     sfile->GetObject("DQMData/EcalRecHitsTask/EcalRecHitsTask, Endcap RecSimHit Ratio gt 3.5 GeV;1",newmeEERecHitSimHitRatioGt35_);
     newmeEERecHitSimHitRatioGt35_;
     newmeEERecHitSimHitRatioGt35_->SetLineColor(scolor);
     

     // --------------------------------
     TCanvas *BarrelSimRec = new TCanvas("BarrelSimRec","BarrelSimRec",800,800);
     if ( meEBRecHitSimHitRatio_ && newmeEBRecHitSimHitRatio_ ) {
       meEBRecHitSimHitRatio_->Draw();
       newmeEBRecHitSimHitRatio_->Draw("same");
       myPV->PVCompute( meEBRecHitSimHitRatio_ , newmeEBRecHitSimHitRatio_ , te );       
     }
     BarrelSimRec->Print("Barrel_SimRecHitsRatio_compare.eps");
     delete BarrelSimRec;


     TCanvas *EndcapSimRec = new TCanvas("EndcapSimRec","EndcapSimRec",800,800);
     if ( meEERecHitSimHitRatio_ && newmeEERecHitSimHitRatio_ ) {
       meEERecHitSimHitRatio_->Draw();
       newmeEERecHitSimHitRatio_->Draw("same");
       myPV->PVCompute( meEERecHitSimHitRatio_ , newmeEERecHitSimHitRatio_ , te );       
     }
     EndcapSimRec->Print("Endcap_SimRecHitsRatio_compare.eps");
     delete EndcapSimRec;


     TCanvas *PreshowerSimRec = new TCanvas("PreshowerSimRec","PreshowerSimRec",800,800);
     if ( meESRecHitSimHitRatio_ && newmeESRecHitSimHitRatio_ ) {
       meESRecHitSimHitRatio_->Draw();
       newmeESRecHitSimHitRatio_->Draw("same");
       myPV->PVCompute( meESRecHitSimHitRatio_ , newmeESRecHitSimHitRatio_ , te );       
     }
     PreshowerSimRec->Print("Preshower_SimRecHitsRatio_compare.eps");
     delete PreshowerSimRec;


     TCanvas *BarrelSimRecGt35 = new TCanvas("BarrelSimRecGt35","BarrelSimRecGt35",800,800);
     if ( meEBRecHitSimHitRatioGt35_ && newmeEBRecHitSimHitRatioGt35_ ) {
       meEBRecHitSimHitRatioGt35_   ->Draw();
       newmeEBRecHitSimHitRatioGt35_->Draw("same");
       myPV->PVCompute( meEBRecHitSimHitRatioGt35_ , newmeEBRecHitSimHitRatioGt35_ , te );       
     }
     BarrelSimRecGt35->Print("Barrel_SimRecHitsRatioGt35_compare.eps");
     delete BarrelSimRecGt35;


     TCanvas *EndcapSimRecGt35 = new TCanvas("EndcapSimRecGt35","EndcapSimRecGt35",800,800);
     if ( meEERecHitSimHitRatioGt35_ && newmeEERecHitSimHitRatioGt35_ ) {
       meEERecHitSimHitRatioGt35_   ->Draw();
       newmeEERecHitSimHitRatioGt35_->Draw("same");
       myPV->PVCompute( meEERecHitSimHitRatioGt35_ , newmeEERecHitSimHitRatioGt35_ , te );       
     }
     EndcapSimRecGt35->Print("Endcap_SimRecHitsRatioGt35_compare.eps");
     delete EndcapSimRecGt35;
   }



 




 // ---------------------------------------------------------------------------------
 cout << endl;
 cout << "Barrel validation" << endl;
 rfile->cd("DQMData/EcalBarrelRecHitsTask");
 gDirectory->ls();
 sfile->cd("DQMData/EcalBarrelRecHitsTask");
 gDirectory->ls();
 
 
 // Barrel validation
 if (1) 
   {    
     TH2 *meEBUncalibRecHitsOccupancy_;
     rfile->GetObject("DQMData/EcalBarrelRecHitsTask/EB Occupancy;1",meEBUncalibRecHitsOccupancy_);
     meEBUncalibRecHitsOccupancy_;
      
     TH1 *meEBUncalibRecHitsAmplitude_;
     rfile->GetObject("DQMData/EcalBarrelRecHitsTask/EB Amplitude;1",meEBUncalibRecHitsAmplitude_);
     meEBUncalibRecHitsAmplitude_;
     meEBUncalibRecHitsAmplitude_->SetLineColor(rcolor);
      
     TH1 *meEBUncalibRecHitsPedestal_;
     rfile->GetObject("DQMData/EcalBarrelRecHitsTask/EB Pedestal;1",meEBUncalibRecHitsPedestal_);
     meEBUncalibRecHitsPedestal_;
     meEBUncalibRecHitsPedestal_->SetLineColor(rcolor);
     
     TH1 *meEBUncalibRecHitsJitter_;
     rfile->GetObject("DQMData/EcalBarrelRecHitsTask/EB Jitter;1",meEBUncalibRecHitsJitter_);
     meEBUncalibRecHitsJitter_;
     meEBUncalibRecHitsJitter_->SetLineColor(rcolor);
     
     TH1 *meEBUncalibRecHitsChi2_;
     rfile->GetObject("DQMData/EcalBarrelRecHitsTask/EB Chi2;1",meEBUncalibRecHitsChi2_);
     meEBUncalibRecHitsChi2_;
     meEBUncalibRecHitsChi2_->SetLineColor(rcolor);

     TH1 *meEBUncalibRecHitMaxSampleRatio_;
     rfile->GetObject("DQMData/EcalBarrelRecHitsTask/EB RecHit Max Sample Ratio;1",meEBUncalibRecHitMaxSampleRatio_);
     meEBUncalibRecHitMaxSampleRatio_;
     meEBUncalibRecHitMaxSampleRatio_->SetLineColor(rcolor);

     TH2 *meEBUncalibRecHitsOccupancyGt100adc_;
     rfile->GetObject("DQMData/EcalBarrelRecHitsTask/EB Occupancy gt 100 adc counts;1",meEBUncalibRecHitsOccupancyGt100adc_);
     meEBUncalibRecHitsOccupancyGt100adc_;
      
     TH1 *meEBUncalibRecHitsAmplitudeGt100adc_;
     rfile->GetObject("DQMData/EcalBarrelRecHitsTask/EB Amplitude gt 100 adc counts;1",meEBUncalibRecHitsAmplitudeGt100adc_);
     meEBUncalibRecHitsAmplitudeGt100adc_;
     meEBUncalibRecHitsAmplitudeGt100adc_->SetLineColor(rcolor);
      
     TH1 *meEBUncalibRecHitsPedestalGt100adc_;
     rfile->GetObject("DQMData/EcalBarrelRecHitsTask/EB Pedestal gt 100 adc counts;1",meEBUncalibRecHitsPedestalGt100adc_);
     meEBUncalibRecHitsPedestalGt100adc_;
     meEBUncalibRecHitsPedestalGt100adc_->SetLineColor(rcolor);
     
     TH1 *meEBUncalibRecHitsJitterGt100adc_;
     rfile->GetObject("DQMData/EcalBarrelRecHitsTask/EB Jitter gt 100 adc counts;1",meEBUncalibRecHitsJitterGt100adc_);
     meEBUncalibRecHitsJitterGt100adc_;
     meEBUncalibRecHitsJitterGt100adc_->SetLineColor(rcolor);
     
     TH1 *meEBUncalibRecHitsChi2Gt100adc_;
     rfile->GetObject("DQMData/EcalBarrelRecHitsTask/EB Chi2 gt 100 adc counts;1",meEBUncalibRecHitsChi2Gt100adc_);
     meEBUncalibRecHitsChi2Gt100adc_;
     meEBUncalibRecHitsChi2Gt100adc_->SetLineColor(rcolor);

     TH1 *meEBUncalibRecHitMaxSampleRatioGt100adc_;
     rfile->GetObject("DQMData/EcalBarrelRecHitsTask/EB RecHit Max Sample Ratio gt 100 adc counts;1",meEBUncalibRecHitMaxSampleRatioGt100adc_);
     meEBUncalibRecHitMaxSampleRatioGt100adc_;
     meEBUncalibRecHitMaxSampleRatioGt100adc_->SetLineColor(rcolor);

     TProfile2D *meEBUncalibRecHitsAmpFullMap_;
     rfile->GetObject("DQMData/EcalBarrelRecHitsTask/EB Amplitude Full Map;1",meEBUncalibRecHitsAmpFullMap_);
     meEBUncalibRecHitsAmpFullMap_;

     TProfile2D *meEBUncalibRecHitsPedFullMap_;
     rfile->GetObject("DQMData/EcalBarrelRecHitsTask/EB Pedestal Full Map;1",meEBUncalibRecHitsPedFullMap_);
     meEBUncalibRecHitsPedFullMap_;

     TH2 *newmeEBUncalibRecHitsOccupancy_;
     sfile->GetObject("DQMData/EcalBarrelRecHitsTask/EB Occupancy;1",newmeEBUncalibRecHitsOccupancy_);
     newmeEBUncalibRecHitsOccupancy_;
      
     TH1 *newmeEBUncalibRecHitsAmplitude_;
     sfile->GetObject("DQMData/EcalBarrelRecHitsTask/EB Amplitude;1",newmeEBUncalibRecHitsAmplitude_);
     newmeEBUncalibRecHitsAmplitude_;
     newmeEBUncalibRecHitsAmplitude_->SetLineColor(scolor);
      
     TH1 *newmeEBUncalibRecHitsPedestal_;
     sfile->GetObject("DQMData/EcalBarrelRecHitsTask/EB Pedestal;1",newmeEBUncalibRecHitsPedestal_);
     newmeEBUncalibRecHitsPedestal_;
     newmeEBUncalibRecHitsPedestal_->SetLineColor(scolor);
     
     TH1 *newmeEBUncalibRecHitsJitter_;
     sfile->GetObject("DQMData/EcalBarrelRecHitsTask/EB Jitter;1",newmeEBUncalibRecHitsJitter_);
     newmeEBUncalibRecHitsJitter_;
     newmeEBUncalibRecHitsJitter_->SetLineColor(scolor);
     
     TH1 *newmeEBUncalibRecHitsChi2_;
     sfile->GetObject("DQMData/EcalBarrelRecHitsTask/EB Chi2;1",newmeEBUncalibRecHitsChi2_);
     newmeEBUncalibRecHitsChi2_;
     newmeEBUncalibRecHitsChi2_->SetLineColor(scolor);

     TH1 *newmeEBUncalibRecHitMaxSampleRatio_;
     sfile->GetObject("DQMData/EcalBarrelRecHitsTask/EB RecHit Max Sample Ratio;1",newmeEBUncalibRecHitMaxSampleRatio_);
     newmeEBUncalibRecHitMaxSampleRatio_;
     newmeEBUncalibRecHitMaxSampleRatio_->SetLineColor(scolor);

     TH2 *newmeEBUncalibRecHitsOccupancyGt100adc_;
     sfile->GetObject("DQMData/EcalBarrelRecHitsTask/EB Occupancy gt 100 adc counts;1",newmeEBUncalibRecHitsOccupancyGt100adc_);
     newmeEBUncalibRecHitsOccupancyGt100adc_;
      
     TH1 *newmeEBUncalibRecHitsAmplitudeGt100adc_;
     sfile->GetObject("DQMData/EcalBarrelRecHitsTask/EB Amplitude gt 100 adc counts;1",newmeEBUncalibRecHitsAmplitudeGt100adc_);
     newmeEBUncalibRecHitsAmplitudeGt100adc_;
     newmeEBUncalibRecHitsAmplitudeGt100adc_->SetLineColor(scolor);
      
     TH1 *newmeEBUncalibRecHitsPedestalGt100adc_;
     sfile->GetObject("DQMData/EcalBarrelRecHitsTask/EB Pedestal gt 100 adc counts;1",newmeEBUncalibRecHitsPedestalGt100adc_);
     newmeEBUncalibRecHitsPedestalGt100adc_;
     newmeEBUncalibRecHitsPedestalGt100adc_->SetLineColor(scolor);
     
     TH1 *newmeEBUncalibRecHitsJitterGt100adc_;
     sfile->GetObject("DQMData/EcalBarrelRecHitsTask/EB Jitter gt 100 adc counts;1",newmeEBUncalibRecHitsJitterGt100adc_);
     newmeEBUncalibRecHitsJitterGt100adc_;
     newmeEBUncalibRecHitsJitterGt100adc_->SetLineColor(scolor);
     
     TH1 *newmeEBUncalibRecHitsChi2Gt100adc_;
     sfile->GetObject("DQMData/EcalBarrelRecHitsTask/EB Chi2 gt 100 adc counts;1",newmeEBUncalibRecHitsChi2Gt100adc_);
     newmeEBUncalibRecHitsChi2Gt100adc_;
     newmeEBUncalibRecHitsChi2Gt100adc_->SetLineColor(scolor);

     TH1 *newmeEBUncalibRecHitMaxSampleRatioGt100adc_;
     sfile->GetObject("DQMData/EcalBarrelRecHitsTask/EB RecHit Max Sample Ratio gt 100 adc counts;1",newmeEBUncalibRecHitMaxSampleRatioGt100adc_);
     newmeEBUncalibRecHitMaxSampleRatioGt100adc_;
     newmeEBUncalibRecHitMaxSampleRatioGt100adc_->SetLineColor(scolor);
      
     TProfile2D *newmeEBUncalibRecHitsAmpFullMap_;
     sfile->GetObject("DQMData/EcalBarrelRecHitsTask/EB Amplitude Full Map;1",newmeEBUncalibRecHitsAmpFullMap_);
     newmeEBUncalibRecHitsAmpFullMap_;

     TProfile2D *newmeEBUncalibRecHitsPedFullMap_;
     sfile->GetObject("DQMData/EcalBarrelRecHitsTask/EB Pedestal Full Map;1",newmeEBUncalibRecHitsPedFullMap_);
     newmeEBUncalibRecHitsPedFullMap_;





      // --------------------------------
      TCanvas *BarrelOccupancy = new TCanvas("BarrelOccupancy","BarrelOccupancy",1000,800);
      BarrelOccupancy->Divide(2,1);
      if ( meEBUncalibRecHitsOccupancy_ && newmeEBUncalibRecHitsOccupancy_ ) 
	{
	  BarrelOccupancy->cd(1); meEBUncalibRecHitsOccupancy_   ->Draw("colz"); 
	  BarrelOccupancy->cd(2); newmeEBUncalibRecHitsOccupancy_->Draw("colz"); 
	  myPV->PVCompute( meEBUncalibRecHitsOccupancy_ , newmeEBUncalibRecHitsOccupancy_ , te );       
	}
      BarrelOccupancy->Print("BarrelOccupancy_compare.eps"); 
      delete BarrelOccupancy;


      TCanvas *BarrelOccupancyGt100adc = new TCanvas("BarrelOccupancyGt100adc","BarrelOccupancyGt100adc",1000,800);
      BarrelOccupancyGt100adc->Divide(2,1);
      if ( meEBUncalibRecHitsOccupancyGt100adc_ && newmeEBUncalibRecHitsOccupancyGt100adc_ ) 
	{
	  BarrelOccupancyGt100adc->cd(1); meEBUncalibRecHitsOccupancyGt100adc_   ->Draw("colz"); 
	  BarrelOccupancyGt100adc->cd(2); newmeEBUncalibRecHitsOccupancyGt100adc_->Draw("colz"); 
	  myPV->PVCompute( meEBUncalibRecHitsOccupancyGt100adc_ , newmeEBUncalibRecHitsOccupancyGt100adc_ , te );       
	}
      BarrelOccupancyGt100adc->Print("BarrelOccupancyGt100adc_compare.eps"); 
      delete BarrelOccupancyGt100adc;


      TCanvas *BarrelFullMap = new TCanvas("BarrelFullMap","BarrelFullMap",1000,800);
      BarrelFullMap->Divide(2,2);
      if ( meEBUncalibRecHitsAmpFullMap_ && newmeEBUncalibRecHitsAmpFullMap_ )
	{
	  BarrelFullMap->cd(1); meEBUncalibRecHitsAmpFullMap_   ->Draw("colz"); 
	  BarrelFullMap->cd(2); newmeEBUncalibRecHitsAmpFullMap_->Draw("colz"); 
	  myPV->PVCompute( meEBUncalibRecHitsAmpFullMap_ , newmeEBUncalibRecHitsAmpFullMap_ , te );       

	  BarrelFullMap->cd(3); meEBUncalibRecHitsPedFullMap_   ->Draw("colz"); 
	  BarrelFullMap->cd(4); newmeEBUncalibRecHitsPedFullMap_->Draw("colz"); 
	  myPV->PVCompute( meEBUncalibRecHitsPedFullMap_ , newmeEBUncalibRecHitsPedFullMap_ , te );       
	}
      BarrelFullMap->Print("BarrelFullMap_compare.eps"); 
      delete BarrelFullMap;


      TCanvas *Barrel = new TCanvas("Barrel","Barrel",800,800);
      Barrel->Divide(3,2);
      if ( meEBUncalibRecHitsAmplitude_ && newmeEBUncalibRecHitsAmplitude_ )
	{
	  Barrel->cd(1); 
	  gPad->SetLogy();  
	  meEBUncalibRecHitsAmplitude_->Draw(); 
	  newmeEBUncalibRecHitsAmplitude_->Draw("same"); 
	  myPV->PVCompute( meEBUncalibRecHitsAmplitude_ , newmeEBUncalibRecHitsAmplitude_ , te );       
	}

      if ( meEBUncalibRecHitsPedestal_ && newmeEBUncalibRecHitsPedestal_ )
	{
	  Barrel->cd(2);  
	  meEBUncalibRecHitsPedestal_->Draw(); 
	  newmeEBUncalibRecHitsPedestal_->Draw("same");
	  myPV->PVCompute( meEBUncalibRecHitsPedestal_ , newmeEBUncalibRecHitsPedestal_ , te );        
	}

      if ( meEBUncalibRecHitsJitter_ && newmeEBUncalibRecHitsJitter_ )
	{
	  Barrel->cd(3);  
	  meEBUncalibRecHitsJitter_->Draw(); 
	  newmeEBUncalibRecHitsJitter_->Draw("same");
	  myPV->PVCompute( meEBUncalibRecHitsJitter_ , newmeEBUncalibRecHitsJitter_ , te );       
	}

      if ( meEBUncalibRecHitsChi2_ && newmeEBUncalibRecHitsChi2_ )
	{
	  Barrel->cd(4);  
	  meEBUncalibRecHitsChi2_->Draw(); 
	  newmeEBUncalibRecHitsChi2_->Draw("same"); 
	  myPV->PVCompute( meEBUncalibRecHitsChi2_ , newmeEBUncalibRecHitsChi2_ , te );       
	}

      if ( meEBUncalibRecHitMaxSampleRatio_ && newmeEBUncalibRecHitMaxSampleRatio_ )
	{
	  Barrel->cd(5);  
	  meEBUncalibRecHitMaxSampleRatio_->Draw(); 
	  newmeEBUncalibRecHitMaxSampleRatio_->Draw("same"); 
	  myPV->PVCompute( meEBUncalibRecHitMaxSampleRatio_ , newmeEBUncalibRecHitMaxSampleRatio_ , te );       
	}

      Barrel->Print("Barrel_compare.eps");
      delete Barrel;
   

      TCanvas *BarrelGt100adc = new TCanvas("BarrelGt100adc","BarrelGt100adc",800,800);
      BarrelGt100adc->Divide(3,2);
      if ( meEBUncalibRecHitsAmplitudeGt100adc_ && newmeEBUncalibRecHitsAmplitudeGt100adc_ )
	{
	  BarrelGt100adc->cd(1); 
	  gPad->SetLogy();  
	  meEBUncalibRecHitsAmplitudeGt100adc_->Draw(); 
	  newmeEBUncalibRecHitsAmplitudeGt100adc_->Draw("same"); 
	  myPV->PVCompute( meEBUncalibRecHitsAmplitudeGt100adc_ , newmeEBUncalibRecHitsAmplitudeGt100adc_ , te );       
	}

      if ( meEBUncalibRecHitsPedestalGt100adc_ && newmeEBUncalibRecHitsPedestalGt100adc_ )
	{
	  BarrelGt100adc->cd(2);  
	  meEBUncalibRecHitsPedestalGt100adc_->Draw(); 
	  newmeEBUncalibRecHitsPedestalGt100adc_->Draw("same");
	  myPV->PVCompute( meEBUncalibRecHitsPedestalGt100adc_ , newmeEBUncalibRecHitsPedestalGt100adc_ , te );        
	}

      if ( meEBUncalibRecHitsJitterGt100adc_ && newmeEBUncalibRecHitsJitterGt100adc_ )
	{
	  BarrelGt100adc->cd(3);  
	  meEBUncalibRecHitsJitterGt100adc_->Draw(); 
	  newmeEBUncalibRecHitsJitterGt100adc_->Draw("same");
	  myPV->PVCompute( meEBUncalibRecHitsJitterGt100adc_ , newmeEBUncalibRecHitsJitterGt100adc_ , te );       
	}

      if ( meEBUncalibRecHitsChi2Gt100adc_ && newmeEBUncalibRecHitsChi2Gt100adc_ )
	{
	  BarrelGt100adc->cd(4);  
	  meEBUncalibRecHitsChi2Gt100adc_->Draw(); 
	  newmeEBUncalibRecHitsChi2Gt100adc_->Draw("same"); 
	  myPV->PVCompute( meEBUncalibRecHitsChi2Gt100adc_ , newmeEBUncalibRecHitsChi2Gt100adc_ , te );       
	}

      if ( meEBUncalibRecHitMaxSampleRatioGt100adc_ && newmeEBUncalibRecHitMaxSampleRatioGt100adc_ )
	{
	  BarrelGt100adc->cd(5);  
	  meEBUncalibRecHitMaxSampleRatioGt100adc_->Draw(); 
	  newmeEBUncalibRecHitMaxSampleRatioGt100adc_->Draw("same"); 
	  myPV->PVCompute( meEBUncalibRecHitMaxSampleRatioGt100adc_ , newmeEBUncalibRecHitMaxSampleRatioGt100adc_ , te );       
	}

      BarrelGt100adc->Print("BarrelGt100adc_compare.eps");
      delete BarrelGt100adc;
   }





 // -----------------------------------------------------------------
 cout << endl;
 cout << "Endcap validation" << endl;
 rfile->cd("DQMData/EcalEndcapRecHitsTask");
 gDirectory->ls();
 sfile->cd("DQMData/EcalEndcapRecHitsTask");
 gDirectory->ls();
 

 
 // Endcap validation
 if (1) 
   {    
     TH2 *meEEUncalibRecHitsOccupancyPlus_;
     rfile->GetObject("DQMData/EcalEndcapRecHitsTask/EE+ Occupancy;1",meEEUncalibRecHitsOccupancyPlus_);
     meEEUncalibRecHitsOccupancyPlus_;

     TH2 *meEEUncalibRecHitsOccupancyMinus_;
     rfile->GetObject("DQMData/EcalEndcapRecHitsTask/EE- Occupancy;1",meEEUncalibRecHitsOccupancyMinus_);
     meEEUncalibRecHitsOccupancyMinus_;
      
     TH1 *meEEUncalibRecHitsAmplitude_;
     rfile->GetObject("DQMData/EcalEndcapRecHitsTask/EE Amplitude;1",meEEUncalibRecHitsAmplitude_);
     meEEUncalibRecHitsAmplitude_;
     meEEUncalibRecHitsAmplitude_->SetLineColor(rcolor);
      
     TH1 *meEEUncalibRecHitsPedestal_;
     rfile->GetObject("DQMData/EcalEndcapRecHitsTask/EE Pedestal;1",meEEUncalibRecHitsPedestal_);
     meEEUncalibRecHitsPedestal_;
     meEEUncalibRecHitsPedestal_->SetLineColor(rcolor);
     
     TH1 *meEEUncalibRecHitsJitter_;
     rfile->GetObject("DQMData/EcalEndcapRecHitsTask/EE Jitter;1",meEEUncalibRecHitsJitter_);
     meEEUncalibRecHitsJitter_;
     meEEUncalibRecHitsJitter_->SetLineColor(rcolor);
     
     TH1 *meEEUncalibRecHitsChi2_;
     rfile->GetObject("DQMData/EcalEndcapRecHitsTask/EE Chi2;1",meEEUncalibRecHitsChi2_);
     meEEUncalibRecHitsChi2_;
     meEEUncalibRecHitsChi2_->SetLineColor(rcolor);

     TH1 *meEEUncalibRecHitMaxSampleRatio_;
     rfile->GetObject("DQMData/EcalEndcapRecHitsTask/EE RecHit Max Sample Ratio;1",meEEUncalibRecHitMaxSampleRatio_);
     meEEUncalibRecHitMaxSampleRatio_;
     meEEUncalibRecHitMaxSampleRatio_->SetLineColor(rcolor);

     TH2 *meEEUncalibRecHitsOccupancyPlusGt60adc_;
     rfile->GetObject("DQMData/EcalEndcapRecHitsTask/EE+ Occupancy gt 60 adc counts;1",meEEUncalibRecHitsOccupancyPlusGt60adc_);
     meEEUncalibRecHitsOccupancyPlusGt60adc_;

     TH2 *meEEUncalibRecHitsOccupancyMinusGt60adc_;
     rfile->GetObject("DQMData/EcalEndcapRecHitsTask/EE- Occupancy gt 60 adc counts;1",meEEUncalibRecHitsOccupancyMinusGt60adc_);
     meEEUncalibRecHitsOccupancyMinusGt60adc_;
      
     TH1 *meEEUncalibRecHitsAmplitudeGt60adc_;
     rfile->GetObject("DQMData/EcalEndcapRecHitsTask/EE Amplitude gt 60 adc counts;1",meEEUncalibRecHitsAmplitudeGt60adc_);
     meEEUncalibRecHitsAmplitudeGt60adc_;
     meEEUncalibRecHitsAmplitudeGt60adc_->SetLineColor(rcolor);
      
     TH1 *meEEUncalibRecHitsPedestalGt60adc_;
     rfile->GetObject("DQMData/EcalEndcapRecHitsTask/EE Pedestal gt 60 adc counts;1",meEEUncalibRecHitsPedestalGt60adc_);
     meEEUncalibRecHitsPedestalGt60adc_;
     meEEUncalibRecHitsPedestalGt60adc_->SetLineColor(rcolor);
     
     TH1 *meEEUncalibRecHitsJitterGt60adc_;
     rfile->GetObject("DQMData/EcalEndcapRecHitsTask/EE Jitter gt 60 adc counts;1",meEEUncalibRecHitsJitterGt60adc_);
     meEEUncalibRecHitsJitterGt60adc_;
     meEEUncalibRecHitsJitterGt60adc_->SetLineColor(rcolor);
     
     TH1 *meEEUncalibRecHitsChi2Gt60adc_;
     rfile->GetObject("DQMData/EcalEndcapRecHitsTask/EE Chi2 gt 60 adc counts;1",meEEUncalibRecHitsChi2Gt60adc_);
     meEEUncalibRecHitsChi2Gt60adc_;
     meEEUncalibRecHitsChi2Gt60adc_->SetLineColor(rcolor);

     TH1 *meEEUncalibRecHitMaxSampleRatioGt60adc_;
     rfile->GetObject("DQMData/EcalEndcapRecHitsTask/EE RecHit Max Sample Ratio gt 60 adc counts;1",meEEUncalibRecHitMaxSampleRatioGt60adc_);
     meEEUncalibRecHitMaxSampleRatioGt60adc_;
     meEEUncalibRecHitMaxSampleRatioGt60adc_->SetLineColor(rcolor);

     TProfile2D *meEEUncalibRecHitsAmpFullMap_;
     rfile->GetObject("DQMData/EcalEndcapRecHitsTask/EE Amplitude Full Map;1",meEEUncalibRecHitsAmpFullMap_);
     meEEUncalibRecHitsAmpFullMap_;

     TProfile2D *meEEUncalibRecHitsPedFullMap_;
     rfile->GetObject("DQMData/EcalEndcapRecHitsTask/EE Pedestal Full Map;1",meEEUncalibRecHitsPedFullMap_);
     meEEUncalibRecHitsPedFullMap_;

     TH2 *newmeEEUncalibRecHitsOccupancyPlus_;
     sfile->GetObject("DQMData/EcalEndcapRecHitsTask/EE+ Occupancy;1",newmeEEUncalibRecHitsOccupancyPlus_);
     newmeEEUncalibRecHitsOccupancyPlus_;

     TH2 *newmeEEUncalibRecHitsOccupancyMinus_;
     sfile->GetObject("DQMData/EcalEndcapRecHitsTask/EE- Occupancy;1",newmeEEUncalibRecHitsOccupancyMinus_);
     newmeEEUncalibRecHitsOccupancyMinus_;

     TH1 *newmeEEUncalibRecHitsAmplitude_;
     sfile->GetObject("DQMData/EcalEndcapRecHitsTask/EE Amplitude;1",newmeEEUncalibRecHitsAmplitude_);
     newmeEEUncalibRecHitsAmplitude_;
     newmeEEUncalibRecHitsAmplitude_->SetLineColor(scolor);
      
     TH1 *newmeEEUncalibRecHitsPedestal_;
     sfile->GetObject("DQMData/EcalEndcapRecHitsTask/EE Pedestal;1",newmeEEUncalibRecHitsPedestal_);
     newmeEEUncalibRecHitsPedestal_;
     newmeEEUncalibRecHitsPedestal_->SetLineColor(scolor);
     
     TH1 *newmeEEUncalibRecHitsJitter_;
     sfile->GetObject("DQMData/EcalEndcapRecHitsTask/EE Jitter;1",newmeEEUncalibRecHitsJitter_);
     newmeEEUncalibRecHitsJitter_;
     newmeEEUncalibRecHitsJitter_->SetLineColor(scolor);
     
     TH1 *newmeEEUncalibRecHitsChi2_;
     sfile->GetObject("DQMData/EcalEndcapRecHitsTask/EE Chi2;1",newmeEEUncalibRecHitsChi2_);
     newmeEEUncalibRecHitsChi2_;
     newmeEEUncalibRecHitsChi2_->SetLineColor(scolor);
      
     TH1 *newmeEEUncalibRecHitMaxSampleRatio_;
     sfile->GetObject("DQMData/EcalEndcapRecHitsTask/EE RecHit Max Sample Ratio;1",newmeEEUncalibRecHitMaxSampleRatio_);
     newmeEEUncalibRecHitMaxSampleRatio_;
     newmeEEUncalibRecHitMaxSampleRatio_->SetLineColor(scolor);

     TH2 *newmeEEUncalibRecHitsOccupancyPlusGt60adc_;
     sfile->GetObject("DQMData/EcalEndcapRecHitsTask/EE+ Occupancy gt 60 adc counts;1",newmeEEUncalibRecHitsOccupancyPlusGt60adc_);
     newmeEEUncalibRecHitsOccupancyPlusGt60adc_;

     TH2 *newmeEEUncalibRecHitsOccupancyMinusGt60adc_;
     sfile->GetObject("DQMData/EcalEndcapRecHitsTask/EE- Occupancy gt 60 adc counts;1",newmeEEUncalibRecHitsOccupancyMinusGt60adc_);
     newmeEEUncalibRecHitsOccupancyMinusGt60adc_;

     TH1 *newmeEEUncalibRecHitsAmplitudeGt60adc_;
     sfile->GetObject("DQMData/EcalEndcapRecHitsTask/EE Amplitude gt 60 adc counts;1",newmeEEUncalibRecHitsAmplitudeGt60adc_);
     newmeEEUncalibRecHitsAmplitudeGt60adc_;
     newmeEEUncalibRecHitsAmplitudeGt60adc_->SetLineColor(scolor);
      
     TH1 *newmeEEUncalibRecHitsPedestalGt60adc_;
     sfile->GetObject("DQMData/EcalEndcapRecHitsTask/EE Pedestal gt 60 adc counts;1",newmeEEUncalibRecHitsPedestalGt60adc_);
     newmeEEUncalibRecHitsPedestalGt60adc_;
     newmeEEUncalibRecHitsPedestalGt60adc_->SetLineColor(scolor);
     
     TH1 *newmeEEUncalibRecHitsJitterGt60adc_;
     sfile->GetObject("DQMData/EcalEndcapRecHitsTask/EE Jitter gt 60 adc counts;1",newmeEEUncalibRecHitsJitterGt60adc_);
     newmeEEUncalibRecHitsJitterGt60adc_;
     newmeEEUncalibRecHitsJitterGt60adc_->SetLineColor(scolor);
     
     TH1 *newmeEEUncalibRecHitsChi2Gt60adc_;
     sfile->GetObject("DQMData/EcalEndcapRecHitsTask/EE Chi2 gt 60 adc counts;1",newmeEEUncalibRecHitsChi2Gt60adc_);
     newmeEEUncalibRecHitsChi2Gt60adc_;
     newmeEEUncalibRecHitsChi2Gt60adc_->SetLineColor(scolor);
      
     TH1 *newmeEEUncalibRecHitMaxSampleRatioGt60adc_;
     sfile->GetObject("DQMData/EcalEndcapRecHitsTask/EE RecHit Max Sample Ratio gt 60 adc counts;1",newmeEEUncalibRecHitMaxSampleRatioGt60adc_);
     newmeEEUncalibRecHitMaxSampleRatioGt60adc_;
     newmeEEUncalibRecHitMaxSampleRatioGt60adc_->SetLineColor(scolor);

     TProfile2D *newmeEEUncalibRecHitsAmpFullMap_;
     sfile->GetObject("DQMData/EcalEndcapRecHitsTask/EE Amplitude Full Map;1",newmeEEUncalibRecHitsAmpFullMap_);
     newmeEEUncalibRecHitsAmpFullMap_;

     TProfile2D *newmeEEUncalibRecHitsPedFullMap_;
     sfile->GetObject("DQMData/EcalEndcapRecHitsTask/EE Pedestal Full Map;1",newmeEEUncalibRecHitsPedFullMap_);
     newmeEEUncalibRecHitsPedFullMap_;



      // --------------------------------
      TCanvas *EndcapOccupancy = new TCanvas("EndcapOccupancy","EndcapOccupancy",1000,800);
      EndcapOccupancy->Divide(2,2);
      if ( meEEUncalibRecHitsOccupancyPlus_ && newmeEEUncalibRecHitsOccupancyPlus_ ) 
	{
	  EndcapOccupancy->cd(1); meEEUncalibRecHitsOccupancyPlus_->Draw("colz"); 
	  EndcapOccupancy->cd(2); newmeEEUncalibRecHitsOccupancyPlus_->Draw("colz"); 
	  myPV->PVCompute( meEEUncalibRecHitsOccupancyPlus_ , newmeEEUncalibRecHitsOccupancyPlus_ , te );       
	}

      if ( meEEUncalibRecHitsOccupancyMinus_ && newmeEEUncalibRecHitsOccupancyMinus_ ) 
	{
	  EndcapOccupancy->cd(3); meEEUncalibRecHitsOccupancyMinus_->Draw("colz"); 
	  EndcapOccupancy->cd(4); newmeEEUncalibRecHitsOccupancyMinus_->Draw("colz"); 
	  myPV->PVCompute( meEEUncalibRecHitsOccupancyMinus_ , newmeEEUncalibRecHitsOccupancyMinus_ , te );       
	}

      EndcapOccupancy->Print("EndcapOccupancy_compare.eps"); 
      delete EndcapOccupancy;


      TCanvas *EndcapOccupancyGt60adc = new TCanvas("EndcapOccupancyGt60adc","EndcapOccupancyGt60adc",1000,800);
      EndcapOccupancyGt60adc->Divide(2,2);
      if ( meEEUncalibRecHitsOccupancyPlusGt60adc_ && newmeEEUncalibRecHitsOccupancyPlusGt60adc_ ) 
	{
	  EndcapOccupancyGt60adc->cd(1); meEEUncalibRecHitsOccupancyPlusGt60adc_->Draw("colz"); 
	  EndcapOccupancyGt60adc->cd(2); newmeEEUncalibRecHitsOccupancyPlusGt60adc_->Draw("colz"); 
	  myPV->PVCompute( meEEUncalibRecHitsOccupancyPlusGt60adc_ , newmeEEUncalibRecHitsOccupancyPlusGt60adc_ , te );       
	}

      if ( meEEUncalibRecHitsOccupancyMinusGt60adc_ && newmeEEUncalibRecHitsOccupancyMinusGt60adc_ ) 
	{
	  EndcapOccupancyGt60adc->cd(3); meEEUncalibRecHitsOccupancyMinusGt60adc_->Draw("colz"); 
	  EndcapOccupancyGt60adc->cd(4); newmeEEUncalibRecHitsOccupancyMinusGt60adc_->Draw("colz"); 
	  myPV->PVCompute( meEEUncalibRecHitsOccupancyMinusGt60adc_ , newmeEEUncalibRecHitsOccupancyMinusGt60adc_ , te );       
	}

      EndcapOccupancyGt60adc->Print("EndcapOccupancyGt60adc_compare.eps"); 
      delete EndcapOccupancyGt60adc;


      TCanvas *EndcapFullMap = new TCanvas("EndcapFullMap","EndcapFullMap",1000,800);
      EndcapFullMap->Divide(2,2);
      if ( meEEUncalibRecHitsAmpFullMap_ && newmeEEUncalibRecHitsAmpFullMap_ )
	{
	  EndcapFullMap->cd(1); meEEUncalibRecHitsAmpFullMap_   ->Draw("colz"); 
	  EndcapFullMap->cd(2); newmeEEUncalibRecHitsAmpFullMap_->Draw("colz"); 
	  myPV->PVCompute( meEEUncalibRecHitsAmpFullMap_ , newmeEEUncalibRecHitsAmpFullMap_ , te );       

	  EndcapFullMap->cd(3); meEEUncalibRecHitsPedFullMap_   ->Draw("colz"); 
	  EndcapFullMap->cd(4); newmeEEUncalibRecHitsPedFullMap_->Draw("colz"); 
	  myPV->PVCompute( meEEUncalibRecHitsPedFullMap_ , newmeEEUncalibRecHitsPedFullMap_ , te );       
	}
      EndcapFullMap->Print("EndcapFullMap_compare.eps"); 
      delete EndcapFullMap;


      TCanvas *Endcap = new TCanvas("Endcap","Endcap",800,800);
      Endcap->Divide(2,2);
      if ( meEEUncalibRecHitsAmplitude_ && newmeEEUncalibRecHitsAmplitude_ )
	{
	  Endcap->cd(1); 
	  gPad->SetLogy();  
	  meEEUncalibRecHitsAmplitude_->Draw(); 
	  newmeEEUncalibRecHitsAmplitude_->Draw("same"); 
	  myPV->PVCompute( meEEUncalibRecHitsAmplitude_ , newmeEEUncalibRecHitsAmplitude_ , te );       
	}

      if ( meEEUncalibRecHitsPedestal_ && newmeEEUncalibRecHitsPedestal_ )
	{
	  Endcap->cd(2);  
	  meEEUncalibRecHitsPedestal_->Draw(); 
	  newmeEEUncalibRecHitsPedestal_->Draw("same");
	  myPV->PVCompute( meEEUncalibRecHitsPedestal_ , newmeEEUncalibRecHitsPedestal_ , te );        
	}

      if ( meEEUncalibRecHitsJitter_ && newmeEEUncalibRecHitsJitter_ )
	{
	  Endcap->cd(3);  
	  meEEUncalibRecHitsJitter_->Draw(); 
	  newmeEEUncalibRecHitsJitter_->Draw("same");
	  myPV->PVCompute( meEEUncalibRecHitsJitter_ , newmeEEUncalibRecHitsJitter_ , te );       
	}

      if ( meEEUncalibRecHitsChi2_ && newmeEEUncalibRecHitsChi2_ )
	{
	  Endcap->cd(4);  
	  meEEUncalibRecHitsChi2_->Draw(); 
	  newmeEEUncalibRecHitsChi2_->Draw("same"); 
	  myPV->PVCompute( meEEUncalibRecHitsChi2_ , newmeEEUncalibRecHitsChi2_ , te );       
	}
      Endcap->Print("Endcap_compare.eps");
      delete Endcap;


      TCanvas *EndcapGt60adc = new TCanvas("EndcapGt60adc","EndcapGt60adc",800,800);
      EndcapGt60adc->Divide(2,2);
      if ( meEEUncalibRecHitsAmplitudeGt60adc_ && newmeEEUncalibRecHitsAmplitudeGt60adc_ )
	{
	  EndcapGt60adc->cd(1); 
	  gPad->SetLogy();  
	  meEEUncalibRecHitsAmplitudeGt60adc_->Draw(); 
	  newmeEEUncalibRecHitsAmplitudeGt60adc_->Draw("same"); 
	  myPV->PVCompute( meEEUncalibRecHitsAmplitudeGt60adc_ , newmeEEUncalibRecHitsAmplitudeGt60adc_ , te );       
	}

      if ( meEEUncalibRecHitsPedestalGt60adc_ && newmeEEUncalibRecHitsPedestalGt60adc_ )
	{
	  EndcapGt60adc->cd(2);  
	  meEEUncalibRecHitsPedestalGt60adc_->Draw(); 
	  newmeEEUncalibRecHitsPedestalGt60adc_->Draw("same");
	  myPV->PVCompute( meEEUncalibRecHitsPedestalGt60adc_ , newmeEEUncalibRecHitsPedestalGt60adc_ , te );        
	}

      if ( meEEUncalibRecHitsJitterGt60adc_ && newmeEEUncalibRecHitsJitterGt60adc_ )
	{
	  EndcapGt60adc->cd(3);  
	  meEEUncalibRecHitsJitterGt60adc_->Draw(); 
	  newmeEEUncalibRecHitsJitterGt60adc_->Draw("same");
	  myPV->PVCompute( meEEUncalibRecHitsJitterGt60adc_ , newmeEEUncalibRecHitsJitterGt60adc_ , te );       
	}

      if ( meEEUncalibRecHitsChi2Gt60adc_ && newmeEEUncalibRecHitsChi2Gt60adc_ )
	{
	  EndcapGt60adc->cd(4);  
	  meEEUncalibRecHitsChi2Gt60adc_->Draw(); 
	  newmeEEUncalibRecHitsChi2Gt60adc_->Draw("same"); 
	  myPV->PVCompute( meEEUncalibRecHitsChi2Gt60adc_ , newmeEEUncalibRecHitsChi2Gt60adc_ , te );       
	}
      EndcapGt60adc->Print("EndcapGt60adc_compare.eps");
      delete EndcapGt60adc;
   }







 // ----------------------------------------------------------------
 cout << endl;
 cout << "Preshower validation" << endl;
 rfile->cd("DQMData/EcalPreshowerRecHitsTask");
 gDirectory->ls();
 sfile->cd("DQMData/EcalPreshowerRecHitsTask");
 gDirectory->ls();

  
 // Preshower validation
 if (1) 
   {    
     TH1 *meESRecHitsEnergy_;
     rfile->GetObject("DQMData/EcalPreshowerRecHitsTask/ES Energy;1",meESRecHitsEnergy_);
     meESRecHitsEnergy_;
     meESRecHitsEnergy_->SetLineColor(rcolor);
     
     TH1 *meESRecHitsEnergy_zp1st_;
     rfile->GetObject("DQMData/EcalPreshowerRecHitsTask/ES Energy Plane1 Side+;1",meESRecHitsEnergy_zp1st_);
     meESRecHitsEnergy_zp1st_;
     meESRecHitsEnergy_zp1st_->SetLineColor(rcolor);

     TH1 *meESRecHitsEnergy_zp2nd_;
     rfile->GetObject("DQMData/EcalPreshowerRecHitsTask/ES Energy Plane2 Side+;1",meESRecHitsEnergy_zp2nd_);
     meESRecHitsEnergy_zp2nd_;
     meESRecHitsEnergy_zp2nd_->SetLineColor(rcolor);

     TH1 *meESRecHitsEnergy_zm1st_;
     rfile->GetObject("DQMData/EcalPreshowerRecHitsTask/ES Energy Plane1 Side-;1",meESRecHitsEnergy_zm1st_);
     meESRecHitsEnergy_zm1st_;
     meESRecHitsEnergy_zm1st_->SetLineColor(rcolor);

     TH1 *meESRecHitsEnergy_zm2nd_;
     rfile->GetObject("DQMData/EcalPreshowerRecHitsTask/ES Energy Plane2 Side-;1",meESRecHitsEnergy_zm2nd_);
     meESRecHitsEnergy_zm2nd_;
     meESRecHitsEnergy_zm2nd_->SetLineColor(rcolor);

     TH1 *meESRecHitsMultip_;
     rfile->GetObject("DQMData/EcalPreshowerRecHitsTask/ES Multiplicity;1",meESRecHitsMultip_);
     meESRecHitsMultip_;
     meESRecHitsMultip_->SetLineColor(rcolor);

     TH1 *meESRecHitsMultip_zp1st_;
     rfile->GetObject("DQMData/EcalPreshowerRecHitsTask/ES Multiplicity Plane1 Side+;1",meESRecHitsMultip_zp1st_);
     meESRecHitsMultip_zp1st_;
     meESRecHitsMultip_zp1st_->SetLineColor(rcolor);

     TH1 *meESRecHitsMultip_zp2nd_;
     rfile->GetObject("DQMData/EcalPreshowerRecHitsTask/ES Multiplicity Plane2 Side+;1",meESRecHitsMultip_zp2nd_);
     meESRecHitsMultip_zp2nd_;
     meESRecHitsMultip_zp2nd_->SetLineColor(rcolor);

     TH1 *meESRecHitsMultip_zm1st_;
     rfile->GetObject("DQMData/EcalPreshowerRecHitsTask/ES Multiplicity Plane1 Side-;1",meESRecHitsMultip_zm1st_);
     meESRecHitsMultip_zm1st_;
     meESRecHitsMultip_zm1st_->SetLineColor(rcolor);

     TH1 *meESRecHitsMultip_zm2nd_;
     rfile->GetObject("DQMData/EcalPreshowerRecHitsTask/ES Multiplicity Plane1 Side-;1",meESRecHitsMultip_zm2nd_);
     meESRecHitsMultip_zm2nd_;
     meESRecHitsMultip_zm2nd_->SetLineColor(rcolor);

     TH1 *meESEERecHitsEnergy_zp_;
     rfile->GetObject("DQMData/EcalPreshowerRecHitsTask/Preshower EE vs ES energy Side+;1",meESEERecHitsEnergy_zp_);
     meESEERecHitsEnergy_zp_;
  
     TH1 *meESEERecHitsEnergy_zm_;
     rfile->GetObject("DQMData/EcalPreshowerRecHitsTask/Preshower EE vs ES energy Side-;1",meESEERecHitsEnergy_zm_);
     meESEERecHitsEnergy_zm_;

     TH1 *newmeESRecHitsEnergy_;
     sfile->GetObject("DQMData/EcalPreshowerRecHitsTask/ES Energy;1",newmeESRecHitsEnergy_);
     newmeESRecHitsEnergy_;
     newmeESRecHitsEnergy_->SetLineColor(scolor);
     
     TH1 *newmeESRecHitsEnergy_zp1st_;
     sfile->GetObject("DQMData/EcalPreshowerRecHitsTask/ES Energy Plane1 Side+;1",newmeESRecHitsEnergy_zp1st_);
     newmeESRecHitsEnergy_zp1st_;
     newmeESRecHitsEnergy_zp1st_->SetLineColor(scolor);

     TH1 *newmeESRecHitsEnergy_zp2nd_;
     sfile->GetObject("DQMData/EcalPreshowerRecHitsTask/ES Energy Plane2 Side+;1",newmeESRecHitsEnergy_zp2nd_);
     newmeESRecHitsEnergy_zp2nd_;
     newmeESRecHitsEnergy_zp2nd_->SetLineColor(scolor);

     TH1 *newmeESRecHitsEnergy_zm1st_;
     sfile->GetObject("DQMData/EcalPreshowerRecHitsTask/ES Energy Plane1 Side-;1",newmeESRecHitsEnergy_zm1st_);
     newmeESRecHitsEnergy_zm1st_;
     newmeESRecHitsEnergy_zm1st_->SetLineColor(scolor);

     TH1 *newmeESRecHitsEnergy_zm2nd_;
     sfile->GetObject("DQMData/EcalPreshowerRecHitsTask/ES Energy Plane2 Side-;1",newmeESRecHitsEnergy_zm2nd_);
     newmeESRecHitsEnergy_zm2nd_;
     newmeESRecHitsEnergy_zm2nd_->SetLineColor(scolor);

     TH1 *newmeESRecHitsMultip_;
     sfile->GetObject("DQMData/EcalPreshowerRecHitsTask/ES Multiplicity;1",newmeESRecHitsMultip_);
     newmeESRecHitsMultip_;
     newmeESRecHitsMultip_->SetLineColor(scolor);

     TH1 *newmeESRecHitsMultip_zp1st_;
     sfile->GetObject("DQMData/EcalPreshowerRecHitsTask/ES Multiplicity Plane1 Side+;1",newmeESRecHitsMultip_zp1st_);
     newmeESRecHitsMultip_zp1st_;
     newmeESRecHitsMultip_zp1st_->SetLineColor(scolor);

     TH1 *newmeESRecHitsMultip_zp2nd_;
     sfile->GetObject("DQMData/EcalPreshowerRecHitsTask/ES Multiplicity Plane2 Side+;1",newmeESRecHitsMultip_zp2nd_);
     newmeESRecHitsMultip_zp2nd_;
     newmeESRecHitsMultip_zp2nd_->SetLineColor(scolor);

     TH1 *newmeESRecHitsMultip_zm1st_;
     sfile->GetObject("DQMData/EcalPreshowerRecHitsTask/ES Multiplicity Plane1 Side-;1",newmeESRecHitsMultip_zm1st_);
     newmeESRecHitsMultip_zm1st_;
     newmeESRecHitsMultip_zm1st_->SetLineColor(scolor);

     TH1 *newmeESRecHitsMultip_zm2nd_;
     sfile->GetObject("DQMData/EcalPreshowerRecHitsTask/ES Multiplicity Plane1 Side-;1",newmeESRecHitsMultip_zm2nd_);
     newmeESRecHitsMultip_zm2nd_;
     newmeESRecHitsMultip_zm2nd_->SetLineColor(scolor);

     TH1 *newmeESEERecHitsEnergy_zp_;
     sfile->GetObject("DQMData/EcalPreshowerRecHitsTask/Preshower EE vs ES energy Side+;1",newmeESEERecHitsEnergy_zp_);
     newmeESEERecHitsEnergy_zp_;
  
     TH1 *newmeESEERecHitsEnergy_zm_;
     sfile->GetObject("DQMData/EcalPreshowerRecHitsTask/Preshower EE vs ES energy Side-;1",newmeESEERecHitsEnergy_zm_);
     newmeESEERecHitsEnergy_zm_;

     
     
     // ----------------------------------------------
     TCanvas *ESEnergyAll = new TCanvas("ESEnergyAll","ESEnergyAll",800,800);
     gPad->SetLogy(); 
     if ( meESRecHitsEnergy_ && newmeESRecHitsEnergy_ ) 
       {
	 meESRecHitsEnergy_->Draw(); 
	 newmeESRecHitsEnergy_->Draw("same");
	 myPV->PVCompute( meESRecHitsEnergy_ , newmeESRecHitsEnergy_ , te );       
       }
     ESEnergyAll->Print("PreshowerEnergyAll_compare.eps");
     delete ESEnergyAll;


     TCanvas *ESEnergy = new TCanvas("ESEnergy","ESEnergy",800,800);
     ESEnergy->Divide(2,2);
     if ( meESRecHitsEnergy_zp1st_ && newmeESRecHitsEnergy_zp1st_ )
       {
	 ESEnergy->cd(1); gPad->SetLogy(); 
	 meESRecHitsEnergy_zp1st   _->Draw(); 
	 newmeESRecHitsEnergy_zp1st_->Draw("same"); 
       }
     if ( meESRecHitsEnergy_zp2nd_ && newmeESRecHitsEnergy_zp2nd_ )
       {
	 ESEnergy->cd(2); gPad->SetLogy(); 
	 meESRecHitsEnergy_zp2nd_   ->Draw(); 
	 newmeESRecHitsEnergy_zp2nd_->Draw("same"); 
       }
     if ( meESRecHitsEnergy_zm1st_ && newmeESRecHitsEnergy_zm1st_ )
       {
	 ESEnergy->cd(3); gPad->SetLogy(); 
	 meESRecHitsEnergy_zm1st_   ->Draw(); 
	 newmeESRecHitsEnergy_zm1st_->Draw("same"); 
       }
     if ( meESRecHitsEnergy_zm2nd_ && newmeESRecHitsEnergy_zm2nd_ )
       {
	 ESEnergy->cd(4); gPad->SetLogy(); 
	 meESRecHitsEnergy_zm2nd_   ->Draw(); 
	 newmeESRecHitsEnergy_zm2nd_->Draw("same"); 
       }
     ESEnergy->Print("PreshowerEnergy_compare.eps");
     delete ESEnergy;


     TCanvas *ESMultipAll = new TCanvas("ESMultipAll","ESMultipAll",800,800);
     if ( meESRecHitsMultip_ && newmeESRecHitsMultip_ ) 
       {
	 meESRecHitsMultip_->Draw(); 
	 newmeESRecHitsMultip_->Draw("same");
	 myPV->PVCompute( meESRecHitsMultip_ , newmeESRecHitsMultip_ , te );       
       }
     ESMultipAll->Print("PreshowerMultipAll_compare.eps");
     delete ESMultipAll;



     TCanvas *ESMultip = new TCanvas("ESMultip","ESMultip",800,800);
     ESMultip->Divide(2,2);
     if ( meESRecHitsMultip_zp1st_ && newmeESRecHitsMultip_zp1st_ )
       {
	 ESMultip->cd(1); 
	 meESRecHitsMultip_zp1st   _->Draw(); 
	 newmeESRecHitsMultip_zp1st_->Draw("same"); 
       }
     if ( meESRecHitsMultip_zp2nd_ && newmeESRecHitsMultip_zp2nd_ )
       {
	 ESMultip->cd(2); 
	 meESRecHitsMultip_zp2nd_   ->Draw(); 
	 newmeESRecHitsMultip_zp2nd_->Draw("same"); 
       }
     if ( meESRecHitsMultip_zm1st_ && newmeESRecHitsMultip_zm1st_ )
       {
	 ESMultip->cd(3); 
	 meESRecHitsMultip_zm1st_   ->Draw(); 
	 newmeESRecHitsMultip_zm1st_->Draw("same"); 
       }
     if ( meESRecHitsMultip_zm2nd_ && newmeESRecHitsMultip_zm2nd_ )
       {
	 ESMultip->cd(4); 
	 meESRecHitsMultip_zm2nd_   ->Draw(); 
	 newmeESRecHitsMultip_zm2nd_->Draw("same"); 
       }
     ESMultip->Print("PreshowerMultip_compare.eps");
     delete ESMultip;



     TCanvas *ESvsEE = new TCanvas("ESvsEE","ESvsEE",800,800);
     ESvsEE->Divide(2,2);
     if ( meESEERecHitsEnergy_zp_ && newmeESEERecHitsEnergy_zp_ )
       {
	 ESvsEE->cd(1);   meESEERecHitsEnergy_zp   _->Draw();
	 ESvsEE->cd(2);   newmeESEERecHitsEnergy_zp_->Draw();
	 myPV->PVCompute( meESEERecHitsEnergy_zp_ , newmeESEERecHitsEnergy_zp_ , te);
       }
     if ( meESEERecHitsEnergy_zm_ && newmeESEERecHitsEnergy_zm_ )
       {
	 ESvsEE->cd(3);   meESEERecHitsEnergy_zm   _->Draw();
	 ESvsEE->cd(4);   newmeESEERecHitsEnergy_zm_->Draw();
	 myPV->PVCompute( meESEERecHitsEnergy_zm_ , newmeESEERecHitsEnergy_zm_ , te);
       }
     ESvsEE->Print("ESvsEE_compare.eps");
     delete ESvsEE;
   }
 

}

