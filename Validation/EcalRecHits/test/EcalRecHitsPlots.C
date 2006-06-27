#include "TFile.h"
#include "TTree.h"
#include "TText.h"

void EcalRecHitsPlots()
{  
  gROOT ->Reset();
  char*  rfilename = "EcalRecHitsValidation.root";
  
  delete gROOT->GetListOfFiles()->FindObject(rfilename);

  TText *te    = new TText();
  TFile *rfile = new TFile(rfilename);

  TF1 *gausa = new TF1 ("gausa","([0]*exp(-1*(x-[1])*(x-[1])/2/[2]/[2]))",0.,5.);

  cout << "General validation" << endl;
  rfile->cd("DQMData/EcalRecHitsTask");
  gDirectory->ls();
  
  Char_t histo[200];
  
  gStyle->SetOptStat("nemruoi");

  //////////////////////////////////////////////////////////////


  // ---------------------------------------------------------
  // General class: particle Gun 
  if (1) 
    {
      TH1* meGunEnergy_;
      rfile->GetObject("DQMData/EcalRecHitsTask/EcalRecHitsTask, Gun Momentum;1",meGunEnergy_);
      meGunEnergy_;
      
      TH1* meGunEta_;
      rfile->GetObject("DQMData/EcalRecHitsTask/EcalRecHitsTask, Gun Eta;1",meGunEta_);
      meGunEta_;
      
      TH1* meGunPhi_; 
      rfile->GetObject("DQMData/EcalRecHitsTask/EcalRecHitsTask, Gun Phi;1",meGunPhi_);
      meGunPhi_; 
      
      
      // --------------------------------
      TCanvas *Ecal = new TCanvas("Ecal","Ecal",1000,800);
      Ecal->Divide(1,3);
      Ecal->cd(1); if ( meGunEnergy_ ) meGunEnergy_->Draw(); 
      Ecal->cd(2); if ( meGunEta_ )    meGunEta_->Draw(); 
      Ecal->cd(3); if ( meGunPhi_ )    meGunPhi_->Draw(); 
      Ecal->Print("ParticleGun.eps");
      delete Ecal;
    }
  



  // General class: sim/rec hit ratios
  if (1) 
    {
      TH1 *meEBRecHitSimHitRatio_;
      rfile->GetObject("DQMData/EcalRecHitsTask/EcalRecHitsTask, Barrel RecSimHit Ratio;1",meEBRecHitSimHitRatio_);
      meEBRecHitSimHitRatio_;
      
      TH1 *meEERecHitSimHitRatio_;
      rfile->GetObject("DQMData/EcalRecHitsTask/EcalRecHitsTask, Endcap RecSimHit Ratio;1",meEERecHitSimHitRatio_);
      meEERecHitSimHitRatio_;

      TH1 *meESRecHitSimHitRatio_;
      rfile->GetObject("DQMData/EcalRecHitsTask/EcalRecHitsTask, Preshower RecSimHit Ratio;1",meESRecHitSimHitRatio_);
      meESRecHitSimHitRatio_;

      TH1 *meEBRecHitSimHitRatioGt35_;
      rfile->GetObject("DQMData/EcalRecHitsTask/EcalRecHitsTask, Barrel RecSimHit Ratio gt 3.5 GeV;1",meEBRecHitSimHitRatioGt35_);
      meEBRecHitSimHitRatioGt35_;
      
      TH1 *meEERecHitSimHitRatioGt35_;
      rfile->GetObject("DQMData/EcalRecHitsTask/EcalRecHitsTask, Endcap RecSimHit Ratio gt 3.5 GeV;1",meEERecHitSimHitRatioGt35_);
      meEERecHitSimHitRatioGt35_;


      
      
      // --------------------------------
      TCanvas *BarrelSimRec = new TCanvas("BarrelSimRec","BarrelSimRec",800,800);
      if ( meEBRecHitSimHitRatio_ ) { meEBRecHitSimHitRatio_->Draw(); }
      BarrelSimRec->Print("Barrel_SimRecHitsRatio.eps");
      delete BarrelSimRec;

      TCanvas *EndcapSimRec = new TCanvas("EndcapSimRec","EndcapSimRec",800,800);
      if ( meEERecHitSimHitRatio_ ) { meEERecHitSimHitRatio_->Draw(); }
      EndcapSimRec->Print("Endcap_SimRecHitsRatio.eps");
      delete EndcapSimRec;

      TCanvas *PreshowerSimRec = new TCanvas("PreshowerSimRec","PreshowerSimRec",800,800);
      if ( meESRecHitSimHitRatio_ ) meESRecHitSimHitRatio_->Draw();
      PreshowerSimRec->Print("Preshower_SimRecHitsRatio.eps");
      delete PreshowerSimRec;

      TCanvas *BarrelSimRecGt35 = new TCanvas("BarrelSimRecGt35","BarrelSimRecGt35",800,800);
      if ( meEBRecHitSimHitRatioGt35_ )
	{ 
	  meEBRecHitSimHitRatioGt35_->Draw(); 
	  double ebMean = meEBRecHitSimHitRatioGt35_-> GetMean();
	  double ebRms  = meEBRecHitSimHitRatioGt35_-> GetRMS();
	  double ebNorm = meEBRecHitSimHitRatioGt35_-> GetMaximum();
	  gausa->SetParameters(ebNorm, ebMean, ebRms);
	  meEBRecHitSimHitRatioGt35_->Fit("gausa", "", "", ebMean-3*ebRms, ebMean+3*ebRms);
	}
      BarrelSimRecGt35->Print("Barrel_SimRecHitsRatioGt35.eps");
      delete BarrelSimRecGt35;

      TCanvas *EndcapSimRecGt35 = new TCanvas("EndcapSimRecGt35","EndcapSimRecGt35",800,800);
      if ( meEERecHitSimHitRatioGt35_ )
	{ 
	  meEERecHitSimHitRatioGt35_->Draw(); 
	  double eeMean = meEERecHitSimHitRatioGt35_-> GetMean();
	  double eeRms  = meEERecHitSimHitRatioGt35_-> GetRMS();
	  double eeNorm = meEERecHitSimHitRatioGt35_-> GetMaximum();
	  gausa->SetParameters(eeNorm, eeMean, eeRms);
	  meEERecHitSimHitRatioGt35_->Fit("gausa", "", "", eeMean-3*eeRms, eeMean+3*eeRms);
	}
      EndcapSimRecGt35->Print("Endcap_SimRecHitsRatioGt35.eps");
      delete EndcapSimRecGt35;
    }
  




  // ----------------------------------------------------------
  cout << endl;
  cout << "Barrel validation" << endl;
  rfile->cd("DQMData/EcalBarrelRecHitsTask");
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
      
      TH1 *meEBUncalibRecHitsPedestal_;
      rfile->GetObject("DQMData/EcalBarrelRecHitsTask/EB Pedestal;1",meEBUncalibRecHitsPedestal_);
      meEBUncalibRecHitsPedestal_;
      
      TH1 *meEBUncalibRecHitsJitter_;
      rfile->GetObject("DQMData/EcalBarrelRecHitsTask/EB Jitter;1",meEBUncalibRecHitsJitter_);
      meEBUncalibRecHitsJitter_;
      
      TH1 *meEBUncalibRecHitsChi2_;
      rfile->GetObject("DQMData/EcalBarrelRecHitsTask/EB Chi2;1",meEBUncalibRecHitsChi2_);
      meEBUncalibRecHitsChi2_;

      TH1 *meEBUncalibRecHitMaxSampleRatio_;
      rfile->GetObject("DQMData/EcalBarrelRecHitsTask/EB RecHit Max Sample Ratio;1",meEBUncalibRecHitMaxSampleRatio_);
      meEBUncalibRecHitMaxSampleRatio_;

      TH2 *meEBUncalibRecHitsOccupancyGt100adc_;
      rfile->GetObject("DQMData/EcalBarrelRecHitsTask/EB Occupancy gt 100 adc counts;1",meEBUncalibRecHitsOccupancyGt100adc_);
      meEBUncalibRecHitsOccupancyGt100adc_;
      
      TH1 *meEBUncalibRecHitsAmplitudeGt100adc_;
      rfile->GetObject("DQMData/EcalBarrelRecHitsTask/EB Amplitude gt 100 adc counts;1",meEBUncalibRecHitsAmplitudeGt100adc_);
      meEBUncalibRecHitsAmplitudeGt100adc_;
      
      TH1 *meEBUncalibRecHitsPedestalGt100adc_;
      rfile->GetObject("DQMData/EcalBarrelRecHitsTask/EB Pedestal gt 100 adc counts;1",meEBUncalibRecHitsPedestalGt100adc_);
      meEBUncalibRecHitsPedestalGt100adc_;
      
      TH1 *meEBUncalibRecHitsJitterGt100adc_;
      rfile->GetObject("DQMData/EcalBarrelRecHitsTask/EB Jitter gt 100 adc counts;1",meEBUncalibRecHitsJitterGt100adc_);
      meEBUncalibRecHitsJitterGt100adc_;
      
      TH1 *meEBUncalibRecHitsChi2Gt100adc_;
      rfile->GetObject("DQMData/EcalBarrelRecHitsTask/EB Chi2 gt 100 adc counts;1",meEBUncalibRecHitsChi2Gt100adc_);
      meEBUncalibRecHitsChi2Gt100adc_;

      TH1 *meEBUncalibRecHitMaxSampleRatioGt100adc_;
      rfile->GetObject("DQMData/EcalBarrelRecHitsTask/EB RecHit Max Sample Ratio gt 100 adc counts;1",meEBUncalibRecHitMaxSampleRatioGt100adc_);
      meEBUncalibRecHitMaxSampleRatioGt100adc_;

      TProfile2D *meEBUncalibRecHitsAmpFullMap_;
      rfile->GetObject("DQMData/EcalBarrelRecHitsTask/EB Amplitude Full Map;1",meEBUncalibRecHitsAmpFullMap_);
      meEBUncalibRecHitsAmpFullMap_;
      
      TProfile2D *meEBUncalibRecHitsPedFullMap_;
      rfile->GetObject("DQMData/EcalBarrelRecHitsTask/EB Pedestal Full Map;1",meEBUncalibRecHitsPedFullMap_);
      meEBUncalibRecHitsPedFullMap_;
      
      TProfile2D *meEBUncalibRecHitAmplMap_[36];
      TProfile2D *meEBUncalibRecHitPedMap_[36];
      for (int ii=0; ii<36 ; ii++) 
	{
	  sprintf(histo, "DQMData/EcalBarrelRecHitsTask/EB Amp SM%02d;1", ii+1);
	  rfile->GetObject(histo,meEBUncalibRecHitAmplMap_[ii]);
	  meEBUncalibRecHitAmplMap_[ii];
	  
	  sprintf(histo, "DQMData/EcalBarrelRecHitsTask/EB Ped SM%02d;1", ii+1);
	  rfile->GetObject(histo,meEBUncalibRecHitPedMap_[ii]);
	  meEBUncalibRecHitPedMap_[ii];
	}
      



      // --------------------------------
      TCanvas *BarrelOccupancy = new TCanvas("BarrelOccupancy","BarrelOccupancy",1000,800);
      if ( meEBUncalibRecHitsOccupancy_ ) meEBUncalibRecHitsOccupancy_->Draw("colz"); 
      BarrelOccupancy->Print("BarrelOccupancy.eps"); 
      delete BarrelOccupancy;

      TCanvas *BarrelOccupancyGt100adc = new TCanvas("BarrelOccupancyGt100adc","BarrelOccupancyGt100adc",1000,800);
      if ( meEBUncalibRecHitsOccupancyGt100adc_ ) meEBUncalibRecHitsOccupancyGt100adc_->Draw("colz"); 
      BarrelOccupancyGt100adc->Print("BarrelOccupancyGt100adc.eps"); 
      delete BarrelOccupancyGt100adc;

      TCanvas *Barrel = new TCanvas("Barrel","Barrel",800,800);
      Barrel->Divide(3,2);
      Barrel->cd(1); 
      gPad->SetLogy();   if ( meEBUncalibRecHitsAmplitude_ )     meEBUncalibRecHitsAmplitude_->Draw(); 
      Barrel->cd(2);     if ( meEBUncalibRecHitsPedestal_ )      meEBUncalibRecHitsPedestal_->Draw(); 
      Barrel->cd(3);     if ( meEBUncalibRecHitsJitter_ )        meEBUncalibRecHitsJitter_->Draw(); 
      Barrel->cd(4);     if ( meEBUncalibRecHitsChi2_ )          meEBUncalibRecHitsChi2_->Draw(); 
      Barrel->cd(5);     if ( meEBUncalibRecHitMaxSampleRatio_ ) meEBUncalibRecHitMaxSampleRatio_->Draw(); 
      Barrel->Print("Barrel.eps");
      delete Barrel;

      TCanvas *BarrelGt100adc = new TCanvas("BarrelGt100adc","BarrelGt100adc",800,800);
      BarrelGt100adc->Divide(3,2);
      BarrelGt100adc->cd(1); 
      gPad->SetLogy();           if ( meEBUncalibRecHitsAmplitudeGt100adc_ )     meEBUncalibRecHitsAmplitudeGt100adc_->Draw(); 
      BarrelGt100adc->cd(2);     if ( meEBUncalibRecHitsPedestalGt100adc_ )      meEBUncalibRecHitsPedestalGt100adc_->Draw(); 
      BarrelGt100adc->cd(3);     if ( meEBUncalibRecHitsJitterGt100adc_ )        meEBUncalibRecHitsJitterGt100adc_->Draw(); 
      BarrelGt100adc->cd(4);     if ( meEBUncalibRecHitsChi2Gt100adc_ )          meEBUncalibRecHitsChi2Gt100adc_->Draw(); 
      BarrelGt100adc->cd(5);     if ( meEBUncalibRecHitMaxSampleRatioGt100adc_ ) meEBUncalibRecHitMaxSampleRatioGt100adc_->Draw(); 
      BarrelGt100adc->Print("BarrelGt100adc.eps");
      delete BarrelGt100adc;

      TCanvas *BarrelFullMaps = new TCanvas("BarrelFullMaps","BarrelFullMaps",800,800);
      BarrelFullMaps->Divide(1,2);
      BarrelFullMaps->cd(1);  if ( meEBUncalibRecHitsAmpFullMap_ ) meEBUncalibRecHitsAmpFullMap_ ->Draw("colz");
      BarrelFullMaps->cd(2);  if ( meEBUncalibRecHitsPedFullMap_ ) meEBUncalibRecHitsPedFullMap_ ->Draw("colz");
      BarrelFullMaps->Print("BarrelFullMaps.eps");
      delete BarrelFullMaps;

      TCanvas *BarrelMapAmpl   = new TCanvas("BarrelMapAmpl","BarrelMapAmpl",800,800);
      TCanvas *BarrelMapPed    = new TCanvas("BarrelMapPed","BarrelMapPed",800,800);
      BarrelMapAmpl->Divide(9,4);
      BarrelMapPed ->Divide(9,4);
      for (int ii=0; ii<36; ii++) 
	{
	  BarrelMapAmpl->cd(ii+1); if( meEBUncalibRecHitAmplMap_[ii] )  meEBUncalibRecHitAmplMap_[ii]->Draw("colz"); 
	  BarrelMapPed ->cd(ii+1); if( meEBUncalibRecHitPedMap_[ii] )   meEBUncalibRecHitPedMap_[ii]->Draw("colz"); 
	} 
      BarrelMapAmpl->Print("BarrelAmplitudeMap.eps"); 
      BarrelMapPed ->Print("BarrelPedestalMap.eps");
      delete BarrelMapAmpl;
      delete BarrelMapPed;
    }






  // -------------------------------------------------------
  cout << endl;
  cout << "Endcap validation" << endl;
  rfile->cd("DQMData/EcalEndcapRecHitsTask");
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

      TH1 *meEEUncalibRecHitsPedestal_;
      rfile->GetObject("DQMData/EcalEndcapRecHitsTask/EE Pedestal;1",meEEUncalibRecHitsPedestal_);
      meEEUncalibRecHitsPedestal_;
      
      TH1 *meEEUncalibRecHitsJitter_;
      rfile->GetObject("DQMData/EcalEndcapRecHitsTask/EE Jitter;1",meEEUncalibRecHitsJitter_);
      meEEUncalibRecHitsJitter_;
      
      TH1 *meEEUncalibRecHitsChi2_;
      rfile->GetObject("DQMData/EcalEndcapRecHitsTask/EE Chi2;1",meEEUncalibRecHitsChi2_);
      meEEUncalibRecHitsChi2_;
      
      TH1 *meEEUncalibRecHitMaxSampleRatio_;
      rfile->GetObject("DQMData/EcalEndcapRecHitsTask/EE RecHit Max Sample Ratio;1",meEEUncalibRecHitMaxSampleRatio_);
      meEEUncalibRecHitMaxSampleRatio_;

      TH2 *meEEUncalibRecHitsOccupancyPlusGt60adc_;
      rfile->GetObject("DQMData/EcalEndcapRecHitsTask/EE+ Occupancy gt 60 adc counts;1",meEEUncalibRecHitsOccupancyPlusGt60adc_);
      meEEUncalibRecHitsOccupancyPlusGt60adc_;

      TH2 *meEEUncalibRecHitsOccupancyMinusGt60adc_;
      rfile->GetObject("DQMData/EcalEndcapRecHitsTask/EE- Occupancy gt 60 adc counts;1",meEEUncalibRecHitsOccupancyMinusGt60adc_);
      meEEUncalibRecHitsOccupancyMinusGt60adc_;
      
      TH1 *meEEUncalibRecHitsAmplitudeGt60adc_;
      rfile->GetObject("DQMData/EcalEndcapRecHitsTask/EE Amplitude gt 60 adc counts;1",meEEUncalibRecHitsAmplitudeGt60adc_);
      meEEUncalibRecHitsAmplitudeGt60adc_;

      TH1 *meEEUncalibRecHitsPedestalGt60adc_;
      rfile->GetObject("DQMData/EcalEndcapRecHitsTask/EE Pedestal gt 60 adc counts;1",meEEUncalibRecHitsPedestalGt60adc_);
      meEEUncalibRecHitsPedestalGt60adc_;
      
      TH1 *meEEUncalibRecHitsJitterGt60adc_;
      rfile->GetObject("DQMData/EcalEndcapRecHitsTask/EE Jitter gt 60 adc counts;1",meEEUncalibRecHitsJitterGt60adc_);
      meEEUncalibRecHitsJitterGt60adc_;
      
      TH1 *meEEUncalibRecHitsChi2Gt60adc_;
      rfile->GetObject("DQMData/EcalEndcapRecHitsTask/EE Chi2 gt 60 adc counts;1",meEEUncalibRecHitsChi2Gt60adc_);
      meEEUncalibRecHitsChi2Gt60adc_;
      
      TH1 *meEEUncalibRecHitMaxSampleRatioGt60adc_;
      rfile->GetObject("DQMData/EcalEndcapRecHitsTask/EE RecHit Max Sample Ratio gt 60 adc counts;1",meEEUncalibRecHitMaxSampleRatioGt60adc_);
      meEEUncalibRecHitMaxSampleRatioGt60adc_;

      TProfile2D *meEEUncalibRecHitsAmpFullMap_;
      rfile->GetObject("DQMData/EcalEndcapRecHitsTask/EE Amplitude Full Map;1",meEEUncalibRecHitsAmpFullMap_);
      meEEUncalibRecHitsAmpFullMap_;

      TProfile2D *meEEUncalibRecHitsPedFullMap_;
      rfile->GetObject("DQMData/EcalEndcapRecHitsTask/EE Pedestal Full Map;1",meEEUncalibRecHitsPedFullMap_);
      meEEUncalibRecHitsPedFullMap_;




      // --------------------------------
      TCanvas *EndcapOccupancy = new TCanvas("EndcapOccupancy","EndcapOccupancy",1000,800);
      EndcapOccupancy->Divide(1,2);
      EndcapOccupancy->cd(1);  if ( meEEUncalibRecHitsOccupancyPlus_ )  meEEUncalibRecHitsOccupancyPlus_->Draw("colz"); 
      EndcapOccupancy->cd(2);  if ( meEEUncalibRecHitsOccupancyMinus_ ) meEEUncalibRecHitsOccupancyMinus_->Draw("colz"); 
      EndcapOccupancy->Print("EndcapOccupancy.eps"); 
      delete EndcapOccupancy;

      TCanvas *EndcapOccupancyGt60adc = new TCanvas("EndcapOccupancyGt60adc","EndcapOccupancyGt60adc",1000,800);
      EndcapOccupancyGt60adc->Divide(1,2);
      EndcapOccupancyGt60adc->cd(1);  if ( meEEUncalibRecHitsOccupancyPlusGt60adc_ )  meEEUncalibRecHitsOccupancyPlusGt60adc_->Draw("colz"); 
      EndcapOccupancyGt60adc->cd(2);  if ( meEEUncalibRecHitsOccupancyMinusGt60adc_ ) meEEUncalibRecHitsOccupancyMinusGt60adc_->Draw("colz"); 
      EndcapOccupancyGt60adc->Print("EndcapOccupancyGt60adc.eps"); 
      delete EndcapOccupancyGt60adc;

      TCanvas *EndcapFullMaps = new TCanvas("EndcapFullMaps","EndcapFullMaps",800,800);
      EndcapFullMaps->Divide(1,2);
      EndcapFullMaps->cd(1);  if ( meEEUncalibRecHitsAmpFullMap_ ) meEEUncalibRecHitsAmpFullMap_ ->Draw("colz");
      EndcapFullMaps->cd(2);  if ( meEEUncalibRecHitsPedFullMap_ ) meEEUncalibRecHitsPedFullMap_ ->Draw("colz");
      EndcapFullMaps->Print("EndcapFullMaps.eps");
      delete EndcapFullMaps;

      TCanvas *Endcap = new TCanvas("Endcap","Endcap",800,800);
      Endcap->Divide(3,2);
      Endcap->cd(1); 
      gPad->SetLogy(); if ( meEEUncalibRecHitsAmplitude_ )     meEEUncalibRecHitsAmplitude_->Draw(); 
      Endcap->cd(2);   if ( meEEUncalibRecHitsPedestal_ )      meEEUncalibRecHitsPedestal_->Draw(); 
      Endcap->cd(3);   if ( meEEUncalibRecHitsJitter_ )        meEEUncalibRecHitsJitter_->Draw(); 
      Endcap->cd(4);   if ( meEEUncalibRecHitsChi2_ )          meEEUncalibRecHitsChi2_->Draw(); 
      Endcap->cd(5);   if ( meEEUncalibRecHitMaxSampleRatio_ ) meEEUncalibRecHitMaxSampleRatio_->Draw(); 
      Endcap->Print("Endcap.eps");
      delete Endcap;

      TCanvas *EndcapGt60adc = new TCanvas("EndcapGt60adc","EndcapGt60adc",800,800);
      EndcapGt60adc->Divide(3,2);
      EndcapGt60adc->cd(1); 
      gPad->SetLogy();        if ( meEEUncalibRecHitsAmplitudeGt60adc_ )     meEEUncalibRecHitsAmplitudeGt60adc_->Draw(); 
      EndcapGt60adc->cd(2);   if ( meEEUncalibRecHitsPedestalGt60adc_ )      meEEUncalibRecHitsPedestalGt60adc_->Draw(); 
      EndcapGt60adc->cd(3);   if ( meEEUncalibRecHitsJitterGt60adc_ )        meEEUncalibRecHitsJitterGt60adc_->Draw(); 
      EndcapGt60adc->cd(4);   if ( meEEUncalibRecHitsChi2Gt60adc_ )          meEEUncalibRecHitsChi2Gt60adc_->Draw(); 
      EndcapGt60adc->cd(5);   if ( meEEUncalibRecHitMaxSampleRatioGt60adc_ ) meEEUncalibRecHitMaxSampleRatioGt60adc_->Draw(); 
      EndcapGt60adc->Print("EndcapGt60adc.eps");
      delete EndcapGt60adc;
    }








  // ----------------------------------------------------------
  cout << endl;
  cout << "Preshower validation" << endl;
  rfile->cd("DQMData/EcalPreshowerRecHitsTask");
  gDirectory->ls();
  
  // Preshower validation
  if (1) 
    {    
      TH1 *meESRecHitsEnergy_;
      rfile->GetObject("DQMData/EcalPreshowerRecHitsTask/ES Energy;1",meESRecHitsEnergy_);
      meESRecHitsEnergy_;

      TH1 *meESRecHitsEnergy_zp1st_;
      rfile->GetObject("DQMData/EcalPreshowerRecHitsTask/ES Energy Plane1 Side+;1",meESRecHitsEnergy_zp1st_);
      meESRecHitsEnergy_zp1st_;

      TH1 *meESRecHitsEnergy_zp2nd_;
      rfile->GetObject("DQMData/EcalPreshowerRecHitsTask/ES Energy Plane2 Side+;1",meESRecHitsEnergy_zp2nd_);
      meESRecHitsEnergy_zp2nd_;

      TH1 *meESRecHitsEnergy_zm1st_;
      rfile->GetObject("DQMData/EcalPreshowerRecHitsTask/ES Energy Plane1 Side-;1",meESRecHitsEnergy_zm1st_);
      meESRecHitsEnergy_zm1st_;

      TH1 *meESRecHitsEnergy_zm2nd_;
      rfile->GetObject("DQMData/EcalPreshowerRecHitsTask/ES Energy Plane2 Side-;1",meESRecHitsEnergy_zm2nd_);
      meESRecHitsEnergy_zm2nd_;

      TH1 *meESRecHitsMultip_;
      rfile->GetObject("DQMData/EcalPreshowerRecHitsTask/ES Multiplicity;1",meESRecHitsMultip_);
      meESRecHitsMultip_;

      TH1 *meESRecHitsMultip_zp1st_;
      rfile->GetObject("DQMData/EcalPreshowerRecHitsTask/ES Multiplicity Plane1 Side+;1",meESRecHitsMultip_zp1st_);
      meESRecHitsMultip_zp1st_;

      TH1 *meESRecHitsMultip_zp2nd_;
      rfile->GetObject("DQMData/EcalPreshowerRecHitsTask/ES Multiplicity Plane2 Side+;1",meESRecHitsMultip_zp2nd_);
      meESRecHitsMultip_zp2nd_;

      TH1 *meESRecHitsMultip_zm1st_;
      rfile->GetObject("DQMData/EcalPreshowerRecHitsTask/ES Multiplicity Plane1 Side-;1",meESRecHitsMultip_zm1st_);
      meESRecHitsMultip_zm1st_;

      TH1 *meESRecHitsMultip_zm2nd_;
      rfile->GetObject("DQMData/EcalPreshowerRecHitsTask/ES Multiplicity Plane2 Side-;1",meESRecHitsMultip_zm2nd_);
      meESRecHitsMultip_zm2nd_;

      TH1 *meESEERecHitsEnergy_zp_;
      rfile->GetObject("DQMData/EcalPreshowerRecHitsTask/Preshower EE vs ES energy Side+;1",meESEERecHitsEnergy_zp_);
      meESEERecHitsEnergy_zp_;

      TH1 *meESEERecHitsEnergy_zm_;
      rfile->GetObject("DQMData/EcalPreshowerRecHitsTask/Preshower EE vs ES energy Side-;1",meESEERecHitsEnergy_zm_);
      meESEERecHitsEnergy_zm_;

      TH2 *meESRecHitsStripOccupancy_zp1st_[32];
      TH2 *meESRecHitsStripOccupancy_zp2nd_[32];
      TH2 *meESRecHitsStripOccupancy_zm1st_[32];
      TH2 *meESRecHitsStripOccupancy_zm2nd_[32];
      for (int kk=0; kk<32; kk++)
	{ 
	  sprintf(histo, "DQMData/EcalPreshowerRecHitsTask/ES Occupancy Plane1 Side+ Strip%02d", kk+1);
	  rfile->GetObject(histo,meESRecHitsStripOccupancy_zp1st_[kk]);
	  meESRecHitsStripOccupancy_zp1st_[kk];

	  sprintf(histo, "DQMData/EcalPreshowerRecHitsTask/ES Occupancy Plane2 Side+ Strip%02d", kk+1);
	  rfile->GetObject(histo,meESRecHitsStripOccupancy_zp2nd_[kk]);
	  meESRecHitsStripOccupancy_zp2nd_[kk];

	  sprintf(histo, "DQMData/EcalPreshowerRecHitsTask/ES Occupancy Plane1 Side- Strip%02d", kk+1);
	  rfile->GetObject(histo,meESRecHitsStripOccupancy_zm1st_[kk]);
	  meESRecHitsStripOccupancy_zm1st_[kk];

	  sprintf(histo, "DQMData/EcalPreshowerRecHitsTask/ES Occupancy Plane2 Side- Strip%02d", kk+1);
	  rfile->GetObject(histo,meESRecHitsStripOccupancy_zm2nd_[kk]);
	  meESRecHitsStripOccupancy_zm2nd_[kk];
	}




      // --------------------------------
      TCanvas *ESEnergyAll = new TCanvas("ESEnergyAll","ESEnergyAll",800,800);
      gPad->SetLogy(); 
      if ( meESRecHitsEnergy_ ) meESRecHitsEnergy_->Draw(); 
      ESEnergyAll->Print("PreshowerEnergyAll.eps");
      delete ESEnergyAll;


      TCanvas *ESEnergy = new TCanvas("ESEnergy","ESEnergy",800,800);
      ESEnergy->Divide(2,2);
      ESEnergy->cd(1); gPad->SetLogy(); 
      if ( meESRecHitsEnergy_zp1st_ ) meESRecHitsEnergy_zp1st_->Draw(); 
      ESEnergy->cd(2); gPad->SetLogy(); 
      if ( meESRecHitsEnergy_zp2nd_ ) meESRecHitsEnergy_zp2nd_->Draw(); 
      ESEnergy->cd(3); gPad->SetLogy(); 
      if ( meESRecHitsEnergy_zm1st_ ) meESRecHitsEnergy_zm1st_->Draw(); 
      ESEnergy->cd(4); gPad->SetLogy(); 
      if ( meESRecHitsEnergy_zm2nd_ ) meESRecHitsEnergy_zm2nd_->Draw(); 
      ESEnergy->Print("PreshowerEnergy.eps");
      delete ESEnergy;


      TCanvas *ESMultipAll = new TCanvas("ESMultipAll","ESMultipAll",800,800);
      if ( meESRecHitsMultip_ ) meESRecHitsMultip_->Draw(); 
      ESMultipAll->Print("PreshowerMultipAll.eps");
      delete ESMultipAll;


      TCanvas *ESMultip = new TCanvas("ESMultip","ESMultip",800,800);
      ESMultip->Divide(2,2);
      ESMultip->cd(1);  if ( meESRecHitsMultip_zp1st_ ) meESRecHitsMultip_zp1st_->Draw(); 
      ESMultip->cd(2);  if ( meESRecHitsMultip_zp2nd_ ) meESRecHitsMultip_zp2nd_->Draw(); 
      ESMultip->cd(3);  if ( meESRecHitsMultip_zm1st_ ) meESRecHitsMultip_zm1st_->Draw(); 
      ESMultip->cd(4);  if ( meESRecHitsMultip_zm2nd_ ) meESRecHitsMultip_zm2nd_->Draw(); 
      ESMultip->Print("PreshowerMultip.eps");
      delete ESMultip;

      
      TCanvas *ESvsEE = new TCanvas("ESvsEE","ESvsEE",800,800);
      ESvsEE->Divide(2,1);
      ESvsEE->cd(1);   if ( meESEERecHitsEnergy_zp_ ) meESEERecHitsEnergy_zp_->Draw();
      ESvsEE->cd(2);   if ( meESEERecHitsEnergy_zm_ ) meESEERecHitsEnergy_zm_->Draw();
      ESvsEE->Print("ESvsEE.eps");
      delete ESvsEE;


      TCanvas *ESOccupancy_zp1st = new TCanvas("ESOccupancy_zp1st", "EE occupancy, plane 1, side+", 800,800);
      TCanvas *ESOccupancy_zp2nd = new TCanvas("ESOccupancy_zp2nd", "EE occupancy, plane 2, side+", 800,800);
      TCanvas *ESOccupancy_zm1st = new TCanvas("ESOccupancy_zm1st", "EE occupancy, plane 1, side-", 800,800);
      TCanvas *ESOccupancy_zm2nd = new TCanvas("ESOccupancy_zm2nd", "EE occupancy, plane 2, side-", 800,800);
      ESOccupancy_zp1st -> Divide(8,4);
      ESOccupancy_zp2nd -> Divide(8,4);
      ESOccupancy_zm1st -> Divide(8,4);
      ESOccupancy_zm2nd -> Divide(8,4);
      for (int kk=0; kk<32; kk++)
	{
	  ESOccupancy_zp1st->cd(kk+1); if(meESRecHitsStripOccupancy_zp1st_[kk]) meESRecHitsStripOccupancy_zp1st_[kk]->Draw("colz"); 
	  ESOccupancy_zp2nd->cd(kk+1); if(meESRecHitsStripOccupancy_zp2nd_[kk]) meESRecHitsStripOccupancy_zp2nd_[kk]->Draw("colz"); 
	  ESOccupancy_zm1st->cd(kk+1); if(meESRecHitsStripOccupancy_zm1st_[kk]) meESRecHitsStripOccupancy_zm1st_[kk]->Draw("colz"); 
	  ESOccupancy_zm2nd->cd(kk+1); if(meESRecHitsStripOccupancy_zm2nd_[kk]) meESRecHitsStripOccupancy_zm2nd_[kk]->Draw("colz"); 
	} 
      ESOccupancy_zp1st -> Print("PreshowerOccupancy_zp_plane1.eps"); 
      ESOccupancy_zp2nd -> Print("PreshowerOccupancy_zp_plane2.eps"); 
      ESOccupancy_zm1st -> Print("PreshowerOccupancy_zm_plane1.eps"); 
      ESOccupancy_zm2nd -> Print("PreshowerOccupancy_zm_plane2.eps"); 
      delete ESOccupancy_zp1st;
      delete ESOccupancy_zp2nd;
      delete ESOccupancy_zm1st;
      delete ESOccupancy_zm2nd;
    }

}
