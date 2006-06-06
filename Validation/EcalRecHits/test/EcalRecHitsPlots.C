#include "TFile.h"
#include "TTree.h"
#include "TText.h"

void EcalRecHitsPlots()
{
  
  gROOT ->Reset();
  char*  rfilename = "EcalRecHitsValidation.root";
  
  delete gROOT->GetListOfFiles()->FindObject(rfilename);
  
  TText* te = new TText();
  TFile * rfile = new TFile(rfilename);

  cout << "General validation" << endl;
  rfile->cd("DQMData/EcalRecHitsTask");
  gDirectory->ls();
  
  Char_t histo[200];
  
  gStyle->SetOptStat("nemruoi");

  //////////////////////////////////////////////////////////////


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
      Ecal->cd(2); if ( meGunEta_ ) meGunEta_->Draw(); 
      Ecal->cd(3); if ( meGunPhi_ ) meGunPhi_->Draw(); 
      Ecal->Print("ParticleGun.eps");
      delete Ecal;
    }
  
  // General class: barrel, sim/rec hit ratios
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
      
      
      // --------------------------------
      TCanvas *BarrelSimRec = new TCanvas("BarrelSimRec","BarrelSimRec",800,800);
      if ( meEBRecHitSimHitRatio_ ) meEBRecHitSimHitRatio_->Draw();
      BarrelSimRec->Print("Barrel_SimRecHitsRatio.eps");
      delete BarrelSimRec;

      TCanvas *EndcapSimRec = new TCanvas("EndcapSimRec","EndcapSimRec",800,800);
      if ( meEERecHitSimHitRatio_ ) meEERecHitSimHitRatio_->Draw();
      EndcapSimRec->Print("Endcap_SimRecHitsRatio.eps");
      delete EndcapSimRec;

      TCanvas *PreshowerSimRec = new TCanvas("PreshowerSimRec","PreshowerSimRec",800,800);
      if ( meESRecHitSimHitRatio_ ) meESRecHitSimHitRatio_->Draw();
      PreshowerSimRec->Print("Preshower_SimRecHitsRatio.eps");
      delete PreshowerSimRec;
    }
  

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

      TCanvas *Barrel = new TCanvas("Barrel","Barrel",800,800);
      Barrel->Divide(2,2);
      Barrel->cd(1); if ( meEBUncalibRecHitsAmplitude_ ) meEBUncalibRecHitsAmplitude_->Draw(); 
      Barrel->cd(2); if ( meEBUncalibRecHitsPedestal_ )  meEBUncalibRecHitsPedestal_->Draw(); 
      Barrel->cd(3); if ( meEBUncalibRecHitsJitter_ )    meEBUncalibRecHitsJitter_->Draw(); 
      Barrel->cd(4); if ( meEBUncalibRecHitsChi2_ )      meEBUncalibRecHitsChi2_->Draw(); 
      Barrel->Print("Barrel.eps");
      delete Barrel;


      TCanvas *BarrelMapAmpl   = new TCanvas("BarrelMapAmpl","BarrelMapAmpl",800,800);
      TCanvas *BarrelMapPed    = new TCanvas("BarrelMapPed","BarrelMapPed",800,800);
      BarrelMapAmpl->Divide(9,4);
      BarrelMapPed->Divide(9,4);
      for (int ii=0; ii<36; ii++) 
	{
	  BarrelMapAmpl->cd(ii+1); if ( meEBUncalibRecHitAmplMap_[ii] ) meEBUncalibRecHitAmplMap_[ii]->Draw("colz"); 
	  BarrelMapPed->cd(ii+1);  if ( meEBUncalibRecHitPedMap_[ii] )  meEBUncalibRecHitPedMap_[ii]->Draw("colz"); 
	} 
      BarrelMapAmpl->Print("BarrelAmplitudeMap.eps"); 
      BarrelMapPed ->Print("BarrelPedestalMap.eps");
      delete BarrelMapAmpl;
      delete BarrelMapPed;
    }


  cout << endl;
  cout << "Endcap validation" << endl;
  rfile->cd("DQMData/EcalEndcapRecHitsTask");
  gDirectory->ls();



  // Endcap validation
  if (1) 
    {    
      TH2 *meEEUncalibRecHitsOccupancy_;
      rfile->GetObject("DQMData/EcalEndcapRecHitsTask/EE Occupancy;1",meEEUncalibRecHitsOccupancy_);
      meEEUncalibRecHitsOccupancy_;
      
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
      

      // --------------------------------
      TCanvas *EndcapOccupancy = new TCanvas("EndcapOccupancy","EndcapOccupancy",1000,800);
      if ( meEEUncalibRecHitsOccupancy_ ) meEEUncalibRecHitsOccupancy_->Draw("colz"); 
      EndcapOccupancy->Print("EndcapOccupancy.eps"); 
      delete EndcapOccupancy;

      TCanvas *Endcap = new TCanvas("Endcap","Endcap",800,800);
      Endcap->Divide(2,2);
      Endcap->cd(1); if ( meEEUncalibRecHitsAmplitude_ ) meEEUncalibRecHitsAmplitude_->Draw(); 
      Endcap->cd(2); if ( meEEUncalibRecHitsPedestal_ )  meEEUncalibRecHitsPedestal_->Draw(); 
      Endcap->cd(3); if ( meEEUncalibRecHitsJitter_ )    meEEUncalibRecHitsJitter_->Draw(); 
      Endcap->cd(4); if ( meEEUncalibRecHitsChi2_ )      meEEUncalibRecHitsChi2_->Draw(); 
      Endcap->Print("Endcap.eps");
      delete Endcap;
    }


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
      
      TH2 *meESRecHitsStripOccupancy_[2][2][32];
      for (int ii=0; ii<2; ii++) {
	for (int jj=0; jj<2; jj++) {
	  int pp;
	  if ( jj == 0 ){ pp = -1; } else { pp = 1; } 
	  for (int kk=0; kk<32; kk++)
	    { 
	      sprintf(histo, "DQMData/EcalPreshowerRecHitsTask/ES Occupancy Plane%01d Side%01d Strip%02d", ii+1, pp, kk+1);
	      rfile->GetObject(histo,meESRecHitsStripOccupancy_[ii][jj][kk]);
	      meESRecHitsStripOccupancy_[ii][jj][kk];
	    }}}
      

      // --------------------------------
      TCanvas *PreshowerEnergy = new TCanvas("PreshowerEnergy","PreshowerEnergy",800,800);
      if ( meESRecHitsEnergy_ ) meESRecHitsEnergy_->Draw(); 
      PreshowerEnergy->Print("PreshowerEnergy.eps");
      delete PreshowerEnergy;

      TCanvas *PreshowerOccupancy_P1S1  = new TCanvas("PreshowerOccupancy_P1S1", "EE occupancy, plane 1, side+1", 800,800);
      TCanvas *PreshowerOccupancy_P1Sm1 = new TCanvas("PreshowerOccupancy_P1Sm1","EE occupancy, plane 1, side-1", 800,800);
      TCanvas *PreshowerOccupancy_P2S1  = new TCanvas("PreshowerOccupancy_P2S1", "EE occupancy, plane 2, side+1", 800,800);
      TCanvas *PreshowerOccupancy_P2Sm1 = new TCanvas("PreshowerOccupancy_P2Sm1","EE occupancy, plane 2, side-1", 800,800);
      PreshowerOccupancy_P1S1 ->Divide(8,4);
      PreshowerOccupancy_P1Sm1->Divide(8,4);
      PreshowerOccupancy_P2S1 ->Divide(8,4);
      PreshowerOccupancy_P2Sm1->Divide(8,4);
      for (int ii=0; ii<32; ii++) 
	{
	  PreshowerOccupancy_P1Sm1->cd(ii+1); 
	  if ( meESRecHitsStripOccupancy_[0][0][ii] ) meESRecHitsStripOccupancy_[0][0][ii]->Draw("colz"); 

	  PreshowerOccupancy_P1S1->cd(ii+1); 
	  if ( meESRecHitsStripOccupancy_[0][1][ii] ) meESRecHitsStripOccupancy_[0][1][ii]->Draw("colz"); 

	  PreshowerOccupancy_P2Sm1->cd(ii+1); 
	  if ( meESRecHitsStripOccupancy_[1][0][ii] ) meESRecHitsStripOccupancy_[1][0][ii]->Draw("colz"); 

	  PreshowerOccupancy_P2S1->cd(ii+1); 
	  if ( meESRecHitsStripOccupancy_[1][1][ii] ) meESRecHitsStripOccupancy_[1][1][ii]->Draw("colz"); 
	} 
      PreshowerOccupancy_P1Sm1->Print("PreshowerOccupancy_plane1_sidem1.eps"); 
      PreshowerOccupancy_P1S1 ->Print("PreshowerOccupancy_plane1_side1.eps"); 
      PreshowerOccupancy_P2Sm1->Print("PreshowerOccupancy_plane2_sidem1.eps"); 
      PreshowerOccupancy_P2S1 ->Print("PreshowerOccupancy_plane2_side1.eps"); 
      delete PreshowerOccupancy_P1Sm1;
      delete PreshowerOccupancy_P1S1;
      delete PreshowerOccupancy_P2Sm1;
      delete PreshowerOccupancy_P2S1;
    }




}
