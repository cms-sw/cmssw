// Commands executed in a GLOBAL scope, e.g. created hitograms aren't erased...
#include "TH1.h"
#include "TH2.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TProfile.h"
#include "TPaveStats.h"
#include "TFile.h"
#include "TString.h"
#include "TList.h"
#include "TStyle.h"
#include "TClass.h"
#include "TKey.h"
#include "TDirectory.h"

#include <cstdio>
#include <string>
#include <iostream>

#include "rootlogon.h"

TDirectory* fileDirectory( TDirectory *target, std::string s);
void SinglePi(const TString ref_vers="330pre6", const TString val_vers="330pre6", bool fastsim=false);

int main(int argn, char **argv)
{
    if(argn == 3)      SinglePi(argv[1], argv[2]);
    else if(argn == 4) SinglePi(argv[1], argv[2], strcmp(argv[3], "fastsim") == 0);
    else               printf("Usage: ./SinglePi.exe ref_ver val_ver [fastsim]\n");
}

void SinglePi(const TString ref_vers, const TString val_vers, bool fastsim){

    //Warning!!! This rootlogon hacks the root color pallate
   setColors();

   TString ref_file = "pi50scan"+ref_vers+"_ECALHCAL_CaloTowers.root";
   TString val_file = "pi50scan"+val_vers+"_ECALHCAL_CaloTowers.root";
      
   TFile f1(ref_file);
   TFile f2(val_file);
   
   // service variables
   //
   //Profiles
   const int Nprof   = 15;

   TProfile* f1_prof[Nprof];
   TProfile* f2_prof[Nprof];

   char labelp[Nprof][64];

   //1D Histos
   const int Nhist1  = 11;

   TH1* f1_hist1[Nhist1];
   TH1* f2_hist1[Nhist1];

   char label1[Nhist1][64];

   //Labels
   //Profiles
   sprintf(labelp[0], "CaloTowersTask_emean_vs_ieta_E1.gif");
   sprintf(labelp[1], "CaloTowersTask_emean_vs_ieta_H1.gif");
   sprintf(labelp[2], "CaloTowersTask_emean_vs_ieta_EH1.gif");

   sprintf(labelp[3], "RecHitsTask_emean_vs_ieta_E.gif");
   sprintf(labelp[4], "RecHitsTask_emean_vs_ieta_H.gif");
   sprintf(labelp[5], "RecHitsTask_emean_vs_ieta_EH.gif");
   if (!fastsim) {
       sprintf(labelp[6], "SimHitsTask_emean_vs_ieta_E.gif");
       sprintf(labelp[7], "SimHitsTask_emean_vs_ieta_H.gif");
       sprintf(labelp[8], "SimHitsTask_emean_vs_ieta_EH.gif");
   }
   sprintf(labelp[9], "RecHitsTask_timing_vs_energy_profile_HB.gif");
   sprintf(labelp[10], "RecHitsTask_timing_vs_energy_profile_HE.gif");
   sprintf(labelp[11], "RecHitsTask_timing_vs_energy_profile_HF.gif");

   sprintf(labelp[12], "CaloTowersTask_emean_vs_ieta_E.gif");
   sprintf(labelp[13], "CaloTowersTask_emean_vs_ieta_H.gif");
   sprintf(labelp[14], "CaloTowersTask_emean_vs_ieta_EH.gif");

   //1D Histos
   sprintf(label1[0], "N_calotowers_HB.gif");
   sprintf(label1[1], "N_calotowers_HE.gif");
   sprintf(label1[2], "N_calotowers_HF.gif");
   
   sprintf(label1[3], "RecHits_energy_HB.gif");
   sprintf(label1[4], "RecHits_energy_HE.gif");
   sprintf(label1[5], "RecHits_energy_HO.gif");
   sprintf(label1[6], "RecHits_energy_HF.gif");

   sprintf(label1[7], "Ndigis_HB.gif" );
   sprintf(label1[8], "Ndigis_HE.gif" );
   sprintf(label1[9], "Ndigis_HO.gif" );
   sprintf(label1[10], "Ndigis_HF.gif" );

   // REFERENCE FILE

   TDirectory *td = fileDirectory(&f1, "CaloTowersTask");
   //f1.cd("DQMData/CaloTowersV/CaloTowersTask");
   //gDirectory->pwd();
   td->pwd();
   f1_prof[0] = (TProfile*)td->Get("emean_vs_ieta_E1");
   f1_prof[1] = (TProfile*)td->Get("emean_vs_ieta_H1");
   f1_prof[2] = (TProfile*)td->Get("emean_vs_ieta_EH1");

   f1_prof[12] = (TProfile*)td->Get("emean_vs_ieta_E");
   f1_prof[13] = (TProfile*)td->Get("emean_vs_ieta_H");
   f1_prof[14] = (TProfile*)td->Get("emean_vs_ieta_EH");

   f1_hist1[0] = (TH1*)td->Get("CaloTowersTask_number_of_fired_towers_HB");
   f1_hist1[1] = (TH1*)td->Get("CaloTowersTask_number_of_fired_towers_HE");
   f1_hist1[2] = (TH1*)td->Get("CaloTowersTask_number_of_fired_towers_HF");

   td = fileDirectory(&f1, "HcalRecHitTask");
   //f1.cd("DQMData/HcalRecHitsV/HcalRecHitTask");
   f1_prof[3] = (TProfile*)td->Get("HcalRecHitTask_En_rechits_cone_profile_vs_ieta_all_depths_E");
   f1_prof[4] = (TProfile*)td->Get("HcalRecHitTask_En_rechits_cone_profile_vs_ieta_all_depths");
   f1_prof[5] = (TProfile*)td->Get("HcalRecHitTask_En_rechits_cone_profile_vs_ieta_all_depths_EH");

   f1_prof[9] = (TProfile*)td->Get("HcalRecHitTask_timing_vs_energy_profile_HB");
   f1_prof[10] = (TProfile*)td->Get("HcalRecHitTask_timing_vs_energy_profile_Low_HE");
   f1_prof[11] = (TProfile*)td->Get("HcalRecHitTask_timing_vs_energy_profile_Low_HF");

   f1_hist1[3] = (TH1*)td->Get("HcalRecHitTask_energy_of_rechits_HB");
   f1_hist1[4] = (TH1*)td->Get("HcalRecHitTask_energy_of_rechits_HE");
   f1_hist1[5] = (TH1*)td->Get("HcalRecHitTask_energy_of_rechits_HO");
   f1_hist1[6] = (TH1*)td->Get("HcalRecHitTask_energy_of_rechits_HF");   


   td = fileDirectory(&f1, "HcalDigiTask");
   f1_hist1[7] =  (TH1*)td->Get("HcalDigiTask_Ndigis_HB");
   f1_hist1[8] =  (TH1*)td->Get("HcalDigiTask_Ndigis_HE");
   f1_hist1[9] =  (TH1*)td->Get("HcalDigiTask_Ndigis_HO");
   f1_hist1[10] =  (TH1*)td->Get("HcalDigiTask_Ndigis_HF");


   if (!fastsim) {
       td = fileDirectory(&f1, "HcalSimHitTask");
       //f1.cd("DQMData/HcalSimHitsV/HcalSimHitTask");
       f1_prof[6] = (TProfile*)td->Get("HcalSimHitTask_En_simhits_cone_profile_vs_ieta_all_depths_E");
       f1_prof[7] = (TProfile*)td->Get("HcalSimHitTask_En_simhits_cone_profile_vs_ieta_all_depths");
       f1_prof[8] = (TProfile*)td->Get("HcalSimHitTask_En_simhits_cone_profile_vs_ieta_all_depths_EH");
   }

   //   NEW FILE

   td = fileDirectory(&f2, "CaloTowersTask");
   //f2.cd("DQMData/CaloTowersV/CaloTowersTask");
   //gDirectory->pwd();
   td->pwd();
   f2_prof[0] = (TProfile*)td->Get("emean_vs_ieta_E1");
   f2_prof[1] = (TProfile*)td->Get("emean_vs_ieta_H1");
   f2_prof[2] = (TProfile*)td->Get("emean_vs_ieta_EH1");

   f2_prof[12] = (TProfile*)td->Get("emean_vs_ieta_E");
   f2_prof[13] = (TProfile*)td->Get("emean_vs_ieta_H");
   f2_prof[14] = (TProfile*)td->Get("emean_vs_ieta_EH");

   f2_hist1[0] = (TH1*)td->Get("CaloTowersTask_number_of_fired_towers_HB");
   f2_hist1[1] = (TH1*)td->Get("CaloTowersTask_number_of_fired_towers_HE");
   f2_hist1[2] = (TH1*)td->Get("CaloTowersTask_number_of_fired_towers_HF");

   td = fileDirectory(&f2, "HcalRecHitTask");
   //f2.cd("DQMData/HcalRecHitsV/HcalRecHitTask");
   f2_prof[3] = (TProfile*)td->Get("HcalRecHitTask_En_rechits_cone_profile_vs_ieta_all_depths_E");
   f2_prof[4] = (TProfile*)td->Get("HcalRecHitTask_En_rechits_cone_profile_vs_ieta_all_depths");
   f2_prof[5] = (TProfile*)td->Get("HcalRecHitTask_En_rechits_cone_profile_vs_ieta_all_depths_EH");

   f2_prof[9] = (TProfile*)td->Get("HcalRecHitTask_timing_vs_energy_profile_HB");
   f2_prof[10] = (TProfile*)td->Get("HcalRecHitTask_timing_vs_energy_profile_Low_HE");
   f2_prof[11] = (TProfile*)td->Get("HcalRecHitTask_timing_vs_energy_profile_Low_HF"); 

   f2_hist1[3] = (TH1*)td->Get("HcalRecHitTask_energy_of_rechits_HB");
   f2_hist1[4] = (TH1*)td->Get("HcalRecHitTask_energy_of_rechits_HE");
   f2_hist1[5] = (TH1*)td->Get("HcalRecHitTask_energy_of_rechits_HO");
   f2_hist1[6] = (TH1*)td->Get("HcalRecHitTask_energy_of_rechits_HF");

   td = fileDirectory(&f1, "HcalDigiTask");
   f2_hist1[7] =  (TH1*)td->Get("HcalDigiTask_Ndigis_HB");
   f2_hist1[8] =  (TH1*)td->Get("HcalDigiTask_Ndigis_HE");
   f2_hist1[9] =  (TH1*)td->Get("HcalDigiTask_Ndigis_HO");
   f2_hist1[10] =  (TH1*)td->Get("HcalDigiTask_Ndigis_HF");

   if (!fastsim) {
       td = fileDirectory(&f2, "HcalSimHitTask");
       //f2.cd("DQMData/HcalSimHitsV/HcalSimHitTask");
       f2_prof[6] = (TProfile*)td->Get("HcalSimHitTask_En_simhits_cone_profile_vs_ieta_all_depths_E");
       f2_prof[7] = (TProfile*)td->Get("HcalSimHitTask_En_simhits_cone_profile_vs_ieta_all_depths");
       f2_prof[8] = (TProfile*)td->Get("HcalSimHitTask_En_simhits_cone_profile_vs_ieta_all_depths_EH");
   }



   //Profiles  titles
   f1_prof[0]->GetXaxis()->SetTitle("CaloTowers eE (GeV) vs ieta 1 Tower");
   f1_prof[1]->GetXaxis()->SetTitle("CaloTowers hE (GeV) vs ieta 1 Tower");
   f1_prof[2]->GetXaxis()->SetTitle("CaloTowers eE+hE (GeV) vs ieta 1 Tower");


   f1_prof[3]->GetXaxis()->SetTitle("RecHits eE (GeV) vs ieta R = 0.3 Cone");
   f1_prof[4]->GetXaxis()->SetTitle("RecHits hE (GeV) vs ieta R = 0.3 Cone");
   f1_prof[5]->GetXaxis()->SetTitle("RecHits eE+hE (GeV) vs ieta R = 0.3 Cone");

   if (!fastsim) {
     f1_prof[6]->GetXaxis()->SetTitle("SimHits eE (GeV) vs ieta R = 0.3 Cone");
     f1_prof[7]->GetXaxis()->SetTitle("SimHits hE (GeV) vs ieta R = 0.3 Cone");
     f1_prof[8]->GetXaxis()->SetTitle("SimHits eE+hE (GeV) vs ieta R = 0.3 Cone");
   }

   f1_prof[9]->GetXaxis()->SetTitle("HB RecHits timing (ns) vs Energy (GeV)");
   f1_prof[10]->GetXaxis()->SetTitle("HE RecHits timing (ns) vs Energy (GeV)");
   f1_prof[11]->GetXaxis()->SetTitle("HF RecHits timing (ns) vs Energy (GeV)");

   f1_prof[12]->GetXaxis()->SetTitle("CaloTowers eE Rcone sum (GeV) vs ieta");
   f1_prof[13]->GetXaxis()->SetTitle("CaloTowers hE Rcone sumn (GeV) vs ieta");
   f1_prof[14]->GetXaxis()->SetTitle("CaloTowers eE+hE Rcone sum (GeV) vs ieta ");


   //1D Histos titles
   f1_hist1[0]->GetXaxis()->SetTitle("Number of HB CaloTowers");
   f1_hist1[1]->GetXaxis()->SetTitle("Number of HE CaloTowers");
   f1_hist1[2]->GetXaxis()->SetTitle("Number of HF CaloTowers");

   f1_hist1[3]->GetXaxis()->SetTitle("HB RecHits energy (GeV)");
   f1_hist1[4]->GetXaxis()->SetTitle("HE RecHits energy (GeV)");
   f1_hist1[5]->GetXaxis()->SetTitle("HO RecHits energy (GeV)");
   f1_hist1[6]->GetXaxis()->SetTitle("HF RecHits energy (GeV)");

   f1_hist1[7]->GetXaxis()->SetTitle("N_HB Digis");
   f1_hist1[8]->GetXaxis()->SetTitle("N_HE Digis");
   f1_hist1[9]->GetXaxis()->SetTitle("N_H0 Digis");
   f1_hist1[10]->GetXaxis()->SetTitle("N_HF Digis");



   //
   f1_prof[0]->SetMaximum(20.);  // CaloTowers 1  
   f1_prof[1]->SetMaximum(40.);
   f1_prof[2]->SetMaximum(40.);
   f1_prof[0]->SetMinimum(0.);  // idem 
   f1_prof[1]->SetMinimum(0.); 
   f1_prof[2]->SetMinimum(0.);

   f1_prof[3]->SetMaximum(30.); //  RecHits R==0.3
   f1_prof[4]->SetMaximum(50.);
   f1_prof[5]->SetMaximum(60.);
   f1_prof[3]->SetMinimum(0.);
   f1_prof[4]->SetMinimum(0.);
   f1_prof[5]->SetMinimum(0.);


   if (!fastsim) {                // SimHits R=0.3 
     f1_prof[6]->SetMinimum(0.);
     f1_prof[7]->SetMinimum(0.);
     f1_prof[8]->SetMinimum(0.);
     f1_prof[6]->SetMaximum(20.); 
     f1_prof[7]->SetMaximum(60.);
     f1_prof[8]->SetMaximum(60.);
   }


   f1_prof[9]->SetMinimum(-25.);  // RecHits Timing
   f1_prof[10]->SetMinimum(-25.);
   f1_prof[11]->SetMinimum(-25.);
   f1_prof[9]->SetMaximum(25.);  
   f1_prof[10]->SetMaximum(25.);
   f1_prof[11]->SetMaximum(25.);


   f1_prof[12]->SetMaximum(30.); // CaloTowers R=0.3 added
   f1_prof[13]->SetMaximum(50.);
   f1_prof[14]->SetMaximum(60.);
   f1_prof[12]->SetMinimum(0.);
   f1_prof[13]->SetMinimum(0.);
   f1_prof[14]->SetMinimum(0.);


   f1_prof[9]->GetXaxis()->SetRangeUser(0.,75.);
   f1_prof[10]->GetXaxis()->SetRangeUser(0.,75.);
   f1_prof[11]->GetXaxis()->SetRangeUser(0.,75.);

 
  

   //   1D HISTOS 

   f1_hist1[0]->GetXaxis()->SetRangeUser(0.,200.);   // N_CaloTowers
   f2_hist1[0]->GetXaxis()->SetRangeUser(0.,200.);

   f1_hist1[1]->GetXaxis()->SetRangeUser(0.,150.);
   f2_hist1[1]->GetXaxis()->SetRangeUser(0.,150.);

   f1_hist1[2]->GetXaxis()->SetRangeUser(0.,500.);
   f2_hist1[2]->GetXaxis()->SetRangeUser(0.,500.);

   f1_hist1[3]->GetXaxis()->SetRangeUser(0.,100.);  // RecHits spectra
   f2_hist1[3]->GetXaxis()->SetRangeUser(0.,100.);

   f1_hist1[4]->GetXaxis()->SetRangeUser(0.,100.);
   f2_hist1[4]->GetXaxis()->SetRangeUser(0.,100.);

   f1_hist1[5]->GetXaxis()->SetRangeUser(0.,100.);
   f2_hist1[5]->GetXaxis()->SetRangeUser(0.,100.);

   f1_hist1[6]->GetXaxis()->SetRangeUser(0.,100.);
   f2_hist1[6]->GetXaxis()->SetRangeUser(0.,100.);

   f1_hist1[3]->SetMaximum(1.e8);
   f1_hist1[4]->SetMaximum(1.e8);
   f1_hist1[5]->SetMaximum(1.e8);
   f1_hist1[6]->SetMaximum(1.e8);

   f1_hist1[7]->GetXaxis()->SetRangeUser(0.,1000);   // N_Digis
   f2_hist1[7]->GetXaxis()->SetRangeUser(0.,1000);

   f1_hist1[8]->GetXaxis()->SetRangeUser(0.,200);
   f2_hist1[8]->GetXaxis()->SetRangeUser(0.,200);

   f1_hist1[9]->GetXaxis()->SetRangeUser(0.,100);
   f2_hist1[9]->GetXaxis()->SetRangeUser(0.,100);

   f1_hist1[10]->GetXaxis()->SetRangeUser(0.,3500);
   f2_hist1[10]->GetXaxis()->SetRangeUser(0.,3500);


   //   gStyle->SetErrorX(0);


   //  1D-histo

   for (int i = 0; i < Nhist1; i++){

     TCanvas *myc = new TCanvas("myc","",800,600);
     gStyle->SetOptStat(1111);

     if(i > 2 && i < 7) myc->SetLogy();
     
     f1_hist1[i]->SetStats(kTRUE);   // stat box  
     f2_hist1[i]->SetStats(kTRUE);  

     f1_hist1[i]->SetTitle("");
     f2_hist1[i]->SetTitle("");
     
     f1_hist1[i]->SetLineWidth(2); 
     f2_hist1[i]->SetLineWidth(2); 
     
     // diffferent histo colors and styles
     f1_hist1[i]->SetLineColor(41);
     f1_hist1[i]->SetLineStyle(1); 
     
     f2_hist1[i]->SetLineColor(43);
     f2_hist1[i]->SetLineStyle(2);  
     
     //Set maximum to the larger of the two
     if (f1_hist1[i]->GetMaximum() < f2_hist1[i]->GetMaximum()) f1_hist1[i]->SetMaximum(1.05 * f2_hist1[i]->GetMaximum());

     TLegend *leg = new TLegend(0.2, 0.91, 0.6, 0.99, "","brNDC");

     leg->SetBorderSize(2);
     //  leg->SetFillColor(51); // see new color definition above
     leg->SetFillStyle(1001); //
     leg->AddEntry(f1_hist1[i],"CMSSW_"+ref_vers,"l");
     leg->AddEntry(f2_hist1[i],"CMSSW_"+val_vers,"l");


     TPaveStats *ptstats = new TPaveStats(0.85,0.86,0.98,0.98,"brNDC");
     ptstats->SetTextColor(41);
     f1_hist1[i]->GetListOfFunctions()->Add(ptstats);
     ptstats->SetParent(f1_hist1[i]->GetListOfFunctions());
     TPaveStats *ptstats2 = new TPaveStats(0.85,0.74,0.98,0.86,"brNDC");
     ptstats2->SetTextColor(43);
     f2_hist1[i]->GetListOfFunctions()->Add(ptstats2);
     ptstats2->SetParent(f2_hist1[i]->GetListOfFunctions());
         
     f1_hist1[i]->Draw(""); // "stat"   
     f2_hist1[i]->Draw("hist sames");   
     
     leg->Draw();   
     
     myc->SaveAs(label1[i]);

     if(myc) delete myc;

     std::cout << "1D histos " << i << " produced" << std::endl; 

   }     



  //  Profiles
  for (int i = 0; i < Nprof; i++){

    TCanvas *myc = new TCanvas("myc","",800,600);

    bool skipHisto = false;
    if (fastsim && i>=6 && i<=8) skipHisto = true;   // SimHits to exclude

    if (!skipHisto) {
      f1_prof[i]->SetStats(kFALSE);   
      f2_prof[i]->SetStats(kFALSE); 
      
      f1_prof[i]->SetTitle("");
      f2_prof[i]->SetTitle("");
      

      f1_prof[i]->SetLineColor(41);
      f1_prof[i]->SetLineStyle(1);     
      f1_prof[i]->SetLineWidth(1); 
      f1_prof[i]->SetMarkerColor(41);
      f1_prof[i]->SetMarkerStyle(21);
      f1_prof[i]->SetMarkerSize(1.0);  
      
      f2_prof[i]->SetLineColor(43);
      f2_prof[i]->SetLineStyle(1);  
      f2_prof[i]->SetLineWidth(1); 
      f2_prof[i]->SetMarkerColor(43);
      f2_prof[i]->SetMarkerStyle(20);
      f2_prof[i]->SetMarkerSize(0.8);  
      
      if(i > 8  && i < 12) {               // Timing 
	f1_prof[i]->SetMarkerSize(0.1);
	f2_prof[i]->SetMarkerSize(0.3);  
      }
      
      myc->SetGrid();
       
      if( i <= 8  ||  i >= 12) {            // Other than RecHits timing
	f1_prof[i]->Draw("histpl ");   
	f2_prof[i]->Draw("histplsame");  //    
      }
      else {
	f1_prof[i]->Draw("pl");            // RecHits Timing
	f2_prof[i]->Draw("pl same");       // 
      }

       f1_prof[i]->GetOption();
       f2_prof[i]->GetOption();

      
      TLegend *leg = new TLegend(0.40, 0.91, 0.74, 0.99, "","brNDC");    
      leg->SetBorderSize(2);
      leg->SetFillStyle(1001); 
      leg->AddEntry(f1_prof[i],"CMSSW_"+ref_vers,"pl");
      leg->AddEntry(f2_prof[i],"CMSSW_"+val_vers,"pl");
      
      leg->Draw("");   
     
      myc->SaveAs(labelp[i]);
    }
    if(myc) delete myc;

    std::cout << "Profile " << i << " produced" << std::endl; 


  }



  // RATIO CaloTower 1 

  
  TCanvas *myc1 = new TCanvas("myc1","",800,600);

  TProfile* ratio1 = (TProfile*)f2_prof[2]->Clone();
  ratio1->Divide(f1_prof[2]);
  ratio1->SetMaximum(1.2);
  ratio1->SetMinimum(0.8);
  myc1->SetGrid();  
  ratio1->Draw("hist pl");

  TLegend *leg1 = new TLegend(0.20, 0.91, 0.70, 0.99, "","brNDC");
  leg1->SetBorderSize(2);
  leg1->SetFillStyle(1001);
  leg1->AddEntry(ratio1,"CaloTowers scale (pi50) ratio "+val_vers+"/"+ref_vers+" vs ieta","pl");
  leg1->Draw("");

  myc1->SaveAs("Ratio.gif");
  

 //  RATIO HCAL RecHits in R=0.3

  TCanvas *myc2 = new TCanvas("myc2","",800,600);

  TProfile* ratio2 = (TProfile*)f2_prof[4]->Clone();
  ratio2->Divide(f1_prof[4]);
  ratio2->SetMaximum(1.2);
  ratio2->SetMinimum(0.8);
  myc2->SetGrid();  
  ratio2->Draw("hist pl");

  TLegend *leg2 = new TLegend(0.10, 0.91, 0.80, 0.99, "","brNDC");
  leg2->SetBorderSize(2);
  leg2->SetFillStyle(1001);
  leg2->AddEntry(ratio2,"HCAL sum ratio "+val_vers+"/"+ref_vers+" vs ieta","pl");
  leg2->Draw("");

  myc2->SaveAs("Ratio_Hcone.gif");


  // RATIO CaloTowers H sum in R=0.3

  TCanvas *myc3 = new TCanvas("myc3","",800,600);

  TProfile* ratio3 = (TProfile*)f2_prof[13]->Clone();
  ratio3->Divide(f1_prof[13]);
  ratio3->SetMaximum(1.2);
  ratio3->SetMinimum(0.8);
  myc3->SetGrid();  
  ratio3->Draw("hist pl");

  TLegend *leg3 = new TLegend(0.10, 0.91, 0.80, 0.99, "","brNDC");
  leg3->SetBorderSize(2);
  leg3->SetFillStyle(1001);
  leg3->AddEntry(ratio3,"CaloTowers HAD in R=0.3 ratio "+val_vers+"/"+ref_vers+" vs ieta","pl");
  leg3->Draw("");
  myc3->SaveAs("Ratio_CaloTowers_Hcone.gif");




   // close ROOT files ===========================================
   //
   f1.Close() ;  
   f2.Close() ;
   
   return ;  
     
}

TDirectory* fileDirectory( TDirectory *target, std::string s) 
{
    TDirectory *retval = 0;

    // loop over all keys in this directory
    TIter nextkey(target->GetListOfKeys());
    TKey *key, *oldkey=0;
    while((key = (TKey*)nextkey())) 
    {
	//keep only the highest cycle number for each key
	if (oldkey && !strcmp(oldkey->GetName(),key->GetName())) continue;

	// read object from file
	target->cd();
	TObject *obj = key->ReadObj();
	
	if(obj->IsA()->InheritsFrom(TDirectory::Class())) 
	{
	    // it's a subdirectory
	    //cout << "Found subdirectory " << obj->GetName() << endl;
	    if(strcmp(s.c_str(), obj->GetName()) == 0) return (TDirectory*)obj;
	    
	    if((retval = fileDirectory((TDirectory*)obj, s))) break;
	    
	}
	else break;
    }
    return retval;
}

