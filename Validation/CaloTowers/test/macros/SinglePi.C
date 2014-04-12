// Commands executed in a GLOBAL scope, e.g. created hitograms aren't erased...

void SinglePi(const TString ref_vers="330pre6", const TString val_vers="330pre6", bool fastsim=false){

   TString ref_file = "pi50scan"+ref_vers+"_ECALHCAL_CaloTowers.root";
   TString val_file = "pi50scan"+val_vers+"_ECALHCAL_CaloTowers.root";
      
   TFile f1(ref_file);
   TFile f2(val_file);
   
   // service variables
   //
   //Profiles
   const int Nprof   = 12;

   TProfile* f1_prof[Nprof];
   TProfile* f2_prof[Nprof];

   char *labelp[Nprof];

   //1D Histos
   const int Nhist1  = 7;

   TH1F* f1_hist1[Nhist1];
   TH1F* f2_hist1[Nhist1];

   char *label1[Nhist1];

   //Labels
   //Profiles
   labelp[0] = &"CaloTowersTask_emean_vs_ieta_E1.gif";
   labelp[1] = &"CaloTowersTask_emean_vs_ieta_H1.gif";
   labelp[2] = &"CaloTowersTask_emean_vs_ieta_EH1.gif";

   labelp[3] = &"RecHitsTask_emean_vs_ieta_E.gif";
   labelp[4] = &"RecHitsTask_emean_vs_ieta_H.gif";
   labelp[5] = &"RecHitsTask_emean_vs_ieta_EH.gif";
   if (!fastsim) {
     labelp[6] = &"SimHitsTask_emean_vs_ieta_E.gif";
     labelp[7] = &"SimHitsTask_emean_vs_ieta_H.gif";
     labelp[8] = &"SimHitsTask_emean_vs_ieta_EH.gif";
   }
   labelp[9]  = &"RecHitsTask_timing_vs_energy_profile_HB.gif";
   labelp[10] = &"RecHitsTask_timing_vs_energy_profile_HE.gif";
   labelp[11] = &"RecHitsTask_timing_vs_energy_profile_HF.gif";



   //1D Histos
   label1[0] = &"N_calotowers_HB.gif";
   label1[1] = &"N_calotowers_HE.gif";
   label1[2] = &"N_calotowers_HF.gif";
   
   label1[3] = &"RecHits_energy_HB.gif";
   label1[4] = &"RecHits_energy_HE.gif";
   label1[5] = &"RecHits_energy_HO.gif";
   label1[6] = &"RecHits_energy_HF.gif";


   f1.cd("DQMData/CaloTowersV/CaloTowersTask");
   gDirectory->pwd();
   f1_prof[0] = emean_vs_ieta_E1;
   f1_prof[1] = emean_vs_ieta_H1;
   f1_prof[2] = emean_vs_ieta_EH1;

   f1_hist1[0] = CaloTowersTask_number_of_fired_towers_HB;
   f1_hist1[1] = CaloTowersTask_number_of_fired_towers_HE;
   f1_hist1[2] = CaloTowersTask_number_of_fired_towers_HF;

   f1.cd("DQMData/HcalRecHitsV/HcalRecHitTask");
   f1_prof[3] = HcalRecHitTask_En_rechits_cone_profile_vs_ieta_all_depths_E;
   f1_prof[4] = HcalRecHitTask_En_rechits_cone_profile_vs_ieta_all_depths;
   f1_prof[5] = HcalRecHitTask_En_rechits_cone_profile_vs_ieta_all_depths_EH;

   f1_prof[9] = HcalRecHitTask_timing_vs_energy_profile_HB;   
   f1_prof[10] = HcalRecHitTask_timing_vs_energy_profile_Low_HE;   
   f1_prof[11] = HcalRecHitTask_timing_vs_energy_profile_Low_HF;   

   f1_hist1[3] = HcalRecHitTask_energy_of_rechits_HB;
   f1_hist1[4] = HcalRecHitTask_energy_of_rechits_HE;
   f1_hist1[5] = HcalRecHitTask_energy_of_rechits_HO;
   f1_hist1[6] = HcalRecHitTask_energy_of_rechits_HF;   

   if (!fastsim) {
     f1.cd("DQMData/HcalSimHitsV/HcalSimHitTask");
     f1_prof[6] = HcalSimHitTask_En_simhits_cone_profile_vs_ieta_all_depths_E;
     f1_prof[7] = HcalSimHitTask_En_simhits_cone_profile_vs_ieta_all_depths;
     f1_prof[8] = HcalSimHitTask_En_simhits_cone_profile_vs_ieta_all_depths_EH;
   }

   f2.cd("DQMData/CaloTowersV/CaloTowersTask");
   gDirectory->pwd();
   f2_prof[0] = emean_vs_ieta_E1;
   f2_prof[1] = emean_vs_ieta_H1;
   f2_prof[2] = emean_vs_ieta_EH1;

   f2_hist1[0] = CaloTowersTask_number_of_fired_towers_HB;
   f2_hist1[1] = CaloTowersTask_number_of_fired_towers_HE;
   f2_hist1[2] = CaloTowersTask_number_of_fired_towers_HF;

   f2.cd("DQMData/HcalRecHitsV/HcalRecHitTask");
   f2_prof[3] = HcalRecHitTask_En_rechits_cone_profile_vs_ieta_all_depths_E;
   f2_prof[4] = HcalRecHitTask_En_rechits_cone_profile_vs_ieta_all_depths;
   f2_prof[5] = HcalRecHitTask_En_rechits_cone_profile_vs_ieta_all_depths_EH;

   f2_prof[9] = HcalRecHitTask_timing_vs_energy_profile_HB;   
   f2_prof[10] = HcalRecHitTask_timing_vs_energy_profile_Low_HE;   
   f2_prof[11] = HcalRecHitTask_timing_vs_energy_profile_Low_HF;   

   f2_hist1[3] = HcalRecHitTask_energy_of_rechits_HB;
   f2_hist1[4] = HcalRecHitTask_energy_of_rechits_HE;
   f2_hist1[5] = HcalRecHitTask_energy_of_rechits_HO;
   f2_hist1[6] = HcalRecHitTask_energy_of_rechits_HF;

   if (!fastsim) {
     f2.cd("DQMData/HcalSimHitsV/HcalSimHitTask");
     f2_prof[6] = HcalSimHitTask_En_simhits_cone_profile_vs_ieta_all_depths_E;
     f2_prof[7] = HcalSimHitTask_En_simhits_cone_profile_vs_ieta_all_depths;
     f2_prof[8] = HcalSimHitTask_En_simhits_cone_profile_vs_ieta_all_depths_EH;
   }

   //Profiles
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


   //1D Histos
   f1_hist1[0]->GetXaxis()->SetTitle("Number of HB CaloTowers");
   f1_hist1[1]->GetXaxis()->SetTitle("Number of HE CaloTowers");
   f1_hist1[2]->GetXaxis()->SetTitle("Number of HF CaloTowers");

   f1_hist1[3]->GetXaxis()->SetTitle("HB RecHits energy (GeV)");
   f1_hist1[4]->GetXaxis()->SetTitle("HE RecHits energy (GeV)");
   f1_hist1[5]->GetXaxis()->SetTitle("HO RecHits energy (GeV)");
   f1_hist1[6]->GetXaxis()->SetTitle("HF RecHits energy (GeV)");

   //
   f1_prof[0]->SetMaximum(20.);
   f1_prof[1]->SetMaximum(40.);
   f1_prof[2]->SetMaximum(40.);
   f1_prof[4]->SetMaximum(50.);
   f1_prof[5]->SetMaximum(50.);

   f1_prof[0]->SetMinimum(0.);
   f1_prof[1]->SetMinimum(0.);
   f1_prof[2]->SetMinimum(0.);
   f1_prof[3]->SetMinimum(0.);
   f1_prof[4]->SetMinimum(0.);
   f1_prof[5]->SetMinimum(0.);

   if (!fastsim) {
     f1_prof[6]->SetMinimum(0.);
     f1_prof[7]->SetMinimum(0.);
     f1_prof[8]->SetMinimum(0.);
   }

   f1_prof[9]->GetXaxis()->SetRangeUser(0.,75.);
   f1_prof[10]->GetXaxis()->SetRangeUser(0.,75.);
   f1_prof[11]->GetXaxis()->SetRangeUser(0.,75.);

   /* 
   f1_prof[9]->SetMinimum(0.);
   f1_prof[10]->SetMinimum(0.);
   f1_prof[11]->SetMinimum(0.);
   */



   // f1_hist[2]->GetXaxis()->SetRangeUser(0.,1200.);
   // f1_hist[7]->GetXaxis()->SetRangeUser(0.,160.);
   // hist1->GetXaxis()->SetNdivisions(-21);
   // hist1->GetYaxis()->SetNdivisions(-1003);

   f1_hist1[0]->GetXaxis()->SetRangeUser(0.,100.);
   f2_hist1[0]->GetXaxis()->SetRangeUser(0.,100.);

   f1_hist1[1]->GetXaxis()->SetRangeUser(0.,150.);
   f2_hist1[1]->GetXaxis()->SetRangeUser(0.,150.);

   f1_hist1[2]->GetXaxis()->SetRangeUser(0.,100.);
   f2_hist1[2]->GetXaxis()->SetRangeUser(0.,100.);

   f1_hist1[3]->GetXaxis()->SetRangeUser(0.,100.);
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



   //  1D-histo

   for (int i = 0; i < Nhist1; i++){

     TCanvas *myc = new TCanvas("myc","",800,600);
     gStyle->SetOptStat(1111);

     if(i > 2) myc->SetLogy();
     
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
     TPaveStats *ptstats = new TPaveStats(0.85,0.74,0.98,0.86,"brNDC");
     ptstats->SetTextColor(43);
     f2_hist1[i]->GetListOfFunctions()->Add(ptstats);
     ptstats->SetParent(f2_hist1[i]->GetListOfFunctions());
         
     f1_hist1[i]->Draw(""); // "stat"   
     f2_hist1[i]->Draw("histsames");   
     
     leg->Draw();   
     
     myc->SaveAs(label1[i]);

     if(myc) delete myc;
   }     


  //  Profiles
  for (int i = 0; i < Nprof; i++){

    TCanvas *myc = new TCanvas("myc","",800,600);

    bool skipHisto = false;
    if (fastsim && i>=6 && i<=8) skipHisto = true;

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
      
      if(i > 8 ) {
	f1_prof[i]->SetMarkerSize(0.1);
	f2_prof[i]->SetMarkerSize(0.3);  
      }
      
      myc->SetGrid();
      
      if( i <= 8) {
	f1_prof[i]->Draw("histpl");   
	f2_prof[i]->Draw("histplsame"); // esame   
      }
      else {
	f1_prof[i]->Draw("pl");   
	f2_prof[i]->Draw("plsame"); // esame   
      }

      
      TLegend *leg = new TLegend(0.40, 0.91, 0.74, 0.99, "","brNDC");    
      leg->SetBorderSize(2);
      leg->SetFillStyle(1001); 
      leg->AddEntry(f1_prof[i],"CMSSW_"+ref_vers,"pl");
      leg->AddEntry(f2_prof[i],"CMSSW_"+val_vers,"pl");
      
      leg->Draw("");   
     
      myc->SaveAs(labelp[i]);
    }
    if(myc) delete myc;
  }

  TCanvas *myc = new TCanvas("myc","",800,600);

  TProfile* ratio1 = f2_prof[2]->Clone();
  ratio1->Divide(f1_prof[2]);
  ratio1->SetMaximum(1.2);
  ratio1->SetMinimum(0.8);
  myc->SetGrid();  
  ratio1->Draw("hist pl");

  TLegend *leg = new TLegend(0.20, 0.91, 0.70, 0.99, "","brNDC");
  leg->SetBorderSize(2);
  leg->SetFillStyle(1001);
  leg->AddEntry(ratio1,"CaloTowers scale (pi50) ratio "+val_vers+"/"+ref_vers+" vs ieta","pl");
  leg->Draw("");


  //  f1_prof[2]->Draw();
  myc->SaveAs("Ratio.gif");


   // close ROOT files
   //
   f1.Close() ;  
   f2.Close() ;
   
   return ;  
     
}
