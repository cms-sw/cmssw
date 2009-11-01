// Commands executed in a GLOBAL scope, e.g. created hitograms aren't erased...

void SinglePi(const TString ref_vers="218", const TString val_vers="218"){

   TCanvas *myc = new TCanvas("myc","",800,600);

   TString ref_file = "pi50scan"+ref_vers+"_ECALHCAL_CaloTowers.root";
   TString val_file = "pi50scan"+val_vers+"_ECALHCAL_CaloTowers.root";
      
   TFile f1(ref_file);
   TFile f2(val_file);
   
   // service variables
   //
   const int Nprof   = 6;

   TProfile* f1_prof[Nprof];
   TProfile* f2_prof[Nprof];

   char *labelp[Nprof];

   labelp[0] = &"CaloTowersTask_emean_vs_ieta_E1.gif";
   labelp[1] = &"CaloTowersTask_emean_vs_ieta_H1.gif";
   labelp[2] = &"CaloTowersTask_emean_vs_ieta_EH1.gif";
   labelp[3] = &"RecHitsTask_emean_vs_ieta_E.gif";
   labelp[4] = &"RecHitsTask_emean_vs_ieta_H.gif";
   labelp[5] = &"RecHitsTask_emean_vs_ieta_EH.gif";

   f1.cd("DQMData/CaloTowersV/CaloTowersTask");
   gDirectory->pwd();
   f1_prof[0] = emean_vs_ieta_E1;
   f1_prof[1] = emean_vs_ieta_H1;
   f1_prof[2] = emean_vs_ieta_EH1;

   
   f1.cd("DQMData/HcalRecHitsV/HcalRecHitTask");
   f1_prof[3] = HcalRecHitTask_En_rechits_cone_profile_vs_ieta_all_depths_E;
   f1_prof[4] = HcalRecHitTask_En_rechits_cone_profile_vs_ieta_all_depths;
   f1_prof[5] = HcalRecHitTask_En_rechits_cone_profile_vs_ieta_all_depths_EH;

   f2.cd("DQMData/CaloTowersV/CaloTowersTask");
   gDirectory->pwd();
   f2_prof[0] = emean_vs_ieta_E1;
   f2_prof[1] = emean_vs_ieta_H1;
   f2_prof[2] = emean_vs_ieta_EH1;

   f2.cd("DQMData/HcalRecHitsV/HcalRecHitTask");
   f2_prof[3] = HcalRecHitTask_En_rechits_cone_profile_vs_ieta_all_depths_E;
   f2_prof[4] = HcalRecHitTask_En_rechits_cone_profile_vs_ieta_all_depths;
   f2_prof[5] = HcalRecHitTask_En_rechits_cone_profile_vs_ieta_all_depths_EH;

   //
   f1_prof[0]->GetXaxis()->SetTitle("CaloTowers eE (GeV) vs ieta 1 Tower");
   f1_prof[1]->GetXaxis()->SetTitle("CaloTowers hE (GeV) vs ieta 1 Tower");
   f1_prof[2]->GetXaxis()->SetTitle("CaloTowers eE+hE (GeV) vs ieta 1 Tower");
   f1_prof[3]->GetXaxis()->SetTitle("RecHits eE (GeV) vs ieta R = 0.3 Cone");
   f1_prof[4]->GetXaxis()->SetTitle("RecHits hE (GeV) vs ieta R = 0.3 Cone");
   f1_prof[5]->GetXaxis()->SetTitle("RecHits eE+hE (GeV) vs ieta R = 0.3 Cone");

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

   // f1_hist[2]->GetXaxis()->SetRangeUser(0.,1200.);
   // f1_hist[7]->GetXaxis()->SetRangeUser(0.,160.);
   // hist1->GetXaxis()->SetNdivisions(-21);
   // hist1->GetYaxis()->SetNdivisions(-1003);


   //  Profiles
  for (int i = 0; i < Nprof; i++){

    f1_prof[i]->SetStats(kFALSE);   
    f2_prof[i]->SetStats(kFALSE); 
    
    f1_prof[i]->SetTitle("");
    f2_prof[i]->SetTitle("");


    f1_prof[i]->SetLineColor(41);
    f1_prof[i]->SetLineStyle(1);     
    f1_prof[i]->SetLineWidth(1); 
    f1_prof[i]->SetMarkerColor(41);
    f1_prof[i]->SetMarkerStyle(21);
    f1_prof[i]->SetMarkerSize(0.8);  

    f2_prof[i]->SetLineColor(43);
    f2_prof[i]->SetLineStyle(2);  
    f2_prof[i]->SetLineWidth(1); 
    f2_prof[i]->SetMarkerColor(43);
    f2_prof[i]->SetMarkerStyle(22);
    f2_prof[i]->SetMarkerSize(1.0);  

    myc->SetGrid();

    f1_prof[i]->Draw("hist pl");   
    f2_prof[i]->Draw("hist pl same"); // esame   

    TLegend *leg = new TLegend(0.40, 0.91, 0.74, 0.99, "","brNDC");    
     leg->SetBorderSize(2);
     leg->SetFillStyle(1001); 
     leg->AddEntry(f1_prof[i],"CMSSW_"+ref_vers,"pl");
     leg->AddEntry(f2_prof[i],"CMSSW_"+val_vers,"pl");

     leg->Draw("");   
     
     myc->SaveAs(labelp[i]);

  }

  TProfile* ratio1 = f2_prof[2]->Clone();
  ratio1->Divide(f1_prof[2]);
  ratio1->SetMaximum(2.0);
  ratio1->SetMinimum(0.0);
  ratio1->Draw("hist pl");

  TLegend *leg = new TLegend(0.20, 0.91, 0.70, 0.99, "","brNDC");
  leg->SetBorderSize(2);
  leg->SetFillStyle(1001);
  leg->AddEntry(ratio1,"CaloTowers scale (pi50) ratio 320/310 vs ieta","pl");
  leg->Draw("");


  //  f1_prof[2]->Draw();
  myc->SaveAs("Ratio.gif");


   // close ROOT files
   //
   f1.Close() ;  
   f2.Close() ;
   
   return ;  
     
}
