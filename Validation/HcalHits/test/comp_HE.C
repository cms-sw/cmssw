// Commands executed in a GLOBAL scope, e.g. created hitograms aren't erased...
{

   TCanvas *myc = new TCanvas("myc","",800,600);
      
   //

   TFile f1("HE_histo_208.root");
   TFile f2("HE_histo_210pre5.root");

   //   TFile f3("HE_histo_180.root");
   //   TFile f4("HB_QGSP_4821.root");
   //   TFile f5("HB_QGSP_EMV_4821.root");

   
   // service variables
   //
   const int Nhist = 9;
   TH1F* f1_hist[Nhist];
   TH1F* f2_hist[Nhist];
   //   TH1F* f3_hist[Nhist];
   //   TH1F* f4_hist[Nhist];
   //   TH1F* f5_hist[Nhist];


   char *label[Nhist];

   label[0] = &"layer0.gif";
   label[1] = &"Lprofile.gif";
   label[2] = &"Nhits.gif";
   label[3] = &"Thits.gif";
   label[4] = &"ThitsEw.gif";
   label[5] = &"NxNfract.gif";
   label[6] = &"layer7.gif";
   label[7] = &"Erec_cone.gif";
   label[8] = &"Edep_tot.gif";

   f1_hist[0] = (TH1F*)f1.Get("hl0") ;
   f2_hist[0] = (TH1F*)f2.Get("hl0") ;
   //   f3_hist[0] = (TH1F*)f3.Get("hl0") ;
   //   f4_hist[0] = (TH1F*)f4.Get("hl0") ;
   //   f5_hist[0] = (TH1F*)f5.Get("hl0") ;

   f1_hist[1] = (TH1F*)f1.Get("h45") ;
   f2_hist[1] = (TH1F*)f2.Get("h45") ;
   //   f3_hist[1] = (TH1F*)f3.Get("h45") ;
   //   f4_hist[1] = (TH1F*)f4.Get("h45") ;
   //   f5_hist[1] = (TH1F*)f5.Get("h45") ;

   f1_hist[2] = (TH1F*)f1.Get("h43") ;
   f2_hist[2] = (TH1F*)f2.Get("h43") ;
   //   f3_hist[2] = (TH1F*)f3.Get("h43") ;
   //   f4_hist[2] = (TH1F*)f4.Get("h43") ;
   //   f5_hist[2] = (TH1F*)f5.Get("h43") ;


   f1_hist[3] = (TH1F*)f1.Get("h40") ;
   f2_hist[3] = (TH1F*)f2.Get("h40") ;
   //   f3_hist[3] = (TH1F*)f3.Get("h40") ;
   //   f4_hist[3] = (TH1F*)f4.Get("h40") ;
   //   f5_hist[3] = (TH1F*)f5.Get("h40") ;

   f1_hist[4] = (TH1F*)f1.Get("h41") ;
   f2_hist[4] = (TH1F*)f2.Get("h41") ;
   //   f3_hist[4] = (TH1F*)f3.Get("h41") ;
   //   f4_hist[4] = (TH1F*)f4.Get("h41") ;
   //   f5_hist[4] = (TH1F*)f5.Get("h41") ;

   f1_hist[5] = (TH1F*)f1.Get("h39") ;
   f2_hist[5] = (TH1F*)f2.Get("h39") ;
   //   f3_hist[5] = (TH1F*)f3.Get("h39") ;
   //   f4_hist[5] = (TH1F*)f4.Get("h39") ;
   //   f5_hist[5] = (TH1F*)f5.Get("h39") ;

   f1_hist[6] = (TH1F*)f1.Get("hl6") ;
   f2_hist[6] = (TH1F*)f2.Get("hl6") ;
   //   f3_hist[6] = (TH1F*)f3.Get("hl6") ;
   //   f4_hist[6] = (TH1F*)f4.Get("hl6") ;
   //   f5_hist[6] = (TH1F*)f5.Get("hl6") ;

   f1_hist[7] = (TH1F*)f1.Get("h6") ;
   f2_hist[7] = (TH1F*)f2.Get("h6") ;
   //   f3_hist[7] = (TH1F*)f3.Get("h6") ;
   //   f4_hist[7] = (TH1F*)f4.Get("h6") ;
   //   f5_hist[7] = (TH1F*)f5.Get("h6") ;

   f1_hist[8] = (TH1F*)f1.Get("h46") ;
   f2_hist[8] = (TH1F*)f2.Get("h46") ;
   //   f3_hist[8] = (TH1F*)f3.Get("h46") ;
   //   f4_hist[8] = (TH1F*)f4.Get("h46") ;
   //   f5_hist[8] = (TH1F*)f5.Get("h46") ;

   f1_hist[0]->GetXaxis()->SetTitle("L0 Deposited energy (GeV)");
   f1_hist[0]->GetYaxis()->SetTitle("N_{ev}/bin");

   f1_hist[1]->GetXaxis()->SetTitle("N_{layer}");
   f1_hist[1]->GetYaxis()->SetTitle("Deposited energy (GeV)");

   f1_hist[2]->GetXaxis()->SetTitle("N_{hits}");
   f1_hist[2]->GetYaxis()->SetTitle("N_{ev}");

   f1_hist[3]->GetXaxis()->SetTitle("t_{hits} (ns)");
   f1_hist[3]->GetYaxis()->SetTitle("N_{ev}/bin");

   f1_hist[4]->GetXaxis()->SetTitle("t_{hist} (ns)");
   f1_hist[4]->GetYaxis()->SetTitle("Energy/bin (MeV)");

   f1_hist[5]->GetXaxis()->SetTitle("NxN square number");
   f1_hist[5]->GetYaxis()->SetTitle("Fraction of energy");

   f1_hist[6]->GetXaxis()->SetTitle("L7 Deposited energy (GeV)");
   f1_hist[6]->GetYaxis()->SetTitle("N_{ev}/bin");

   f1_hist[7]->GetXaxis()->SetTitle("Pseudo-reconstructed energy (GeV)");
   f1_hist[7]->GetYaxis()->SetTitle("N_{ev}/bin");

   f1_hist[8]->GetXaxis()->SetTitle("Deposited energy (GeV)");
   f1_hist[8]->GetYaxis()->SetTitle("N_{ev}/bin");

   /*

   f1_hist[0]->SetMaximum(2500.);
   f1_hist[1]->SetMaximum(300.);
   f1_hist[2]->SetMaximum(150.);
   //   f1_hist[3]->SetMaximum(300.);
   f1_hist[4]->SetMaximum(70000.);
   f1_hist[5]->SetMinimum(0.7);
   f1_hist[5]->SetMaximum(1.02.);
   f1_hist[6]->SetMaximum(1000.);
   //   f1_hist[6]->SetMaximum(600.);
   f1_hist[7]->SetMaximum(350.);
   f1_hist[8]->SetMaximum(350.);
   */

   f1_hist[2]->GetXaxis()->SetRangeUser(0.,1000.);
   f1_hist[7]->GetXaxis()->SetRangeUser(0.,100.);
   f1_hist[8]->GetXaxis()->SetRangeUser(0.,1.);
  // hist1->GetXaxis()->SetNdivisions(-21);
  //  hist1->GetYaxis()->SetNdivisions(-1003);

   for (int i = 0; i < Nhist; i++){

    f1_hist[i]->SetStats(kFALSE);   
    f2_hist[i]->SetStats(kFALSE); 
    //    f3_hist[i]->SetStats(kFALSE); 
    //    f4_hist[i]->SetStats(kFALSE); 
    //    f5_hist[i]->SetStats(kFALSE); 

 
     f1_hist[i]->SetTitle("");
     f2_hist[i]->SetTitle("");
     //     f3_hist[i]->SetTitle("");
     //     f4_hist[i]->SetTitle("");
     //     f5_hist[i]->SetTitle("");

     f1_hist[i]->SetLineWidth(2); 
     f2_hist[i]->SetLineWidth(3); 
     //     f3_hist[i]->SetLineWidth(3); 
     //     f4_hist[i]->SetLineWidth(3); 
     //     f5_hist[i]->SetLineWidth(3); 


     // diffferent histo colors and styles
     f1_hist[i]->SetLineColor(41);
     f1_hist[i]->SetLineStyle(1); 

     f2_hist[i]->SetLineColor(42);
     f2_hist[i]->SetLineStyle(2);  

     /*     
     f3_hist[i]->SetLineColor(44);
     f3_hist[i]->SetLineStyle(3);  

    
     f4_hist[i]->SetLineColor(44);
     f4_hist[i]->SetLineStyle(4);  

     f5_hist[i]->SetLineColor(46);
     f5_hist[i]->SetLineStyle(5);  
     */
     
     //...Set axis title - sizes are absolute 
     f1_hist[i]->GetXaxis()->SetTickLength(-0.01);
     f1_hist[i]->GetYaxis()->SetTickLength(-0.01);
     f1_hist[i]->GetXaxis()->SetTitleOffset(1.5);
     f1_hist[i]->GetYaxis()->SetTitleOffset(1.5);
     f1_hist[i]->GetXaxis()->SetLabelOffset(0.02);
     f1_hist[i]->GetYaxis()->SetLabelOffset(0.02);
     f1_hist[i]->GetXaxis()->SetLabelSize(0.04);
     f1_hist[i]->GetYaxis()->SetLabelSize(0.04);
     f1_hist[i]->GetXaxis()->SetTitleSize(0.045);
     f1_hist[i]->GetYaxis()->SetTitleSize(0.045);

    
     if(i == 0 || i == 1 || i == 2 || i == 3 || i == 4 || i == 5 || i == 6 || i == 8) {
       TLegend *leg = new TLegend(0.67, 0.55, 0.97, 0.8, "","brNDC");
     }
     else {
       TLegend *leg = new TLegend(0.08, 0.7, 0.38, 0.85, "","brNDC");
     }

     leg->SetBorderSize(2);
     //  leg->SetFillColor(51); // see new color definition above
     leg->SetFillStyle(1001); //
     leg->AddEntry(f1_hist[i],"CMSSW_2_0_8","l");
     leg->AddEntry(f2_hist[i],"CMSSW_2_1_0pre5","l");
     //     leg->AddEntry(f3_hist[i],"CMSSW_1_8_0","l");
     //     leg->AddEntry(f3_hist[i],"CMSSW_1_2_0_4821","l");
     //     leg->AddEntry(f4_hist[i],"QGSP_4.8.2p1","l");
     //     leg->AddEntry(f5_hist[i],"QGSP_EMV_4.8.2p1","l");
     
     f1_hist[i]->Draw("hist");   
     f2_hist[i]->Draw("hist same");   
     //     f3_hist[i]->Draw("hist same");   
     //     f4_hist[i]->Draw("hist same");   
     //     f5_hist[i]->Draw("hist same");   
     leg->Draw();   
     
     myc->SaveAs(label[i]);
   }     


   // close ROOT files
   //
   f1.Close() ;  
   f2.Close() ;
   //   f3.Close() ;
   //   f4.Close() ;
   //   f5.Close() ;
   
   return ;  
     
}
