// Commands executed in a GLOBAL scope, e.g. created hitograms aren't erased...
{
  //***************************************************************************


   TCanvas *myc = new TCanvas("myc","",800,600);
      
   //
   //   TFile f1("histo_HF_164.root.root");
   //   TFile f2("simg4hcal_500_ref344.root");
   TFile f3("HF_histo_208.root");
   TFile f4("HF_histo_210pre5.root");
   //   TFile f5("HF_histo_180.root");

   
   // service variables
   //
   const int Nhist = 4;
   //   TH1F* f1_hist[Nhist];
   //   TH1F* f2_hist[Nhist];
   TH1F* f3_hist[Nhist];
   TH1F* f4_hist[Nhist];
   //   TH1F* f5_hist[Nhist];

   //   cout << "1" << endl;

   char *label[Nhist];

   label[0] = &"Thits.gif";
   label[1] = &"Nhits.gif";
   label[2] = &"Long_pe.gif";
   label[3] = &"Short_pe.gif";

   //   f1_hist[0] = (TH1F*)f1.Get("h15") ;
   //   f2_hist[0] = (TH1F*)f2.Get("h15") ;
   f3_hist[0] = (TH1F*)f3.Get("h15") ;
   f4_hist[0] = (TH1F*)f4.Get("h15") ;
   //   f5_hist[0] = (TH1F*)f5.Get("h15") ;

   //  f1_hist[3] = (TH1F*)f1.Get("h17") ;
   //  f2_hist[3] = (TH1F*)f2.Get("h17") ;
   f3_hist[1] = (TH1F*)f3.Get("h17") ;
   f4_hist[1] = (TH1F*)f4.Get("h17") ;
   //   f5_hist[1] = (TH1F*)f5.Get("h17") ;

   //   f1_hist[1] = (TH1F*)f1.Get("h18") ;
   //   f2_hist[1] = (TH1F*)f2.Get("h18") ;
   f3_hist[2] = (TH1F*)f3.Get("h18") ;
   f4_hist[2] = (TH1F*)f4.Get("h18") ;
   //   f5_hist[2] = (TH1F*)f5.Get("h18") ;

   //   f1_hist[2] = (TH1F*)f1.Get("h19") ;
   //   f2_hist[2] = (TH1F*)f2.Get("h19") ;
   f3_hist[3] = (TH1F*)f3.Get("h19") ;
   f4_hist[3] = (TH1F*)f4.Get("h19") ;
   //   f5_hist[3] = (TH1F*)f5.Get("h19") ;


   f3_hist[0]->GetXaxis()->SetTitle("T_{hits} (ns)");
   f3_hist[0]->GetYaxis()->SetTitle("N_{ev}/bin");

   f3_hist[1]->GetXaxis()->SetTitle("N_{hits}");
   f3_hist[1]->GetYaxis()->SetTitle("N_{ev}/bin");

   f3_hist[2]->GetXaxis()->SetTitle("Long Fibers p.e.");
   f3_hist[2]->GetYaxis()->SetTitle("N_{ev}/bin");

   f3_hist[3]->GetXaxis()->SetTitle("Short Fibers p.e. ");
   f3_hist[3]->GetYaxis()->SetTitle("N_{ev}/bin");


   //  f3_hist[0]->SetMaximum(5000.);
   // f1_hist[5]->SetMinimum(0.9);
   // f1_hist[5]->SetMaximum(1.02.);

   for (int i = 0; i < Nhist; i++){

     //    f1_hist[i]->SetStats(kFALSE);   
     //    f2_hist[i]->SetStats(kFALSE); 
    f3_hist[i]->SetStats(kFALSE);
    f4_hist[i]->SetStats(kFALSE);
    //    f5_hist[i]->SetStats(kFALSE);

    //     f1_hist[i]->SetTitle("");
    //     f2_hist[i]->SetTitle("");
     f3_hist[i]->SetTitle("");
     f4_hist[i]->SetTitle("");
     //     f5_hist[i]->SetTitle("");

     /*
     f1_hist[i]->SetLineColor(45);
     f1_hist[i]->SetLineWidth(3); 
     f1_hist[i]->SetLineStyle(1); 
     f2_hist[i]->SetLineColor(41);
     f2_hist[i]->SetLineWidth(3); 
     f2_hist[i]->SetLineStyle(2);  
     */

     f3_hist[i]->SetLineColor(41);
     f3_hist[i]->SetLineWidth(2); 
     f3_hist[i]->SetLineStyle(1); 
     f4_hist[i]->SetLineColor(42);
     f4_hist[i]->SetLineWidth(3); 
     f4_hist[i]->SetLineStyle(2);
     //     f5_hist[i]->SetLineColor(44);
     //     f5_hist[i]->SetLineWidth(3); 
     //     f5_hist[i]->SetLineStyle(3);
     //  h->SetMinimum(0.01);
     //  h->SetMinimum(0.8);

     
     //...Set axis title - sizes are absolute 
     f3_hist[i]->GetXaxis()->SetTickLength(-0.01);
     f3_hist[i]->GetYaxis()->SetTickLength(-0.01);
     f3_hist[i]->GetXaxis()->SetTitleOffset(1.5);
     f3_hist[i]->GetYaxis()->SetTitleOffset(1.5);
     f3_hist[i]->GetXaxis()->SetLabelOffset(0.02);
     f3_hist[i]->GetYaxis()->SetLabelOffset(0.02);
     f3_hist[i]->GetXaxis()->SetLabelSize(0.04);
     f3_hist[i]->GetYaxis()->SetLabelSize(0.04);
     f3_hist[i]->GetXaxis()->SetTitleSize(0.045);
     f3_hist[i]->GetYaxis()->SetTitleSize(0.045);

     //...Legend
     TLegend *leg = new TLegend(0.65, 0.8, 0.93, 0.95, "","brNDC");
     leg->SetBorderSize(2);
     //  leg->SetFillColor(51); // see new color definition above
     leg->SetFillStyle(1001); //
     //     leg->AddEntry(f1_hist[i],"OSCAR_3_9_82","l");
     //     leg->AddEntry(f2_hist[i],"OSCAR_5_0_0","l");
     leg->AddEntry(f3_hist[i],"CMSSW_2_0_8","l");
     leg->AddEntry(f4_hist[i],"CMSSW_2_1_0pre5","l");
     //     leg->AddEntry(f5_hist[i],"CMSSW_1_8_0_re5","l");
     
     //     f1_hist[i]->Draw("hist");   
     //     f2_hist[i]->Draw("hist same");   
     f3_hist[i]->Draw("hist");   
     f4_hist[i]->Draw("hist same");   
     //     f5_hist[i]->Draw("hist same");   
     leg->Draw();   

     
     myc->SaveAs(label[i]);
   }     


   // close ROOT files
   //
   //   f1.Close() ;  
   //   f2.Close() ;
   f3.Close() ;
   f4.Close() ;
   //   f5.Close() ;
   
   
   return ;  
     
}
