// Commands executed in a GLOBAL scope, e.g. created hitograms aren't erased...
{

   TCanvas *myc = new TCanvas("myc","",800,600);
      
   //

   TFile f1("HE_histo_pre5.root");
   TFile f2("HE_histo_pre6.root");

   // service variables
   //
   const int Nhist = 9;
   TH1F* f1_hist[Nhist];
   TH1F* f2_hist[Nhist];
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

   f1_hist[1] = (TH1F*)f1.Get("h45") ;
   f2_hist[1] = (TH1F*)f2.Get("h45") ;

   f1_hist[2] = (TH1F*)f1.Get("h43") ;
   f2_hist[2] = (TH1F*)f2.Get("h43") ;

   f1_hist[3] = (TH1F*)f1.Get("h40") ;
   f2_hist[3] = (TH1F*)f2.Get("h40") ;

   f1_hist[4] = (TH1F*)f1.Get("h41") ;
   f2_hist[4] = (TH1F*)f2.Get("h41") ;

   f1_hist[5] = (TH1F*)f1.Get("h39") ;
   f2_hist[5] = (TH1F*)f2.Get("h39") ;

   f1_hist[6] = (TH1F*)f1.Get("hl6") ;
   f2_hist[6] = (TH1F*)f2.Get("hl6") ;

   f1_hist[7] = (TH1F*)f1.Get("h6") ;
   f2_hist[7] = (TH1F*)f2.Get("h6") ;

   f1_hist[8] = (TH1F*)f1.Get("h46") ;
   f2_hist[8] = (TH1F*)f2.Get("h46") ;

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


   f1_hist[2]->GetXaxis()->SetRangeUser(0.,2000.);
   f1_hist[7]->GetXaxis()->SetRangeUser(0.,100.);
   f1_hist[8]->GetXaxis()->SetRangeUser(0.,1.);
  // hist1->GetXaxis()->SetNdivisions(-21);
  //  hist1->GetYaxis()->SetNdivisions(-1003);

   for (int i = 0; i < Nhist; i++){

     if(i == 5 ) { 
       f1_hist[i]->SetStats(kFALSE);   
       f2_hist[i]->SetStats(kFALSE); 
     }
     else {
       f1_hist[i]->SetStats(kTRUE);   
       f2_hist[i]->SetStats(kTRUE); 
     }
 
     f1_hist[i]->SetTitle("");
     f2_hist[i]->SetTitle("");

     f1_hist[i]->SetLineWidth(2); 
     f2_hist[i]->SetLineWidth(3); 

     // diffferent histo colors and styles
     f1_hist[i]->SetLineColor(41);
     f1_hist[i]->SetLineStyle(1); 

     f2_hist[i]->SetLineColor(43);
     f2_hist[i]->SetLineStyle(2);  

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

     TLegend *leg = new TLegend(0.55, 0.91, 0.84, 0.99, "","brNDC");

     leg->SetBorderSize(2);
     //  leg->SetFillColor(51); // see new color definition above
     leg->SetFillStyle(1001); //
     leg->AddEntry(f1_hist[i],"CMSSW_300pre5","l");
     leg->AddEntry(f2_hist[i],"CMSSW_300pre6","l");
     
     if (i != 5) {
       TPaveStats *ptstats = new TPaveStats(0.85,0.86,0.98,0.98,"brNDC");
       ptstats->SetTextColor(41);
       f1_hist[i]->GetListOfFunctions()->Add(ptstats);
       ptstats->SetParent(f1_hist[i]->GetListOfFunctions());
       TPaveStats *ptstats = new TPaveStats(0.85,0.74,0.98,0.86,"brNDC");
       ptstats->SetTextColor(43);
       f2_hist[i]->GetListOfFunctions()->Add(ptstats);
       ptstats->SetParent(f2_hist[i]->GetListOfFunctions());

       f1_hist[i]->Draw(""); // "stat"
       f2_hist[i]->Draw("hist sames");
     }
     else {
       f1_hist[i]->Draw("hist");
       f2_hist[i]->Draw("hist same");
     }

     leg->Draw();   

     // Chi2 test
     if(i == 2 || i == 3 || i == 7 || i ==8 ) {
       const float NCHI2MIN = 0.01;
       
       float pval;
       stringstream mystream;
       char tempbuff[30];
       
       pval = f1_hist[i]->Chi2Test(f2_hist[i]);
       //     cout << "i_hist "  << i << " pval = " << pval << endl;
       
       sprintf(tempbuff,"Chi2 p-value: %6.3E%c",pval,'\0');
       mystream<<tempbuff;
       
       TPaveText* ptchi2 = new TPaveText(0.25, 0.91, 0.5, 0.99, "NDC");
       
       if (pval > NCHI2MIN) ptchi2->SetFillColor(kGreen);
       else                 ptchi2->SetFillColor(kRed);
       
       ptchi2->SetTextSize(0.03);
       ptchi2->AddText(mystream.str().c_str()); 
       ptchi2->Draw();

     }
     
     myc->SaveAs(label[i]);
   }     

   // close ROOT files
   //
   f1.Close() ;  
   f2.Close() ;
   
   return ;  
     
}
