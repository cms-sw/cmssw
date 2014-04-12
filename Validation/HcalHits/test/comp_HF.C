// Commands executed in a GLOBAL scope, e.g. created hitograms aren't erased...
{
  //***************************************************************************


   TCanvas *myc = new TCanvas("myc","",800,600);
      
   TFile f3("HF_histo_pre5.root");
   TFile f4("HF_histo_pre6.root");

   const int Nhist = 4;
   TH1F* f3_hist[Nhist];
   TH1F* f4_hist[Nhist];
   //   cout << "1" << endl;

   char *label[Nhist];

   label[0] = &"Thits.gif";
   label[1] = &"Nhits.gif";
   label[2] = &"Long_pe.gif";
   label[3] = &"Short_pe.gif";

   f3_hist[0] = (TH1F*)f3.Get("h15") ;
   f4_hist[0] = (TH1F*)f4.Get("h15") ;

   f3_hist[1] = (TH1F*)f3.Get("h17") ;
   f4_hist[1] = (TH1F*)f4.Get("h17") ;

   f3_hist[2] = (TH1F*)f3.Get("h18") ;
   f4_hist[2] = (TH1F*)f4.Get("h18") ;

   f3_hist[3] = (TH1F*)f3.Get("h19") ;
   f4_hist[3] = (TH1F*)f4.Get("h19") ;

   //
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
 
     f3_hist[i]->SetStats(kTRUE);
     f4_hist[i]->SetStats(kTRUE);

     f3_hist[i]->SetTitle("");
     f4_hist[i]->SetTitle("");
    
     f3_hist[i]->SetLineColor(41);
     f3_hist[i]->SetLineWidth(2); 
     f3_hist[i]->SetLineStyle(1); 
     f4_hist[i]->SetLineColor(43);
     f4_hist[i]->SetLineWidth(3); 
     f4_hist[i]->SetLineStyle(2);
     
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

     // Chi2 test
     const float NCHI2MIN = 0.01;
     
     float pval;
     stringstream mystream;
     char tempbuff[30];
     
     pval = f3_hist[i]->Chi2Test(f4_hist[i]);
     
     sprintf(tempbuff,"Chi2 p-value: %6.3E%c",pval,'\0');
     mystream<<tempbuff;
     
     TPaveText* ptchi2 = new TPaveText(0.25, 0.91, 0.5, 0.99, "NDC");
     
     if (pval > NCHI2MIN) ptchi2->SetFillColor(kGreen);
     else                 ptchi2->SetFillColor(kRed);
     
     ptchi2->SetTextSize(0.03);
     ptchi2->AddText(mystream.str().c_str());
     //



     //...Legend
     TLegend *leg = new TLegend(0.55, 0.91, 0.84, 0.99, "","brNDC");
     leg->SetBorderSize(2);
     leg->SetFillStyle(1001); //

     leg->AddEntry(f3_hist[i],"CMSSW_300pre5","l");
     leg->AddEntry(f4_hist[i],"CMSSW_300pre6","l");


     if (i >= 0) {
       TPaveStats *ptstats = new TPaveStats(0.85,0.86,0.98,0.98,"brNDC");
       ptstats->SetTextColor(41);
       f3_hist[i]->GetListOfFunctions()->Add(ptstats);
       ptstats->SetParent(f3_hist[i]->GetListOfFunctions());
       TPaveStats *ptstats = new TPaveStats(0.85,0.74,0.98,0.86,"brNDC");
       ptstats->SetTextColor(43);
       f4_hist[i]->GetListOfFunctions()->Add(ptstats);
       ptstats->SetParent(f4_hist[i]->GetListOfFunctions());
       
       f3_hist[i]->Draw(""); // "stat"   
       f4_hist[i]->Draw("hist sames");   
     }
     else { 
       f3_hist[i]->Draw("hist");   
       f4_hist[i]->Draw("hist same");   
     }

     ptchi2->Draw();
     leg->Draw();   

     
     myc->SaveAs(label[i]);
   }     


   // close ROOT files

   f3.Close() ;
   f4.Close() ;
   
   return ;  
     
}
