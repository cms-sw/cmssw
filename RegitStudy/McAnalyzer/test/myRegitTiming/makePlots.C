void makePlots(const char* name_std, const char* name_myregit)
{
   TFile *fstd = new TFile(name_std);
   TFile *fmy = new TFile(name_myregit);

   TCanvas *c1 = new TCanvas();

   // peakvsize
   fstd->cd();
   peakvsize->Draw("peakvsize>>hstd(40,0,4000)");
   TH1F *hstd = (TH1F*) gDirectory->Get("hstd");
   hstd->SetMarkerColor(kRed);
   hstd->SetLineColor(kRed);
   hstd->GetXaxis()->SetTitle("Peak memory (MB)");
   hstd->GetYaxis()->SetTitle("Num. events");
   fmy->cd();
   peakvsize->Draw("peakvsize>>hmy(40,0,4000)");
   TH1F *hmy = (TH1F*) gDirectory->Get("hmy");
   hmy->SetMarkerColor(kBlue);
   hmy->SetLineColor(kBlue);

   hstd->Draw("hist");
   hmy->Draw("hist same");

   TLegend *tleg = new TLegend(0.2,0.2,0.5,0.5);
   tleg->AddEntry(hstd,"Default RegIt","l");
   tleg->AddEntry(hmy,"Proposed RegIt","l");
   tleg->SetFillColor(0);
   tleg->SetBorderSize(0);
   tleg->Draw();

   c1->SaveAs("peakvsize.pdf");
   c1->SaveAs("peakvsize.png");

   // average time
   fstd->cd();
   timestats->Draw("avgt>>hstd(50,0,100)");
   hstd = (TH1F*) gDirectory->Get("hstd");
   hstd->SetMarkerColor(kRed);
   hstd->SetLineColor(kRed);
   hstd->GetXaxis()->SetTitle("Average user time (s)");
   hstd->GetYaxis()->SetTitle("Num. events");
   fmy->cd();
   timestats->Draw("avgt>>hmy(50,0,100)");
   hmy = (TH1F*) gDirectory->Get("hmy");
   hmy->SetMarkerColor(kBlue);
   hmy->SetLineColor(kBlue);

   hstd->Draw("hist");
   hmy->Draw("hist same");

   tleg = new TLegend(0.55,0.4,0.85,0.7);
   tleg->AddEntry(hstd,"Default RegIt","l");
   tleg->AddEntry(hmy,"Proposed RegIt","l");
   tleg->SetFillColor(0);
   tleg->SetBorderSize(0);
   tleg->Draw();

   c1->SaveAs("avgt.pdf");
   c1->SaveAs("avgt.png");

   // max time
   fstd->cd();
   timestats->Draw("maxt>>hstd(50,0,1500)");
   hstd = (TH1F*) gDirectory->Get("hstd");
   hstd->SetMarkerColor(kRed);
   hstd->SetLineColor(kRed);
   hstd->GetXaxis()->SetTitle("Maximum user time (s)");
   hstd->GetYaxis()->SetTitle("Num. events");
   fmy->cd();
   timestats->Draw("maxt>>hmy(50,0,1500)");
   hmy = (TH1F*) gDirectory->Get("hmy");
   hmy->SetMarkerColor(kBlue);
   hmy->SetLineColor(kBlue);

   hstd->Draw("hist");
   hmy->Draw("hist same");

   tleg = new TLegend(0.62,0.62,0.92,0.92);
   tleg->AddEntry(hstd,"Default RegIt","l");
   tleg->AddEntry(hmy,"Proposed RegIt","l");
   tleg->SetFillColor(0);
   tleg->SetBorderSize(0);
   tleg->Draw();

   c1->SaveAs("maxt.pdf");
   c1->SaveAs("maxt.png");
}
