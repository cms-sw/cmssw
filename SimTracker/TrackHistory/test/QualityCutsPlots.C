void QualityCutsPlots()
{
    gROOT->SetStyle("Plain");
    TFile * file = new TFile("test.root");

//=========Macro generated from canvas: c1/c1
//=========  (Wed May 16 12:11:16 2007) by ROOT version5.14/00b

    TCanvas *c1 = new TCanvas("c1", "c1",547,72,698,498);

    c1->Divide(2,2);

    c1->cd(1);

    gPad->SetLogy();

    THStack * hs1 = new THStack("hs", "Signed decay length for all tracks.");

    //sdlsdl_fake_tracks->SetLineColor(kGreen);
    sdl_fake_tracks->SetFillColor(kGreen);
    processWithErrors(sdl_fake_tracks);
    hs1->Add(sdl_fake_tracks);

    //sdl_displaced_tracks->SetLineColor(kRed);
    sdl_displaced_tracks->SetFillColor(kRed);
    processWithErrors(sdl_displaced_tracks);
    hs1->Add(sdl_displaced_tracks);

    //sdl_bad_tracks->SetLineColor(kYellow);
    sdl_bad_tracks->SetFillColor(kCyan);
    processWithErrors(sdl_bad_tracks);
    hs1->Add(sdl_bad_tracks);

    // sdl_B_tracks->SetLineColor(kBlue);
    sdl_B_tracks->SetFillColor(kBlue);
    processWithErrors(sdl_B_tracks);
    hs1->Add(sdl_B_tracks);

    //sdl_nonB_tracks->SetLineColor(kBlack);
    sdl_nonB_tracks->SetFillColor(kBlack);
    processWithErrors(sdl_nonB_tracks);
    hs1->Add(sdl_nonB_tracks);

    hs1->Draw("nostack bar");
    hs1->GetXaxis()->SetTitle("Signed decay length [cm]");
    hs1->GetYaxis()->SetTitle("Relative frequency");

    TLegend *leg1 = new TLegend(0.62458,0.692819,0.885747,0.839096,NULL,"brNDC");
    leg1->SetBorderSize(1);
    leg1->SetLineColor(0);
    leg1->SetLineStyle(1);
    leg1->SetLineWidth(1);
    leg1->SetFillColor(0);
    leg1->SetFillStyle(1001);
    leg1->AddEntry(sdl_fake_tracks,"Fake", "f");
    leg1->AddEntry(sdl_bad_tracks,"Bad", "f");
    leg1->AddEntry(sdl_displaced_tracks,"Displaced", "f");
    leg1->AddEntry(sdl_B_tracks,"B", "f");
    leg1->AddEntry(sdl_nonB_tracks,"Non B", "f");
    leg1->Draw();

    c1->cd(2);

    gPad->SetLogy();

    THStack * hs2 = new THStack("hs", "Distance to jet for all tracks.");

    // dta_fake_tracks->SetLineColor(kBlack);
    dta_fake_tracks->SetFillColor(kGreen);
    processWithErrors(dta_fake_tracks);
    hs2->Add(dta_fake_tracks);

    // dta_displaced_tracks->SetLineColor(kWhite);
    dta_displaced_tracks->SetFillColor(kRed);
    processWithErrors(dta_displaced_tracks);
    hs2->Add(dta_displaced_tracks);

    //sdl_bad_tracks->SetLineColor(kYellow);
    dta_bad_tracks->SetFillColor(kCyan);
    processWithErrors(dta_bad_tracks);
    hs2->Add(dta_bad_tracks);

    // dta_B_tracks->SetLineColor(kWhite);
    dta_B_tracks->SetFillColor(kBlue);
    processWithErrors(dta_B_tracks);
    hs2->Add(dta_B_tracks);

    // dta_nonB_tracks->SetLineColor(kWhite);
    dta_nonB_tracks->SetFillColor(kBlack);
    processWithErrors(dta_nonB_tracks);
    hs2->Add(dta_nonB_tracks);

    hs2->Draw("nostack bar");
    hs2->GetXaxis()->SetTitle("Distance to jet [cm]");
    hs2->GetYaxis()->SetTitle("Relative frequency");

    TLegend *leg2 = new TLegend(0.62458,0.692819,0.885747,0.839096,NULL,"brNDC");
    leg2->SetBorderSize(1);
    leg2->SetLineColor(0);
    leg2->SetLineStyle(1);
    leg2->SetLineWidth(1);
    leg2->SetFillColor(0);
    leg2->SetFillStyle(1001);
    leg2->AddEntry(dta_fake_tracks,"Fake", "f");
    leg2->AddEntry(dta_bad_tracks,"Bad", "f");
    leg2->AddEntry(dta_displaced_tracks,"Displaced", "f");
    leg2->AddEntry(dta_B_tracks,"B", "f");
    leg2->AddEntry(dta_nonB_tracks,"Non B", "f");
    leg2->Draw();

    c1->cd(3);

    gPad->SetLogy();

    THStack * hs3 = new THStack("hs", "LIP for all tracks.");

    // lip_fake_tracks->SetLineColor(kBlack);
    lip_fake_tracks->SetFillColor(kGreen);
    processWithErrors(lip_fake_tracks);
    hs3->Add(lip_fake_tracks);

    // lip_displaced_tracks->SetLineColor(kWhite);
    lip_displaced_tracks->SetFillColor(kRed);
    processWithErrors(lip_displaced_tracks);
    hs3->Add(lip_displaced_tracks);

    //sdl_bad_tracks->SetLineColor(kYellow);
    lip_bad_tracks->SetFillColor(kCyan);
    processWithErrors(lip_bad_tracks);
    hs3->Add(lip_bad_tracks);

    // lip_B_tracks->SetLineColor(kWhite);
    lip_B_tracks->SetFillColor(kBlue);
    processWithErrors(lip_B_tracks);
    hs3->Add(lip_B_tracks);

    // lip_nonB_tracks->SetLineColor(kWhite);
    lip_nonB_tracks->SetFillColor(kBlack);
    processWithErrors(lip_nonB_tracks);
    hs3->Add(lip_nonB_tracks);

    hs3->Draw("nostack bar");
    hs3->GetXaxis()->SetTitle("Logitudinal impact parameter [cm]");
    hs3->GetYaxis()->SetTitle("Relative frequency");

    TLegend *leg3 = new TLegend(0.62458,0.692819,0.885747,0.839096,NULL,"brNDC");
    leg3->SetBorderSize(1);
    leg3->SetLineColor(0);
    leg3->SetLineStyle(1);
    leg3->SetLineWidth(1);
    leg3->SetFillColor(0);
    leg3->SetFillStyle(1001);
    leg3->AddEntry(lip_fake_tracks,"Fake", "f");
    leg3->AddEntry(lip_bad_tracks,"Bad", "f");
    leg3->AddEntry(lip_displaced_tracks,"Displaced", "f");
    leg3->AddEntry(lip_B_tracks,"B", "f");
    leg3->AddEntry(lip_nonB_tracks,"Non B", "f");
    leg3->Draw();

    c1->cd(4);

    gPad->SetLogy();

    THStack * hs4 = new THStack("hs", "TIP for all tracks.");

    // tip_fake_tracks->SetLineColor(kBlack);
    tip_fake_tracks->SetFillColor(kGreen);
    processWithErrors(tip_fake_tracks);
    hs4->Add(tip_fake_tracks);

    // tip_displaced_tracks->SetLineColor(kWhite);
    tip_displaced_tracks->SetFillColor(kRed);
    processWithErrors(tip_displaced_tracks);
    hs4->Add(tip_displaced_tracks);

    tip_bad_tracks->SetFillColor(kCyan);
    processWithErrors(tip_bad_tracks);
    hs4->Add(tip_bad_tracks);

    // tip_B_tracks->SetLineColor(kWhite);
    tip_B_tracks->SetFillColor(kBlue);
    processWithErrors(tip_B_tracks);
    hs4->Add(tip_B_tracks);

    // tip_nonB_tracks->SetLineColor(kWhite);
    tip_nonB_tracks->SetFillColor(kBlack);
    processWithErrors(tip_nonB_tracks);
    hs4->Add(tip_nonB_tracks);

    hs4->Draw("nostack bar");
    hs4->GetXaxis()->SetTitle("Transverse impact parameter [cm]");
    hs4->GetYaxis()->SetTitle("Relative frequency");

    TLegend *leg4 = new TLegend(0.62458,0.692819,0.885747,0.839096,NULL,"brNDC");
    leg4->SetBorderSize(1);
    leg4->SetLineColor(0);
    leg4->SetLineStyle(1);
    leg4->SetLineWidth(1);
    leg4->SetFillColor(0);
    leg4->SetFillStyle(1001);
    leg4->AddEntry(tip_fake_tracks,"Fake", "f");
    leg4->AddEntry(tip_bad_tracks,"Bad", "f");
    leg4->AddEntry(tip_displaced_tracks,"Displaced", "f");
    leg4->AddEntry(tip_B_tracks,"B", "f");
    leg4->AddEntry(tip_nonB_tracks,"Non B", "f");
    leg4->Draw();

    TCanvas *c2 = new TCanvas("c2", "c1",547,72,698,498);

    c2->Divide(2,2);

    c2->cd(1);

    gPad->SetLogy();

    THStack * hs5 = new THStack("hs", "Track pt for all tracks.");

    // pt_1Gev_fake_tracks->SetLineColor(kGreen);
    pt_1Gev_fake_tracks->SetFillColor(kGreen);
    processWithErrors(pt_1Gev_fake_tracks);
    hs5->Add(pt_1Gev_fake_tracks);

    pt_1Gev_bad_tracks->SetFillColor(kCyan);
    processWithErrors(pt_1Gev_bad_tracks);
    hs5->Add(pt_1Gev_bad_tracks);

    // pt_1Gev_displaced_tracks->SetLineColor(kRed);
    pt_1Gev_displaced_tracks->SetFillColor(kRed);
    processWithErrors(pt_1Gev_displaced_tracks);
    hs5->Add(pt_1Gev_displaced_tracks);

    // pt_1Gev_nonB_tracks->SetLineColor(kBlack);
    pt_1Gev_nonB_tracks->SetFillColor(kBlack);
    processWithErrors(pt_1Gev_nonB_tracks);
    hs5->Add(pt_1Gev_nonB_tracks);

    // pt_1Gev_B_tracks->SetLineColor(kBlue);
    pt_1Gev_B_tracks->SetFillColor(kBlue);
    processWithErrors(pt_1Gev_B_tracks);
    hs5->Add(pt_1Gev_B_tracks);

    hs5->Draw("nostack bar");
    hs5->GetXaxis()->SetTitle("pt [GeV]");
    hs5->GetYaxis()->SetTitle("Relative frequency");

    TLegend *leg5 = new TLegend(0.62458,0.692819,0.885747,0.839096,NULL,"brNDC");
    leg5->SetBorderSize(1);
    leg5->SetLineColor(0);
    leg5->SetLineStyle(1);
    leg5->SetLineWidth(1);
    leg5->SetFillColor(0);
    leg5->SetFillStyle(1001);
    leg5->AddEntry(pt_1Gev_fake_tracks,"Fake", "f");
    leg5->AddEntry(pt_1Gev_bad_tracks,"Bad", "f");
    leg5->AddEntry(pt_1Gev_displaced_tracks,"Displaced", "f");
    leg5->AddEntry(pt_1Gev_B_tracks,"B", "f");
    leg5->AddEntry(pt_1Gev_nonB_tracks,"Non B", "f");
    leg5->Draw();

    c2->cd(2);

    gPad->SetLogy();

    THStack * hs6 = new THStack("hs", "Track pt for all tracks.");

    //pt_B_tracks->SetLineColor(kBlue);
    pt_B_tracks->SetFillColor(kBlue);
    processWithErrors(pt_B_tracks);
    hs6->Add(pt_B_tracks);

    //pt_fake_tracks->SetLineColor(kGreen);
    pt_fake_tracks->SetFillColor(kGreen);
    processWithErrors(pt_fake_tracks);
    hs6->Add(pt_fake_tracks);

    pt_bad_tracks->SetFillColor(kCyan);
    processWithErrors(pt_bad_tracks);
    hs6->Add(pt_bad_tracks);

    //pt_nonB_tracks->SetLineColor(kBlack);
    pt_nonB_tracks->SetFillColor(kBlack);
    processWithErrors(pt_nonB_tracks);
    hs6->Add(pt_nonB_tracks);

    //pt_displaced_tracks->SetLineColor(kRed);
    pt_displaced_tracks->SetFillColor(kRed);
    processWithErrors(pt_displaced_tracks);
    hs6->Add(pt_displaced_tracks);

    hs6->Draw("nostack bar");
    hs6->GetXaxis()->SetTitle("pt [GeV]");
    hs6->GetYaxis()->SetTitle("Relative frequency");

    TLegend *leg6 = new TLegend(0.62458,0.692819,0.885747,0.839096,NULL,"brNDC");
    leg6->SetBorderSize(1);
    leg6->SetLineColor(0);
    leg6->SetLineStyle(1);
    leg6->SetLineWidth(1);
    leg6->SetFillColor(0);
    leg6->SetFillStyle(1001);
    leg6->AddEntry(pt_fake_tracks,"Fake", "f");
    leg6->AddEntry(pt_bad_tracks,"Bad", "f");
    leg6->AddEntry(pt_displaced_tracks,"Displaced", "f");
    leg6->AddEntry(pt_B_tracks,"B", "f");
    leg6->AddEntry(pt_nonB_tracks,"Non B", "f");
    leg6->Draw();

    c2->cd(3);

    gPad->SetLogy();

    THStack * hs7 = new THStack("hs", "Normilized Chi2 for all tracks.");

    //chi2_fake_tracks->SetLineColor(kGreen);
    chi2_fake_tracks->SetFillColor(kGreen);
    processWithErrors(chi2_fake_tracks);
    hs7->Add(chi2_fake_tracks);

    chi2_bad_tracks->SetFillColor(kCyan);
    processWithErrors(chi2_bad_tracks);
    hs7->Add(chi2_bad_tracks);

    //chi2_displaced_tracks->SetLineColor(kRed);
    chi2_displaced_tracks->SetFillColor(kRed);
    processWithErrors(chi2_displaced_tracks);
    hs7->Add(chi2_displaced_tracks);

    //chi2_B_tracks->SetLineColor(kBlue);
    chi2_B_tracks->SetFillColor(kBlue);
    processWithErrors(chi2_B_tracks);
    hs7->Add(chi2_B_tracks);

    //chi2_nonB_tracks->SetLineColor(kBlack);
    chi2_nonB_tracks->SetFillColor(kBlack);
    processWithErrors(chi2_nonB_tracks);
    hs7->Add(chi2_nonB_tracks);

    hs7->Draw("nostack bar");
    hs7->GetXaxis()->SetTitle("Normilized Chi2");
    hs7->GetYaxis()->SetTitle("Relative frequency");

    TLegend * leg7 = new TLegend(0.62458,0.692819,0.885747,0.839096,NULL,"brNDC");
    leg7->SetBorderSize(1);
    leg7->SetLineColor(0);
    leg7->SetLineStyle(1);
    leg7->SetLineWidth(1);
    leg7->SetFillColor(0);
    leg7->SetFillStyle(1001);
    leg7->AddEntry(chi2_fake_tracks,"Fake", "f");
    leg7->AddEntry(chi2_bad_tracks,"Bad", "f");
    leg7->AddEntry(chi2_displaced_tracks,"Displaced", "f");
    leg7->AddEntry(chi2_B_tracks,"B", "f");
    leg7->AddEntry(chi2_nonB_tracks,"Non B", "f");
    leg7->Draw();

    c2->cd(4);

    gPad->SetLogy();

    THStack * hs8 = new THStack("hs", "Hits for all tracks.");

    hits_fake_tracks->SetLineColor(kGreen);
    //hits_fake_tracks->SetFillColor(kGreen);
    processWithoutErrors(hits_fake_tracks);
    hs8->Add(hits_fake_tracks);

    hits_bad_tracks->SetLineColor(kCyan);
    processWithoutErrors(hits_bad_tracks);
    hs8->Add(hits_bad_tracks);

    hits_displaced_tracks->SetLineColor(kRed);
    //hits_displaced_tracks->SetFillColor(kRed);
    processWithoutErrors(hits_displaced_tracks);
    hs8->Add(hits_displaced_tracks);

    hits_B_tracks->SetLineColor(kBlue);
    //hits_B_tracks->SetFillColor(kBlue);
    processWithoutErrors(hits_B_tracks);
    hs8->Add(hits_B_tracks);

    hits_nonB_tracks->SetLineColor(kBlack);
    //hits_nonB_tracks->SetFillColor(kBlack);
    processWithoutErrors(hits_nonB_tracks);
    hs8->Add(hits_nonB_tracks);

    hs8->Draw("nostack");
    hs8->GetXaxis()->SetTitle("Hits");
    hs8->GetYaxis()->SetTitle("Relative frequency");

    TLegend *leg8 = new TLegend(0.62458,0.692819,0.885747,0.839096,NULL,"brNDC");
    leg8->SetBorderSize(1);
    leg8->SetLineColor(0);
    leg8->SetLineStyle(1);
    leg8->SetLineWidth(1);
    leg8->SetFillColor(0);
    leg8->SetFillStyle(1001);
    leg8->AddEntry(hits_fake_tracks,"Fake", "f");
    leg8->AddEntry(hits_bad_tracks,"Bad", "f");
    leg8->AddEntry(hits_displaced_tracks,"Displaced", "f");
    leg8->AddEntry(hits_B_tracks,"B", "f");
    leg8->AddEntry(hits_nonB_tracks,"Non B", "f");
    leg8->Draw();
}

void processWithErrors(TH1* histogram)
{
    histogram->Scale(1./histogram->GetEntries());
    for (int i = 0; i < histogram->GetNbinsX(); i++)
        histogram->SetBinError(i,sqrt(histogram->GetBinContent(i)*(1-histogram->GetBinContent(i))/histogram->GetEntries()));
}

void processWithoutErrors(TH1* histogram)
{
    histogram->Scale(1./histogram->GetEntries());
}
