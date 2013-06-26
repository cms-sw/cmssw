{

gROOT->LoadMacro("../Tools/NicePlot.C");
InitNicePlot();

TFile file("JetBenchmarkGeneric_Endcap.root");
  
gStyle->SetOptStat(0);
TH2F* histoPF2 = (TH2F*) file.Get("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen/DeltaEtOverEtvsEt");
TH2F* histoCALO2 = (TH2F*) file.Get("DQMData/PFTask/Benchmarks/iterativeCone5CaloJets/Gen/DeltaEtOverEtvsEt");
TH1F* histoPF40_60 = (TH1F*)(histoPF2->ProjectionY("",40,60)->Clone());
TH1F* histoCALO40_60 = (TH1F*)(histoCALO2->ProjectionY("",40,60)->Clone());
histoPF40_60->SetTitle("CMS Preliminary");
histoPF40_60->Rebin(25);
histoPF40_60->GetXaxis().SetRangeUser(-1,1);
//histoPF40_60->Fit("gaus","","",-0.25,0.1);
histoCALO40_60->Rebin(25);
histoCALO40_60->GetXaxis().SetRangeUser(-1,1);
//histoCALO40_60->Fit("gaus","","",-0.7,-0.3);

histoPF40_60->Fit("gaus");
histoCALO40_60->Fit("gaus");
histoPF40_60->GetFunction("gaus")->SetLineColor(2);
histoCALO40_60->GetFunction("gaus")->SetLineColor(4);


TH1F* histoPF300_400 = (TH1F*)(histoPF2->ProjectionY("",300,400)->Clone());
TH1F* histoCALO300_400 = (TH1F*)(histoCALO2->ProjectionY("",300,400)->Clone());
histoPF300_400->SetTitle("CMS Preliminary");
histoPF300_400->Rebin(10);
histoPF300_400->GetXaxis().SetRangeUser(-1,1);
//histoPF300_400->Fit("gaus","","",-0.25,0.1);
histoCALO300_400->Rebin(10);
histoCALO300_400->GetXaxis().SetRangeUser(-1,1);
//histoCALO300_400->Fit("gaus","","",-0.7,-0.3);

histoPF300_400->Fit("gaus");
histoCALO300_400->Fit("gaus");
histoPF300_400->GetFunction("gaus")->SetLineColor(2);
histoCALO300_400->GetFunction("gaus")->SetLineColor(4);

TCanvas* c1 = new TCanvas();
FormatPad(c1,false);
//gPad->SetLogy();

//histoPF40_60->SetLineColor(1);
//histoPF40_60->SetLineWidth(2);  
//histoPF40_60->SetFillStyle(3002);
//histoPF40_60->SetFillColor(2);
histoPF40_60->SetMaximum(200);
FormatHisto(histoPF40_60,spred);
histoPF40_60->SetXTitle("#Delta p_{T}/p_{T}");
histoPF40_60->SetYTitle("Number of jets");
histoPF40_60->Draw();
//histoCALO40_60->SetLineColor(1);
//histoCALO40_60->SetLineWidth(2);
//histoCALO40_60->SetFillStyle(3004);
//histoCALO40_60->SetFillColor(4);
FormatHisto(histoCALO40_60,spblue);
histoCALO40_60->Draw("same");

TLegend *leg=new TLegend(0.65,0.65,0.90,0.85);
leg->AddEntry(histoCALO40_60, "Calo-Jets", "lf");
leg->AddEntry(histoPF40_60, "Particle-Flow Jets", "lf");
leg->SetTextSize(0.03);
leg->Draw();

TLatex text;
text.SetTextColor(1);
text.SetTextSize(0.03);
text.DrawLatex(0.30,110,"p_{T} = 40 - 60 GeV/c");
TLatex text2;
text2.SetTextColor(1);
text2.SetTextSize(0.03);
text2.DrawLatex(0.30,95,"1.5 < |#eta| < 2.5");

gPad->SaveAs("Jet40_60_Endcap.pdf");
gPad->SaveAs("Jet40_60_Endcap.png");

TCanvas* c2 = new TCanvas();
FormatPad(c2,false);
//gPad->SetLogy();


//histoPF300_400->SetLineColor(1);
//histoPF300_400->SetLineWidth(2);  
//histoPF300_400->SetFillStyle(3002);
//histoPF300_400->SetFillColor(2);
FormatHisto(histoPF300_400,spred);
histoPF300_400->SetMaximum(150);
histoPF300_400->SetXTitle("#Delta p_{T}/p_{T}");
histoPF300_400->SetYTitle("Number of jets");
histoPF300_400->Draw();
//histoCALO300_400->SetLineColor(1);
//histoCALO300_400->SetLineWidth(2);
//histoCALO300_400->SetFillStyle(3004);
//histoCALO300_400->SetFillColor(4);
FormatHisto(histoCALO300_400,spblue);
histoCALO300_400->Draw("same");

TLegend *leg=new TLegend(0.65,0.65,0.90,0.85);
leg->AddEntry(histoCALO300_400, "Calo-Jets", "lf");
leg->AddEntry(histoPF300_400, "Particle-Flow Jets", "lf");
leg->SetTextSize(0.03);
leg->Draw();

text.DrawLatex(0.30,85,"p_{T} = 300 - 400 GeV/c");
text2.DrawLatex(0.30,75,"1.5 < |#eta| < 2.5");

gPad->SaveAs("Jet300_400_Endcap.pdf");
gPad->SaveAs("Jet300_400_Endcap.png");
 
}
