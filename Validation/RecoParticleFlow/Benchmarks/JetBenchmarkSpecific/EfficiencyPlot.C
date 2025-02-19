{

gROOT->LoadMacro("../Tools/NicePlot.C");
InitNicePlot();

TFile file("JetBenchmarkGeneric_Efficiency.root");
  
gStyle->SetOptStat(0);
TH1F* EffPFDenBarrel = (TH1F*) file.Get("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen/GenJetVsPtEffBarrel")->Clone();
TH1F* EffPFNumBarrel = (TH1F*) file.Get("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen/RecoJetVsPtEffBarrel")->Clone();
TH1F* EffCaloDenBarrel = (TH1F*) file.Get("DQMData/PFTask/Benchmarks/iterativeCone5CaloJets/Gen/GenJetVsPtEffBarrel")->Clone();
TH1F* EffCaloNumBarrel = (TH1F*) file.Get("DQMData/PFTask/Benchmarks/iterativeCone5CaloJets/Gen/RecoJetVsPtEffBarrel")->Clone();

EffPFDenBarrel->Rebin(4);
EffPFNumBarrel->Rebin(4);
EffCaloDenBarrel->Rebin(4);
EffCaloNumBarrel->Rebin(4);

TH1F* EffPFBarrel=EffPFNumBarrel->Clone();
EffPFBarrel->Divide(EffPFDenBarrel);
TH1F* EffCaloBarrel=EffCaloNumBarrel->Clone();
EffCaloBarrel->Divide(EffCaloDenBarrel);

EffPFBarrel->SetTitle("CMS Preliminary");
EffPFBarrel->GetXaxis().SetRangeUser(0,204);

TH1F* FakePFDenBarrel = (TH1F*) file.Get("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen/GenJetVsPtFakeBarrel")->Clone();
TH1F* FakePFNumBarrel = (TH1F*) file.Get("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen/RecoJetVsPtFakeBarrel")->Clone();
TH1F* FakeCaloDenBarrel = (TH1F*) file.Get("DQMData/PFTask/Benchmarks/iterativeCone5CaloJets/Gen/GenJetVsPtFakeBarrel")->Clone();
TH1F* FakeCaloNumBarrel = (TH1F*) file.Get("DQMData/PFTask/Benchmarks/iterativeCone5CaloJets/Gen/RecoJetVsPtFakeBarrel")->Clone();

FakePFDenBarrel->Rebin(4);
FakePFNumBarrel->Rebin(4);
FakeCaloDenBarrel->Rebin(4);
FakeCaloNumBarrel->Rebin(4);

TH1F* FakePFBarrel=FakePFNumBarrel->Clone();
FakePFBarrel->Divide(FakePFDenBarrel);
TH1F* FakeCaloBarrel=FakeCaloNumBarrel->Clone();
FakeCaloBarrel->Divide(FakeCaloDenBarrel);

FakePFBarrel->SetTitle("CMS Preliminary");
FakePFBarrel->GetXaxis().SetRangeUser(0,204);

TH1F* EffPFDenEndcap = (TH1F*) file.Get("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen/GenJetVsPtEffEndcap")->Clone();
TH1F* EffPFNumEndcap = (TH1F*) file.Get("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen/RecoJetVsPtEffEndcap")->Clone();
TH1F* EffCaloDenEndcap = (TH1F*) file.Get("DQMData/PFTask/Benchmarks/iterativeCone5CaloJets/Gen/GenJetVsPtEffEndcap")->Clone();
TH1F* EffCaloNumEndcap = (TH1F*) file.Get("DQMData/PFTask/Benchmarks/iterativeCone5CaloJets/Gen/RecoJetVsPtEffEndcap")->Clone();

EffPFDenEndcap->Rebin(10);
EffPFNumEndcap->Rebin(10);
EffCaloDenEndcap->Rebin(10);
EffCaloNumEndcap->Rebin(10);

TH1F* EffPFEndcap=EffPFNumEndcap->Clone();
EffPFEndcap->Divide(EffPFDenEndcap);
TH1F* EffCaloEndcap=EffCaloNumEndcap->Clone();
EffCaloEndcap->Divide(EffCaloDenEndcap);

EffPFEndcap->SetTitle("CMS Preliminary");
EffPFEndcap->GetXaxis().SetRangeUser(0,200);

TH1F* FakePFDenEndcap = (TH1F*) file.Get("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen/GenJetVsPtFakeEndcap")->Clone();
TH1F* FakePFNumEndcap = (TH1F*) file.Get("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen/RecoJetVsPtFakeEndcap")->Clone();
TH1F* FakeCaloDenEndcap = (TH1F*) file.Get("DQMData/PFTask/Benchmarks/iterativeCone5CaloJets/Gen/GenJetVsPtFakeEndcap")->Clone();
TH1F* FakeCaloNumEndcap = (TH1F*) file.Get("DQMData/PFTask/Benchmarks/iterativeCone5CaloJets/Gen/RecoJetVsPtFakeEndcap")->Clone();

FakePFDenEndcap->Rebin(10);
FakePFNumEndcap->Rebin(10);
FakeCaloDenEndcap->Rebin(10);
FakeCaloNumEndcap->Rebin(10);

TH1F* FakePFEndcap=FakePFNumEndcap->Clone();
FakePFEndcap->Divide(FakePFDenEndcap);
TH1F* FakeCaloEndcap=FakeCaloNumEndcap->Clone();
FakeCaloEndcap->Divide(FakeCaloDenEndcap);

FakePFEndcap->SetTitle("CMS Preliminary");
FakePFEndcap->GetXaxis().SetRangeUser(0,200);

TCanvas* c1 = new TCanvas();
FormatPad(c1,false);
//gPad->SetLogy();
gPad->SetGridx();
gPad->SetGridy();

TF1* effPFBarrel = new TF1("effPFBarrel","[0]+[1]*exp(-x/[2])",1,1000);
TF1* effCaloBarrel = new TF1("effCaloBarrel","[0]+[1]*exp(-x/[2])",1,1000);
effPFBarrel->SetParameters(1,-0.5,20);
effCaloBarrel->SetParameters(1,-0.5,20);
effPFBarrel->SetLineColor(2);
effCaloBarrel->SetLineColor(4);
EffPFBarrel->Fit("effPFBarrel","","",5,210);
EffCaloBarrel->Fit("effCaloBarrel","","",5,210);

FormatHisto(EffPFBarrel,sback);
EffPFBarrel->SetMaximum(1.2);
EffPFBarrel->SetXTitle("Generated p_{T} [GeV/c]");
EffPFBarrel->SetYTitle("Jet Matching Efficiency [%]");
EffPFBarrel->SetMarkerStyle(22);
EffPFBarrel->SetMarkerSize(1.2);
EffPFBarrel->SetMarkerColor(2);
EffPFBarrel->SetLineColor(2);
EffPFBarrel->SetLineWidth(2);
EffPFBarrel->Draw("P");

EffCaloBarrel->SetMarkerStyle(25);
EffCaloBarrel->SetMarkerSize(1.0);
EffCaloBarrel->SetMarkerColor(4);
EffCaloBarrel->SetLineColor(4);
EffCaloBarrel->SetLineWidth(2);
EffCaloBarrel->Draw("Psame");

TLegend *leg=new TLegend(0.65,0.25,0.90,0.45);
leg->AddEntry(EffPFBarrel, "Particle-Flow Jets", "lp");
leg->AddEntry(EffCaloBarrel, "Calo-Jets", "lp");
leg->SetTextSize(0.03);
leg->Draw();

TLatex text;
text.SetTextColor(1);
text.DrawLatex(145,0.55,"#DeltaR=0.1");

gPad->SaveAs("EfficiencyBarrel.pdf");
gPad->SaveAs("EfficiencyBarrel.png");

TCanvas* c2 = new TCanvas();
FormatPad(c2,false);
c2->cd();
//gPad->SetLogy();
gPad->SetGridx();
gPad->SetGridy();

TF1* fakePFBarrel = new TF1("fakePFBarrel","[0]+[1]*exp(-x/[2])",1,1000);
TF1* fakeCaloBarrel = new TF1("fakeCaloBarrel","[0]+[1]*exp(-x/[2])",1,1000);
fakePFBarrel->SetParameters(0.,0.5,5);
fakeCaloBarrel->SetParameters(0.,0.5,5);
fakePFBarrel->SetLineColor(2);
fakeCaloBarrel->SetLineColor(4);
FakePFBarrel->Fit("fakePFBarrel","","",5,210);
FakeCaloBarrel->Fit("fakeCaloBarrel","","",5,210);

FormatHisto(FakePFBarrel,sback);
FakePFBarrel->SetMaximum(1.0);
FakePFBarrel->SetXTitle("Reconstructed p_{T} [GeV/c]");
FakePFBarrel->SetYTitle("Mismatched Jet Rate [%]");
FakePFBarrel->SetMarkerStyle(22);
FakePFBarrel->SetMarkerSize(1.2);
FakePFBarrel->SetMarkerColor(2);
FakePFBarrel->SetLineColor(2);
FakePFBarrel->SetLineWidth(2);
FakePFBarrel->Draw("P");

FakeCaloBarrel->SetMarkerStyle(25);
FakeCaloBarrel->SetMarkerSize(1.0);
FakeCaloBarrel->SetMarkerColor(4);
FakeCaloBarrel->SetLineColor(4);
FakeCaloBarrel->SetLineWidth(2);
FakeCaloBarrel->Draw("Psame");

TLegend *leg=new TLegend(0.65,0.65,0.90,0.85);
leg->AddEntry(FakeCaloBarrel, "Calo-Jets", "lp");
leg->AddEntry(FakePFBarrel, "Particle-Flow Jets", "lp");
leg->SetTextSize(0.03);
leg->Draw();

text.DrawLatex(145,0.55,"#DeltaR=0.1");

gPad->SaveAs("FakeBarrel.pdf");
gPad->SaveAs("FakeBarrel.png");

/* */
TCanvas* c3 = new TCanvas();
FormatPad(c3,false);
c3->cd();
//gPad->SetLogy();
gPad->SetGridx();
gPad->SetGridy();

TF1* effPFEndcap = new TF1("effPFEndcap","[0]+[1]*exp(-x/[2])",1,1000);
TF1* effCaloEndcap = new TF1("effCaloEndcap","[0]+[1]*exp(-x/[2])",1,1000);
effPFEndcap->SetParameters(1,-0.5,20);
effCaloEndcap->SetParameters(1,-0.5,20);
effPFEndcap->SetLineColor(2);
effCaloEndcap->SetLineColor(4);
EffPFEndcap->Fit("effPFEndcap","","",5,210);
EffCaloEndcap->Fit("effCaloEndcap","","",5,210);

FormatHisto(EffPFEndcap,sback);
EffPFEndcap->SetMaximum(1.2);
EffPFEndcap->SetXTitle("Generated p_{T} [GeV/c]");
EffPFEndcap->SetYTitle("Jet Matching Efficiency [%]");
EffPFEndcap->SetMarkerStyle(22);
EffPFEndcap->SetMarkerSize(1.2);
EffPFEndcap->SetMarkerColor(2);
EffPFEndcap->SetLineColor(2);
EffPFEndcap->SetLineWidth(2);
EffPFEndcap->Draw("P");

EffCaloEndcap->SetMarkerStyle(25);
EffCaloEndcap->SetMarkerSize(1.0);
EffCaloEndcap->SetMarkerColor(4);
EffCaloEndcap->SetLineColor(4);
EffCaloEndcap->SetLineWidth(2);
EffCaloEndcap->Draw("Psame");

TLegend *leg=new TLegend(0.65,0.25,0.90,0.45);
leg->AddEntry(EffPFEndcap, "Particle-Flow Jets", "lp");
leg->AddEntry(EffCaloEndcap, "Calo-Jets", "lp");
leg->SetTextSize(0.03);
leg->Draw();

text.DrawLatex(145,0.55,"#DeltaR=0.1");

gPad->SaveAs("EfficiencyEndcap.pdf");
gPad->SaveAs("EfficiencyEndcap.png");

TCanvas* c4 = new TCanvas();
FormatPad(c4,false);
c4->cd();
//gPad->SetLogy();
gPad->SetGridx();
gPad->SetGridy();

TF1* fakePFEndcap = new TF1("fakePFEndcap","[0]+[1]*exp(-x/[2])",1,1000);
TF1* fakeCaloEndcap = new TF1("fakeCaloEndcap","[0]+[1]*exp(-x/[2])",1,1000);
fakePFEndcap->SetParameters(0.,0.5,5);
fakeCaloEndcap->SetParameters(0.,0.5,5);
fakePFEndcap->SetLineColor(2);
fakeCaloEndcap->SetLineColor(4);
FakePFEndcap->Fit("fakePFEndcap","","",5,210);
FakeCaloEndcap->Fit("fakeCaloEndcap","","",5,210);

FormatHisto(FakePFEndcap,sback);
FakePFEndcap->SetMaximum(1.0);
FakePFEndcap->SetXTitle("Reconstructed p_{T} [GeV/c]");
FakePFEndcap->SetYTitle("Mismatched Jet Rate [%]");
FakePFEndcap->SetMarkerStyle(22);
FakePFEndcap->SetMarkerSize(1.2);
FakePFEndcap->SetMarkerColor(2);
FakePFEndcap->SetLineColor(2);
FakePFEndcap->SetLineWidth(2);
FakePFEndcap->Draw("P");

FakeCaloEndcap->SetMarkerStyle(25);
FakeCaloEndcap->SetMarkerSize(1.0);
FakeCaloEndcap->SetMarkerColor(4);
FakeCaloEndcap->SetLineColor(4);
FakeCaloEndcap->SetLineWidth(2);
FakeCaloEndcap->Draw("Psame");

TLegend *leg=new TLegend(0.65,0.65,0.90,0.85);
leg->AddEntry(FakeCaloEndcap, "Calo-Jets", "lp");
leg->AddEntry(FakePFEndcap, "Particle-Flow Jets", "lp");
leg->SetTextSize(0.03);
leg->Draw();

text.DrawLatex(145,0.55,"#DeltaR=0.1");

gPad->SaveAs("FakeEndcap.pdf");
gPad->SaveAs("FakeEndcap.png");
/* */
}
