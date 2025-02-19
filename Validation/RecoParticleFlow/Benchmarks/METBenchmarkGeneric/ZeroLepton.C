{

gROOT->LoadMacro("../Tools/NicePlot.C");
InitNicePlot();

TFile file("zero-lepton.root");
  
gStyle->SetOptStat(0);
TH1F* DeltaMETPF = (TH1F*) (file.Get("DQMData/PFTask/Benchmarks/pfMet/Gen/DeltaMET")->Clone());
TH1F* DeltaMETTC = (TH1F*) (file.Get("DQMData/PFTask/Benchmarks/pfMet/Gen/DeltaTCMET")->Clone());
TH1F* DeltaPhiPF = (TH1F*) (file.Get("DQMData/PFTask/Benchmarks/pfMet/Gen/DeltaPhi")->Clone());
TH1F* DeltaPhiTC = (TH1F*) (file.Get("DQMData/PFTask/Benchmarks/pfMet/Gen/DeltaTCPhi")->Clone());
TH1F* DeltaLogPF = (TH1F*) (file.Get("DQMData/PFTask/Benchmarks/pfMet/Gen/DeltaMET")->Clone());
TH1F* DeltaLogTC = (TH1F*) (file.Get("DQMData/PFTask/Benchmarks/pfMet/Gen/DeltaTCMET")->Clone());
TH1F* DeltaMEXPF = (TH1F*) (file.Get("DQMData/PFTask/Benchmarks/pfMet/Gen/MEX")->Clone());
TH1F* DeltaMEXTC = (TH1F*) (file.Get("DQMData/PFTask/Benchmarks/pfMet/Gen/TCMEX")->Clone());
TH1F* DeltaLogXPF = (TH1F*) (file.Get("DQMData/PFTask/Benchmarks/pfMet/Gen/MEX")->Clone());
TH1F* DeltaLogXTC = (TH1F*) (file.Get("DQMData/PFTask/Benchmarks/pfMet/Gen/TCMEX")->Clone());

DeltaMETPF->SetTitle("CMS Preliminary");
DeltaMETPF->Rebin(2);
DeltaMETPF->GetXaxis().SetRangeUser(-50.,100.);

DeltaMETTC->Rebin(2);
DeltaMETTC->GetXaxis().SetRangeUser(-50.,100.);

TCanvas* c1 = new TCanvas();
FormatPad(c1,false);
//gPad->SetLogy();

DeltaMETPF->SetMaximum(3500);
FormatHisto(DeltaMETPF,spred);
DeltaMETPF->SetXTitle("#Delta MET [GeV]");
DeltaMETPF->SetYTitle("Number of events");
DeltaMETPF->Draw();
FormatHisto(DeltaMETTC,spblue);
DeltaMETTC->Draw("same");

TLegend *leg=new TLegend(0.60,0.65,0.90,0.85);
leg->AddEntry(DeltaMETTC, "Track-Corrected MET", "lf");
leg->AddEntry(DeltaMETPF, "Particle-Flow MET", "lf");
leg->SetTextSize(0.03);
leg->Draw();

TLatex text;
text.SetTextColor(1);
text.SetTextSize(0.03);
text.DrawLatex(42,2000,"Fully hadronic ttbar events");

gPad->SaveAs("MET_ZeroLepton.pdf");
gPad->SaveAs("MET_ZeroLepton.png");


DeltaLogPF->SetTitle("CMS Preliminary");
DeltaLogPF->Rebin(2);
DeltaLogPF->GetXaxis().SetRangeUser(-80.,160.);
DeltaLogTC->Rebin(2);
DeltaLogTC->GetXaxis().SetRangeUser(-80.,160.);

TCanvas* c2 = new TCanvas();
FormatPad(c2,false);
gPad->SetLogy();

DeltaLogPF->SetMaximum(5000);
FormatHisto(DeltaLogPF,spred);
DeltaLogPF->SetXTitle("#Delta MET [GeV]");
DeltaLogPF->SetYTitle("Number of events");
DeltaLogPF->Draw();
FormatHisto(DeltaLogTC,spblue);
DeltaLogTC->Draw("same");

TLegend *leg=new TLegend(0.65,0.65,0.92,0.85);
leg->AddEntry(DeltaLogTC, "Track-Corrected MET", "lf");
leg->AddEntry(DeltaLogPF, "Particle-Flow MET", "lf");
leg->SetTextSize(0.03);
leg->Draw();

TLatex text;
text.SetTextColor(1);
text.SetTextSize(0.03);
text.DrawLatex(75,100,"Fully hadronic ttbar events");

gPad->SaveAs("METLog_ZeroLepton.pdf");
gPad->SaveAs("METLog_ZeroLepton.png");

DeltaMEXPF->SetTitle("CMS Preliminary");
DeltaMEXPF->Rebin(2);
DeltaMEXPF->GetXaxis().SetRangeUser(-150.,150.);

DeltaMEXTC->Rebin(2);
DeltaMEXTC->GetXaxis().SetRangeUser(-150.,150.);

TCanvas* c3 = new TCanvas();
FormatPad(c3,false);

DeltaMEXPF->SetMaximum(3000);
FormatHisto(DeltaMEXPF,spred);
DeltaMEXPF->SetXTitle("#Delta MEX [GeV]");
DeltaMEXPF->SetYTitle("Number of events");
DeltaMEXPF->Draw();
FormatHisto(DeltaMEXTC,spblue);
DeltaMEXTC->Draw("same");

TLegend *leg=new TLegend(0.60,0.65,0.90,0.85);
leg->AddEntry(DeltaMEXTC, "Track-Corrected MEX", "lf");
leg->AddEntry(DeltaMEXPF, "Particle-Flow MEX", "lf");
leg->SetTextSize(0.03);
leg->Draw();

TLatex text;
text.SetTextColor(1);
text.SetTextSize(0.03);
text.DrawLatex(30,1800,"Fully hadronic ttbar events");

gPad->SaveAs("MEX_ZeroLepton.pdf");
gPad->SaveAs("MEX_ZeroLepton.png");


DeltaLogXPF->SetTitle("CMS Preliminary");
DeltaLogXPF->Rebin(2);
DeltaLogXPF->GetXaxis().SetRangeUser(-300.,300.);
DeltaLogXTC->Rebin(2);
DeltaLogXTC->GetXaxis().SetRangeUser(-300.,300.);

TCanvas* c4 = new TCanvas();
FormatPad(c4,false);
gPad->SetLogy();

DeltaLogXPF->SetMaximum(5000);
FormatHisto(DeltaLogXPF,spred);
DeltaLogXPF->SetXTitle("#Delta MEX [GeV]");
DeltaLogXPF->SetYTitle("Number of events");
DeltaLogXPF->Draw();
FormatHisto(DeltaLogXTC,spblue);
DeltaLogXTC->Draw("same");

TLegend *leg=new TLegend(0.65,0.65,0.92,0.85);
leg->AddEntry(DeltaLogXTC, "Track-Corrected MEX", "lf");
leg->AddEntry(DeltaLogXPF, "Particle-Flow MEX", "lf");
leg->SetTextSize(0.03);
leg->Draw();

TLatex text;
text.SetTextColor(1);
text.SetTextSize(0.03);
text.DrawLatex(90,100,"Fully hadronic ttbar events");

gPad->SaveAs("MEXLog_ZeroLepton.pdf");
gPad->SaveAs("MEXLog_ZeroLepton.png");

DeltaPhiPF->SetTitle("CMS Preliminary");
DeltaPhiPF->Rebin(2);
DeltaPhiPF->GetXaxis().SetRangeUser(-3.,3.);

DeltaPhiTC->Rebin(2);
DeltaPhiTC->GetXaxis().SetRangeUser(-3.,3.);

}
