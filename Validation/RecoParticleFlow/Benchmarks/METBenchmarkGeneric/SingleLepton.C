{

gROOT->LoadMacro("../Tools/NicePlot.C");
InitNicePlot();

TFile file("single-lepton.root");
  
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
DeltaMETPF->GetXaxis().SetRangeUser(-100.,100.);

DeltaMETTC->Rebin(2);
DeltaMETTC->GetXaxis().SetRangeUser(-100.,100.);

TCanvas* c1 = new TCanvas();
FormatPad(c1,false);
//gPad->SetLogy();

DeltaMETPF->SetMaximum(2500);
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
text.DrawLatex(20,1500,"Semi-leptonic ttbar events");

gPad->SaveAs("MET_SingleLepton.pdf");
gPad->SaveAs("MET_SingleLepton.png");


DeltaLogPF->SetTitle("CMS Preliminary");
DeltaLogPF->Rebin(2);
//DeltaLogPF->GetXaxis().SetRangeUser(-300.,300.);
DeltaLogTC->Rebin(2);
//DeltaLogTC->GetXaxis().SetRangeUser(-300.,300.);

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
text.DrawLatex(55,100,"Semi-leptonic ttbar events");

gPad->SaveAs("METLog_SingleLepton.pdf");
gPad->SaveAs("METLog_SingleLepton.png");


DeltaPhiPF->SetTitle("CMS Preliminary");
DeltaPhiPF->Rebin(1);
DeltaPhiPF->GetXaxis().SetRangeUser(-3.14,3.14);

DeltaPhiTC->Rebin(1);
DeltaPhiTC->GetXaxis().SetRangeUser(-3.14,3.14);

TCanvas* c5 = new TCanvas();
FormatPad(c5,false);
//gPad->SetLogy();

DeltaPhiPF->SetMaximum(7000);
FormatHisto(DeltaPhiPF,spred);
DeltaPhiPF->SetXTitle("#Delta #phi [rad]");
DeltaPhiPF->SetYTitle("Number of events");
DeltaPhiPF->Draw();
FormatHisto(DeltaPhiTC,spblue);
DeltaPhiTC->Draw("same");

TLegend *leg=new TLegend(0.68,0.65,0.90,0.85);
leg->AddEntry(DeltaPhiTC, "Track-Corrected", "lf");
leg->AddEntry(DeltaPhiPF, "Particle-Flow", "lf");
leg->SetTextSize(0.03);
leg->Draw();

TLatex text;
text.SetTextColor(1);
text.SetTextSize(0.03);
text.DrawLatex(1,4200,"Semi-leptonic ttbar events");

gPad->SaveAs("Phi_SingleLepton.pdf");
gPad->SaveAs("Phi_SingleLepton.png");

}
