
{ 

#include <vector>

gROOT->LoadMacro("../Tools/NicePlot.C");
InitNicePlot();

gStyle->SetOptStat(0);


TCanvas* c2 = new TCanvas();
FormatPad(c2,false);
c2->cd();

TFile* file = new TFile("JetBenchmarkGeneric.root");
TH2F* histoCALOEta = (TH2F*) file.Get("DQMData/PFTask/Benchmarks/iterativeCone5CaloJets/Gen/DeltaEtOverEtvsEta");
TH2F* histoPFEta = (TH2F*) file.Get("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen/DeltaEtOverEtvsEta");

TH1F* pfEta = (TH1F*) (histoPFEta->ProfileX()->Clone());
TH1F* caloEta = (TH1F*) (histoCALOEta->ProfileX()->Clone());
FormatHisto(pfEta,sback);
 
gPad->SetGridx();
gPad->SetGridy();
  
pfEta->SetTitle( "CMS Preliminary" );
pfEta->SetYTitle( "Jet Response");
pfEta->SetMaximum(0.3);
pfEta->SetMinimum(-0.7);
pfEta->Rebin(2);
//pfEta->GetXaxis().SetRangeUser(-2.8,2.8);

pfEta->SetMarkerStyle(22);						
pfEta->SetMarkerColor(2);						
pfEta->SetLineColor(2);						  
pfEta->SetMarkerSize(1.2);						
pfEta->SetLineWidth(3);						  
pfEta->SetLineStyle(1);
pfEta->Draw();

caloEta->Rebin(2);
//caloEta->GetXaxis().SetRangeUser(-2.8,2.8);
caloEta->SetMarkerStyle(25);						
caloEta->SetMarkerColor(4);						
caloEta->SetMarkerSize(1.0);						
caloEta->SetLineColor(4);						  
caloEta->SetLineWidth(3);						  
caloEta->Draw("same");

TLegend *leg=new TLegend(0.60,0.65,0.85,0.85);
leg->AddEntry(pfEta, "Particle-Flow Jets", "p");
leg->AddEntry(caloEta, "Calo-Jets", "p");
leg->SetTextSize(0.03);
leg->Draw();

gPad->SaveAs("JetResponseEta.pdf");
gPad->SaveAs("JetResponseEta.png");

}
