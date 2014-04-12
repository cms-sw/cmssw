
{ 

#include <vector>

gROOT->LoadMacro("../Tools/NicePlot.C");
InitNicePlot();

gStyle->SetOptStat(0);


TCanvas* c2 = new TCanvas();
FormatPad(c2,false);
c2->cd();

TFile* file = new TFile("JetBenchmarkGeneric.root");
TFile* file2 = new TFile("JetBenchmarkGeneric_pre7.root");
TH2F* histoCALOEta = (TH2F*) file.Get("DQMData/PFTask/Benchmarks/iterativeCone5CaloJets/Gen/DeltaEtOverEtvsEta");
TH2F* histoPFEta = (TH2F*) file.Get("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen/DeltaEtOverEtvsEta");
TH2F* histoCALOEta2 = (TH2F*) file2.Get("DQMData/PFTask/Benchmarks/iterativeCone5CaloJets/Gen/DeltaEtOverEtvsEta");
TH2F* histoPFEta2 = (TH2F*) file2.Get("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen/DeltaEtOverEtvsEta");


TH1F* pfEta = (TH1F*) (histoPFEta->ProfileX()->Clone());
TH1F* caloEta = (TH1F*) (histoCALOEta->ProfileX()->Clone());
TH1F* pfEta2 = (TH1F*) (histoPFEta2->ProfileX()->Clone());
TH1F* caloEta2 = (TH1F*) (histoCALOEta2->ProfileX()->Clone());

 TH1F* pfRes = new TH1F("pfRes","PF pre9",200,-5.,5.);
 TH1F* pfRes2 = new TH1F("pfRes2","PF pre7",200,-5.,5.);
 TH1F* caloRes = new TH1F("caloRes","Calo pre9",200,-5.,5.);
 TH1F* caloRes2 = new TH1F("caloRes2","Calo pre7",200,-5.,5.);

FormatHisto(pfRes,sback);


 for ( unsigned i=0; i<201; ++i ) { 

   double pf = pfEta->GetBinContent(i);
   double pf2 = pfEta2->GetBinContent(i);
   double ca = caloEta->GetBinContent(i);
   double ca2 = caloEta2->GetBinContent(i);

   double pfx = pfEta->GetBinCenter(i);
   double pfx2 = pfEta2->GetBinCenter(i);
   double cax = caloEta->GetBinCenter(i);
   double cax2 = caloEta2->GetBinCenter(i);

   cout << i << " " << pfx << " " << pf  << endl;

   pfRes->Fill(pfx,1+pf);
   pfRes2->Fill(pfx2,1+pf2);
   caloRes->Fill(cax,1+ca);
   caloRes2->Fill(cax2,1+ca2);

 }

 pfRes->Rebin(2);
 pfRes2->Rebin(2);
 caloRes->Rebin(2);
 caloRes2->Rebin(2);
pfRes->Divide(pfRes2);
caloRes->Divide(caloRes2);
 
gPad->SetGridx();
gPad->SetGridy();
  
pfRes->SetTitle( "CMS Preliminary" );
pfRes->SetYTitle( "Jet Response");
pfRes->SetMaximum(1.5);
pfRes->SetMinimum(0.5);
//pfRes->Rebin(2);
//pfRes->GetXaxis().SetRangeUser(-2.8,2.8);

pfRes->SetMarkerStyle(22);						
pfRes->SetMarkerColor(2);						
pfRes->SetLineColor(2);						  
pfRes->SetMarkerSize(1.2);						
pfRes->SetLineWidth(3);						  
pfRes->SetLineStyle(1);
pfRes->Draw();

//caloRes->Rebin(2);
//caloRes->GetXaxis().SetRangeUser(-2.8,2.8);
caloRes->SetMarkerStyle(25);						
caloRes->SetMarkerColor(4);						
caloRes->SetMarkerSize(1.0);						
caloRes->SetLineColor(4);						  
caloRes->SetLineWidth(3);						  
caloRes->Draw("same");

TLegend *leg=new TLegend(0.60,0.65,0.85,0.85);
leg->AddEntry(pfRes, "Particle-Flow Jets", "p");
leg->AddEntry(caloRes, "Calo-Jets", "p");
leg->SetTextSize(0.03);
leg->Draw();

gPad->SaveAs("JetComparisonEta.pdf");
gPad->SaveAs("JetComparisonEta.png");

}
