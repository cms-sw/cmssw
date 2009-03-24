{ 

#include <vector>

gROOT->LoadMacro("../Tools/NicePlot.C");
InitNicePlot();


vector<float> JetPFFull;
vector<float> JetPFFast;
vector<float> jetPt;

jetPt.push_back(29.8);
jetPt.push_back(49.6);
jetPt.push_back(69.6);
jetPt.push_back(89.5);
jetPt.push_back(123.);
jetPt.push_back(173.);
jetPt.push_back(223.);
jetPt.push_back(274.);
jetPt.push_back(346.);
jetPt.push_back(445.);
jetPt.push_back(620.);

JetPFFull.push_back(1-0.0890398);
JetPFFull.push_back(1-0.0710146);
JetPFFull.push_back(1-0.0593942);
JetPFFull.push_back(1-0.052804);
JetPFFull.push_back(1-0.0469199);
JetPFFull.push_back(1-0.0438645);
JetPFFull.push_back(1-0.0422467);
JetPFFull.push_back(1-0.0449852);
JetPFFull.push_back(1-0.0399654);
JetPFFull.push_back(1-0.0354058);
JetPFFull.push_back(1-0.0264279);

JetPFFast.push_back(1-0.0881328);
JetPFFast.push_back(1-0.0808864);
JetPFFast.push_back(1-0.0689337);
JetPFFast.push_back(1-0.0638864);
JetPFFast.push_back(1-0.0527432);
JetPFFast.push_back(1-0.0547869);
JetPFFast.push_back(1-0.0567134);
JetPFFast.push_back(1-0.0520955);
JetPFFast.push_back(1-0.0484314);
JetPFFast.push_back(1-0.0432866);
JetPFFast.push_back(1-0.0325687);


TGraph* grPfFull = new TGraph ( 11, &jetPt[0], &JetPFFull[0] );
TGraph* grPfFast = new TGraph ( 11, &jetPt[0], &JetPFFast[0] );

TCanvas *c = new TCanvas();
FormatPad(c,false);
c->cd();

TH2F *h = new TH2F("Systematics","", 
		   100, 20., 620., 100, 0.0, 1.2 );
h->SetTitle( "CMS Preliminary" );
h->SetXTitle("p_{T} [GeV/c]" );
h->SetYTitle("Jet Energy scale");
h->SetStats(0);
h->Draw();
gPad->SetGridx();
gPad->SetGridy();

grPfFull->SetMarkerColor(2);						
grPfFull->SetMarkerStyle(22);
grPfFull->SetMarkerSize(1.5);						
grPfFull->SetLineWidth(2);
grPfFull->SetLineColor(2);
grPfFull->Draw("CP");

grPfFast->SetMarkerColor(4);						
grPfFast->SetMarkerStyle(25);
grPfFast->SetMarkerSize(1.5);						
grPfFast->SetLineWidth(2);
grPfFast->SetLineColor(4);
grPfFast->Draw("CP");

TLegend *leg=new TLegend(0.65,0.25,0.90,0.45);
leg->AddEntry(grPfFull, "Full Simulation", "lp");
leg->AddEntry(grPfFast, "Fast Simulation", "lp");
leg->SetTextSize(0.03);
leg->Draw();


gPad->SaveAs("SystematicPF2.png");
gPad->SaveAs("SystematicPF2.pdf");

TCanvas *cz = new TCanvas();
FormatPad(cz,false);
cz->cd();

TH2F *h = new TH2F("Systematics","", 
		   100, 20., 620., 100, 0.9, 1.0 );
h->SetTitle( "CMS Preliminary" );
h->SetXTitle("p_{T} [GeV/c]" );
h->SetYTitle("Jet Energy scale");
h->SetStats(0);
h->Draw();
gPad->SetGridx();
gPad->SetGridy();

grPfFull->SetMarkerColor(2);						
grPfFull->SetMarkerStyle(22);
grPfFull->SetMarkerSize(1.5);						
grPfFull->SetLineWidth(2);
grPfFull->SetLineColor(2);
grPfFull->Draw("CP");

grPfFast->SetMarkerColor(4);						
grPfFast->SetMarkerStyle(25);
grPfFast->SetMarkerSize(1.5);						
grPfFast->SetLineWidth(2);
grPfFast->SetLineColor(4);
grPfFast->Draw("CP");

TLegend *legz=new TLegend(0.65,0.25,0.90,0.45);
legz->AddEntry(grPfFull, "Full Simulation", "lp");
legz->AddEntry(grPfFast, "Fast Simulation", "lp");
legz->SetTextSize(0.03);
legz->Draw();


gPad->SaveAs("SystematicPF2_Zoom.png");
gPad->SaveAs("SystematicPF2_Zoom.pdf");


}
