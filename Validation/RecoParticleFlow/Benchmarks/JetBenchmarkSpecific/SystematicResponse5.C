{ 

#include <vector>

gROOT->LoadMacro("../Tools/NicePlot.C");
InitNicePlot();


vector<float> JetTracking0;
vector<float> JetTracking5;
vector<float> jetPt;

jetPt.push_back(17.0);
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

JetTracking0.push_back(1 - 0.0865300);
JetTracking0.push_back(1 - 0.0888898);
JetTracking0.push_back(1 - 0.0711393);
JetTracking0.push_back(1 - 0.0587086);
JetTracking0.push_back(1 - 0.0518305);
JetTracking0.push_back(1 - 0.0467558);
JetTracking0.push_back(1 - 0.0432953);
JetTracking0.push_back(1 - 0.0421117);
JetTracking0.push_back(1 - 0.044124);
JetTracking0.push_back(1 - 0.0396139);
JetTracking0.push_back(1 - 0.0341208);
JetTracking0.push_back(1 - 0.0259526);

JetTracking5.push_back(1 - 0.0954600);
JetTracking5.push_back(1 - 0.107235);
JetTracking5.push_back(1 - 0.0900822);
JetTracking5.push_back(1 - 0.0715365);
JetTracking5.push_back(1 - 0.062945);
JetTracking5.push_back(1 - 0.0575109);
JetTracking5.push_back(1 - 0.0513701);
JetTracking5.push_back(1 - 0.048963);
JetTracking5.push_back(1 - 0.0498712);
JetTracking5.push_back(1 - 0.0445844);
JetTracking5.push_back(1 - 0.0373004);
JetTracking5.push_back(1 - 0.0284231);

TGraph* grPfTracking0 = new TGraph ( 12, &jetPt[0], &JetTracking0[0] );
TGraph* grPfTracking5 = new TGraph ( 12, &jetPt[0], &JetTracking5[0] );

TCanvas *c = new TCanvas();
FormatPad(c,false);
c->cd();

TH2F *h = new TH2F("Systematics","", 
		   100, 20., 620., 100, 0.0, 1.2 );
FormatHisto(h,sback);
h->SetTitle( "CMS Preliminary" );
h->SetXTitle("p_{T} [GeV/c]" );
h->SetYTitle("Jet Response");
h->SetStats(0);
h->Draw();
gPad->SetGridx();
gPad->SetGridy();

grPfTracking0->SetMarkerColor(2);						
grPfTracking0->SetMarkerStyle(22);
grPfTracking0->SetMarkerSize(1.5);						
grPfTracking0->SetLineWidth(2);
grPfTracking0->SetLineColor(2);
grPfTracking0->Draw("CP");

grPfTracking5->SetMarkerColor(4);						
grPfTracking5->SetMarkerStyle(25);
grPfTracking5->SetMarkerSize(1.5);						
grPfTracking5->SetLineWidth(2);
grPfTracking5->SetLineColor(4);
grPfTracking5->Draw("CP");

TLegend *leg=new TLegend(0.65,0.25,0.90,0.45);
leg->AddEntry(grPfTracking0, "Tracking0 Simulation", "lp");
leg->AddEntry(grPfTracking5, "Tracking5 Simulation", "lp");
leg->SetTextSize(0.03);
leg->Draw();


gPad->SaveAs("SystematicPF5.png");
gPad->SaveAs("SystematicPF5.pdf");

TCanvas *cz = new TCanvas();
FormatPad(cz,false);
cz->cd();

TH2F *hz = new TH2F("Systematics","", 
		   100, 15., 620., 100, 0.79, 1.01 );
FormatHisto(hz,sback);
hz->SetTitle( "CMS Preliminary" );
hz->SetXTitle("p_{T} [GeV/c]" );
hz->SetYTitle("Jet Response");
hz->SetStats(0);
hz->Draw();
gPad->SetGridx();
gPad->SetGridy();

grPfTracking0->SetMarkerColor(2);						
grPfTracking0->SetMarkerStyle(22);
grPfTracking0->SetMarkerSize(1.5);						
grPfTracking0->SetLineWidth(2);
grPfTracking0->SetLineColor(2);
grPfTracking0->Draw("CP");

grPfTracking5->SetMarkerColor(4);						
grPfTracking5->SetMarkerStyle(25);
grPfTracking5->SetMarkerSize(1.5);						
grPfTracking5->SetLineWidth(2);
grPfTracking5->SetLineColor(4);
grPfTracking5->Draw("CP");

TLegend *legz=new TLegend(0.55,0.25,0.90,0.45);
legz->AddEntry(grPfTracking0, "Default tracking efficiency", "lp");
legz->AddEntry(grPfTracking5, "Tracking efficiency -5%", "lp");
legz->SetTextSize(0.03);
legz->Draw();


gPad->SaveAs("SystematicPF5_Zoom.png");
gPad->SaveAs("SystematicPF5_Zoom.pdf");


}
