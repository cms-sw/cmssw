{ 

#include <vector>

gROOT->LoadMacro("../Tools/NicePlot.C");
InitNicePlot();


vector<float> JetPFBarrel;
vector<float> SigmaJets;
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
jetPt.push_back(620.);
jetPt.push_back(445.);
jetPt.push_back(346.);
jetPt.push_back(274.);
jetPt.push_back(223.);
jetPt.push_back(173.);
jetPt.push_back(123.);
jetPt.push_back(89.5);
jetPt.push_back(69.6);
jetPt.push_back(49.6);
jetPt.push_back(29.8);
jetPt.push_back(17.0);
jetPt.push_back(17.0);

SigmaJets.push_back(1 - 0.0877100);
SigmaJets.push_back(1 - 0.0924872);
SigmaJets.push_back(1 - 0.0751864);
SigmaJets.push_back(1 - 0.0643899);
SigmaJets.push_back(1 - 0.0578865);
SigmaJets.push_back(1 - 0.0522699);
SigmaJets.push_back(1 - 0.0487406);
SigmaJets.push_back(1 - 0.0471545);
SigmaJets.push_back(1 - 0.048942);
SigmaJets.push_back(1 - 0.0426602);
SigmaJets.push_back(1 - 0.0366337);
SigmaJets.push_back(1 - 0.0269305);
SigmaJets.push_back(1 - 0.0264192);
SigmaJets.push_back(1 - 0.0340472);
SigmaJets.push_back(1 - 0.038924);
SigmaJets.push_back(1 - 0.0427373);
SigmaJets.push_back(1 - 0.0409938);
SigmaJets.push_back(1 - 0.0414219);
SigmaJets.push_back(1 - 0.0438692);
SigmaJets.push_back(1 - 0.0482692);
SigmaJets.push_back(1 - 0.0552161);
SigmaJets.push_back(1 - 0.069731);
SigmaJets.push_back(1 - 0.0868196);
SigmaJets.push_back(1 - 0.0846300);
SigmaJets.push_back(1 - 0.0877100);

JetPFBarrel.push_back(1 - (0.0877100+0.0846300)/2.);
JetPFBarrel.push_back(1 - (0.0924872+0.0868196)/2.);
JetPFBarrel.push_back(1 - (0.0751864+0.069731)/2.);
JetPFBarrel.push_back(1 - (0.0643899+0.0552161)/2.);
JetPFBarrel.push_back(1 - (0.0578865+0.0482692)/2.);
JetPFBarrel.push_back(1 - (0.0522699+0.0438692)/2.);
JetPFBarrel.push_back(1 - (0.0487406+0.0414219)/2.);
JetPFBarrel.push_back(1 - (0.0471545+0.0409938)/2.);
JetPFBarrel.push_back(1 - (0.0489420+0.0427373)/2.);
JetPFBarrel.push_back(1 - (0.0426602+0.038924)/2.);
JetPFBarrel.push_back(1 - (0.0366337+0.0340472)/2.);
JetPFBarrel.push_back(1 - (0.0269305+0.0264192)/2.);

TGraph* grPfBarrel = new TGraph ( 12, &jetPt[0], &JetPFBarrel[0] );
//TGraph* grPfEndcap = new TGraph ( 11, &jetPt[0], &JetPFEndcap[0] );
//TGraph* grPfBarrelP = new TGraph ( 11, &jetPt[0], &JetPFBarrelP[0] );
//TGraph* grPfEndcapP = new TGraph ( 11, &jetPt[0], &JetPFEndcapP[0] );

TPolyLine* systeBarrel = new TPolyLine(25, &jetPt[0], &SigmaJets[0]);
//TPolyLine* systeEndcap = new TPolyMarker(23, &jetPt[0], &JetPFBarrelP[0]);


TCanvas *cBarrel = new TCanvas();
FormatPad(cBarrel,false);
cBarrel->cd();

TH2F *h = new TH2F("Systematics","", 
		   100, 15., 620., 100, 0.79, 1.01 );
                   //100, 15., 620., 100, 0.00, 1.20 );
FormatHisto(h,sback);
h->SetTitle( "CMS Preliminary" );
h->SetXTitle("p_{T} [GeV/c]" );
h->SetYTitle("Jet Response");
h->SetStats(0);
h->Draw();
gPad->SetGridx();
gPad->SetGridy();

systeBarrel->SetLineColor(4);
systeBarrel->SetLineWidth(1.8);
systeBarrel->SetFillColor(4);
systeBarrel->SetFillStyle(3004);
systeBarrel->Draw("f");
systeBarrel->Draw();

grPfBarrel->SetMarkerColor(2);						
grPfBarrel->SetMarkerStyle(22);
grPfBarrel->SetMarkerSize(1.5);						
grPfBarrel->SetLineWidth(2);
grPfBarrel->SetLineColor(2);
grPfBarrel->Draw("CP");

TLegend *legz=new TLegend(0.60,0.25,0.85,0.45);
legz->AddEntry(grPfBarrel, "Default Resolution", "lp");
legz->AddEntry(systeBarrel, "Resolution #pm 50% ", "lf");
legz->SetTextSize(0.03);
legz->Draw();


gPad->SaveAs("SystematicPF4_Zoom.png");
gPad->SaveAs("SystematicPF4_Zoom.pdf");
//gPad->SaveAs("SystematicPF4.png");
//gPad->SaveAs("SystematicPF4.pdf");
}
