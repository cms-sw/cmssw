{ 

#include <vector>

gROOT->LoadMacro("../Tools/NicePlot.C");
InitNicePlot();


vector<float> JetPFBarrel;
vector<float> JetPFBarrelP;
vector<float> JetPFEndcap;
vector<float> JetPFEndcapP;
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

//jetPt.push_back(1000.);
//jetPt.push_back(1620.);
//jetPt.push_back(3000.);
//jetPt.push_back(3000.);
//jetPt.push_back(1620.);
//jetPt.push_back(1000.);


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
jetPt.push_back(29.8);

JetPFBarrelP.push_back(0.94016);
JetPFBarrelP.push_back(0.929155);
JetPFBarrelP.push_back(0.947731);
JetPFBarrelP.push_back(0.953819);
JetPFBarrelP.push_back(0.961707);
JetPFBarrelP.push_back(0.978224);
JetPFBarrelP.push_back(0.985614);
JetPFBarrelP.push_back(0.987815);
JetPFBarrelP.push_back(0.992177);
JetPFBarrelP.push_back(0.999556);
JetPFBarrelP.push_back(1.00401);
JetPFBarrelP.push_back(1.01505);

//JetPFBarrelP.push_back(1.02605);
//JetPFBarrelP.push_back(1.03805);
//JetPFBarrelP.push_back(1.04505);
//JetPFBarrelP.push_back(0.95805);
//JetPFBarrelP.push_back(0.95305);
//JetPFBarrelP.push_back(0.94405);


JetPFBarrelP.push_back(0.935982);
JetPFBarrelP.push_back(0.931657);
JetPFBarrelP.push_back(0.930688);
JetPFBarrelP.push_back(0.928387);
JetPFBarrelP.push_back(0.931187);
JetPFBarrelP.push_back(0.936027);
JetPFBarrelP.push_back(0.940989);
JetPFBarrelP.push_back(0.945885);
JetPFBarrelP.push_back(0.940601);
JetPFBarrelP.push_back(0.924205);
JetPFBarrelP.push_back(0.902948);
JetPFBarrelP.push_back(0.91252);
JetPFBarrelP.push_back(0.94016);

JetPFBarrel.push_back(0.926349);
JetPFBarrel.push_back(0.913542);
JetPFBarrel.push_back(0.929886);
JetPFBarrel.push_back(0.947261);
JetPFBarrel.push_back(0.947005);
JetPFBarrel.push_back(0.956455);
JetPFBarrel.push_back(0.956616);
JetPFBarrel.push_back(0.956872);
JetPFBarrel.push_back(0.957936);
JetPFBarrel.push_back(0.963562);
JetPFBarrel.push_back(0.967255);
JetPFBarrel.push_back(0.975126);

//JetPFBarrel.push_back(0.985126);
//JetPFBarrel.push_back(0.995126);
//JetPFBarrel.push_back(1.001526);

JetPFEndcapP.push_back(0.910001);
JetPFEndcapP.push_back(0.903306);
JetPFEndcapP.push_back(0.910116);
JetPFEndcapP.push_back(0.902537);
JetPFEndcapP.push_back(0.901861);
JetPFEndcapP.push_back(0.905012);
JetPFEndcapP.push_back(0.909464);
JetPFEndcapP.push_back(0.922443);
JetPFEndcapP.push_back(0.925539);
JetPFEndcapP.push_back(0.93416);
JetPFEndcapP.push_back(0.939964);

JetPFEndcap.push_back(0.909817);
JetPFEndcap.push_back(0.903072);
JetPFEndcap.push_back(0.909032);
JetPFEndcap.push_back(0.902627);
JetPFEndcap.push_back(0.90178);
JetPFEndcap.push_back(0.905156);
JetPFEndcap.push_back(0.909015);
JetPFEndcap.push_back(0.92196);
JetPFEndcap.push_back(0.925392);
JetPFEndcap.push_back(0.934317);
JetPFEndcap.push_back(0.939878);

TGraph* grPfBarrel = new TGraph ( 12, &jetPt[0], &JetPFBarrel[0] );
//TGraph* grPfBarrel = new TGraph ( 14, &jetPt[0], &JetPFBarrel[0] );
//TGraph* grPfEndcap = new TGraph ( 11, &jetPt[0], &JetPFEndcap[0] );
//TGraph* grPfBarrelP = new TGraph ( 11, &jetPt[0], &JetPFBarrelP[0] );
//TGraph* grPfEndcapP = new TGraph ( 11, &jetPt[0], &JetPFEndcapP[0] );

TPolyLine* systeBarrel = new TPolyLine(25, &jetPt[0], &JetPFBarrelP[0]);
//TPolyLine* systeBarrel = new TPolyLine(29, &jetPt[0], &JetPFBarrelP[0]);
//TPolyLine* systeEndcap = new TPolyMarker(23, &jetPt[0], &JetPFBarrelP[0]);

TCanvas *cBarrel = new TCanvas();
FormatPad(cBarrel,false);
cBarrel->cd();

TH2F *h = new TH2F("Systematics","", 
		   100, 15., 620., 100, 0.0, 1.2 );
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

/*
grPfEndcap->SetMarkerColor(4);						
grPfEndcap->SetMarkerStyle(23);
grPfEndcap->SetMarkerSize(1.5);						
grPfEndcap->SetLineWidth(2);
grPfEndcap->SetLineColor(4);
grPfEndcap->Draw("CP");

systeEndcap->SetLineColor(4);
systeEndcap->SetLineWidth(2);
systeEndcap->SetFillColor(4);
systeEndcap->SetFillStyle(2002);
systeEndcap->Draw("same");
*/

TLegend *legz=new TLegend(0.60,0.25,0.85,0.45);
legz->AddEntry(grPfBarrel, "Default Correction", "lp");
legz->AddEntry(systeBarrel, "Correction #pm 50% ", "lf");
legz->SetTextSize(0.03);
legz->Draw();


gPad->SaveAs("SystematicPF1.png");
gPad->SaveAs("SystematicPF1.pdf");
}
