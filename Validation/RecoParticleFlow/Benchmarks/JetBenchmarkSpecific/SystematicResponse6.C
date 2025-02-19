{ 

#include <vector>

gROOT->LoadMacro("../Tools/NicePlot.C");
InitNicePlot();


vector<float> JetPFBarrel;
vector<float> JetTracking0;
vector<float> ThreshJets;
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
jetPt.push_back(15.0);
jetPt.push_back(15.0);

ThreshJets.push_back(1 - 0.0994200);
ThreshJets.push_back(1 - 0.102375);
ThreshJets.push_back(1 - 0.0865635);
ThreshJets.push_back(1 - 0.0770886);
ThreshJets.push_back(1 - 0.0689717);
ThreshJets.push_back(1 - 0.0648082);
ThreshJets.push_back(1 - 0.0593238);
ThreshJets.push_back(1 - 0.0577405);
ThreshJets.push_back(1 - 0.0587558);
ThreshJets.push_back(1 - 0.0519235);
ThreshJets.push_back(1 - 0.0456619);
ThreshJets.push_back(1 - 0.0355071);
ThreshJets.push_back(1 - 0.0216278);
ThreshJets.push_back(1 - 0.0288968);
ThreshJets.push_back(1 - 0.0338232);
ThreshJets.push_back(1 - 0.0383845);
ThreshJets.push_back(1 - 0.0361105);
ThreshJets.push_back(1 - 0.0360658);
ThreshJets.push_back(1 - 0.0394715);
ThreshJets.push_back(1 - 0.0432623);
ThreshJets.push_back(1 - 0.0497155);
ThreshJets.push_back(1 - 0.0657936);
ThreshJets.push_back(1 - 0.0842258);
ThreshJets.push_back(1 - 0.08007000);
ThreshJets.push_back(1 - 0.0994200);

JetPFBarrel.push_back(1 - (0.0994200+0.0800700)/2.);
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

TGraph* grPfBarrel = new TGraph ( 12, &jetPt[0], &JetPFBarrel[0] );
//TGraph* grPfEndcap = new TGraph ( 11, &jetPt[0], &JetPFEndcap[0] );
//TGraph* grPfBarrelP = new TGraph ( 11, &jetPt[0], &JetPFBarrelP[0] );
//TGraph* grPfEndcapP = new TGraph ( 11, &jetPt[0], &JetPFEndcapP[0] );

TPolyLine* systeBarrel = new TPolyLine(25, &jetPt[0], &ThreshJets[0]);
//TPolyLine* systeEndcap = new TPolyMarker(23, &jetPt[0], &JetPFBarrelP[0]);

TCanvas *cz = new TCanvas();
FormatPad(cz,false);
cz->cd();


TH2F *hz = new TH2F("Systematics","", 
		    100, 15., 620., 100, 0.79, 1.01 );
                    //100, 15., 620., 100, 0.00, 1.20 );
FormatHisto(hz,sback);
hz->SetTitle( "CMS Preliminary" );
hz->SetXTitle("p_{T} [GeV/c]" );
hz->SetYTitle("Jet Response");
hz->SetStats(0);
hz->Draw();
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
legz->AddEntry(grPfBarrel, "Default Thresholds", "lp");
legz->AddEntry(systeBarrel, "Thresholds #pm 50% ", "lf");
legz->SetTextSize(0.03);
legz->Draw();


gPad->SaveAs("SystematicPF6_Zoom.png");
gPad->SaveAs("SystematicPF6_Zoom.pdf");
//gPad->SaveAs("SystematicPF6.png");
//gPad->SaveAs("SystematicPF6.pdf");
}
