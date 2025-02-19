{ 

#include <vector>

gROOT->LoadMacro("../Tools/NicePlot.C");
InitNicePlot();


vector<float> GluonJets;
vector<float> BJets;
vector<float> CJets;
vector<float> UDSJets;
vector<float> QCDJets;
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

// Gluons
/*
GluonJets.push_back(1-0.0929962);
GluonJets.push_back(1-0.0614971);
GluonJets.push_back(1-0.0461211);
GluonJets.push_back(1-0.0484418);
GluonJets.push_back(1-0.0463932);
GluonJets.push_back(1-0.0469397);
GluonJets.push_back(1-0.0518321);
GluonJets.push_back(1-0.0443672);
GluonJets.push_back(1-0.0395973);
GluonJets.push_back(1-0.0307177);
GluonJets.push_back(1-0.0194428);
*/
GluonJets.push_back(1 - 0.076324);
GluonJets.push_back(1 - 0.092596);
GluonJets.push_back(1 - 0.0614971);
GluonJets.push_back(1 - 0.0458769);
GluonJets.push_back(1 - 0.0513169);
GluonJets.push_back(1 - 0.047205);
GluonJets.push_back(1 - 0.0466603);
GluonJets.push_back(1 - 0.0519459);
GluonJets.push_back(1 - 0.0442138);
GluonJets.push_back(1 - 0.0396926);
GluonJets.push_back(1 - 0.03057);
GluonJets.push_back(1 - 0.0195085);

// uds quarks
/*
UDSJets.push_back(1-0.0374509-0.005);
UDSJets.push_back(1-0.0215221-0.005);
UDSJets.push_back(1-0.020943-0.005);
UDSJets.push_back(1-0.0211214-0.005);
UDSJets.push_back(1-0.0228523-0.005);
UDSJets.push_back(1-0.0251432-0.005);
UDSJets.push_back(1-0.0211393-0.005);
UDSJets.push_back(1-0.0207524-0.005);
UDSJets.push_back(1-0.013009-0.005);
UDSJets.push_back(1-0.00861667-0.005);
UDSJets.push_back(1-0.000191365-0.005);
*/
UDSJets.push_back(1 - 0.046883-0.003);
UDSJets.push_back(1 - 0.037356-0.003);
UDSJets.push_back(1 - 0.021645-0.003);
UDSJets.push_back(1 - 0.021698-0.003);
UDSJets.push_back(1 - 0.021317-0.003);
UDSJets.push_back(1 - 0.023404-0.003);
UDSJets.push_back(1 - 0.0256166-0.003);
UDSJets.push_back(1 - 0.0219457-0.003);
UDSJets.push_back(1 - 0.0210752-0.003);
UDSJets.push_back(1 - 0.0131959-0.003);
UDSJets.push_back(1 - 0.00887791-0.003);
UDSJets.push_back(1 - 0.000369609-0.003);

// c quarks
/*
CJets.push_back(1-0.067642);
CJets.push_back(1-0.0430367);
CJets.push_back(1-0.037203);
CJets.push_back(1-0.0433098);
CJets.push_back(1-0.0442006);
CJets.push_back(1-0.0441374);
CJets.push_back(1-0.0388044);
CJets.push_back(1-0.0375926);
CJets.push_back(1-0.029705);
CJets.push_back(1-0.0200212);
CJets.push_back(1-0.00988312);
*/
CJets.push_back(1 - 0.085036);
CJets.push_back(1 - 0.0658586);
CJets.push_back(1 - 0.0380552);
CJets.push_back(1 - 0.0347716);
CJets.push_back(1 - 0.0387312);
CJets.push_back(1 - 0.0384288);
CJets.push_back(1 - 0.0367445);
CJets.push_back(1 - 0.0316843);
CJets.push_back(1 - 0.030752);
CJets.push_back(1 - 0.0241302);
CJets.push_back(1 - 0.0154405);
CJets.push_back(1 - 0.00853577);

// b quarks
/*
BJets.push_back(1-0.0874662);
BJets.push_back(1-0.0512796);
BJets.push_back(1-0.0466355);
BJets.push_back(1-0.0448335);
BJets.push_back(1-0.0523162);
BJets.push_back(1-0.0527279);
BJets.push_back(1-0.0545219);
BJets.push_back(1-0.0544589);
BJets.push_back(1-0.0463127);
BJets.push_back(1-0.0379773);
BJets.push_back(1-0.0238118);
*/
BJets.push_back(1 - 0.068905);
BJets.push_back(1 - 0.0864237);
BJets.push_back(1 - 0.04851);
BJets.push_back(1 - 0.0387301);
BJets.push_back(1 - 0.0321592);
BJets.push_back(1 - 0.0376053);
BJets.push_back(1 - 0.0365119);
BJets.push_back(1 - 0.0417268);
BJets.push_back(1 - 0.0414892);
BJets.push_back(1 - 0.0351108);
BJets.push_back(1 - 0.029757);
BJets.push_back(1 - 0.0196275);


QCDJets.push_back(0.913542);
QCDJets.push_back(0.929886);
QCDJets.push_back(0.947261);
QCDJets.push_back(0.947005);
QCDJets.push_back(0.956455);
QCDJets.push_back(0.956616);
QCDJets.push_back(0.956872);
QCDJets.push_back(0.957936);
QCDJets.push_back(0.963562);
QCDJets.push_back(0.967255);
QCDJets.push_back(0.975126);

TGraph* grQCD = new TGraph ( 12, &jetPt[0], &QCDJets[0] );
TGraph* grUDS = new TGraph ( 12, &jetPt[0], &UDSJets[0] );
TGraph* grGluon = new TGraph ( 12, &jetPt[0], &GluonJets[0] );
TGraph* grC = new TGraph ( 12, &jetPt[0], &CJets[0] );
TGraph* grB = new TGraph ( 12, &jetPt[0], &BJets[0] );


TCanvas *c = new TCanvas();
FormatPad(c,false);
c->cd();

TH2F *h = new TH2F("Systematics","", 
		   100, 29., 620., 100, 0.0, 1.2 );
FormatHisto(h,sback);
h->SetTitle( "CMS Preliminary" );
h->SetXTitle("p_{T} [GeV/c]" );
h->SetYTitle("Jet Response");
h->SetStats(0);
h->Draw();
gPad->SetGridx();
gPad->SetGridy();

//grQCD->SetMarkerColor(2);						
//grQCD->SetMarkerStyle(22);
//grQCD->SetMarkerSize(1.2);						
//grQCD->SetLineWidth(2);
//grQCD->SetLineColor(2);
//grQCD->Draw("CP");

grUDS->SetMarkerColor(1);						
grUDS->SetMarkerStyle(23);
grUDS->SetMarkerSize(1.2);						
grUDS->SetLineWidth(2);
grUDS->SetLineColor(1);
grUDS->Draw("CP");

grGluon->SetMarkerColor(4);						
grGluon->SetMarkerStyle(24);
grGluon->SetMarkerSize(1.2);						
grGluon->SetLineWidth(2);
grGluon->SetLineColor(4);
grGluon->Draw("CP");

grC->SetMarkerColor(2);						
grC->SetMarkerStyle(25);
grC->SetMarkerSize(1.2);						
grC->SetLineWidth(2);
grC->SetLineColor(2);
grC->Draw("CP");

grB->SetMarkerColor(3);						
grB->SetMarkerStyle(26);
grB->SetMarkerSize(1.2);						
grB->SetLineWidth(2);
grB->SetLineColor(3);
grB->Draw("CP");

TLegend *leg=new TLegend(0.72,0.18,0.88,0.52);
//leg->AddEntry(grQCD, "QCD Jets", "lp");
leg->AddEntry(grGluon, "Gluon Jets", "lp");
leg->AddEntry(grUDS, "uds Jets", "lp");
leg->AddEntry(grC, "c Jets", "lp");
leg->AddEntry(grB, "b Jets", "lp");
leg->SetTextSize(0.03);
leg->Draw();

gPad->SaveAs("SystematicPF3.png");
gPad->SaveAs("SystematicPF3.pdf");

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


//grQCD->Draw("CP");
grUDS->Draw("CP");
grGluon->Draw("CP");
grC->Draw("CP");
grB->Draw("CP");

TLegend *legz=new TLegend(0.72,0.18,0.88,0.52);
//legz->AddEntry(grQCD, "QCD Jets", "lp");
legz->AddEntry(grGluon, "Gluon Jets", "lp");
legz->AddEntry(grUDS, "uds Jets", "lp");
legz->AddEntry(grC, "c Jets", "lp");
legz->AddEntry(grB, "b Jets", "lp");
legz->SetTextSize(0.03);
legz->Draw();

gPad->SaveAs("SystematicPF3_Zoom.png");
gPad->SaveAs("SystematicPF3_Zoom.pdf");



}
