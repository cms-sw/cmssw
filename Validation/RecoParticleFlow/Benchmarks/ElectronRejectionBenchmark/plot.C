{
gSystem->Load("libFWCoreFWLite.so");
gSystem->Load("libValidationRecoParticleFlow.so");

//gROOT->LoadMacro("../Tools/NicePlot.C");
//InitNicePlot();
gROOT->LoadMacro("./tools.C");

gROOT->SetStyle("Plain");
gStyle->SetOptStat(0);
gStyle->SetErrorX(0);
gStyle->SetOptTitle(kFALSE);
gStyle->SetCanvasBorderMode(0);
gStyle->SetFrameBorderMode(0);
gStyle->SetPadRightMargin(0.08);
gStyle->SetPadLeftMargin(0.2);
gStyle->SetPadBottomMargin(0.2);

gStyle->SetCanvasDefH(700);
gStyle->SetCanvasDefW(1400);
gStyle->SetLabelOffset(0.001,"Y");
gStyle->SetTitleOffset(0.9,"X");
gStyle->SetTitleOffset(0.9,"Y");
gStyle->SetTitleXSize(0.07);
gStyle->SetTitleYSize(0.07);
gROOT->ForceStyle();


bool normHists= true;
TString plotDir = "./";
TString plotName = plotDir+"tauBenchmarkElecRej";

TFile f1("tauBenchmarkElecRejection_ztt.root");
TFile f2("tauBenchmarkElecRejection_zee.root");

TString dir = "DQMData/PFTask/Benchmarks/PFTauElecRejection/Gen";

TCanvas c1;
Styles::FormatPad( &c1, false );
c1->Divide(4,2);

Styles styles;
Style* s1 = styles.s1;
Style* s2 = styles.s2;

//////
c1->cd(1);
f1.cd(dir);
TH1F* htau_EoP = (TH1F*) gDirectory.Get("EoverP");
//htau_EoP.Rebin(2);
Styles::FormatHisto(htau_EoP, s1);
htau_EoP.Draw();

f2.cd(dir);
TH1F* helec_EoP = (TH1F*) gDirectory.Get("EoverP");
//helec_EoP.Rebin(2);
Styles::FormatHisto(helec_EoP, s2);
helec_EoP.Draw("same");

c1_1->SetLogy(1);
if (normHists) {
  NormHistos(htau_EoP,helec_EoP);
}

// fill performance graphs
TGraph* eopPerf = new TGraph();
fillPerfGraphESUM(eopPerf,htau_EoP,helec_EoP);


//////
c1->cd(2);
f1.cd(dir);
TH1F* htau_HoP = (TH1F*) gDirectory.Get("HoverP");
//htau_HoP.Rebin(2);
Styles::FormatHisto(htau_HoP, s1);
htau_HoP.Draw();

f2.cd(dir);
TH1F* helec_HoP = (TH1F*) gDirectory.Get("HoverP");
//helec_HoP.Rebin(2);
Styles::FormatHisto(helec_HoP, s2);
helec_HoP.Draw("same");

c1_2->SetLogy(1);
if (normHists) {
  NormHistos(htau_HoP,helec_HoP);
}

// fill performance graphs
TGraph* h3x3Perf = new TGraph();
fillPerfGraphHoP(h3x3Perf,htau_HoP,helec_HoP);

/*
//////
c1->cd(3);
f1.cd(dir);
TH1F* htau_epreid = (TH1F*) gDirectory.Get("ElecPreID");
//htau_epreid.Rebin(2);
Styles::FormatHisto(htau_epreid, s1);
htau_epreid.Draw();

f2.cd(dir);
TH1F* helec_epreid = (TH1F*) gDirectory.Get("ElecPreID");
//helec_epreid.Rebin(2);
Styles::FormatHisto(helec_epreid, s2);
helec_epreid.Draw("same");

c1_3->SetLogy(1);
if (normHists) {
  NormHistos(htau_epreid,helec_epreid);
}

//
TGraph* epreidPerf = new TGraph();
fillPerfGraphEPreID(epreidPerf,htau_epreid,helec_epreid);
*/


//////
c1->cd(4);
f1.cd(dir);
TH1F* htau_etauD = (TH1F*) gDirectory.Get("TauElecDiscriminant");
//htau_etauD.Rebin(2);
Styles::FormatHisto(htau_etauD, s1);
htau_etauD.Draw();

f2.cd(dir);
TH1F* helec_etauD = (TH1F*) gDirectory.Get("TauElecDiscriminant");
//helec_etauD.Rebin(2);
Styles::FormatHisto(helec_etauD, s2);
helec_etauD.Draw("same");

c1_4->SetLogy(1);
if (normHists) {
  NormHistos(htau_etauD,helec_etauD);
}

//
TGraph* discriminantPerf = new TGraph();
fillPerfGraphDiscr(discriminantPerf,htau_etauD,helec_etauD);

//////
c1->cd(5);
f1.cd(dir);
TH2F* htau_hvseop_preid0 = (TH2F*) gDirectory.Get("HoPvsEoP_preid0");
//htau_hvseop_preid0.Rebin(2);
htau_hvseop_preid0.SetLineColor(kBlack);
htau_hvseop_preid0.SetLineWidth(2);
htau_hvseop_preid0.Draw("box");

TLine li1;li1.SetLineWidth(2.);li1.SetLineColor(kBlue);
li1.DrawLine(0.95,0.05,2.,0.05);
TLine li2;li2.SetLineWidth(2.);li2.SetLineColor(kBlue);
li2.DrawLine(0.95,0.,0.95,0.05);

//////
c1->cd(6);
f2.cd(dir);
TH2F* helec_hvseop_preid0 = (TH2F*) gDirectory.Get("HoPvsEoP_preid0");
helec_hvseop_preid0.SetLineColor(kBlue);
helec_hvseop_preid0.SetLineWidth(2);
helec_hvseop_preid0.Draw("box");

TLine li1;li1.SetLineWidth(2.);li1.SetLineColor(kBlue);
li1.DrawLine(0.95,0.05,2.,0.05);
TLine li2;li2.SetLineWidth(2.);li2.SetLineColor(kBlue);
li2.DrawLine(0.95,0.,0.95,0.05);


//////
c1->cd(7);
f1.cd(dir);
TH2F* htau_hvseop_preid1 = (TH2F*) gDirectory.Get("HoPvsEoP_preid1");
htau_hvseop_preid1.SetLineColor(kBlack);
htau_hvseop_preid1.SetLineWidth(2);
htau_hvseop_preid1.Draw("box");

TLine li1;li1.SetLineWidth(2.);li1.SetLineColor(kRed);
li1.DrawLine(0.8,0.15,2.,0.15);
TLine li2;li2.SetLineWidth(2.);li2.SetLineColor(kRed);
li2.DrawLine(0.8,0.,0.8,0.15);

//////
c1->cd(8);
f2.cd(dir);
TH2F* helec_hvseop_preid1 = (TH2F*) gDirectory.Get("HoPvsEoP_preid1");
helec_hvseop_preid1.SetLineColor(kBlue);
helec_hvseop_preid1.SetLineWidth(2);
helec_hvseop_preid1.Draw("box");

TLine li1;li1.SetLineWidth(2.);li1.SetLineColor(kRed);
li1.DrawLine(0.8,0.15,2.,0.15);
TLine li2;li2.SetLineWidth(2.);li2.SetLineColor(kRed);
li2.DrawLine(0.8,0.,0.8,0.15);

//////
c1->cd();
gPad->SaveAs(plotName+"1.gif");


///////////////////

TCanvas c2;
Styles::FormatPad( &c2, false );
c2->Divide(3,2);

//////
c2->cd(1);
f1.cd(dir);
TH1F* htau_deltaEta = (TH1F*) gDirectory.Get("pfcand_deltaEta");
//htau_deltaEta.Rebin(2);
Styles::FormatHisto(htau_deltaEta, s1);
htau_deltaEta.Draw();

f2.cd(dir);
TH1F* helec_deltaEta = (TH1F*) gDirectory.Get("pfcand_deltaEta");
//helec_deltaEta.Rebin(2);
Styles::FormatHisto(helec_deltaEta, s2);
helec_deltaEta.Draw("same");

gPad->SetLogy(1);
if (normHists) {
  NormHistos(htau_deltaEta,helec_deltaEta);
}

//////
c2->cd(2);
f1.cd(dir);
TH1F* htau_deltaPhiOverQ = (TH1F*) gDirectory.Get("pfcand_deltaPhiOverQ");
//htau_deltaPhiOverQ.Rebin(2);
Styles::FormatHisto(htau_deltaPhiOverQ, s1);
htau_deltaPhiOverQ.Draw();

f2.cd(dir);
TH1F* helec_deltaPhiOverQ = (TH1F*) gDirectory.Get("pfcand_deltaPhiOverQ");
//helec_deltaPhiOverQ.Rebin(2);
Styles::FormatHisto(helec_deltaPhiOverQ, s2);
helec_deltaPhiOverQ.Draw("same");

gPad->SetLogy(1);
if (normHists) {
  NormHistos(htau_deltaPhiOverQ,helec_deltaPhiOverQ);
}

//////
c2->cd(4);
f1.cd(dir);
TH1F* htau_leadTk_pt = (TH1F*) gDirectory.Get("leadTk_pt");
//htau_leadTk_pt.Rebin(2);
Styles::FormatHisto(htau_leadTk_pt, s1);
htau_leadTk_pt.Draw();

f2.cd(dir);
TH1F* helec_leadTk_pt = (TH1F*) gDirectory.Get("leadTk_pt");
//helec_leadTk_pt.Rebin(2);
Styles::FormatHisto(helec_leadTk_pt, s2);
helec_leadTk_pt.Draw("same");

gPad->SetLogy(1);
if (normHists) {
  NormHistos(htau_leadTk_pt,helec_leadTk_pt);
}

//////
c2->cd(5);
f1.cd(dir);
TH1F* htau_leadTk_eta = (TH1F*) gDirectory.Get("leadTk_eta");
//htau_leadTk_eta.Rebin(2);
Styles::FormatHisto(htau_leadTk_eta, s1);
htau_leadTk_eta.Draw();

f2.cd(dir);
TH1F* helec_leadTk_eta = (TH1F*) gDirectory.Get("leadTk_eta");
//helec_leadTk_eta.Rebin(2);
Styles::FormatHisto(helec_leadTk_eta, s2);
helec_leadTk_eta.Draw("same");

gPad->SetLogy(1);
if (normHists) {
  NormHistos(htau_leadTk_eta,helec_leadTk_eta);
}

//////
c2->cd(6);
f1.cd(dir);
TH1F* htau_leadTk_phi = (TH1F*) gDirectory.Get("leadTk_phi");
//htau_leadTk_phi.Rebin(2);
Styles::FormatHisto(htau_leadTk_phi, s1);
htau_leadTk_phi.Draw();

f2.cd(dir);
TH1F* helec_leadTk_phi = (TH1F*) gDirectory.Get("leadTk_phi");
//helec_leadTk_phi.Rebin(2);
Styles::FormatHisto(helec_leadTk_phi, s2);
helec_leadTk_phi.Draw("same");

gPad->SetLogy(1);
if (normHists) {
  NormHistos(htau_leadTk_phi,helec_leadTk_phi);
}

//////
c2->cd(3);
f1.cd(dir);
TH1F* htau_mva = (TH1F*) gDirectory.Get("ElecMVA");
//htau_mva.Rebin(2);
Styles::FormatHisto(htau_mva, s1);
htau_mva.Draw();

f2.cd(dir);
TH1F* helec_mva = (TH1F*) gDirectory.Get("ElecMVA");
//helec_mva.Rebin(2);
Styles::FormatHisto(helec_mva, s2);
helec_mva.Draw("same");


// fill performance graphs
TGraph* mvaPerf = new TGraph();
fillPerfGraphMVA(mvaPerf,htau_mva,helec_mva);


gPad->SetLogy(1);
if (normHists) {
  NormHistos(htau_mva,helec_mva);
}


//return;
///////////////////


TCanvas c3;
Styles::FormatPad( &c3, false );
c3->Divide(2,1);

c3->cd(1);
gPad->SetLogy(0);
gPad->DrawFrame(0.58,0.,1.,0.102);
//gPad->DrawFrame(0.,0.,1.,1.);

/*
emfracPerf->SetLineColor(kBlack);
emfracPerf->SetLineWidth(2);
emfracPerf->SetMarkerColor(kBlack);
emfracPerf->SetMarkerStyle(21);
emfracPerf->SetTitle("graph");
emfracPerf->GetXaxis()->SetTitle("#tau_{had} Efficiency");
emfracPerf->GetYaxis()->SetTitle("e Efficiency");
emfracPerf->Draw("LP");
*/

eopPerf->SetLineColor(kRed);
eopPerf->SetLineWidth(2);
eopPerf->SetMarkerColor(kBlack);
eopPerf->SetMarkerStyle(20);
eopPerf->Draw("LP");

//mvaPerf->SetLineColor(kBlue);
//mvaPerf->SetLineWidth(2);
//mvaPerf->SetMarkerColor(kBlack);
//mvaPerf->SetMarkerStyle(21);
//mvaPerf->Draw("LP same");

h3x3Perf->SetLineColor(kGreen);
h3x3Perf->SetLineWidth(2);
h3x3Perf->SetMarkerColor(kBlack);
h3x3Perf->SetMarkerStyle(24);
h3x3Perf->Draw("LP same");

discriminantPerf->SetLineColor(kBlack);
discriminantPerf->SetLineWidth(2);
discriminantPerf->SetMarkerColor(kBlue);
discriminantPerf->SetMarkerStyle(28);
discriminantPerf->SetMarkerSize(2.4);
discriminantPerf->Draw("P same");

//epreidPerf->SetLineColor(kBlack);
//epreidPerf->SetLineWidth(2);
//epreidPerf->SetMarkerColor(kRed);
//epreidPerf->SetMarkerStyle(27);
//epreidPerf->SetMarkerSize(2.4);
//epreidPerf->Draw("LP same");


gPad->Update();
gPad->SetGrid(1);

TLatex l;
l.DrawLatex(0.82,-0.02,"#tau_{had} Efficiency");
l.DrawLatex(0.52,0.105,"electron Efficiency");
//l.SetTextAngle(90.);
//l.SetTextColor(kRed);
//l->PaintLatex(0.89,-0.01,0.,.04,"Efficiency");
//l.PaintLatex(0.6,0.07,0.,.4,"Rejection");
//l.Paint();
l.SetTextSize(0.052);
l.SetTextFont(40);
//l.DrawLatex(0.87,0.105,"CMS Preliminary");
//return;



//////
c3->cd(2);

TLegend* leg = new TLegend(0.1,0.1,0.89,0.89);
leg->SetHeader("Electron vs. tau efficiency");
leg->AddEntry(discriminantPerf,"Discriminant (MVA>-0.1)","p");
//leg->AddEntry(epreidPerf,"Electron Pre-ID","p");
//leg->AddEntry(d2test1Perf,"Emfrac-E/p","p");
//leg->AddEntry(emfracPerf,"EM fraction","pl");
leg->AddEntry(eopPerf,"E/p < [0.7,0.75,0.8,0.85,0.9]","lp");
leg->AddEntry(h3x3Perf,"H_{3x3}/p > [0.25,0.2,0.15,0.1,0.05]","lp");
//leg->AddEntry(mvaPerf,"MVA < [-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.]","pl");
gPad->Clear();
gPad->SetLogy(0);
leg->Draw();
      
gPad->Update();
gPad->Modified();

//////
c3->cd();
gPad->SaveAs(plotName+"2.gif");


}
