{

gROOT->LoadMacro("../Tools/NicePlot.C");
InitNicePlot();

TFile file("JetBenchmark_Full_310pre3.root");
TFile file2("JetBenchmark_Full_Tracking0.root");
  
gStyle->SetOptStat(0);
TH2F* histoPF2 = (TH2F*) file.Get("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen/BRPtvsPt");
TH2F* histoPF3 = (TH2F*) file2.Get("BRPtvsPt");
TH1F* histoPF15_20 = (TH1F*)(histoPF2->ProjectionY("",7,10)->Clone());
TH1F* histoPF15_20_2 = (TH1F*)(histoPF3->ProjectionY("",7,10)->Clone());
TH1F* histoPF20_40 = (TH1F*)(histoPF2->ProjectionY("",10,20)->Clone());
TH1F* histoPF20_40_2 = (TH1F*)(histoPF3->ProjectionY("",10,20)->Clone());
TH1F* histoPF40_60 = (TH1F*)(histoPF2->ProjectionY("",20,30)->Clone());
TH1F* histoPF40_60_2 = (TH1F*)(histoPF3->ProjectionY("",20,30)->Clone());
TH1F* histoPF300_400 = (TH1F*)(histoPF2->ProjectionY("",140,190)->Clone());
TH1F* histoPF300_400_2 = (TH1F*)(histoPF3->ProjectionY("",140,190)->Clone());
// Title
histoPF15_20->SetTitle("CMS Preliminary");
histoPF15_20_2->SetTitle("CMS Preliminary");
histoPF20_40->SetTitle("CMS Preliminary");
histoPF20_40_2->SetTitle("CMS Preliminary");
histoPF40_60->SetTitle("CMS Preliminary");
histoPF40_60_2->SetTitle("CMS Preliminary");
histoPF300_400->SetTitle("CMS Preliminary");
histoPF300_400_2->SetTitle("CMS Preliminary");
// Rebin
histoPF15_20->Rebin(8);
histoPF15_20_2->Rebin(8);
histoPF20_40->Rebin(8);
histoPF20_40_2->Rebin(8);
histoPF40_60->Rebin(5);
histoPF40_60_2->Rebin(5);
histoPF300_400->Rebin(2);
histoPF300_400_2->Rebin(2);
// Fit to a Gaussian
histoPF15_20->Fit("gaus");
histoPF15_20_2->Fit("gaus");
histoPF20_40->Fit("gaus");
histoPF20_40_2->Fit("gaus");
histoPF40_60->Fit("gaus");
histoPF300_400->Fit("gaus");
histoPF40_60_2->Fit("gaus");
histoPF300_400_2->Fit("gaus");
// Set axis
histoPF15_20->GetXaxis().SetRangeUser(-0.7,0.7);
histoPF15_20_2->GetXaxis().SetRangeUser(-0.7,0.7);
histoPF20_40->GetXaxis().SetRangeUser(-0.7,0.7);
histoPF20_40_2->GetXaxis().SetRangeUser(-0.7,0.7);
histoPF40_60->GetXaxis().SetRangeUser(-0.7,0.7);
histoPF40_60_2->GetXaxis().SetRangeUser(-0.7,0.7);
histoPF300_400->GetXaxis().SetRangeUser(-0.7,0.7);
histoPF300_400_2->GetXaxis().SetRangeUser(-0.7,0.7);

TF1* gaus15_20 = histoPF15_20->GetFunction( "gaus" );
TF1* gaus15_20_2 = histoPF15_20_2->GetFunction( "gaus" );
TF1* gaus20_40 = histoPF20_40->GetFunction( "gaus" );
TF1* gaus20_40_2 = histoPF20_40_2->GetFunction( "gaus" );
TF1* gaus40_60 = histoPF40_60->GetFunction( "gaus" );
TF1* gaus40_60_2 = histoPF40_60_2->GetFunction( "gaus" );
TF1* gaus300_400 = histoPF300_400->GetFunction( "gaus" );
TF1* gaus300_400_2 = histoPF300_400_2->GetFunction( "gaus" );

std::cout << "15_20 " << (gaus15_20->GetParameter(2)/(1.+min(0.,gaus15_20->GetParameter(1)))) << std::endl;
std::cout << "15_20_2 " << (gaus15_20_2->GetParameter(2)/(1.+min(0.,gaus15_20_2->GetParameter(1)))) << std::endl;
std::cout << "20_40 " << (gaus20_40->GetParameter(2)/(1.+min(0.,gaus20_40->GetParameter(1)))) << std::endl;
std::cout << "20_40_2 " << (gaus20_40_2->GetParameter(2)/(1.+min(0.,gaus20_40_2->GetParameter(1)))) << std::endl;
std::cout << "40_60 " << (gaus40_60->GetParameter(2)/(1.+min(0.,gaus40_60->GetParameter(1)))) << std::endl;
std::cout << "40_60_2 " << (gaus40_60_2->GetParameter(2)/(1.+min(0.,gaus40_60_2->GetParameter(1)))) << std::endl;
std::cout << "300_400 " << (gaus300_400->GetParameter(2)/(1.+min(0.,gaus300_400->GetParameter(1)))) << std::endl;
std::cout << "300_400_2 " << (gaus300_400_2->GetParameter(2)/(1.+min(0.,gaus300_400_2->GetParameter(1)))) << std::endl;

/* */
TCanvas* c1 = new TCanvas();
FormatPad(c1,false);
//gPad->SetLiny();

histoPF15_20->SetMaximum(180);
FormatHisto(histoPF15_20,spred);
histoPF15_20->SetXTitle("#Delta p_{T}/p_{T}");
histoPF15_20->SetYTitle("Events");
histoPF15_20->Draw();

TLegend *leg=new TLegend(0.57,0.75,0.94,0.85);
leg->AddEntry(histoPF15_20, "Particle-Flow Jets, 15-20 GeV", "lf");
leg->SetTextSize(0.03);
leg->Draw();

gPad->SaveAs("PFJet15_20.pdf");
gPad->SaveAs("PFJet15_20.png");

TCanvas* c2 = new TCanvas();
FormatPad(c2,false);
//gPad->SetLogy();

histoPF20_40->SetMaximum(800);
FormatHisto(histoPF20_40,spred);
histoPF20_40->SetXTitle("#Delta p_{T}/p_{T}");
histoPF20_40->SetYTitle("Events");
histoPF20_40->Draw();

TLegend *leg=new TLegend(0.57,0.75,0.94,0.85);
leg->AddEntry(histoPF20_40, "Particle-Flow Jets, 20-40 GeV", "lf");
leg->SetTextSize(0.03);
leg->Draw();

gPad->SaveAs("PFJet20_40.pdf");
gPad->SaveAs("PFJet20_40.png");

TCanvas* c3 = new TCanvas();
FormatPad(c3,false);

gPad->SetLogy();
histoPF40_60_2->SetMaximum(600.);
histoPF40_60_2->SetMinimum(1.);
FormatHisto(histoPF40_60_2,spred);
histoPF40_60_2->SetXTitle("#Delta p_{T}/p_{T}");
histoPF40_60_2->SetYTitle("Events");
histoPF40_60_2->Draw();

TLegend *leg=new TLegend(0.67,0.75,0.92,0.85);
leg->AddEntry(histoPF40_60_2, "Particle-Flow Jets", "lf");
leg->SetTextSize(0.03);
leg->Draw();

gPad->SaveAs("PFJet40_60_2.pdf");
gPad->SaveAs("PFJet40_60_2.png");

TCanvas* c4 = new TCanvas();
FormatPad(c4,false);

gPad->SetLogy();
histoPF300_400_2->SetMaximum(500.);
histoPF300_400_2->SetMinimum(1);
FormatHisto(histoPF300_400_2,spred);
histoPF300_400_2->SetXTitle("#Delta p_{T}/p_{T}");
histoPF300_400_2->SetYTitle("Events");
histoPF300_400_2->Draw();

TLegend *leg=new TLegend(0.67,0.75,0.92,0.85);
leg->AddEntry(histoPF300_400_2, "Particle-Flow Jets", "lf");
leg->SetTextSize(0.03);
leg->Draw();

gPad->SaveAs("PFJet300_400_2.pdf");
gPad->SaveAs("PFJet300_400_2.png");

/*
TCanvas* c2 = new TCanvas();
FormatPad(c2,false);
//gPad->SetLogy();


//histoPF300_400->SetLineColor(1);
//histoPF300_400->SetLineWidth(2);  
//histoPF300_400->SetFillStyle(3002);
//histoPF300_400->SetFillColor(2);
FormatHisto(histoPF300_400,spred);
histoPF300_400->SetMaximum(500);
histoPF300_400->SetXTitle("#Delta p_{T}/p_{T}");
histoPF300_400->SetYTitle("Events");
histoPF300_400->Draw();
//histoCALO300_400->SetLineColor(1);
//histoCALO300_400->SetLineWidth(2);
//histoCALO300_400->SetFillStyle(3004);
//histoCALO300_400->SetFillColor(4);
FormatHisto(histoCALO300_400,spblue);
histoCALO300_400->Draw("same");

TLegend *leg=new TLegend(0.65,0.65,0.90,0.85);
leg->AddEntry(histoCALO300_400, "Calo-Jets", "lf");
leg->AddEntry(histoPF300_400, "Particle-Flow Jets", "lf");
leg->SetTextSize(0.03);
leg->Draw();

gPad->SaveAs("Jet300_400.pdf");
gPad->SaveAs("Jet300_400.png");
*/
 
}
