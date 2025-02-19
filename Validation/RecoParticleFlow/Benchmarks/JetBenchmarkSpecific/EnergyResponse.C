
{ 

#include <vector>

gROOT->LoadMacro("../Tools/NicePlot.C");
InitNicePlot();

TFile* fileCalo = new TFile("JetBenchmarkGeneric_Barrel.root");

gStyle->SetOptStat(0);

vector<TH1F*> histPF;
vector<TH1F*> histCALO;
vector<Float_t> pts;
TH2F* histoCALO = (TH2F*) fileCalo->Get("DQMData/PFTask/Benchmarks/iterativeCone5CaloJets/Gen/DeltaEtOverEtvsEt")->Clone();
histCALO.push_back( (TH1F*)(histoCALO->ProjectionY("",14,20)->Clone()));
histCALO.push_back( (TH1F*)(histoCALO->ProjectionY("",20,40)->Clone()));
histCALO.push_back( (TH1F*)(histoCALO->ProjectionY("",40,60)->Clone()));
histCALO.push_back( (TH1F*)(histoCALO->ProjectionY("",60,80)->Clone()));
histCALO.push_back( (TH1F*)(histoCALO->ProjectionY("",80,100)->Clone()));
histCALO.push_back( (TH1F*)(histoCALO->ProjectionY("",100,150)->Clone()));
histCALO.push_back( (TH1F*)(histoCALO->ProjectionY("",150,200)->Clone()));
histCALO.push_back( (TH1F*)(histoCALO->ProjectionY("",200,250)->Clone()));
histCALO.push_back( (TH1F*)(histoCALO->ProjectionY("",250,300)->Clone()));
histCALO.push_back( (TH1F*)(histoCALO->ProjectionY("",300,400)->Clone()));
histCALO.push_back( (TH1F*)(histoCALO->ProjectionY("",400,500)->Clone()));
histCALO.push_back( (TH1F*)(histoCALO->ProjectionY("",500,750)->Clone()));

TFile* filePF = new TFile("JetBenchmark_Full_310pre2.root");
TH2F* histoPF = (TH2F*) filePF->Get("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen/BRPtvsPt")->Clone();
histPF.push_back( (TH1F*)(histoPF->ProjectionY("",7,10)->Clone()));
histPF[0].Draw();
histPF.push_back( (TH1F*)(filePF->Get("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen/BRPt20_40")->Clone()) );
histPF.push_back( (TH1F*)(filePF->Get("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen/BRPt40_60")->Clone()) );
histPF.push_back( (TH1F*)(filePF->Get("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen/BRPt60_80")->Clone()) );
histPF.push_back( (TH1F*)(filePF->Get("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen/BRPt80_100")->Clone()) );
histPF.push_back( (TH1F*)(filePF->Get("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen/BRPt100_150")->Clone()) );
histPF.push_back( (TH1F*)(filePF->Get("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen/BRPt150_200")->Clone()) );
histPF.push_back( (TH1F*)(filePF->Get("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen/BRPt200_250")->Clone()) );
histPF.push_back( (TH1F*)(filePF->Get("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen/BRPt250_300")->Clone()) );
histPF.push_back( (TH1F*)(filePF->Get("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen/BRPt300_400")->Clone()) );
histPF.push_back( (TH1F*)(filePF->Get("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen/BRPt400_500")->Clone()) ); 
histPF.push_back( (TH1F*)(filePF->Get("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen/BRPt500_750")->Clone()) );
pts.push_back(17);
pts.push_back(30);
pts.push_back(50);
pts.push_back(70);
pts.push_back(90);
pts.push_back(125);
pts.push_back(175);
pts.push_back(225);
pts.push_back(275);
pts.push_back(350);
pts.push_back(450);
pts.push_back(625);

vector<Float_t> sigmaPF;
vector<Float_t> rmsPF;
vector<Float_t> meanPF;
vector<Float_t> arithPF;
vector<Float_t> sigmaCALO;
vector<Float_t> rmsCALO;
vector<Float_t> meanCALO;
vector<Float_t> arithCALO;

Int_t n = pts.size();
for( unsigned i=0; i<n; ++i) {

  histPF[i]->Fit( "gaus","Q0");
  TF1* gausPF = histPF[i]->GetFunction( "gaus" );
  histCALO[i]->Fit( "gaus","Q0");
  TF1* gausCALO = histCALO[i]->GetFunction( "gaus" );
  
  sigmaPF.push_back(gausPF->GetParameter(2)/(gausPF->GetParameter(1)+1));
  meanPF.push_back(gausPF->GetParameter(1));
  rmsPF.push_back(histPF[i]->GetRMS()/(histPF[i]->GetMean()+1));    
  arithPF.push_back(histPF[i]->GetMean());
  
  sigmaCALO.push_back(gausCALO->GetParameter(2)/(gausCALO->GetParameter(1)+1));
  meanCALO.push_back(gausCALO->GetParameter(1));
  rmsCALO.push_back(histCALO[i]->GetRMS()/(histCALO[i]->GetMean()+1.));    
  arithCALO.push_back(histCALO[i]->GetMean());

  cout << pts[i] << " " << gausPF->GetParameter(1)+1 << " " << gausPF->GetParameter(2)
       << " " << histPF[i]->GetMean()+1 << " " <<  histPF[i]->GetRMS() << endl;
}

TGraph* grPF1 = new TGraph ( n, &pts[0], &sigmaPF[0] );
TGraph* grPF2 = new TGraph ( n, &pts[0], &rmsPF[0] );
TGraph* grPF3 = new TGraph ( n, &pts[0], &meanPF[0] );
TGraph* grPF4 = new TGraph ( n, &pts[0], &arithPF[0] );

TGraph* grCALO1 = new TGraph ( n, &pts[0], &sigmaCALO[0] );
TGraph* grCALO2 = new TGraph ( n, &pts[0], &rmsCALO[0] );
TGraph* grCALO3 = new TGraph ( n, &pts[0], &meanCALO[0] );
TGraph* grCALO4 = new TGraph ( n, &pts[0], &arithCALO[0] );

TH2F *h = new TH2F("h","", 10, 15., 620, 10, -1.0, 0.2 );
FormatHisto(h,sback);

TCanvas* c1 = new TCanvas();
FormatPad(c1,false);
c1->cd();

h->SetTitle( "CMS Preliminary" );
h->SetXTitle( "p_{T} (GeV/c)" );
h->SetYTitle( "Jet Response");
h->SetStats(0);
h->Draw();
 
gPad->SetGridx();
gPad->SetGridy();

TF1* pf4 = new TF1("pf4","[0]+[1]*exp(-x/[2])",15,700);
TF1* calo4 = new TF1("calo4","[0]+[1]*exp(-x/[2])",15,700);
pf4->SetParameters(0.,-0.1,10);
calo4->SetParameters(0.,-0.5,50);
pf4->SetLineColor(2);
calo4->SetLineColor(4);
grPF4->Fit("pf4","","",15,700);
grCALO4->Fit("calo4","","",15,700);

grPF4->SetMarkerStyle(22);						
grPF4->SetMarkerColor(2);						
grPF4->SetLineColor(2);						  
grPF4->SetMarkerSize(1.5);						
grPF4->SetLineWidth(3);						  
grPF4->SetLineStyle(1);
grPF4->Draw("P");


grCALO4->SetMarkerStyle(25);						
grCALO4->SetMarkerColor(4);						
grCALO4->SetMarkerSize(1.5);						
grCALO4->SetLineColor(4);						  
grCALO4->SetLineWidth(3);						  
grCALO4->Draw("P");

TLegend *leg=new TLegend(0.60,0.25,0.85,0.45);
leg->AddEntry(grPF4, "Particle-Flow Jets", "lp");
leg->AddEntry(grCALO4, "Calo-Jets", "lp");
leg->SetTextSize(0.03);
leg->Draw();

TLatex text;
text.SetTextColor(1);
text.SetTextSize(0.03);
//text.DrawLatex(420,-0.92,"Barrel");
text.DrawLatex(410,-0.92,"0 < |#eta| < 1.5");
 
gPad->SaveAs("JetResponse.pdf");
gPad->SaveAs("JetResponse.png");


TCanvas* c2 = new TCanvas();
FormatPad(c2,false);
c2->cd();

TFile* fileCalo2 = new TFile("JetBenchmarkGeneric_AllEta.root");
TH2F* histoCALOEta = (TH2F*) fileCalo2.Get("DQMData/PFTask/Benchmarks/iterativeCone5CaloJets/Gen/DeltaEtOverEtvsEta");
TH2F* histoPFEta = (TH2F*) filePF.Get("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen/RPtvsEta");

TH1F* pfEta = (TH1F*) (histoPFEta->ProfileX()->Clone());
TH1F* caloEta = (TH1F*) (histoCALOEta->ProfileX()->Clone());
FormatHisto(pfEta,sback);
 
gPad->SetGridx();
gPad->SetGridy();
  
pfEta->SetTitle( "CMS Preliminary" );
pfEta->SetYTitle( "Jet Response");
pfEta->SetMaximum(0.3);
pfEta->SetMinimum(-0.5);
pfEta->Rebin(2);
pfEta->GetXaxis().SetRangeUser(-2.8,2.8);

pfEta->SetMarkerStyle(22);						
pfEta->SetMarkerColor(2);						
pfEta->SetLineColor(2);						  
pfEta->SetMarkerSize(1.5);						
pfEta->SetLineWidth(3);						  
pfEta->SetLineStyle(1);
pfEta->Draw();

caloEta->Rebin(2);
caloEta->GetXaxis().SetRangeUser(-2.8,2.8);
caloEta->SetMarkerStyle(25);						
caloEta->SetMarkerColor(4);						
caloEta->SetMarkerSize(1.5);						
caloEta->SetLineColor(4);						  
caloEta->SetLineWidth(3);						  
caloEta->Draw("same");

TLegend *leg=new TLegend(0.60,0.65,0.85,0.85);
leg->AddEntry(grPF4, "Particle-Flow Jets", "p");
leg->AddEntry(grCALO4, "Calo-Jets", "p");
leg->SetTextSize(0.03);
leg->Draw();

gPad->SaveAs("JetResponseEta.pdf");
gPad->SaveAs("JetResponseEta.png");

}
