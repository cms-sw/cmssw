
{ 

#include <vector>

gROOT->LoadMacro("../Tools/NicePlot.C");
InitNicePlot();

TFile fileCalo("JetBenchmarkGeneric_Endcap.root");
TFile filePF("JetBenchmark_Full_310pre2.root");

gStyle->SetOptStat(0);

vector<TH1F*> histPF;
vector<TH1F*> histCALO;
vector<Float_t> pts;
TH2F* histoCALO = (TH2F*) fileCalo.Get("DQMData/PFTask/Benchmarks/iterativeCone5CaloJets/Gen/DeltaEtOverEtvsEt");
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
histPF.push_back( (TH1F*)(filePF.Get("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen/ERPt20_40")->Clone()) );
histPF.push_back( (TH1F*)(filePF.Get("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen/ERPt40_60")->Clone()) );
histPF.push_back( (TH1F*)(filePF.Get("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen/ERPt60_80")->Clone()) );
histPF.push_back( (TH1F*)(filePF.Get("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen/ERPt80_100")->Clone()) );
histPF.push_back( (TH1F*)(filePF.Get("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen/ERPt100_150")->Clone()) );
histPF.push_back( (TH1F*)(filePF.Get("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen/ERPt150_200")->Clone()) );
histPF.push_back( (TH1F*)(filePF.Get("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen/ERPt200_250")->Clone()) );
histPF.push_back( (TH1F*)(filePF.Get("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen/ERPt250_300")->Clone()) );
histPF.push_back( (TH1F*)(filePF.Get("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen/ERPt300_400")->Clone()) );
histPF.push_back( (TH1F*)(filePF.Get("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen/ERPt400_500")->Clone()) ); 
histPF.push_back( (TH1F*)(filePF.Get("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen/ERPt500_750")->Clone()) );
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
}

TGraph* grPF1 = new TGraph ( n, &pts[0], &sigmaPF[0] );
TGraph* grPF2 = new TGraph ( n, &pts[0], &rmsPF[0] );
TGraph* grPF3 = new TGraph ( n, &pts[0], &meanPF[0] );
TGraph* grPF4 = new TGraph ( n, &pts[0], &arithPF[0] );

TGraph* grCALO1 = new TGraph ( n, &pts[0], &sigmaCALO[0] );
TGraph* grCALO2 = new TGraph ( n, &pts[0], &rmsCALO[0] );
TGraph* grCALO3 = new TGraph ( n, &pts[0], &meanCALO[0] );
TGraph* grCALO4 = new TGraph ( n, &pts[0], &arithCALO[0] );

TH2F *h = new TH2F("h","", 10, 0., 700, 10, -1.0, 0.2 );

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

grPF4->SetMarkerStyle(22);						
grPF4->SetMarkerColor(2);						
grPF4->SetLineColor(2);						  
grPF4->SetMarkerSize(1.5);						
grPF4->SetLineWidth(3);						  
grPF4->SetLineStyle(1);
grPF4->Draw("CP");


grCALO4->SetMarkerStyle(25);						
grCALO4->SetMarkerColor(4);						
grCALO4->SetMarkerSize(1.5);						
grCALO4->SetLineColor(4);						  
grCALO4->SetLineWidth(3);						  
grCALO4->Draw("CP");

TLegend *leg=new TLegend(0.60,0.25,0.85,0.45);
leg->AddEntry(grCALO4, "Calo-Jets", "lp");
leg->AddEntry(grPF4, "Particle-Flow Jets", "lp");
leg->SetTextSize(0.03);
leg->Draw();

gPad->SaveAs("JetResponse_Endcap.pdf");
gPad->SaveAs("JetResponse_Endcap.png");


}
