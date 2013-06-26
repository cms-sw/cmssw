
{ 

#include <vector>

gROOT->LoadMacro("../Tools/NicePlot.C");
InitNicePlot();


TFile* fileCalo = new TFile("JetBenchmarkGeneric.root");
TH2F* etaCALO2 = (TH2F*) fileCalo->Get("DQMData/PFTask/Benchmarks/iterativeCone5CaloJets/Gen/DeltaEtavsEt");
TH2F* phiCALO2 = (TH2F*) fileCalo->Get("DQMData/PFTask/Benchmarks/iterativeCone5CaloJets/Gen/DeltaPhivsEt");
 etaCALO2->Draw();

vector<TH1F*> etaCALO;
vector<TH1F*> phiCALO;

etaCALO.push_back( (TH1F*)(etaCALO2->ProjectionY("",7,10)->Clone()));
etaCALO.push_back( (TH1F*)(etaCALO2->ProjectionY("",10,20)->Clone()));
etaCALO.push_back( (TH1F*)(etaCALO2->ProjectionY("",20,30)->Clone()));
etaCALO.push_back( (TH1F*)(etaCALO2->ProjectionY("",30,40)->Clone()));
etaCALO.push_back( (TH1F*)(etaCALO2->ProjectionY("",40,50)->Clone()));
etaCALO.push_back( (TH1F*)(etaCALO2->ProjectionY("",50,75)->Clone()));
etaCALO.push_back( (TH1F*)(etaCALO2->ProjectionY("",75,100)->Clone()));
etaCALO.push_back( (TH1F*)(etaCALO2->ProjectionY("",100,125)->Clone()));
etaCALO.push_back( (TH1F*)(etaCALO2->ProjectionY("",125,150)->Clone()));
etaCALO.push_back( (TH1F*)(etaCALO2->ProjectionY("",150,200)->Clone()));
etaCALO.push_back( (TH1F*)(etaCALO2->ProjectionY("",200,250)->Clone()));

phiCALO.push_back( (TH1F*)(phiCALO2->ProjectionY("",7,10)->Clone()));
phiCALO.push_back( (TH1F*)(phiCALO2->ProjectionY("",10,20)->Clone()));
phiCALO.push_back( (TH1F*)(phiCALO2->ProjectionY("",20,30)->Clone()));
phiCALO.push_back( (TH1F*)(phiCALO2->ProjectionY("",30,40)->Clone()));
phiCALO.push_back( (TH1F*)(phiCALO2->ProjectionY("",40,50)->Clone()));
phiCALO.push_back( (TH1F*)(phiCALO2->ProjectionY("",50,75)->Clone()));
phiCALO.push_back( (TH1F*)(phiCALO2->ProjectionY("",75,100)->Clone()));
phiCALO.push_back( (TH1F*)(phiCALO2->ProjectionY("",100,125)->Clone()));
phiCALO.push_back( (TH1F*)(phiCALO2->ProjectionY("",125,150)->Clone()));
phiCALO.push_back( (TH1F*)(phiCALO2->ProjectionY("",150,200)->Clone()));
phiCALO.push_back( (TH1F*)(phiCALO2->ProjectionY("",200,250)->Clone()));

TFile* filePF = new TFile("JetBenchmark_Full_310pre2.root");
TH2F* etaPF2 = (TH2F*) filePF->Get("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen/BDEtavsPt");
TH2F* phiPF2 = (TH2F*) filePF->Get("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen/BDPhivsPt");

gStyle->SetOptStat(0);

vector<TH1F*> etaPF;
vector<TH1F*> phiPF;
vector<Float_t> pts;

etaPF.push_back( (TH1F*)(etaPF2->ProjectionY("",7,10)->Clone()));
etaPF.push_back( (TH1F*)(etaPF2->ProjectionY("",10,20)->Clone()));
etaPF.push_back( (TH1F*)(etaPF2->ProjectionY("",20,30)->Clone()));
etaPF.push_back( (TH1F*)(etaPF2->ProjectionY("",30,40)->Clone()));
etaPF.push_back( (TH1F*)(etaPF2->ProjectionY("",40,50)->Clone()));
etaPF.push_back( (TH1F*)(etaPF2->ProjectionY("",50,75)->Clone()));
etaPF.push_back( (TH1F*)(etaPF2->ProjectionY("",75,100)->Clone()));
etaPF.push_back( (TH1F*)(etaPF2->ProjectionY("",100,125)->Clone()));
etaPF.push_back( (TH1F*)(etaPF2->ProjectionY("",125,150)->Clone()));
etaPF.push_back( (TH1F*)(etaPF2->ProjectionY("",150,200)->Clone()));
etaPF.push_back( (TH1F*)(etaPF2->ProjectionY("",200,250)->Clone()));

phiPF.push_back( (TH1F*)(phiPF2->ProjectionY("",7,10)->Clone()));
phiPF.push_back( (TH1F*)(phiPF2->ProjectionY("",10,20)->Clone()));
phiPF.push_back( (TH1F*)(phiPF2->ProjectionY("",20,30)->Clone()));
phiPF.push_back( (TH1F*)(phiPF2->ProjectionY("",30,40)->Clone()));
phiPF.push_back( (TH1F*)(phiPF2->ProjectionY("",40,50)->Clone()));
phiPF.push_back( (TH1F*)(phiPF2->ProjectionY("",50,75)->Clone()));
phiPF.push_back( (TH1F*)(phiPF2->ProjectionY("",75,100)->Clone()));
phiPF.push_back( (TH1F*)(phiPF2->ProjectionY("",100,125)->Clone()));
phiPF.push_back( (TH1F*)(phiPF2->ProjectionY("",125,150)->Clone()));
phiPF.push_back( (TH1F*)(phiPF2->ProjectionY("",150,200)->Clone()));
phiPF.push_back( (TH1F*)(phiPF2->ProjectionY("",200,250)->Clone()));

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
//pts.push_back(625);

vector<Float_t> sigmaEtaPF;
vector<Float_t> sigmaPhiPF;
vector<Float_t> sigmaEtaCALO;
vector<Float_t> sigmaPhiCALO;
vector<Float_t> rmsEtaPF;
vector<Float_t> rmsPhiPF;
vector<Float_t> rmsEtaCALO;
vector<Float_t> rmsPhiCALO;
Int_t n = pts.size();
for( unsigned i=0; i<n; ++i) {

  if ( i == 0 ) etaCALO[i]->Draw();
  rmsEtaPF.push_back(etaPF[i]->GetRMS());    
  rmsPhiPF.push_back(phiPF[i]->GetRMS());    
  rmsEtaCALO.push_back(etaCALO[i]->GetRMS());    
  rmsPhiCALO.push_back(phiCALO[i]->GetRMS());    

  etaPF[i]->Fit( "gaus","Q0");
  TF1* gausEtaPF = etaPF[i]->GetFunction( "gaus" );
  phiPF[i]->Fit( "gaus","Q0");
  TF1* gausPhiPF = phiPF[i]->GetFunction( "gaus" );
  etaCALO[i]->Fit( "gaus","Q0");
  TF1* gausEtaCALO = etaCALO[i]->GetFunction( "gaus" );
  phiCALO[i]->Fit( "gaus","Q0");
  TF1* gausPhiCALO = phiCALO[i]->GetFunction( "gaus" );
  sigmaEtaPF.push_back(gausEtaPF->GetParameter(2));
  sigmaPhiPF.push_back(gausPhiPF->GetParameter(2));  
  sigmaEtaCALO.push_back(gausEtaCALO->GetParameter(2));
  sigmaPhiCALO.push_back(gausPhiCALO->GetParameter(2));

  cout << i << " " << etaPF[i]->GetRMS() << " " << rmsEtaPF.back() << endl;
  
}

TGraph* grPF1 = new TGraph ( n, &pts[0], &sigmaEtaPF[0] );
TGraph* grPF2 = new TGraph ( n, &pts[0], &rmsEtaPF[0] );
TGraph* grPF3 = new TGraph ( n, &pts[0], &sigmaPhiPF[0] );
TGraph* grPF4 = new TGraph ( n, &pts[0], &rmsPhiPF[0] );

TGraph* grCALO1 = new TGraph ( n, &pts[0], &sigmaEtaCALO[0] );
TGraph* grCALO2 = new TGraph ( n, &pts[0], &rmsEtaCALO[0] );
TGraph* grCALO3 = new TGraph ( n, &pts[0], &sigmaPhiCALO[0] );
TGraph* grCALO4 = new TGraph ( n, &pts[0], &rmsPhiCALO[0] );

TCanvas* c1 = new TCanvas();
FormatPad(c1,false);
c1->cd();

TH2F *h = new TH2F("h","", 100, 15., 500, 10, 0.0, 0.16 );
FormatHisto(h,sback);
h->SetTitle( "CMS Preliminary" );
h->SetXTitle( "p_{T} (GeV/c)" );
h->SetYTitle( "#eta Resolution");
h->SetStats(0);
h->Draw();
 
gPad->SetLogx();
gPad->SetGridx();
gPad->SetGridy();

TF1* pf2 = new TF1("pf2","[0]+[1]*exp(-x/[2])",15,700);
TF1* calo2 = new TF1("calo2","[0]+[1]*exp(-x/[2])",15,700);
pf2->SetParameters(0.01,0.03,10);
calo2->SetParameters(0.02,0.1,50);
pf2->SetLineColor(2);
calo2->SetLineColor(4);
grPF2->Fit("pf2","","",15,700);
grCALO2->Fit("calo2","","",15,700);

grPF2->SetMarkerStyle(22);						
grPF2->SetMarkerColor(2);						
grPF2->SetLineColor(2);						  
grPF2->SetMarkerSize(1.2);						
grPF2->SetLineWidth(3);						  
grPF2->SetLineStyle(1);
grPF2->Draw("P");
//grPF1->Draw("C*");

grCALO2->SetMarkerStyle(25);						
grCALO2->SetMarkerColor(4);						
grCALO2->SetMarkerSize(1.2);						
grCALO2->SetLineColor(4);						  
grCALO2->SetLineWidth(3);						  
grCALO2->Draw("P");
//grCALO1->Draw("C*");

TLegend *leg=new TLegend(0.60,0.65,0.85,0.85);
leg->AddEntry(grCALO2, "Calo-Jets", "p");
leg->AddEntry(grPF2, "Particle-Flow Jets", "p");
leg->SetTextSize(0.03);
leg->Draw();

TLatex text;
text.SetTextColor(1);
text.SetTextSize(0.03);
text.DrawLatex(150,0.095,"0 < |#eta| < 1.5");

gPad->SaveAs("EtaResolution.png");
gPad->SaveAs("EtaResolution.pdf");



TCanvas* c2 = new TCanvas();
FormatPad(c2,false);
c2->cd();

TH2F *h2 = new TH2F("h2","", 100, 15., 500, 10, 0.0, 0.16 );
FormatHisto(h2,sback);
h2->SetTitle( "CMS Preliminary" );
h2->SetXTitle( "p_{T} (GeV/c)" );
h2->SetYTitle( "#phi Resolution");
h2->SetStats(0);
h2->Draw();
 
gPad->SetLogx();
gPad->SetGridx();
gPad->SetGridy();

TF1* pf4 = new TF1("pf4","[0]+[1]*exp(-x/[2])",15,700);
TF1* calo4 = new TF1("calo4","[0]+[1]*exp(-x/[2])",15,700);
pf4->SetParameters(0.01,0.03,10);
calo4->SetParameters(0.02,0.1,50);
pf4->SetLineColor(2);
calo4->SetLineColor(4);
grPF4->Fit("pf4","","",15,700);
grCALO4->Fit("calo4","","",15,700);

grPF4->SetMarkerStyle(22);						
grPF4->SetMarkerColor(2);						
grPF4->SetLineColor(2);						  
grPF4->SetMarkerSize(1.2);						
grPF4->SetLineWidth(3);						  
grPF4->SetLineStyle(1);
grPF4->Draw("P");
//grPF3->Draw("C*");


grCALO4->SetMarkerStyle(25);						
grCALO4->SetMarkerColor(4);						
grCALO4->SetMarkerSize(1.2);						
grCALO4->SetLineColor(4);						  
grCALO4->SetLineWidth(3);						  
grCALO4->Draw("P");
//grCALO3->Draw("C*");

TLegend *leg=new TLegend(0.60,0.65,0.85,0.85);
leg->AddEntry(grCALO4, "Calo-Jets", "lp");
leg->AddEntry(grPF4, "Particle-Flow Jets", "lp");
leg->SetTextSize(0.03);
leg->Draw();

text.DrawLatex(150,0.095,"0 < |#eta| < 1.5");
 
gPad->SaveAs("PhiResolution.png");
gPad->SaveAs("PhiResolution.pdf");

}
 
