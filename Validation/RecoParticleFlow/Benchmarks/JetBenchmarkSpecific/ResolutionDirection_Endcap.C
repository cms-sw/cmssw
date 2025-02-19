
{ 

#include <vector>

  gROOT->LoadMacro(" ../../../Validation/RecoParticleFlow/Benchmarks/Tools/NicePlot.C");
  InitNicePlot();


TFile* filePF = new TFile("pfjetBenchmark.root");
TH2F* etaPF2 = (TH2F*) filePF->Get("EDEtavsPt");
TH2F* phiPF2 = (TH2F*) filePF->Get("EDPhivsPt");

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
etaPF.push_back( (TH1F*)(etaPF2->ProjectionY("",250,350)->Clone()));
etaPF.push_back( (TH1F*)(etaPF2->ProjectionY("",350,500)->Clone()));
etaPF.push_back( (TH1F*)(etaPF2->ProjectionY("",500,1000)->Clone()));
//etaPF.push_back( (TH1F*)(etaPF2->ProjectionY("",1000,2500)->Clone()));

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
phiPF.push_back( (TH1F*)(phiPF2->ProjectionY("",250,350)->Clone()));
phiPF.push_back( (TH1F*)(phiPF2->ProjectionY("",350,500)->Clone()));
phiPF.push_back( (TH1F*)(phiPF2->ProjectionY("",500,1000)->Clone()));
//phiPF.push_back( (TH1F*)(phiPF2->ProjectionY("",1000,2500)->Clone()));

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
pts.push_back(600);
pts.push_back(850);
pts.push_back(1500);
//pts.push_back(3500);

vector<Float_t> sigmaEtaPF;
vector<Float_t> sigmaPhiPF;
vector<Float_t> rmsEtaPF;
vector<Float_t> rmsPhiPF;
Int_t n = pts.size();
 cout << "Nb de points = " << n << endl;
for( unsigned i=0; i<n; ++i) {

  etaPF[i]->Rebin(2);
  phiPF[i]->Rebin(2);
  rmsEtaPF.push_back(etaPF[i]->GetRMS());    
  rmsPhiPF.push_back(phiPF[i]->GetRMS());    
  cout << i << " " << etaPF[i]->GetRMS() << " " << rmsEtaPF.back() << endl;

  if ( etaPF[i]->GetRMS() == 0.0 &&
       phiPF[i]->GetRMS() == 0.0 ) { 
    sigmaEtaPF.push_back(0);
    sigmaPhiPF.push_back(0);  
    continue;
  } else {
    etaPF[i]->Fit( "gaus","Q0");
    TF1* gausEtaPF = etaPF[i]->GetFunction( "gaus" );
    phiPF[i]->Fit( "gaus","Q0");
    TF1* gausPhiPF = phiPF[i]->GetFunction( "gaus" );
    sigmaEtaPF.push_back(gausEtaPF->GetParameter(2));
    sigmaPhiPF.push_back(gausPhiPF->GetParameter(2));  
    cout << i << " " << gausEtaPF->GetParameter(2) << " " << sigmaEtaPF.back() << endl;
  }
}

TGraph* grPF1 = new TGraph ( n, &pts[0], &sigmaEtaPF[0] );
TGraph* grPF2 = new TGraph ( n, &pts[0], &rmsEtaPF[0] );
TGraph* grPF3 = new TGraph ( n, &pts[0], &sigmaPhiPF[0] );
TGraph* grPF4 = new TGraph ( n, &pts[0], &rmsPhiPF[0] );

TCanvas* c1 = new TCanvas();
FormatPad(c1,false);
c1->cd();

TH2F *h = new TH2F("h","", 100, 15., 2000, 10, 0.0, 0.05 );
FormatHisto(h,sback);
h->SetTitle( "CMS Preliminary" );
h->SetXTitle( "p_{T} (GeV/c)" );
h->SetYTitle( "#eta Resolution");
h->SetStats(0);
h->Draw();
 
gPad->SetLogx();
gPad->SetGridx();
gPad->SetGridy();

TF1* pf1 = new TF1("pf1","[0]+[1]*exp(-x/[2])",15,2000);
pf1->SetParameters(0.01,0.03,10);
pf1->SetLineColor(2);
pf1->SetLineStyle(1);
grPF1->Fit("pf1","","",15,2000);

TF1* pf2 = new TF1("pf2","[0]+[1]*exp(-x/[2])",15,2000);
pf2->SetParameters(0.01,0.03,10);
pf2->SetLineColor(4);
pf2->SetLineStyle(2);
grPF2->Fit("pf2","","",15,2000);

grPF2->SetMarkerStyle(25);						
grPF2->SetMarkerColor(4);						
grPF2->SetLineColor(4);						  
grPF2->SetMarkerSize(1.2);						
grPF2->SetLineWidth(3);						  
grPF2->SetLineStyle(2);
grPF2->Draw("P");

grPF1->SetMarkerStyle(22);						
grPF1->SetMarkerColor(2);						
grPF1->SetLineColor(2);						  
grPF1->SetMarkerSize(1.2);						
grPF1->SetLineWidth(3);						  
grPF1->SetLineStyle(1);
grPF1->Draw("P");

TLegend *leg=new TLegend(0.55,0.65,0.90,0.85);
leg->AddEntry(grPF2, "Particle-Flow Jets (RMS)", "p");
leg->AddEntry(grPF1, "Particle-Flow Jets (Gaussian)", "p");
leg->SetTextSize(0.03);
leg->Draw();

TLatex text;
text.SetTextColor(1);
text.SetTextSize(0.03);
text.DrawLatex(150,0.095,"0 < |#eta| < 1.5");

gPad->SaveAs("EtaResolution_Endcap.png");
gPad->SaveAs("EtaResolution_Endcap.pdf");



TCanvas* c2 = new TCanvas();
FormatPad(c2,false);
c2->cd();

TH2F *h2 = new TH2F("h2","", 100, 15., 2000, 10, 0.0, 0.05 );
FormatHisto(h2,sback);
h2->SetTitle( "CMS Preliminary" );
h2->SetXTitle( "p_{T} (GeV/c)" );
h2->SetYTitle( "#phi Resolution");
h2->SetStats(0);
h2->Draw();
 
gPad->SetLogx();
gPad->SetGridx();
gPad->SetGridy();

TF1* pf3 = new TF1("pf3","[0]+[1]*exp(-x/[2])",15,2000);
pf3->SetParameters(0.01,0.03,10);
pf3->SetLineColor(2);
pf3->SetLineStyle(1);
grPF3->Fit("pf3","","",15,2000);

TF1* pf4 = new TF1("pf4","[0]+[1]*exp(-x/[2])",15,2000);
pf4->SetParameters(0.01,0.03,10);
pf4->SetLineColor(4);
pf4->SetLineStyle(2);
grPF4->Fit("pf4","","",15,2000);

grPF4->SetMarkerStyle(25);						
grPF4->SetMarkerColor(4);						
grPF4->SetLineColor(4);						  
grPF4->SetMarkerSize(1.2);						
grPF4->SetLineWidth(3);						  
grPF4->SetLineStyle(2);
grPF4->Draw("P");

grPF3->SetMarkerStyle(22);						
grPF3->SetMarkerColor(2);						
grPF3->SetLineColor(2);						  
grPF3->SetMarkerSize(1.2);						
grPF3->SetLineWidth(3);						  
grPF3->SetLineStyle(1);
grPF3->Draw("P");

TLegend *leg=new TLegend(0.55,0.65,0.90,0.85);
leg->AddEntry(grPF4, "Particle-Flow Jets (RMS)", "lp");
leg->AddEntry(grPF3, "Particle-Flow Jets (Gaussian)", "lp");
leg->SetTextSize(0.03);
leg->Draw();

text.DrawLatex(150,0.095,"0 < |#eta| < 1.5");
 
gPad->SaveAs("PhiResolution_Endcap.png");
gPad->SaveAs("PhiResolution_Endcap.pdf");

}
