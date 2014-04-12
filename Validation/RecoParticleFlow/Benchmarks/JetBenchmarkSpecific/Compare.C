#include <vector>

void PlotHisto(char* hname, TFile* f1, TFile* f2, unsigned algo1=0, unsigned algo2=0)
{

  if ( algo1 == 1 ) 
    f1->cd("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen"); 
  else if ( algo1 == 0 ) 
    f1->cd("DQMData/PFTask/Benchmarks/ak5PFJets/Gen"); 
  TH1F* h1 = (TH1F*) gDirectory->Get(hname);
  if ( algo2 == 1 ) 
    f2->cd("DQMData/PFTask/Benchmarks/iterativeCone5PFJets/Gen"); 
  else if ( algo2 == 0 ) 
    f2->cd("DQMData/PFTask/Benchmarks/ak5PFJets/Gen"); 
  TH1F* h2 = (TH1F*) gDirectory->Get(hname);
  h1->SetTitle(hname);
  h1->SetStats(0);
  h1->SetMinimum(0);
  
  if ( h1->GetEntries() > 0 && h2->GetEntries() > 0 ) h1->Scale( h2->GetEntries()/h1->GetEntries() );
  
  h1->SetMarkerStyle(22);
  h1->SetMarkerSize(0.6);
  h1->SetMarkerColor(2);
  h1->Draw("error");
  h2->SetLineColor(4);
  h2->SetLineWidth(1.5);
  h2->Draw("same");
}

void Compare(unsigned barrel, 
	     const char* fastInput, 
	     const char* fullInput,
	     const char* output,
	     const char* title,
	     unsigned algo1=0,
	     unsigned algo2=0)
{
  gROOT->Reset();
  TFile *f1 = new TFile(fastInput);
  TFile *f2 = new TFile(fullInput);
  vector< string > hists;
  if ( barrel == 1) {
    hists.push_back( "BRPt20_40") ;
    hists.push_back( "BRPt40_60");
    hists.push_back( "BRPt60_80");
    hists.push_back( "BRPt80_100");
    hists.push_back( "BRPt100_150");
    hists.push_back( "BRPt150_200");
    hists.push_back( "BRPt200_250");
    hists.push_back( "BRPt250_300");
    hists.push_back( "BRPt300_400");
    hists.push_back( "BRPt400_500");
    hists.push_back( "BRPt500_750");
  } else if ( barrel == 2 ) { 
    hists.push_back( "ERPt20_40") ;
    hists.push_back( "ERPt40_60");
    hists.push_back( "ERPt60_80");
    hists.push_back( "ERPt80_100");
    hists.push_back( "ERPt100_150");
    hists.push_back( "ERPt150_200");
    hists.push_back( "ERPt200_250");
    hists.push_back( "ERPt250_300");
    hists.push_back( "ERPt300_400");
    hists.push_back( "ERPt400_500");
    hists.push_back( "ERPt500_750");
  } else if ( barrel == 3 ) { 
    hists.push_back( "FRPt20_40") ;
    hists.push_back( "FRPt40_60");
    hists.push_back( "FRPt60_80");
    hists.push_back( "FRPt80_100");
    hists.push_back( "FRPt100_150");
    hists.push_back( "FRPt150_200");
    hists.push_back( "FRPt200_250");
    hists.push_back( "FRPt250_300");
    hists.push_back( "FRPt300_400");
    hists.push_back( "FRPt400_500");
    hists.push_back( "FRPt500_750");
  }

  /* */
  TCanvas *c = new TCanvas("c","",1000, 600);
  c->Divide(3,4);
  for( unsigned i=0; i<hists.size(); ++i) {
    c->cd(i+1);
    PlotHisto( hists[i].c_str(), f1, f2, algo1, algo2 );
  }
  /* */

  c->cd(12);
  TH1F* dummy = new TH1F();
  dummy->SetTitle(title);
  dummy->SetStats(0);
  dummy->Draw();
  TLegend *leg=new TLegend(0.1,0.6,0.9,0.9);
  leg->AddEntry( dummy, title,"lp");
  leg->SetTextSize(0.1);
  leg->Draw();
  

  c->SaveAs(output);

  /*
  TCanvas *c1 = new TCanvas("c1","",1000,600);
  c1->Divide(2,2);
  string hist1 = "BRneut";
  c1->cd(1);
  PlotHisto( hist1.c_str(), f1, f2 );
  string hist2 = "BRCHE";
  c1->cd(2);
  PlotHisto( hist2.c_str(), f1, f2 );
  string hist3 = "ERneut";
  c1->cd(3);
  PlotHisto( hist3.c_str(), f1, f2 );
  string hist4 = "ERCHE";
  c1->cd(4);
  PlotHisto( hist4.c_str(), f1, f2 );
  */
}
