{
  //#include <tdrstyle.C>
//author J.Weng
//This macro can be used to reproduce the tau benchmark plot
//for tau jet reconstruction studies from the validation file 
//created with PFBenchmarkAnalyzer

gROOT->Reset();
TFile *fCalo = new TFile("TauBenchmark.root","READ");
TFile *fPF = new TFile("TauBenchmark.root","READ");
//setTDRStyle();

TH1F* h_deltaETvisible_MCPF  = fPF->Get("DQMData/PFTask/Benchmarks/ParticleFlow/Gen/DeltaEt");
TH1F* h_deltaETvisible_MCEHT = fCalo->Get("DQMData/PFTask/Benchmarks/Calo/Gen/DeltaEt");
h_deltaETvisible_MCPF->SetStats(00000000); 
h_deltaETvisible_MCPF->GetYaxis()->SetTitle("#");
h_deltaETvisible_MCPF->SetTitle("#");
// h_deltaETvisible_MCPF->GetXaxis()->SetTitleOffset(1.3);
// h_deltaETvisible_MCPF->GetXaxis()->SetTitleOffset(0.5);
// h_deltaETvisible_MCPF->GetYaxis()->SetTitleOffset(1.3);
// h_deltaETvisible_MCPF->GetYaxis()->SetTitleOffset(0.5);
h_deltaETvisible_MCPF->SetTitle("");
h_deltaETvisible_MCPF->SetLineColor(2);
h_deltaETvisible_MCPF->Draw();
h_deltaETvisible_MCEHT->Draw("same");

Double_t x_1=0.152; Double_t y_1 = 0.51;
Double_t x_2=0.483; Double_t y_2 = 0.646;

TLegend *leg = new TLegend(x_1,y_1,x_2,y_2,NULL,"brNDC");
leg->SetTextSize(0.031);
leg->SetFillStyle(0);
leg->SetFillColor(0);
leg->SetTextFont(52);
leg->SetTextAlign(32);

leg->AddEntry(h_deltaETvisible_MCPF,"Particle Flow Jets","l");
leg->AddEntry(h_deltaETvisible_MCEHT,"caloTower Jets","l");
leg->Draw();

c1->Print("DeltaEt.eps");
c1->Print("DeltaEt.gif");
c1->Print("DeltaEt.root");
gApplication->Terminate();
}
