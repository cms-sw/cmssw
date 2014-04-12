import re, sys, os

file = open("output", "r")

lines = file.readlines()
file.close()

variable = []
eta1 = []
eta2 = []
mean = []
error= []
effS = []

for line in lines:
    elements = re.split("\t", line)
    variable += [elements[1],]
    
    eta1 += [re.split(">", re.split("&&", elements[2])[0])[1],]
    eta2 += [re.split("<", elements[2])[1],]

    mean += [elements[3],]
    error+= [elements[4],]
    effS += [elements[5][:-1],]

header = """void plot_EffSigmaVsEta(){
   TCanvas *c1        = new TCanvas("c1","EffSigma vs Eta", 800, 1000);
   TH1F* h_emCorr_et  = new TH1F("h_emCorr_et","",150,0,1.5);
   TH1F* h_em_et      = new TH1F("h_em_et","",150,0,1.5);
   TH1F* add          = new TH1F("add","",150,0,1.5);
   TH1F* ratio        = new TH1F("ratio","",150,0,1.5);

   c1->Divide(1,2);
   c1->cd(1);
"""
file = open("plot_EffSigmaVsEta.C", "w")
file.write(header)

for i in ("emCorr_et", "em_et"):
    for j in range(0, len(eta1)):
        if variable[j] <> i:
            continue
        bin = str(int((float(eta1[j]) + float(eta2[j]))*50))
        file.write("  h_" + i + "->SetBinContent(" + bin + ", " + effS[j] + ");\n")
        file.write("  h_" + i + "->SetBinError  (" + bin + ", 1e-10);\n")

file.write("  h_emCorr_et->SetMarkerStyle(23);\n")
file.write("  h_em_et    ->SetMarkerStyle(20);\n")

file.write("  h_emCorr_et->SetMarkerColor(4);\n")
file.write("  h_em_et    ->SetMarkerColor(1);\n")

file.write("  gStyle->SetOptStat(0);\n")

file.write("  h_em_et    ->Draw();\n")
file.write("  h_emCorr_et    ->Draw(\"SAME\");\n")

header ="""  

  TAxis* ax = h_em_et->GetXaxis();
  ax->SetTitle("#eta");
  TAxis* ay = h_em_et->GetYaxis();
  ay->SetTitle("#sigma_{EFF}");
  
  TLegend *leg = new TLegend(0.2, 0.2, 0.4, 0.4);
  leg->AddEntry(h_em_et, "Before correction");
  leg->AddEntry(h_emCorr_et, "After correction  ");
  leg->Draw("");

  c1->cd(2);
  add->Add(h_emCorr_et, h_em_et, -1, 1);
  ratio->Divide(add, h_em_et);
  
  ratio->SetMarkerStyle(21);
  ratio->SetMarkerColor(4);
  ratio->GetYaxis()->SetTitle("Improvement in #sigma_{eff}");
  ratio->GetXaxis()->SetTitle("#eta");
  ratio->Draw();
  
  c1->Print("EffSigmaVsEta.ps");
  gROOT->ProcessLine(".q");
"""
file.write(header)

file.write("}\n")
