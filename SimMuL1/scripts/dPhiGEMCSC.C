TCut ok_lct1 = "(has_lct&1) > 0";
TCut ok_lct2 = "(has_lct&2) > 0";


TCut ok_pad1 = "(has_gem_pad&1) > 0";
TCut ok_pad2 = "(has_gem_pad&2) > 0";



TFile* gf = 0;
TTree* gt = 0;
TString gf_name = "";


TTree* getTree(TString f_name)
{
if (gf_name == f_name && gt != 0) return gt;
cout<<f_name<<endl;
gf_name = f_name;
gf = TFile::Open(f_name);
gt = (TTree*) gf->Get("GEMCSCAnalyzer/trk_eff");
return gt;
}


void drawplot_dphi()
{



  txtHeader = new TLegend(.13,.935,.97,1.);
  txtHeader->SetFillColor(kWhite);
  txtHeader->SetFillStyle(0);
  txtHeader->SetBorderSize(0);
  txtHeader->SetTextFont(42);
  txtHeader->SetTextSize(0.045);
  txtHeader->SetTextAlign(22);
  txtHeader->SetHeader("|#phi(CSC strip)-#phi(GEM Pad4)| in odd numbered chambers");


  legend = new TLegend(.4,.60,1.2,.92);
  legend->SetFillColor(kWhite);
  legend->SetFillStyle(0);
  legend->SetBorderSize(0);
  legend->SetTextSize(0.045);
  legend->SetMargin(0.13);


  t = getTree("gem_alfredo/gem_csc_delta_pt5_pad4.root");
  t1 = getTree("gem_alfredo/gem_csc_delta_pt20_pad4.root");


  TH1F *dphi_odd_pt5 = new TH1F("dphi_odd_pt5","",600,0.0,0.03);
  TH1F *dphi_odd_pt20 = new TH1F("dphi_odd_pt20","",600,0.0,0.03);

  TCanvas* cDphi_odd = new TCanvas("cDphi_odd","cDphi_odd",700,450);
  
  t->Draw("TMath::Abs(dphi_pad_odd)>>dphi_odd_pt5" , ok_pad1 && ok_lct1);
  t1->Draw("TMath::Abs(dphi_pad_odd)>>dphi_odd_pt20" , ok_pad1 && ok_lct1);

  dphi_odd_pt5->Scale(1/dphi_odd_pt5->Integral());
  dphi_odd_pt20->Scale(1/dphi_odd_pt20->Integral());


  dphi_odd_pt5->SetLineColor(kRed);
  dphi_odd_pt20->SetLineColor(kBlue);
  dphi_odd_pt5->SetLineWidth(2);
  dphi_odd_pt20->SetLineWidth(2);

  dphi_odd_pt20->GetXaxis()->SetTitle("|#phi(CSC half-strip) - #phi(GEM pad)| [rad]");
  dphi_odd_pt20->GetYaxis()->SetTitle("Scaled to unity");


  legend->AddEntry(dphi_odd_pt5,"Muons with p_{T}=5 GeV","L");
  legend->AddEntry(dphi_odd_pt20,"Muons with p_{T}=20 GeV","L");

  dphi_odd_pt20->Draw();
  dphi_odd_pt5->Draw("same");
  txtHeader->Draw("same");
  legend->Draw("same");
  cDphi_odd->SaveAs("Dphi_odd_chambers.png","recrate");


  TH1F *dphi_even_pt5 = new TH1F("dphi_even_pt5","",600,0.0,0.03);
  TH1F *dphi_even_pt20 = new TH1F("dphi_even_pt20","",600,0.0,0.03);

  TCanvas* cDphi_even = new TCanvas("cDphi_even","cDphi_even",700,450);
  
  t->Draw("TMath::Abs(dphi_pad_even)>>dphi_even_pt5" , ok_pad2 && ok_lct2);
  t1->Draw("TMath::Abs(dphi_pad_even)>>dphi_even_pt20" , ok_pad2 && ok_lct2);

  dphi_even_pt5->Scale(1/dphi_even_pt5->Integral());
  dphi_even_pt20->Scale(1/dphi_even_pt20->Integral());

  dphi_even_pt5->SetLineColor(kRed);
  dphi_even_pt20->SetLineColor(kBlue);
  dphi_even_pt5->SetLineWidth(2);
  dphi_even_pt20->SetLineWidth(2);

  dphi_even_pt20->GetXaxis()->SetTitle("|#phi(CSC half-strip) - #phi(GEM pad)| [rad]");
  dphi_even_pt20->GetYaxis()->SetTitle("Scaled to unity");


  txtHeader->SetHeader("|#phi(CSC strip)-#phi(GEM Pad4)| in even numbered chambers");


  dphi_even_pt20->Draw();
  dphi_even_pt5->Draw("same");
  txtHeader->Draw("same");
  legend->Draw("same");
  cDphi_even->SaveAs("Dphi_even_chambers.png","recrate");

  
  


  


}
