//gROOT->ProcessLine(".L effFunctions.C");


gROOT->SetBatch(1);


TString filesDir = "files/";
TString plotDir = "plots/efficiency/";
TString ext = ".pdf";

TCut ok_sh1 = "(has_csc_sh&1) > 0";
TCut ok_sh2 = "(has_csc_sh&2) > 0";
TCut ok_st1 = "(has_csc_strips&1) > 0";
TCut ok_st2 = "(has_csc_strips&2) > 0";
TCut ok_w1 = "(has_csc_wires&1) > 0";
TCut ok_w2 = "(has_csc_wires&2) > 0";
TCut ok_digi1 = (ok_st1 && ok_w1);
TCut ok_digi2 = (ok_st2 && ok_w2);
TCut ok_lct1 = "(has_lct&1) > 0";
TCut ok_lct2 = "(has_lct&2) > 0";
TCut ok_lcths1 = ok_lct1 && "hs_lct_odd > 4 && hs_lct_odd < 125";
TCut ok_lcths2 = ok_lct2 && "hs_lct_even > 4 && hs_lct_even < 125";

TCut ok_gsh1 = "(has_gem_sh&1) > 0";
TCut ok_gsh2 = "(has_gem_sh&2) > 0";
TCut ok_g2sh1 = "(has_gem_sh2&1) > 0";
TCut ok_g2sh2 = "(has_gem_sh2&2) > 0";
TCut ok_gdg1 = "(has_gem_dg&1) > 0";
TCut ok_gdg2 = "(has_gem_dg&2) > 0";
TCut ok_pad1 = "(has_gem_pad&1) > 0";
TCut ok_pad2 = "(has_gem_pad&2) > 0";
TCut ok_2pad1 = "(has_gem_pad2&1) > 0";
TCut ok_2pad2 = "(has_gem_pad2&2) > 0";
TCut ok_pad1_overlap = ok_pad1 || (ok_lct2 && ok_pad2);
TCut ok_pad2_overlap = ok_pad2 || (ok_lct1 && ok_pad1);
TCut ok_copad1 = "(has_gem_copad&1) > 0";
TCut ok_copad2 = "(has_gem_copad&2) > 0";


TCut Qp = "charge > 0";
TCut Qn = "charge < 0";

TCut Ep = "endcap > 0";
TCut En = "endcap < 0";

TCut ok_eta = "TMath::Abs(eta)>1.64 && TMath::Abs(eta)<2.14";
TCut ok_pt = "pt > 20";



// TCut ok_dphi1 = "dphi_pad_odd < 10.";
// TCut ok_dphi2 = "dphi_pad_even < 10.";

// enum {GEM_EFF95 = 0, GEM_EFF98, GEM_EFF99};
// enum {DPHI_PT10 = 0, DPHI_PT15, DPHI_PT20, DPHI_PT30, DPHI_PT40};

double mymod(double x, double y) 
{
  return fmod(x,y);
}

// void setDPhi(int label_pt, int label_eff = GEM_EFF98)
// {
//   float dphi_odd[3][20]  = {
//     {0.009887, 0.006685, 0.005194, 0.003849, 0.003133},
//     {0.01076 , 0.007313, 0.005713, 0.004263, 0.003513},
//     {0.011434, 0.007860, 0.006162, 0.004615, 0.003891} };
//   float dphi_even[3][20] = {
//     {0.004418, 0.003274, 0.002751, 0.002259, 0.002008},
//     {0.004863, 0.003638, 0.003063, 0.002563, 0.002313},
//     {0.005238, 0.003931, 0.003354, 0.002809, 0.002574} };
//   ok_dphi1 = Form("TMath::Abs(dphi_pad_odd) < %f", dphi_odd[label_eff][label_pt]);
//   ok_dphi2 = Form("TMath::Abs(dphi_pad_even) < %f", dphi_even[label_eff][label_pt]);
// }


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

void eff_hs_dphi(TString f_name, TString p_name)
{
  // efficiency vs half-strip  - separate odd-even
  TCut ok_eta = "TMath::Abs(eta)>1.64 && TMath::Abs(eta)<2.12";
  //TCut ok_eta = "TMath::Abs(eta)>1.64 && TMath::Abs(eta)<1.74";
  //TCut ok_eta = "TMath::Abs(eta)>1.94 && TMath::Abs(eta)<2.12";

  TTree *t = getTree(f_name);
  setDPhi(DPHI_PT40, GEM_EFF98);

  //  TH1F* ho = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip;Eff.", "h_odd", "(130,0.5,130.5)", "hs_lct_odd", ok_lct1 && ok_eta && ok_gsh1 , ok_pad1, "", kRed);
  //  TH1F* he = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip;Eff.", "h_evn", "(130,0.5,130.5)", "hs_lct_even", ok_lct2 && ok_eta && ok_gsh2, ok_pad2, "same");

  //TH1F* ho = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip;Eff.", "h_odd", "(130,0.5,130.5)", "hs_lct_odd", ok_lct1 && ok_eta, ok_gsh1, "", kRed);
  //TH1F* he = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip;Eff.", "h_evn", "(130,0.5,130.5)", "hs_lct_even", ok_lct2 && ok_eta, ok_gsh2, "same");

  //TH1F* ho = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip;Eff.", "h_odd", "(130,0.5,130.5)", "hs_lct_odd", ok_lct1 && ok_eta && Qneg, ok_pad1 && ok_dphi1, "", kRed);
  //TH1F* he = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip;Eff.", "h_evn", "(130,0.5,130.5)", "hs_lct_even", ok_lct2 && ok_eta && Qneg, ok_pad2 && ok_dphi2, "same");
  //TH1F* ho = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip;Eff.", "h_odd", "(130,-0.2,0.2)", "mymod(phi+TMath::Pi()/36., TMath::Pi()/18.)", ok_eta, ok_lct1, "", kRed);
  //TH1F* he = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip;Eff.", "h_evn", "(130,-0.2,0.2)", "mymod(phi+TMath::Pi()/36., TMath::Pi()/18.)",ok_eta, ok_lct2, "same");
  //TH1F* ho = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip;Eff.", "h_odd", "(130,-0.2,0.2)", "mymod(phi+TMath::Pi()/36., TMath::Pi()/18.)", ok_eta, ok_lct1 || ok_lct2, "", kRed);
  //TH1F* hg = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip;Eff.", "h_odd", "(130,-0.2,0.2)", "mymod(phi+TMath::Pi()/36., TMath::Pi()/18.)", ok_eta, ok_pad1 || ok_pad2, "same");
  //TH1F* hgp = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip;Eff.", "h_odd", "(130,-0.2,0.2)", "mymod(phi+TMath::Pi()/36., TMath::Pi()/18.)", ok_eta&&Qpos, ok_pad1 || ok_pad2, "same", kGreen);


  //TH1F* ho = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip;Eff.", "h_odd", "(384,0.,384.)", "strip_gemsh_odd", ok_gsh1 && ok_eta, ok_pad1, "", kRed);
  //TH1F* he = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip;Eff.", "h_evn", "(384,0.,384.)", "strip_gemsh_even", ok_gsh2 && ok_eta, ok_pad2, "same");
  //TH1F* he = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip;Eff.", "h_evn", "(384,0.,384.)", "strip_gemsh_even", ok_gsh2 && ok_eta, ok_pad2, "");

  //TH1F* ho = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip;Eff.", "h_odd", "(384,0.,384.)", "strip_gemsh_odd", ok_gsh1 && ok_eta, ok_gdg1, "", kRed);
  //TH1F* he = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip;Eff.", "h_evn", "(384,0.,384.)", "strip_gemsh_even", ok_gsh2 && ok_eta, ok_gdg2, "same");

  //TH1F* ho = draw_eff(t, "Eff. for track with GEM digi to have GEM pad in chamber;digi strip;Eff.", "h_odd", "(384,0.5,384.5)", "strip_gemdg_odd", ok_gdg1 && ok_eta, ok_pad1, "", kRed);
  //TH1F* he = draw_eff(t, "Eff. for track with GEM digi to have GEM pad in chamber;digi strip;Eff.", "h_evn", "(384,0.5,384.5)", "strip_gemdg_even", ok_gdg2 && ok_eta, ok_pad2, "same");

  //gPad->Print(p_name);
}


void efficiency_1(TString f_name, TString p_name, TString pt, bool overlap)
{
  
  gStyle->SetTitleStyle(0);
  gStyle->SetTitleAlign(13); // coord in top left
  gStyle->SetTitleX(0.);
  gStyle->SetTitleY(1.);
  gStyle->SetTitleW(1);
  gStyle->SetTitleH(0.058);
  gStyle->SetTitleBorderSize(0);
    
  gStyle->SetPadLeftMargin(0.126);
  gStyle->SetPadRightMargin(0.04);
  gStyle->SetPadTopMargin(0.06);
  gStyle->SetPadBottomMargin(0.13);
  gStyle->SetOptStat(0);
  gStyle->SetMarkerStyle(1);

  // efficiency vs half-strip  - separate odd-even
  TCut ok_eta = "TMath::Abs(eta)>1.64 && TMath::Abs(eta)<2.12";
  TCut cut1;
  TCut cut2;
  if (overlap)
  {
    cut1 = ok_pad1_overlap;
    cut2 = ok_pad2_overlap;
  }
  else
  {
    cut1 = ok_pad1;
    cut2 = ok_pad2;
  }

  TTree *t = getTree(f_name);
  TH1F* ho = draw_eff(t, "         GEM reconstruction efficiency               CMS Simulation Preliminary;LCT half-strip number;Efficiency", "h_odd", "(130,0.5,130.5)", "hs_lct_odd", ok_lct1 && ok_eta , cut1, "", kRed);
  TH1F* he = draw_eff(t, "         GEM reconstruction efficiency               CMS Simulation Preliminary;LCT half-strip number;Efficiency", "h_evn", "(130,0.5,130.5)", "hs_lct_even", ok_lct2 && ok_eta , cut2, "same");
  ho->SetMinimum(0.9);
  ho->GetXaxis()->SetLabelSize(0.06);
  ho->GetYaxis()->SetLabelSize(0.06);


  TLegend *leg = new TLegend(0.25,0.23,.75,0.5, NULL, "brNDC");
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  leg->SetTextSize(0.06);
  leg->AddEntry((TObject*)0,"muon p_{T} = " + pt + " GeV/c",""); 
  leg->AddEntry(he, "\"Close\" chamber pairs","l");
  leg->AddEntry(ho, "\"Far\" chamber pairs","l");
  leg->Draw();

  // TLatex* tex2 = new TLatex(.67,.8,"   L1 Trigger");
  // tex2->SetTextSize(0.05);
  // tex2->SetNDC();
  // tex2->Draw();
    
  TLatex *  tex = new TLatex(.66,.73,"1.64<|#eta|<2.12");
  tex->SetTextSize(0.05);
  tex->SetNDC();
  tex->Draw();

  gPad->Print(p_name);
}


void efficiency_2(TString f_name, TString p_name, TString pt, bool overlap)
{
  // efficiency vs half-strip  - including overlaps in odd&even
  TCut ok_eta = "TMath::Abs(eta)>1.64 && TMath::Abs(eta)<2.12";
  TCut cut1;
  TCut cut2;
  if (overlap)
  {
    cut1 = ok_pad1_overlap;
    cut2 = ok_pad2_overlap;
  }
  else
  {
    cut1 = ok_pad1;
    cut2 = ok_pad2;
  }
  
  TTree *t = getTree(f_name);

  // latest instructions by Vadim: 21-08-2013
  TGraphAsymmErrors* hgn = draw_geff(t, "         GEM pad matching             CMS Simulation Preliminary;Generated muon #phi [deg];Efficiency", "h_odd", "(40,-10,10)", "mymod(phi*180./TMath::Pi(), 360/18.)", ok_eta&&Qn, ok_pad1 || ok_pad2,"", kRed);
  TGraphAsymmErrors* hgp = draw_geff(t, "         GEM pad matching             CMS Simulation Preliminary;Generated muon #phi [deg];Effciency", "h_odd", "(40,-10,10)", "mymod(phi*180./TMath::Pi(), 360/18.)", ok_eta&&Qp, ok_pad1 || ok_pad2,"same", kBlue);
  double maxi = 1.1;
  double mini = 0.0;
  hgn->SetMinimum(mini);
  hgn->SetMaximum(maxi);
  hgn->GetXaxis()->SetLabelSize(0.05);
  hgn->GetYaxis()->SetLabelSize(0.05);
  hgp->GetXaxis()->SetLabelSize(0.05);
  hgp->GetYaxis()->SetLabelSize(0.05);

  TLine *l1 = new TLine(-5,mini,-5,maxi);
  l1->SetLineStyle(2);
  l1->Draw();
  TLine *l1 = new TLine(5,mini,5,maxi);
  l1->SetLineStyle(2);
  l1->Draw();

  // hgn->Fit("pol0","R0","",-10,10);
  // hgp->Fit("pol0","R0","",-10,10);
  // double eff_neg = (hgn->GetFunction("pol0"))->GetParameter(0);
  // double eff_pos = (hgp->GetFunction("pol0"))->GetParameter(0);
  // std::cout << "negative efficiency" << eff_neg << " " 
  // 	    << "positive efficiency" << eff_pos << std::endl;

  // ho = draw_eff(t, "         GEM reconstruction efficiency               CMS Simulation;local #phi [deg];Efficiency", "h_odd", "(150,-10,10)", "mymod(phi+TMath::Pi()/36., TMath::Pi()/18.) * 180./TMath::Pi()", ok_eta&&Qn, ok_pad1 || ok_pad2, "", kRed);
  // he = draw_eff(t, "         GEM reconstruction efficiency               CMS Simulation;local #phi [deg];Efficiency", "h_odd", "(150,-10,10)", "mymod(phi+TMath::Pi()/36., TMath::Pi()/18.) * 180./TMath::Pi()", ok_eta&&Qp, ok_pad1 || ok_pad2, "same",kBlue);

  // TH1F* ho = draw_eff(t, "         GEM reconstruction efficiency               CMS Simulation;local #phi [deg];Efficiency", "h_odd", "(130,-5,5)", "fmod(180*phi/TMath::Pi(),5)", ok_lct1 && ok_eta , cut1, "", kRed);
  // TH1F* he = draw_eff(t, "         GEM reconstruction efficiency               CMS Simulation;local #phi [deg];Efficiency", "h_evn", "(130,-5,5)", "fmod(180*phi/TMath::Pi(),5)", ok_lct2 && ok_eta , cut2, "same");
  //  ho->SetMinimum(0.);

  TLegend *leg = new TLegend(0.25,0.23,.75,0.5, NULL, "brNDC");
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  leg->SetTextSize(0.06);
  leg->AddEntry((TObject*)0,"muon p_{T} = " + pt + " GeV/c",""); 
  leg->AddEntry(hgp, "Postive muons","l");
  leg->AddEntry(hgn, "Negative muons","l");
  leg->Draw();

  // Print additional information
  /*
  TLatex* tex2 = new TLatex(.67,.8,"   L1 Trigger");
  tex2->SetTextSize(0.05);
  tex2->SetNDC();
  tex2->Draw();
  */
  
  //  TLatex *  tex = new TLatex(.66,.73,"1.64<|#eta|<2.12");
  TLatex *  tex = new TLatex(.7,.2,"1.64<|#eta|<2.12");
  tex->SetTextSize(0.05);
  tex->SetNDC();
  tex->Draw();

  gPad->Print(p_name);
}

void drawplot_eff_eta()
{

  // gROOT->ProcessLine(".L effFunctions.C");

  TCanvas* cEff = new TCanvas("cEff","cEff",700,450);
  cEff->SetGridx(1);
  cEff->SetGridy(1);

  TTree *gt = getTree(filesDir + "gem_csc_delta_pt40_pad4.root");

  //ht = draw_geff(gt, "Eff. for a SimTrack to have an associated LCT;SimTrack |#eta|;Eff.", "h_odd", "(100,1.54,2.2)", "TMath::Abs(eta)", "", ok_lct1 || ok_lct2, "P", kRed);
  //hh = draw_geff(gt, "Eff. for a SimTrack to have an associated LCT;SimTrack |#eta|;Eff.", "h_odd", "(100,1.54,2.2)", "TMath::Abs(eta)", "", ok_sh1 || ok_sh2, "P same", kViolet);
  h1 = draw_geff(gt, "Eff. for a SimTrack to have an associated ME1/b LCT;SimTrack |#eta|;Eff.", "h_odd", "(70,1.54,2.2)", "TMath::Abs(eta)", ok_sh1, ok_lct1, "P", kRed);
  h2 = draw_geff(gt, "Eff. for a SimTrack to have an associated ME1/b LCT;SimTrack |#eta|;Eff.", "h_odd", "(70,1.54,2.2)", "TMath::Abs(eta)", ok_sh2, ok_lct2, "P same");
  eff_base->GetYaxis()->SetRangeUser(0.6,1.05);
  TLegend *leg = new TLegend(0.42,0.23,.96,0.4, NULL, "brNDC");
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  leg->AddEntry(h1, "odd chambers","l");
  leg->AddEntry(h2, "even chambers","l");
  leg->Draw();
  TLatex *  tex = new TLatex(0.17, 0.16,"No Pile-Up");
  tex->SetNDC();
  tex->Draw();
  cEff->Print(plotDir + "lct_eff_for_Trk_vsTrkEta_pt40" + ext);


  h1 = draw_geff(gt, "Eff. for a SimTrack to have an associated ME1/b LCT and GEM Pad;SimTrack |#eta|;Eff.", "h_odd", "(70,1.54,2.2)", "TMath::Abs(eta)", ok_sh1, ok_lct1 && ok_pad1, "P", kRed);
  h2 = draw_geff(gt, "Eff. for a SimTrack to have an associated ME1/b LCT and GEM Pad;SimTrack |#eta|;Eff.", "h_odd", "(70,1.54,2.2)", "TMath::Abs(eta)", ok_sh2, ok_lct2 && ok_pad2, "P same");
  eff_base->GetYaxis()->SetRangeUser(0.6,1.05);
  TLegend *leg = new TLegend(0.42,0.23,.96,0.4, NULL, "brNDC");
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  leg->AddEntry(h1, "odd chambers","l");
  leg->AddEntry(h2, "even chambers","l");
  leg->Draw();
  TLatex *  tex = new TLatex(0.17, 0.16,"No Pile-Up");
  tex->SetNDC();
  tex->Draw();
  cEff->Print(plotDir + "gem_pad_and_lct_eff_for_Trk_vsTrkEta_pt40" + ext);

  return;

  h1 = draw_geff(gt, "Eff. for a SimTrack to have an associated GEM Pad;SimTrack |#eta|;Eff.", "h_odd", "(70,1.54,2.2)", "TMath::Abs(eta)", "", ok_pad1 || ok_pad2, "P", kViolet);
  eff_base->GetYaxis()->SetRangeUser(0.6,1.05);
  TLatex *  tex = new TLatex(0.17, 0.16,"No Pile-Up");
  tex->SetNDC();
  tex->Draw();
  cEff->Print(plotDir + "gem_pad0_eff_for_Trk_vsTrkEta_pt40" + ext);




  TTree *gt15 = getTree(filesDir + "gem_csc_delta_pt15_pad4.root");
  h1 = draw_geff(gt15, "Eff. for a SimTrack to have an associated LCT;SimTrack |#eta|;Eff.", "h_odd", "(140,1.54,2.2)", "TMath::Abs(eta)", "", ok_lct1 || ok_lct2, "P", kViolet+2);
  cEff->Print(plotDir + "lct_eff_for_Trk_vsTrkEta_pt15" + ext);


  ho = draw_geff(gt, "Eff. for track with LCT to have GEM pad in chamber;LCT |#eta|;Eff.", "h_odd", "(140,1.5,2.2)", "TMath::Abs(eta_lct_odd)", ok_lct1, ok_pad1, "P", kRed);
  he = draw_geff(gt, "Eff. for track with LCT to have GEM pad in chamber;LCT |#eta|;Eff.", "h_evn", "(140,1.5,2.2)", "TMath::Abs(eta_lct_even)", ok_lct2, ok_pad2, "P same");
  TLegend *leg = new TLegend(0.42,0.23,.96,0.4, NULL, "brNDC");
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  leg->AddEntry(he, "\"Close\" chambers","l");
  leg->AddEntry(ho, "\"Far\" chambers","l");
  leg->Draw();
  cEff->Print(plotDir + "gem_pad_eff_for_LCT_vsLCTEta_pt40" + ext);

  draw_geff(gt, "Eff. for track with LCT to have GEM pad in chamber;SimTrack |#eta|;Eff.", "h_odd", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_lct1, ok_pad1, "P", kRed);
  draw_geff(gt, "Eff. for track with LCT to have GEM pad in chamber;SimTrack |#eta|;Eff.", "h_evn", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_lct2, ok_pad2, "P same");
  leg->Draw();
  cEff->Print(plotDir + "gem_pad_eff_for_LCT_vsTrkEta_pt40" + ext);

  draw_geff(gt, "Eff. for track with LCT to have GEM pad in chamber;LCT |#eta|;Eff.", "h_odd", "(140,1.5,2.2)", "TMath::Abs(eta_lct_odd)", ok_lct1, ok_pad1_overlap, "P", kRed);
  draw_geff(gt, "Eff. for track with LCT to have GEM pad in chamber;LCT |#eta|;Eff.", "h_evn", "(140,1.5,2.2)", "TMath::Abs(eta_lct_even)", ok_lct2, ok_pad2_overlap, "P same");
  leg->Draw();
  cEff->Print(plotDir + "gem_pad_eff_for_LCT_vsLCTEta_pt40_overlap" + ext);

  draw_geff(gt, "Eff. for track with LCT to have GEM pad in chamber;SimTrack |#eta|;Eff.", "h_odd", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_lct1, ok_pad1_overlap, "P", kRed);
  draw_geff(gt, "Eff. for track with LCT to have GEM pad in chamber;SimTrack |#eta|;Eff.", "h_evn", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_lct2, ok_pad2_overlap, "P same");
  leg->Draw();
  cEff->Print(plotDir + "gem_pad_eff_for_LCT_vsTrkEta_pt40_overlap" + ext);

  //draw_geff(gt, "Eff. for track with LCT to have GEM pad in chamber;z SimTrack |#eta|;Eff.", "h_odd", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_lct1 && Ep, ok_pad1_overlap, "P", kRed);
  //draw_geff(gt, "Eff. for track with LCT to have GEM pad in chamber;z SimTrack |#eta|;Eff.", "h_evn", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_lct2 && Ep, ok_pad2_overlap, "P same");
  //draw_geff(gt, "Eff. for track with LCT to have GEM pad in chamber;z SimTrack |#eta|;Eff.", "h_odd", "(140,1.5,2.2)", "TMath::Abs(eta_gemsh_odd)", ok_gsh1, ok_gdg1, "P", kRed);
  h1 = draw_geff(gt, "Eff. for a SimTrack to have an associated GEM pad;SimTrack |#eta|;Eff.", "h_odd", "(140,1.54,2.2)", "TMath::Abs(eta)", "", ok_pad1 || ok_pad2, "P", kViolet);
  h2 = draw_geff(gt, "Eff. for a SimTrack to have an associated GEM pad;SimTrack |#eta|;Eff.", "h_odd", "(140,1.54,2.2)", "TMath::Abs(eta)", "", ok_2pad1 || ok_2pad2, "P same", kViolet-6);
  TLegend *leg = new TLegend(0.42,0.23,.96,0.4, NULL, "brNDC");
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  leg->AddEntry(h1, "at least one pad","l");
  leg->AddEntry(he, "two pads in two GEMs","l");
  leg->Draw();
  cEff->Print(plotDir + "gem_pad_eff_for_Trk_vsTrkEta_pt40" + ext);

  return;
  draw_geff(gt, "Eff. for a SimTrack to have an associated GEM pad;SimTrack |#eta|;Eff.", "h_odd", "(140,1.54,2.2)", "TMath::Abs(eta)", "", ok_gsh1 || ok_gsh2, "P", kViolet);
  draw_geff(gt, "Eff. for a SimTrack to have an associated GEM pad;SimTrack |#eta|;Eff.", "h_odd", "(140,1.54,2.2)", "TMath::Abs(eta)", "", ok_g2sh1 || ok_g2sh2 , "P", kOrange);
  draw_geff(gt, "Eff. for a SimTrack to have an associated GEM pad;SimTrack |#eta|;Eff.", "h_odd", "(140,1.54,2.2)", "TMath::Abs(eta)", "", ok_copad1 || ok_copad2 , "P same", kRed);


}

void drawplot_eff()
{
  gROOT->ProcessLine(".L effFunctions.C");
  gROOT->ProcessLine(".L tdrstyle.C");
  //setTDRStyle();
  
  gStyle->SetTitleStyle(0);
  gStyle->SetTitleAlign(13);// coord in top left
  gStyle->SetTitleX(0.);
  gStyle->SetTitleY(1.);
  gStyle->SetTitleW(1);
  gStyle->SetTitleH(0.058);
  gStyle->SetTitleBorderSize(0);
  
  TCanvas* cEff = new TCanvas("cEff","cEff",700,500);

  // efficiency_1(filesDir + "gem_csc_delta_pt5_pad4.root",  plotDir + "gem_pad_eff_for_LCT_vs_HS_pt05" + ext, "5", false);
  // efficiency_1(filesDir + "gem_csc_delta_pt10_pad4.root", plotDir + "gem_pad_eff_for_LCT_vs_HS_pt10" + ext, "10", false);
  // efficiency_1(filesDir + "gem_csc_delta_pt15_pad4.root", plotDir + "gem_pad_eff_for_LCT_vs_HS_pt15" + ext, "15", false);
  // efficiency_1(filesDir + "gem_csc_delta_pt20_pad4.root", plotDir + "gem_pad_eff_for_LCT_vs_HS_pt20" + ext, "20", false);
  // efficiency_1(filesDir + "gem_csc_delta_pt30_pad4.root", plotDir + "gem_pad_eff_for_LCT_vs_HS_pt30" + ext, "30", false);
  // efficiency_1(filesDir + "gem_csc_delta_pt40_pad4.root", plotDir + "gem_pad_eff_for_LCT_vs_HS_pt40" + ext, "40", false);

  // efficiency_1(filesDir + "gem_csc_delta_pt5_pad4.root",  plotDir + "gem_pad_eff_for_LCT_vs_HS_pt05_overlap" + ext, "5", true);
  // efficiency_1(filesDir + "gem_csc_delta_pt10_pad4.root", plotDir + "gem_pad_eff_for_LCT_vs_HS_pt10_overlap" + ext, "10", true);
  // efficiency_1(filesDir + "gem_csc_delta_pt15_pad4.root", plotDir + "gem_pad_eff_for_LCT_vs_HS_pt15_overlap" + ext, "15", true);
  // efficiency_1(filesDir + "gem_csc_delta_pt20_pad4.root", plotDir + "gem_pad_eff_for_LCT_vs_HS_pt20_overlap" + ext, "20", true);
  // efficiency_1(filesDir + "gem_csc_delta_pt30_pad4.root", plotDir + "gem_pad_eff_for_LCT_vs_HS_pt30_overlap" + ext, "30", true);
  // efficiency_1(filesDir + "gem_csc_delta_pt40_pad4.root", plotDir + "gem_pad_eff_for_LCT_vs_HS_pt40_overlap" + ext, "40", true);

  // efficiency_2(filesDir + "gem_csc_delta_pt5_pad4.root",  plotDir + "gem_pad_eff_for_LCT_vs_phi_pt05" + ext, "5", false);
  // efficiency_2(filesDir + "gem_csc_delta_pt10_pad4.root", plotDir + "gem_pad_eff_for_LCT_vs_phi_pt10" + ext, "10", false);
  // efficiency_2(filesDir + "gem_csc_delta_pt15_pad4.root", plotDir + "gem_pad_eff_for_LCT_vs_phi_pt15" + ext, "15", false);
  // efficiency_2(filesDir + "gem_csc_delta_pt20_pad4.root", plotDir + "gem_pad_eff_for_LCT_vs_phi_pt20" + ext, "20", false);
  // efficiency_2(filesDir + "gem_csc_delta_pt30_pad4.root", plotDir + "gem_pad_eff_for_LCT_vs_phi_pt30" + ext, "30", false);
  // efficiency_2(filesDir + "gem_csc_delta_pt40_pad4.root", plotDir + "gem_pad_eff_for_LCT_vs_phi_pt40" + ext, "40", false);

  // efficiency_2(filesDir + "gem_csc_delta_pt5_pad4.root",  plotDir + "gem_pad_eff_for_LCT_vs_phi_pt05_overlap" + ext, "5", true);
  // efficiency_2(filesDir + "gem_csc_delta_pt10_pad4.root", plotDir + "gem_pad_eff_for_LCT_vs_phi_pt10_overlap" + ext, "10", true);
  // efficiency_2(filesDir + "gem_csc_delta_pt15_pad4.root", plotDir + "gem_pad_eff_for_LCT_vs_phi_pt15_overlap" + ext, "15", true);

  efficiency_2(filesDir + "gem_csc_delta_pt20_pad4.root", plotDir + "gem_pad_eff_for_LCT_vs_phi_pt20_overlap" + ext, "20", true);


  // efficiency_2(filesDir + "gem_csc_delta_pt30_pad4.root", plotDir + "gem_pad_eff_for_LCT_vs_phi_pt30_overlap" + ext, "30", true);
  // efficiency_2(filesDir + "gem_csc_delta_pt40_pad4.root", plotDir + "gem_pad_eff_for_LCT_vs_phi_pt40_overlap" + ext, "40", true);

  //  drawplot_eff_eta();
}


/*
  TTree *gt = getTree("gem_csc_delta_pt40_pad4.root");


  draw_eff(gt, "Eff. ;p_{T}, GeV/c;Eff.", "hname", "(45,0.5,45.5)", "pt", ok_sh1 && ok_eta , ok_digi1)
  draw_eff(gt, "Eff. ;p_{T}, GeV/c;Eff.", "hname", "(45,0.5,45.5)", "pt", ok_digi1 && ok_eta , ok_lct1)
  draw_eff(gt, "Eff. ;p_{T}, GeV/c;Eff.", "hname", "(50,0.,50.)", "pt", ok_sh1 && ok_eta , ok_lct1)


  draw_eff(gt, "Eff. of |CLCT pattern bend| selection for matched LCTs;p_{T}, GeV/c;Eff.", "hname2", "(50,0.,50.)", "pt", (ok_lct1||ok_lct2) && ok_eta, "TMath::Abs(bend_lct_odd)<2 || TMath::Abs(bend_lct_even)<2")
  draw_eff(gt, "Eff. of |CLCT pattern bend| selection for matched LCTs;p_{T}, GeV/c;Eff.", "hname1", "(50,0.,50.)", "pt", (ok_lct1||ok_lct2) && ok_eta, "TMath::Abs(bend_lct_odd)<1 || TMath::Abs(bend_lct_even)<1","same",kBlack)
  draw_eff(gt, "Eff. of |CLCT bend|<3 selection for matched LCTs;p_{T}, GeV/c;Eff.", "hname3", "(50,0.,50.)", "pt", (ok_lct1||ok_lct2) && ok_eta, "TMath::Abs(bend_lct_odd)<3 || TMath::Abs(bend_lct_even)<3","same",kGreen+2)
  draw_eff(gt, "Eff. of |CLCT bend|<4 selection for matched LCTs;p_{T}, GeV/c;Eff.", "hname4", "(50,0.,50.)", "pt", (ok_lct1||ok_lct2) && ok_eta, "TMath::Abs(bend_lct_odd)<4 || TMath::Abs(bend_lct_even)<4","same",kRed)

  draw_eff(gt, "Eff. for track with LCT to have GEM pad in chamber;|#eta|;Eff.", "hname", "(90,1.5,2.2)", "TMath::Abs(eta)", ok_lct2, ok_pad2 )
  draw_eff(gt, "Eff. for track with LCT to have GEM pad in chamber;|#eta|;Eff.", "hname", "(90,1.5,2.2)", "TMath::Abs(eta)", ok_lct1, ok_pad1, "same" )

  draw_eff(gt, "Eff. for track with LCT to have GEM pad in chamber;|#phi|;Eff.", "hname", "(128,0,3.2)", "TMath::Abs(phi)", ok_lct2 && ok_eta && ok_pt , ok_pad2 )
  draw_eff(gt, "Eff. for track with LCT to have GEM pad in chamber;|#phi|;Eff.", "hname", "(128,0,3.2)", "TMath::Abs(phi)", ok_lct1 && ok_eta && ok_pt , ok_pad1, "same" )


  draw_eff(gt, "Eff. for track with LCT to have GEM pad in chamber;p_{T}, GeV/c;Eff.", "h_odd", "(50,0.,50.)", "pt", ok_lct1 && ok_eta , ok_pad1)
  draw_eff(gt, "Eff. for track with LCT to have GEM pad in chamber;p_{T}, GeV/c;Eff.", "h_evn", "(50,0.,50.)", "pt", ok_lct2 && ok_eta , ok_pad2, "same")




  // efficiency vs half-strip  - separate odd-even
  TCut ok_eta = "TMath::Abs(eta)>1.64 && TMath::Abs(eta)<1.9"
  draw_eff(gt, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip;Eff.", "h_odd", "(130,0.5,130.5)", "hs_lct_odd", ok_lct1 && ok_eta , ok_pad1, "", kRed)
  draw_eff(gt, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip;Eff.", "h_evn", "(130,0.5,130.5)", "hs_lct_even", ok_lct2 && ok_eta , ok_pad2, "same")

  // efficiency vs half-strip  - including overlaps in odd&even
  TCut ok_eta = "TMath::Abs(eta)>1.64 && TMath::Abs(eta)<1.9"
  TCut ok_pad1_overlap = ok_pad1 || (ok_lct2 && ok_pad2);
  TCut ok_pad2_overlap = ok_pad2 || (ok_lct1 && ok_pad1);
  TTree *t = getTree("gem_csc_delta_pt20_pad4.root");

  draw_eff(gt, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip;Eff.", "h_odd", "(130,0.5,130.5)", "hs_lct_odd", ok_lct1 && ok_eta , ok_pad1_overlap, "", kRed)
  draw_eff(gt, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip;Eff.", "h_evn", "(130,0.5,130.5)", "hs_lct_even", ok_lct2 && ok_eta , ok_pad2_overlap, "same")


  draw_eff(gt, "Eff. for track with LCT to have GEM pad in chamber;LCT |#eta|;Eff.", "h_odd", "(140,1.5,2.2)", "TMath::Abs(eta_lct_odd)", ok_lct1, ok_pad1, "", kRed)
  draw_eff(gt, "Eff. for track with LCT to have GEM pad in chamber;LCT |#eta|;Eff.", "h_evn", "(140,1.5,2.2)", "TMath::Abs(eta_lct_even)", ok_lct2, ok_pad2, "same")

  draw_eff(gt, "Eff. for track with LCT to have GEM pad in chamber;trk |#eta|;Eff.", "h_odd", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_lct1, ok_pad1, "", kRed)
  draw_eff(gt, "Eff. for track with LCT to have GEM pad in chamber;trk |#eta|;Eff.", "h_evn", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_lct2, ok_pad2, "same")


  draw_eff(gt, "Eff. for track with LCT to have matched GEM pad;p_{T}, GeV/c;Eff.", "h_odd", "(50,0.,50.)", "pt", ok_lct1 && ok_eta , ok_pad1 && ok_dphi1)
  draw_eff(gt, "Eff. for track with LCT to have matched GEM pad;p_{T}, GeV/c;Eff.", "h_env", "(50,0.,50.)", "pt", ok_lct2 && ok_eta , ok_pad2 && ok_dphi2, "same")

  draw_eff(gt, "Eff. for track with LCT to have matched GEM pad;p_{T}, GeV/c;Eff.", "h_odd", "(50,0.,50.)", "pt", ok_lct1 && ok_eta && ok_pad1, ok_dphi1)
  draw_eff(gt, "Eff. for track with LCT to have matched GEM pad;p_{T}, GeV/c;Eff.", "h_evn", "(50,0.,50.)", "pt", ok_lct2 && ok_eta && ok_pad2, ok_dphi2, "same")





  // 98% pt10
  TCut ok_dphi1 = "TMath::Abs(dphi_pad_odd) < 0.01076"
  TCut ok_dphi2 = "TMath::Abs(dphi_pad_even) < 0.004863"
  draw_eff(gt, "Eff. for track with LCT to have matched GEM pad;p_{T}, GeV/c;Eff.", "h_odd_10", "(50,0.,50.)", "pt", ok_lct1 && ok_eta , ok_pad1 && ok_dphi1, "", kRed)
  draw_eff(gt, "Eff. for track with LCT to have matched GEM pad;p_{T}, GeV/c;Eff.", "h_evn_10", "(50,0.,50.)", "pt", ok_lct2 && ok_eta , ok_pad2 && ok_dphi2, "same")
  // 98% pt30
  TCut ok_dphi1 = "TMath::Abs(dphi_pad_odd) < 0.00571"
  TCut ok_dphi2 = "TMath::Abs(dphi_pad_even) < 0.00306"
  draw_eff(gt, "Eff. for track with LCT to have matched GEM pad;p_{T}, GeV/c;Eff.", "h_odd_20", "(50,0.,50.)", "pt", ok_lct1 && ok_eta , ok_pad1 && ok_dphi1, "same", kRed)
  draw_eff(gt, "Eff. for track with LCT to have matched GEM pad;p_{T}, GeV/c;Eff.", "h_evn_20", "(50,0.,50.)", "pt", ok_lct2 && ok_eta , ok_pad2 && ok_dphi2, "same")
  // 98% pt30
  TCut ok_dphi1 = "TMath::Abs(dphi_pad_odd) < 0.00426"
  TCut ok_dphi2 = "TMath::Abs(dphi_pad_even) < 0.00256"
  draw_eff(gt, "Eff. for track with LCT to have matched GEM pad;p_{T}, GeV/c;Eff.", "h_odd_30", "(50,0.,50.)", "pt", ok_lct1 && ok_eta , ok_pad1 && ok_dphi1, "same", kRed)
  draw_eff(gt, "Eff. for track with LCT to have matched GEM pad;p_{T}, GeV/c;Eff.", "h_evn_30", "(50,0.,50.)", "pt", ok_lct2 && ok_eta , ok_pad2 && ok_dphi2, "same")
  // 98% pt40
  TCut ok_dphi1 = "TMath::Abs(dphi_pad_odd) < 0.00351"
  TCut ok_dphi2 = "TMath::Abs(dphi_pad_even) < 0.00231"
  draw_eff(gt, "Eff. for track with LCT to have matched GEM pad;p_{T}, GeV/c;Eff.", "h_odd_40", "(50,0.,50.)", "pt", ok_lct1 && ok_eta , ok_pad1 && ok_dphi1, "same", kRed)
  draw_eff(gt, "Eff. for track with LCT to have matched GEM pad;p_{T}, GeV/c;Eff.", "h_evn_40", "(50,0.,50.)", "pt", ok_lct2 && ok_eta , ok_pad2 && ok_dphi2, "same")


  |#Delta#phi_{odd}(LCT,Pad)| < 5.5 mrad
  |#Delta#phi_{even}(LCT,Pad)| < 3.1 mrad
  |#Delta#eta(LCT,Pad)| < 0.08

  draw_eff(gt, "Eff. for track with LCT to have matched GEM pad;p_{T}, GeV/c;Eff.", "hname", "(45,0.5,45.5)", "pt", ok_sh1 && ok_eta , ok_lct1 && ok_pad1 && ok_dphi1)
  draw_eff(gt, "Eff. for track with LCT to have matched GEM pad;p_{T}, GeV/c;Eff.", "hname", "(45,0.5,45.5)", "pt", ok_sh2 && ok_eta , ok_lct2 && ok_pad2 && ok_dphi2, "same")




  draw_eff(gt, "title;pt;Eff.", "hname", "(45,0.5,45.5)", "pt", ok_lct1 && ok_eta , ok_pad1 && ok_dphi1)

  draw_eff(gt, "title;pt;Eff.", "hname", "(45,0.5,45.5)", "pt", ok_lct2 && ok_eta , ok_pad2 && ok_dphi2, "same")


  draw_eff(gt, "title;|#eta|;Eff.", "hname", "(45,1.5,2.2)", "TMath::Abs(eta)", ok_sh1 , ok_lct1 && ok_pad1 )


  draw_eff(gt, "title;|#phi|;Eff.", "hname", "(128,0,3.2)", "TMath::Abs(phi)", ok_sh2 , ok_lct2 && ok_pad2 )




  gt->Draw("TMath::Abs(eta)", ok_sh1, "");
  gt->Draw("TMath::Abs(eta)", ok_sh1 && ok_lct1, "same");
  gt->Draw("TMath::Abs(eta)", ok_sh1 && ok_lct1 && ok_pad1, "same");

  gt->Draw("TMath::Abs(phi)", ok_sh1 && ok_lct1 && ok_pad1, "");
  gt->Draw("TMath::Abs(phi)", ok_sh1, "");
  gt->Draw("TMath::Abs(eta)", ok_sh1 && ok_lct1, "same");

  gt->Draw("TMath::Abs(phi)", ok_sh1, "");
  gt->Draw("TMath::Abs(phi)", ok_sh1 && ok_lct1, "same");
  gt->Draw("TMath::Abs(phi)", ok_sh1 && ok_lct1 && ok_pad1, "same");


  gt->Draw("pt", ok_sh1, "");
  gt->Draw("pt", ok_sh1 && ok_lct1, "same");
  gt->Draw("pt", ok_sh1 && ok_lct1 && ok_pad1, "same");
  gt->Draw("pt", ok_sh1, "");
  gt->Draw("pt", ok_sh1 && ok_lct1 && ok_pad1, "same");

  gt->Draw("pt>>h1", ok_sh1, "");
  dn=(TH1F*)h1->Clone("dn")
  gt->Draw("pt>>h2", ok_sh1 && ok_lct1 && ok_pad1, "same");
  nm=(TH1F*)h2->Clone("nm")

  dn->Draw()
  nm->Draw("same")

  nm->Divide(dn)
  nm->Draw()

  gt->Draw("pt>>h2", ok_sh1 && ok_lct1, "same");
  nmlct=(TH1F*)h2->Clone("nmlct")
  nmlct->Divide(dn)

  nm->Draw()
  nmlct->Draw("same")
*/
