TLatex* drawEtaLabel(TString minEta, TString maxEta, float x=0.17, float y=0.35, float font_size=0.)
{
  TString label(minEta + " < |#eta| < " + maxEta);
  TLatex *  tex = new TLatex(x, y,label);
  if (font_size > 0.) tex->SetFontSize(font_size);
  tex->SetTextSize(0.04);
  tex->SetNDC();
  tex->Draw();
  return tex;
}

TLatex* drawLumiLabel(float x=0.17, float y=0.35)
{
  TLatex *  tex = new TLatex(x, y,"L = 4*10^{34} cm^{-2} s^{-1}");
  tex->SetTextSize(0.04);
  tex->SetNDC();
  tex->Draw();
  return tex;
}

TLatex* drawL1Label(float x=0.17, float y=0.35)
{
  TLatex *  tex = new TLatex(x, y,"L1 trigger in 2012 configuration");
  tex->SetTextSize(0.04);
  tex->SetNDC();
  tex->Draw();
  return tex;
}

void produceRatePlots(TString ext)
{
  gem_dir = "files/"; 
  gem_label = "gem98";

  TString the_ttl = "        L1 trigger rates in 1.64<|#eta|<2.14 region, CMS Simulation;p_{T}^{cut} [GeV/c];rate [kHz]";
  TString plots = "plots/rate/";

  //gStyle->SetStatW(0.13);
  //gStyle->SetStatH(0.08);
  gStyle->SetStatW(0.07);
  gStyle->SetStatH(0.06);

  gStyle->SetOptStat(0);

  gStyle->SetTitleStyle(0);
  gStyle->SetTitleAlign(13);// coord in top left
  gStyle->SetTitleX(0.);
  gStyle->SetTitleY(1.);
  gStyle->SetTitleW(1);
  gStyle->SetTitleH(0.058);
  gStyle->SetTitleBorderSize(0);

  gStyle->SetPadLeftMargin(0.126);
  gStyle->SetPadRightMargin(0.04);
  gStyle->SetPadTopMargin(0.06);
  gStyle->SetPadBottomMargin(0.13);

  gStyle->SetMarkerStyle(1);

  // ********** PAT2 **********

  getPTHistos("minbias_pt06_pat2");
  hh = (TH1D*)result_def_3s1b->Clone("gem_new");
  for (int b = hh->FindBin(6.01); b <= hh->GetNbinsX(); ++b) hh->SetBinContent(b, 0);
  hh_all = (TH1D*)result_def_eta_all_3s1b->Clone("gem_new_eta_all");
  for (int b = hh_all->FindBin(6.01); b <= hh_all->GetNbinsX(); ++b) hh_all->SetBinContent(b, 0);
  hh_no1a = (TH1D*)result_def_eta_no1a_3s1b->Clone("gem_new_eta_no1a");
  for (int b = hh_no1a->FindBin(6.01); b <= hh_no1a->GetNbinsX(); ++b) hh_no1a->SetBinContent(b, 0);
  hh_2s1b = (TH1D*)result_def_2s1b->Clone("gem_new_2s1b");
  for (int b = hh_2s1b->FindBin(6.01); b <= hh_2s1b->GetNbinsX(); ++b) hh_2s1b->SetBinContent(b, 0);

  h06 = (TH1D*)result_gem->Clone("gem_new_06");
  for (int b = h06->FindBin(6.01); b < h06->FindBin(10.01); ++b) {hh->SetBinContent(b, h06->GetBinContent(b)); hh->SetBinError(b, h06->GetBinError(b));}
  h06_all = (TH1D*)result_gem_eta_all->Clone("gem_new_eta_all_06");
  for (int b = h06_all->FindBin(6.01); b < h06_all->FindBin(10.01); ++b) {hh_all->SetBinContent(b, h06_all->GetBinContent(b)); hh_all->SetBinError(b, h06_all->GetBinError(b));}
  h06_no1a = (TH1D*)result_gem_eta_no1a->Clone("gem_new_eta_no1a_06");
  for (int b = h06_no1a->FindBin(6.01); b < h06_no1a->FindBin(10.01); ++b) {hh_no1a->SetBinContent(b, h06_no1a->GetBinContent(b)); hh_no1a->SetBinError(b, h06_no1a->GetBinError(b));}
  h06_2s1b = (TH1D*)result_gem_2s1b->Clone("gem_new_2s1b_06");
  for (int b = h06_2s1b->FindBin(6.01); b < h06_2s1b->FindBin(10.01); ++b) {hh_2s1b->SetBinContent(b, h06_2s1b->GetBinContent(b)); hh_2s1b->SetBinError(b, h06_2s1b->GetBinError(b));}

  getPTHistos("minbias_pt10_pat2");
  h10 = (TH1D*)result_gem->Clone("gem10");
  for (int b = h10->FindBin(10.01); b < h10->FindBin(15.01); ++b) {hh->SetBinContent(b, h10->GetBinContent(b)); hh->SetBinError(b, h10->GetBinError(b));}
  h10_all = (TH1D*)result_gem_eta_all->Clone("gem_new_eta_all_10");
  for (int b = h10_all->FindBin(10.01); b < h10_all->FindBin(15.01); ++b) {hh_all->SetBinContent(b, h10_all->GetBinContent(b)); hh_all->SetBinError(b, h10_all->GetBinError(b));}
  h10_no1a = (TH1D*)result_gem_eta_no1a->Clone("gem_new_eta_no1a_10");
  for (int b = h10_no1a->FindBin(10.01); b < h10_no1a->FindBin(15.01); ++b) {hh_no1a->SetBinContent(b, h10_no1a->GetBinContent(b)); hh_no1a->SetBinError(b, h10_no1a->GetBinError(b));}
  h10_2s1b = (TH1D*)result_gem_2s1b->Clone("gem_new_2s1b_10");
  for (int b = h10_2s1b->FindBin(10.01); b < h10_2s1b->FindBin(15.01); ++b) {hh_2s1b->SetBinContent(b, h10_2s1b->GetBinContent(b)); hh_2s1b->SetBinError(b, h10_2s1b->GetBinError(b));}

  getPTHistos("minbias_pt15_pat2");
  h15 = (TH1D*)result_gem->Clone("gem15");
  for (int b = h15->FindBin(15.01); b < h15->FindBin(20.01); ++b) {hh->SetBinContent(b, h15->GetBinContent(b)); hh->SetBinError(b, h15->GetBinError(b));}
  h15_all = (TH1D*)result_gem_eta_all->Clone("gem_new_eta_all_15");
  for (int b = h15_all->FindBin(15.01); b < h15_all->FindBin(20.01); ++b) {hh_all->SetBinContent(b, h15_all->GetBinContent(b)); hh_all->SetBinError(b, h15_all->GetBinError(b));}
  h15_no1a = (TH1D*)result_gem_eta_no1a->Clone("gem_new_eta_no1a_15");
  for (int b = h15_no1a->FindBin(15.01); b < h15_no1a->FindBin(20.01); ++b) {hh_no1a->SetBinContent(b, h15_no1a->GetBinContent(b)); hh_no1a->SetBinError(b, h15_no1a->GetBinError(b));}
  h15_2s1b = (TH1D*)result_gem_2s1b->Clone("gem_new_2s1b_15");
  for (int b = h15_2s1b->FindBin(15.01); b < h15_2s1b->FindBin(20.01); ++b) {hh_2s1b->SetBinContent(b, h15_2s1b->GetBinContent(b)); hh_2s1b->SetBinError(b, h15_2s1b->GetBinError(b));}

  getPTHistos("minbias_pt20_pat2");
  h20 = (TH1D*)result_gem->Clone("gem20");
  for (int b = h20->FindBin(20.01); b < h20->FindBin(30.01); ++b) {hh->SetBinContent(b, h20->GetBinContent(b)); hh->SetBinError(b, h20->GetBinError(b));}
  h20_all = (TH1D*)result_gem_eta_all->Clone("gem_new_eta_all_20");
  for (int b = h20_all->FindBin(20.01); b < h20_all->FindBin(30.01); ++b) {hh_all->SetBinContent(b, h20_all->GetBinContent(b)); hh_all->SetBinError(b, h20_all->GetBinError(b));}
  h20_no1a = (TH1D*)result_gem_eta_no1a->Clone("gem_new_eta_no1a_20");
  for (int b = h20_no1a->FindBin(20.01); b < h20_no1a->FindBin(30.01); ++b) {hh_no1a->SetBinContent(b, h20_no1a->GetBinContent(b)); hh_no1a->SetBinError(b, h20_no1a->GetBinError(b));}
  h20_2s1b = (TH1D*)result_gem_2s1b->Clone("gem_new_2s1b_20");
  for (int b = h20_2s1b->FindBin(20.01); b < h20_2s1b->FindBin(30.01); ++b) {hh_2s1b->SetBinContent(b, h20_2s1b->GetBinContent(b)); hh_2s1b->SetBinError(b, h20_2s1b->GetBinError(b));}

  getPTHistos("minbias_pt30_pat2");
  h30 = (TH1D*)result_gem->Clone("gem30");
  for (int b = h30->FindBin(30.01); b <= h30->FindBin(40.01); ++b) {hh->SetBinContent(b, h30->GetBinContent(b)); hh->SetBinError(b, h30->GetBinError(b));}
  h30_all = (TH1D*)result_gem_eta_all->Clone("gem_new_eta_all_30");
  for (int b = h30_all->FindBin(30.01); b < h30_all->FindBin(40.01); ++b) {hh_all->SetBinContent(b, h30_all->GetBinContent(b)); hh_all->SetBinError(b, h30_all->GetBinError(b));}
  h30_no1a = (TH1D*)result_gem_eta_no1a->Clone("gem_new_eta_no1a_30");
  for (int b = h30_no1a->FindBin(30.01); b < h30_no1a->FindBin(40.01); ++b) {hh_no1a->SetBinContent(b, h30_no1a->GetBinContent(b)); hh_no1a->SetBinError(b, h30_no1a->GetBinError(b));}
  h30_2s1b = (TH1D*)result_gem_2s1b->Clone("gem_new_2s1b_30");
  for (int b = h30_2s1b->FindBin(30.01); b < h30_2s1b->FindBin(40.01); ++b) {hh_2s1b->SetBinContent(b, h30_2s1b->GetBinContent(b)); hh_2s1b->SetBinError(b, h30_2s1b->GetBinError(b));}

  getPTHistos("minbias_pt40_pat2");
  h40 = (TH1D*)result_gem->Clone("gem30");
  for (int b = h40->FindBin(40.01); b <= h40->GetNbinsX(); ++b) {hh->SetBinContent(b, h40->GetBinContent(b)); hh->SetBinError(b, h40->GetBinError(b));}
  h40_all = (TH1D*)result_gem_eta_all->Clone("gem_new_eta_all_40");
  for (int b = h40_all->FindBin(40.01); b < h40_all->GetNbinsX(); ++b) {hh_all->SetBinContent(b, h40_all->GetBinContent(b)); hh_all->SetBinError(b, h40_all->GetBinError(b));}
  h40_no1a = (TH1D*)result_gem_eta_no1a->Clone("gem_new_eta_no1a_40");
  for (int b = h40_no1a->FindBin(40.01); b < h40_no1a->GetNbinsX(); ++b) {hh_no1a->SetBinContent(b, h40_no1a->GetBinContent(b)); hh_no1a->SetBinError(b, h40_no1a->GetBinError(b));}
  h40_2s1b = (TH1D*)result_gem_2s1b->Clone("gem_new_2s1b_40");
  for (int b = h40_2s1b->FindBin(40.01); b < h40_2s1b->GetNbinsX(); ++b) {hh_2s1b->SetBinContent(b, h40_2s1b->GetBinContent(b)); hh_2s1b->SetBinError(b, h40_2s1b->GetBinError(b));}

  for (int b = 1; b <= hh->GetNbinsX(); ++b) if (hh->GetBinContent(b)==0) hh->SetBinError(b, 0.);
  for (int b = 1; b <= hh_all->GetNbinsX(); ++b) if (hh_all->GetBinContent(b)==0) hh_all->SetBinError(b, 0.);
  for (int b = 1; b <= hh_no1a->GetNbinsX(); ++b) if (hh_no1a->GetBinContent(b)==0) hh_no1a->SetBinError(b, 0.);
  for (int b = 1; b <= hh_2s1b->GetNbinsX(); ++b) if (hh_2s1b->GetBinContent(b)==0) hh_2s1b->SetBinError(b, 0.);

  hh = setPTHisto(hh, the_ttl, kGreen+3, 1, 1);
  hh_all = setPTHisto(hh_all, the_ttl, kGreen+3, 1, 1);
  hh_no1a = setPTHisto(hh_no1a, the_ttl, kGreen+3, 1, 1);
  hh_2s1b = setPTHisto(hh_2s1b, the_ttl, kGreen+3, 1, 1);

  result_gmtsing = setPTHisto(result_gmtsing, the_ttl, kAzure+1, 1, 1);

  result_def = setPTHisto(result_def, the_ttl, kAzure+9, 1, 1);
  result_def_2s = setPTHisto(result_def_2s, the_ttl, kAzure+9, 1, 1);
  result_def_3s1b = setPTHisto(result_def_3s1b, the_ttl, kAzure+9, 1, 1);
  result_def_2s1b = setPTHisto(result_def_2s1b, the_ttl, kAzure+9, 1, 1);
  result_def_eta_all = setPTHisto(result_def_eta_all, the_ttl, kAzure+9, 1, 1);
  result_def_eta_all_3s1b = setPTHisto(result_def_eta_all_3s1b, the_ttl, kAzure+9, 1, 1);
  result_def_eta_no1a = setPTHisto(result_def_eta_no1a, the_ttl, kAzure+9, 1, 1);
  result_def_eta_no1a_3s1b = setPTHisto(result_def_eta_no1a_3s1b, the_ttl, kAzure+9, 1, 1);

  result_def_2s__pat2 = (TH1D*) result_def_2s->Clone("result_def_2s__pat2");
  result_def_3s__pat2 = (TH1D*) result_def->Clone("result_def_2s__pat2");
  result_def_2s1b__pat2 = (TH1D*) result_def_2s1b->Clone("result_def_2s1b__pat2");
  result_def_3s1b__pat2 = (TH1D*) result_def_3s1b->Clone("result_def_3s1b__pat2");
  result_gmtsing__pat2 = (TH1D*) result_gmtsing->Clone("result_gmtsing__pat2");;

  result_gem_2s1b__pat2 = (TH1D*) hh_2s1b->Clone("result_gem_2s1b__pat2");
  result_gem_3s1b__pat2 = (TH1D*) hh->Clone("result_gem_2s1b__pat2");

  // ********** PAT8 **********

  getPTHistos("minbias_pt06_pat8");
  hh = (TH1D*)result_def_3s1b->Clone("gem_new");
  for (int b = hh->FindBin(6.01); b <= hh->GetNbinsX(); ++b) hh->SetBinContent(b, 0);
  hh_all = (TH1D*)result_def_eta_all_3s1b->Clone("gem_new_eta_all");
  for (int b = hh_all->FindBin(6.01); b <= hh_all->GetNbinsX(); ++b) hh_all->SetBinContent(b, 0);
  hh_no1a = (TH1D*)result_def_eta_no1a_3s1b->Clone("gem_new_eta_no1a");
  for (int b = hh_no1a->FindBin(6.01); b <= hh_no1a->GetNbinsX(); ++b) hh_no1a->SetBinContent(b, 0);
  hh_2s1b = (TH1D*)result_def_2s1b->Clone("gem_new_2s1b");
  for (int b = hh_2s1b->FindBin(6.01); b <= hh_2s1b->GetNbinsX(); ++b) hh_2s1b->SetBinContent(b, 0);

  h06 = (TH1D*)result_gem->Clone("gem_new_06");
  for (int b = h06->FindBin(6.01); b < h06->FindBin(10.01); ++b) {hh->SetBinContent(b, h06->GetBinContent(b)); hh->SetBinError(b, h06->GetBinError(b));}
  h06_all = (TH1D*)result_gem_eta_all->Clone("gem_new_eta_all_06");
  for (int b = h06_all->FindBin(6.01); b < h06_all->FindBin(10.01); ++b) {hh_all->SetBinContent(b, h06_all->GetBinContent(b)); hh_all->SetBinError(b, h06_all->GetBinError(b));}
  h06_no1a = (TH1D*)result_gem_eta_no1a->Clone("gem_new_eta_no1a_06");
  for (int b = h06_no1a->FindBin(6.01); b < h06_no1a->FindBin(10.01); ++b) {hh_no1a->SetBinContent(b, h06_no1a->GetBinContent(b)); hh_no1a->SetBinError(b, h06_no1a->GetBinError(b));}
  h06_2s1b = (TH1D*)result_gem_2s1b->Clone("gem_new_2s1b_06");
  for (int b = h06_2s1b->FindBin(6.01); b < h06_2s1b->FindBin(10.01); ++b) {hh_2s1b->SetBinContent(b, h06_2s1b->GetBinContent(b)); hh_2s1b->SetBinError(b, h06_2s1b->GetBinError(b));}

  getPTHistos("minbias_pt10_pat8");
  h10 = (TH1D*)result_gem->Clone("gem10");
  for (int b = h10->FindBin(10.01); b < h10->FindBin(15.01); ++b) {hh->SetBinContent(b, h10->GetBinContent(b)); hh->SetBinError(b, h10->GetBinError(b));}
  h10_all = (TH1D*)result_gem_eta_all->Clone("gem_new_eta_all_10");
  for (int b = h10_all->FindBin(10.01); b < h10_all->FindBin(15.01); ++b) {hh_all->SetBinContent(b, h10_all->GetBinContent(b)); hh_all->SetBinError(b, h10_all->GetBinError(b));}
  h10_no1a = (TH1D*)result_gem_eta_no1a->Clone("gem_new_eta_no1a_10");
  for (int b = h10_no1a->FindBin(10.01); b < h10_no1a->FindBin(15.01); ++b) {hh_no1a->SetBinContent(b, h10_no1a->GetBinContent(b)); hh_no1a->SetBinError(b, h10_no1a->GetBinError(b));}
  h10_2s1b = (TH1D*)result_gem_2s1b->Clone("gem_new_2s1b_10");
  for (int b = h10_2s1b->FindBin(10.01); b < h10_2s1b->FindBin(15.01); ++b) {hh_2s1b->SetBinContent(b, h10_2s1b->GetBinContent(b)); hh_2s1b->SetBinError(b, h10_2s1b->GetBinError(b));}

  getPTHistos("minbias_pt15_pat8");
  h15 = (TH1D*)result_gem->Clone("gem15");
  for (int b = h15->FindBin(15.01); b < h15->FindBin(20.01); ++b) {hh->SetBinContent(b, h15->GetBinContent(b)); hh->SetBinError(b, h15->GetBinError(b));}
  h15_all = (TH1D*)result_gem_eta_all->Clone("gem_new_eta_all_15");
  for (int b = h15_all->FindBin(15.01); b < h15_all->FindBin(20.01); ++b) {hh_all->SetBinContent(b, h15_all->GetBinContent(b)); hh_all->SetBinError(b, h15_all->GetBinError(b));}
  h15_no1a = (TH1D*)result_gem_eta_no1a->Clone("gem_new_eta_no1a_15");
  for (int b = h15_no1a->FindBin(15.01); b < h15_no1a->FindBin(20.01); ++b) {hh_no1a->SetBinContent(b, h15_no1a->GetBinContent(b)); hh_no1a->SetBinError(b, h15_no1a->GetBinError(b));}
  h15_2s1b = (TH1D*)result_gem_2s1b->Clone("gem_new_2s1b_15");
  for (int b = h15_2s1b->FindBin(15.01); b < h15_2s1b->FindBin(20.01); ++b) {hh_2s1b->SetBinContent(b, h15_2s1b->GetBinContent(b)); hh_2s1b->SetBinError(b, h15_2s1b->GetBinError(b));}

  getPTHistos("minbias_pt20_pat8");
  h20 = (TH1D*)result_gem->Clone("gem20");
  for (int b = h20->FindBin(20.01); b < h20->FindBin(30.01); ++b) {hh->SetBinContent(b, h20->GetBinContent(b)); hh->SetBinError(b, h20->GetBinError(b));}
  h20_all = (TH1D*)result_gem_eta_all->Clone("gem_new_eta_all_20");
  for (int b = h20_all->FindBin(20.01); b < h20_all->FindBin(30.01); ++b) {hh_all->SetBinContent(b, h20_all->GetBinContent(b)); hh_all->SetBinError(b, h20_all->GetBinError(b));}
  h20_no1a = (TH1D*)result_gem_eta_no1a->Clone("gem_new_eta_no1a_20");
  for (int b = h20_no1a->FindBin(20.01); b < h20_no1a->FindBin(30.01); ++b) {hh_no1a->SetBinContent(b, h20_no1a->GetBinContent(b)); hh_no1a->SetBinError(b, h20_no1a->GetBinError(b));}
  h20_2s1b = (TH1D*)result_gem_2s1b->Clone("gem_new_2s1b_20");
  for (int b = h20_2s1b->FindBin(20.01); b < h20_2s1b->FindBin(30.01); ++b) {hh_2s1b->SetBinContent(b, h20_2s1b->GetBinContent(b)); hh_2s1b->SetBinError(b, h20_2s1b->GetBinError(b));}

  getPTHistos("minbias_pt30_pat8");
  h30 = (TH1D*)result_gem->Clone("gem30");
  for (int b = h30->FindBin(30.01); b <= h30->FindBin(40.01); ++b) {hh->SetBinContent(b, h30->GetBinContent(b)); hh->SetBinError(b, h30->GetBinError(b));}
  h30_all = (TH1D*)result_gem_eta_all->Clone("gem_new_eta_all_30");
  for (int b = h30_all->FindBin(30.01); b < h30_all->FindBin(40.01); ++b) {hh_all->SetBinContent(b, h30_all->GetBinContent(b)); hh_all->SetBinError(b, h30_all->GetBinError(b));}
  h30_no1a = (TH1D*)result_gem_eta_no1a->Clone("gem_new_eta_no1a_30");
  for (int b = h30_no1a->FindBin(30.01); b < h30_no1a->FindBin(40.01); ++b) {hh_no1a->SetBinContent(b, h30_no1a->GetBinContent(b)); hh_no1a->SetBinError(b, h30_no1a->GetBinError(b));}
  h30_2s1b = (TH1D*)result_gem_2s1b->Clone("gem_new_2s1b_30");
  for (int b = h30_2s1b->FindBin(30.01); b < h30_2s1b->FindBin(40.01); ++b) {hh_2s1b->SetBinContent(b, h30_2s1b->GetBinContent(b)); hh_2s1b->SetBinError(b, h30_2s1b->GetBinError(b));}

  getPTHistos("minbias_pt40_pat8");
  h40 = (TH1D*)result_gem->Clone("gem30");
  for (int b = h40->FindBin(40.01); b <= h40->GetNbinsX(); ++b) {hh->SetBinContent(b, h40->GetBinContent(b)); hh->SetBinError(b, h40->GetBinError(b));}
  h40_all = (TH1D*)result_gem_eta_all->Clone("gem_new_eta_all_40");
  for (int b = h40_all->FindBin(40.01); b < h40_all->GetNbinsX(); ++b) {hh_all->SetBinContent(b, h40_all->GetBinContent(b)); hh_all->SetBinError(b, h40_all->GetBinError(b));}
  h40_no1a = (TH1D*)result_gem_eta_no1a->Clone("gem_new_eta_no1a_40");
  for (int b = h40_no1a->FindBin(40.01); b < h40_no1a->GetNbinsX(); ++b) {hh_no1a->SetBinContent(b, h40_no1a->GetBinContent(b)); hh_no1a->SetBinError(b, h40_no1a->GetBinError(b));}
  h40_2s1b = (TH1D*)result_gem_2s1b->Clone("gem_new_2s1b_40");
  for (int b = h40_2s1b->FindBin(40.01); b < h40_2s1b->GetNbinsX(); ++b) {hh_2s1b->SetBinContent(b, h40_2s1b->GetBinContent(b)); hh_2s1b->SetBinError(b, h40_2s1b->GetBinError(b));}

  for (int b = 1; b <= hh->GetNbinsX(); ++b) if (hh->GetBinContent(b)==0) hh->SetBinError(b, 0.);
  for (int b = 1; b <= hh_all->GetNbinsX(); ++b) if (hh_all->GetBinContent(b)==0) hh_all->SetBinError(b, 0.);
  for (int b = 1; b <= hh_no1a->GetNbinsX(); ++b) if (hh_no1a->GetBinContent(b)==0) hh_no1a->SetBinError(b, 0.);
  for (int b = 1; b <= hh_2s1b->GetNbinsX(); ++b) if (hh_2s1b->GetBinContent(b)==0) hh_2s1b->SetBinError(b, 0.);

  hh = setPTHisto(hh, the_ttl, kGreen+3, 1, 1);
  hh_all = setPTHisto(hh_all, the_ttl, kGreen+3, 1, 1);
  hh_no1a = setPTHisto(hh_no1a, the_ttl, kGreen+3, 1, 1);
  hh_2s1b = setPTHisto(hh_2s1b, the_ttl, kGreen+3, 1, 1);

  result_gmtsing = setPTHisto(result_gmtsing, the_ttl, kAzure+1, 1, 1);

  result_def = setPTHisto(result_def, the_ttl, kAzure+9, 1, 1);
  result_def_2s = setPTHisto(result_def_2s, the_ttl, kAzure+9, 1, 1);
  result_def_3s1b = setPTHisto(result_def_3s1b, the_ttl, kAzure+9, 1, 1);
  result_def_2s1b = setPTHisto(result_def_2s1b, the_ttl, kAzure+9, 1, 1);
  result_def_eta_all = setPTHisto(result_def_eta_all, the_ttl, kAzure+9, 1, 1);
  result_def_eta_all_3s1b = setPTHisto(result_def_eta_all_3s1b, the_ttl, kAzure+9, 1, 1);
  result_def_eta_no1a = setPTHisto(result_def_eta_no1a, the_ttl, kAzure+9, 1, 1);
  result_def_eta_no1a_3s1b = setPTHisto(result_def_eta_no1a_3s1b, the_ttl, kAzure+9, 1, 1);

  result_def_2s__pat8 = (TH1D*) result_def_2s->Clone("result_def_2s__pat8");
  result_def_3s__pat8 = (TH1D*) result_def->Clone("result_def_2s__pat8");
  result_def_2s1b__pat8 = (TH1D*) result_def_2s1b->Clone("result_def_2s1b__pat8");
  result_def_3s1b__pat8 = (TH1D*) result_def_3s1b->Clone("result_def_3s1b__pat8");
  result_gmtsing__pat8 = (TH1D*) result_gmtsing->Clone("result_gmtsing__pat8");;

  result_gem_2s1b__pat8 = (TH1D*) hh_2s1b->Clone("result_gem_2s1b__pat8");
  result_gem_3s1b__pat8 = (TH1D*) hh->Clone("result_gem_2s1b__pat8");

  ////////////////////////
  // PLOTS FOR APPROVAL //
  ////////////////////////
  {
    result_gmtsing__pat2->SetFillColor(kRed);
    result_gmtsing__pat8->SetFillColor(kRed);

    result_def_2s__pat2->SetFillColor(kViolet+1);
    result_def_2s1b__pat2->SetFillColor(kAzure+1);
    result_gem_2s1b__pat2->SetFillColor(kGreen-2);

    result_def_3s__pat2->SetFillColor(kViolet+1);
    result_def_3s1b__pat2->SetFillColor(kAzure+1);
    result_gem_3s1b__pat2->SetFillColor(kGreen-2);

    result_def_2s__pat8->SetFillColor(kViolet+1);
    result_def_2s1b__pat8->SetFillColor(kAzure+1);
    result_gem_2s1b__pat8->SetFillColor(kGreen-2);

    result_def_3s__pat8->SetFillColor(kViolet+1);
    result_def_3s1b__pat8->SetFillColor(kAzure+1);
    result_gem_3s1b__pat8->SetFillColor(kGreen-2);

    // GMT; CSCTF 2 stubs; CSCTF 2 stubs + ME1/b; CSCTF 2 stubs + ME1/b + GEM -- LOOSE -- Absolute + ratio
    TCanvas* c = new TCanvas("c","c",800,800);
    c->Clear();
    TPad *pad1 = new TPad("pad1","top pad",0.0,0.25,1.0,1.0);
    pad1->Draw();
    TPad *pad2 = new TPad("pad2","bottom pad",0,0.,1.0,.30);
    pad2->Draw();

    pad1->cd();
    pad1->SetLogx(1);
    pad1->SetLogy(1);
    pad1->SetGridx(1);
    pad1->SetGridy(1);
    pad1->SetFrameBorderMode(0);
    pad1->SetFillColor(kWhite);

    result_gmtsing__pat2->Draw("e3");
    result_def_2s__pat2->Draw("same e3");
    result_def_2s1b__pat2->Draw("same e3");
    result_gem_2s1b__pat2->Draw("same e3");
    result_gmtsing__pat2->Draw("same e3");
    result_gmtsing__pat2->GetYaxis()->SetRangeUser(0.1, 10000.);
    result_gmtsing__pat2->GetXaxis()->SetTitle("");
 
    TLegend *leg0 = new TLegend(0.15,0.88,.9,0.93,NULL,"brNDC");
    leg0->SetBorderSize(0);
    leg0->SetFillStyle(0);
    leg0->SetTextSize(0.04);
    leg0->SetFillStyle(1001);
    leg0->SetFillColor(kWhite);
    leg0->AddEntry(result_gmtsing__pat2,"Single muon trigger in 2012 configuration [GMT]","f");
    leg0->Draw();
 
    TLegend *leg = new TLegend(0.45,0.68,.93,0.86,NULL,"brNDC");
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextSize(0.04);
    leg->SetFillStyle(1001);
    leg->SetFillColor(kWhite);
    leg->AddEntry((TObject*)0,          "L1 Selections (#geq2 stations):","");
    leg->AddEntry(result_def_2s__pat2,  "CSC, loose","f");
    leg->AddEntry(result_def_2s1b__pat2,"CSC, tight","f");
    leg->AddEntry(result_gem_2s1b__pat2,"GEM+CSC Integrated Trigger","f");
    leg->Draw();

    drawLumiLabel(0.17,.3);

    pad2->cd();
    pad2->SetLogx(1);
    pad2->SetLogy(1);
    pad2->SetGridx(1);
    pad2->SetGridy(1);
    pad2->SetFillColor(kWhite);
    pad2->SetFrameBorderMode(0);
    pad2->SetLeftMargin(0.126);
    pad2->SetRightMargin(0.04);
    pad2->SetTopMargin(0.06);
    pad2->SetBottomMargin(0.4);
    
    hh_ratio = setHistoRatio(result_gem_2s1b__pat2, result_def_2s1b__pat2, "", 0.01,1.1,kAzure+1);

    hh_ratio->GetXaxis()->SetTitle("p_{T}^{cut} [GeV/c]");
    hh_ratio->Draw("P");

    hh_ratio_gmt = setHistoRatio(result_gem_2s1b__pat2, result_gmtsing__pat2, "", 0.01,1.1,kRed);
    hh_ratio_gmt->Draw("P same");

    leg = new TLegend(0.15,0.45,.5,0.7,NULL,"brNDC");
    leg->SetBorderSize(0);
    leg->SetTextSize(0.1);
    leg->SetFillStyle(1001);
    leg->SetFillColor(kWhite);
    leg->AddEntry(hh_ratio_gmt, "GEM+CSC/GMT","p");
    leg->AddEntry(hh_ratio,     "GEM+CSC/CSC tight","p");
    leg->Draw("same");

    c->SaveAs(plots + "rates_vs_pt__PU100__def_2s_2s1b_2s1bgem__loose" + ext);
  }
  {
    // GMT; CSCTF 2 stubs; CSCTF 2 stubs + ME1/b; CSCTF 2 stubs + ME1/b + GEM -- TIGHT + GEM -- Absolute
    pad1->cd();
    pad1->SetLogx(1);
    pad1->SetLogy(1);
    pad1->SetGridx(1);
    pad1->SetGridy(1);
    pad1->SetFrameBorderMode(0);
    pad1->SetFillColor(kWhite);

    result_gmtsing__pat8->Draw("e3");
    result_def_2s__pat8->Draw("same e3");
    result_def_2s1b__pat8->Draw("same e3");
    result_gem_2s1b__pat8->Draw("same e3");
    result_gmtsing__pat8->Draw("same e3");
    result_gmtsing__pat8->GetYaxis()->SetRangeUser(0.1, 10000.);
    result_gmtsing__pat8->GetXaxis()->SetTitle("");
 
    TLegend *leg0 = new TLegend(0.15,0.88,.9,0.93,NULL,"brNDC");
    leg0->SetBorderSize(0);
    leg0->SetFillStyle(0);
    leg0->SetTextSize(0.04);
    leg0->SetFillStyle(1001);
    leg0->SetFillColor(kWhite);
    leg0->AddEntry(result_gmtsing__pat8,"Single muon trigger in 2012 configuration [GMT]","f");
    leg0->Draw();
 
    TLegend *leg = new TLegend(0.45,0.68,.93,0.86,NULL,"brNDC");
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextSize(0.04);
    leg->SetFillStyle(1001);
    leg->SetFillColor(kWhite);
    leg->AddEntry((TObject*)0,          "L1 Selections (#geq2 stations):","");
    leg->AddEntry(result_def_2s__pat8,  "CSC, loose","f");
    leg->AddEntry(result_def_2s1b__pat8,"CSC, tight","f");
    leg->AddEntry(result_gem_2s1b__pat8,"GEM+CSC Integrated Trigger","f");
    leg->Draw();

    drawLumiLabel(0.17,.3);

    pad2->cd();
    pad2->SetLogx(1);
    pad2->SetLogy(1);
    pad2->SetGridx(1);
    pad2->SetGridy(1);
    pad2->SetFillColor(kWhite);
    pad2->SetFrameBorderMode(0);
    pad2->SetLeftMargin(0.126);
    pad2->SetRightMargin(0.04);
    pad2->SetTopMargin(0.06);
    pad2->SetBottomMargin(0.4);
    
    hh_ratio = setHistoRatio(result_gem_2s1b__pat8, result_def_2s1b__pat8, "", 0.01,1.1,kAzure+1);
    hh_ratio->GetXaxis()->SetTitle("p_{T}^{cut} [GeV/c]");
    hh_ratio->Draw("P");
    hh_ratio_gmt = setHistoRatio(result_gem_2s1b__pat8, result_gmtsing__pat8, "", 0.01,1.1,kRed);
    hh_ratio_gmt->Draw("P same");

    leg = new TLegend(0.15,0.45,.5,0.7,NULL,"brNDC");
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextSize(0.1);
    leg->SetFillStyle(1001);
    leg->SetFillColor(kWhite);
    leg->AddEntry(hh_ratio_gmt, "GEM+CSC/GMT","p");
    leg->AddEntry(hh_ratio,     "GEM+CSC/CSC tight","p");
    leg->Draw("same");

    c->SaveAs(plots + "rates_vs_pt__PU100__def_2s_2s1b_2s1bgem__tight" + ext);
  }
  {
    // GMT; CSCTF 3 stubs; CSCTF 3 stubs + ME1/b; CSCTF 3 stubs + ME1/b + GEM -- LOOSE -- Absolute
    pad1->cd();
    pad1->SetLogx(1);
    pad1->SetLogy(1);
    pad1->SetGridx(1);
    pad1->SetGridy(1);
    pad1->SetFrameBorderMode(0);
    pad1->SetFillColor(kWhite);

    result_gmtsing__pat2->Draw("e3");
    result_def_3s__pat2->Draw("same e3");
    result_def_3s1b__pat2->Draw("same e3");
    result_gem_3s1b__pat2->Draw("same e3");
    result_gmtsing__pat2->Draw("same e3");
    result_gmtsing__pat2->GetYaxis()->SetRangeUser(0.01, 10000.);
    result_gmtsing__pat2->GetXaxis()->SetTitle("");
 
    TLegend *leg0 = new TLegend(0.15,0.88,.9,0.93,NULL,"brNDC");
    leg0->SetBorderSize(0);
    leg0->SetFillStyle(0);
    leg0->SetTextSize(0.04);
    leg0->SetFillStyle(1001);
    leg0->SetFillColor(kWhite);
    leg0->AddEntry(result_gmtsing__pat2,"Single muon trigger in 2012 configuration [GMT]","f");
    leg0->Draw();
 
    TLegend *leg = new TLegend(0.45,0.68,.93,0.86,NULL,"brNDC");
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextSize(0.04);
    leg->SetFillStyle(1001);
    leg->SetFillColor(kWhite);
    leg->AddEntry((TObject*)0,          "L1 Selections (#geq3 stations):","");
    leg->AddEntry(result_def_3s__pat2,  "CSC, loose","f");
    leg->AddEntry(result_def_3s1b__pat2,"CSC, tight","f");
    leg->AddEntry(result_gem_3s1b__pat2,"GEM+CSC Integrated Trigger","f");
    leg->Draw();

    drawLumiLabel(0.17,.3);

    pad2->cd();
    pad2->SetLogx(1);
    pad2->SetLogy(1);
    pad2->SetGridx(1);
    pad2->SetGridy(1);
    pad2->SetFillColor(kWhite);
    pad2->SetFrameBorderMode(0);
    pad2->SetLeftMargin(0.126);
    pad2->SetRightMargin(0.04);
    pad2->SetTopMargin(0.06);
    pad2->SetBottomMargin(0.4);
    
    hh_ratio = setHistoRatio(result_gem_3s1b__pat2, result_def_3s1b__pat2, "", 0.01,1.1,kAzure+1);
    hh_ratio->GetXaxis()->SetTitle("p_{T}^{cut} [GeV/c]");
    hh_ratio->Draw("P");
    hh_ratio_gmt = setHistoRatio(result_gem_3s1b__pat2, result_gmtsing__pat2, "", 0.01,1.1,kRed);
    hh_ratio_gmt->Draw("P same");

    leg = new TLegend(0.15,0.45,.5,0.7,NULL,"brNDC");
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextSize(0.1);
    leg->SetFillStyle(1001);
    leg->SetFillColor(kWhite);
    leg->AddEntry(hh_ratio_gmt, "GEM+CSC/GMT","p");
    leg->AddEntry(hh_ratio,     "GEM+CSC/CSC tight","p");
    leg->Draw("same");

    c->SaveAs(plots + "rates_vs_pt__PU100__def_3s_3s1b_3s1bgem__loose" + ext);
  }
  {
    // GMT; CSCTF 3 stubs; CSCTF 3 stubs + ME1/b; CSCTF 3 stubs + ME1/b + GEM -- TIGHT + GEM -- Absolute
    pad1->cd();
    pad1->SetLogx(1);
    pad1->SetLogy(1);
    pad1->SetGridx(1);
    pad1->SetGridy(1);
    pad1->SetFrameBorderMode(0);
    pad1->SetFillColor(kWhite);

    result_gmtsing__pat8->Draw("e3");
    result_def_3s__pat8->Draw("same e3");
    result_def_3s1b__pat8->Draw("same e3");
    result_gem_3s1b__pat8->Draw("same e3");
    result_gmtsing__pat8->Draw("same e3");
    result_gmtsing__pat8->GetYaxis()->SetRangeUser(0.01, 10000.);
    result_gmtsing__pat8->GetXaxis()->SetTitle("");

    TLegend *leg0 = new TLegend(0.15,0.88,.9,0.93,NULL,"brNDC");
    leg0->SetBorderSize(0);
    leg0->SetFillStyle(0);
    leg0->SetTextSize(0.04);
    leg0->SetFillStyle(1001);
    leg0->SetFillColor(kWhite);
    leg0->AddEntry(result_gmtsing__pat8,"Single muon trigger in 2012 configuration [GMT]","f");
    leg0->Draw();
 
    TLegend *leg = new TLegend(0.45,0.68,.93,0.86,NULL,"brNDC");
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextSize(0.04);
    leg->SetFillStyle(1001);
    leg->SetFillColor(kWhite);
    leg->AddEntry((TObject*)0,          "L1 Selections (#geq3 stations):","");
    leg->AddEntry(result_def_3s__pat8,  "CSC, loose","f");
    leg->AddEntry(result_def_3s1b__pat8,"CSC, tight","f");
    leg->AddEntry(result_gem_3s1b__pat8,"GEM+CSC Integrated Trigger","f");
    leg->Draw();

    drawLumiLabel(0.17,.3);

    pad2->cd();
    pad2->SetLogx(1);
    pad2->SetLogy(1);
    pad2->SetGridx(1);
    pad2->SetGridy(1);
    pad2->SetFillColor(kWhite);
    pad2->SetFrameBorderMode(0);
    pad2->SetLeftMargin(0.126);
    pad2->SetRightMargin(0.04);
    pad2->SetTopMargin(0.06);
    pad2->SetBottomMargin(0.4);
    
    hh_ratio = setHistoRatio(result_gem_3s1b__pat8, result_def_3s1b__pat8, "", 0.01,1.1,kAzure+1);
    hh_ratio->GetXaxis()->SetTitle("p_{T}^{cut} [GeV/c]");
    hh_ratio->Draw("P");
    hh_ratio_gmt = setHistoRatio(result_gem_3s1b__pat8, result_gmtsing__pat8, "", 0.01,1.1,kRed);
    hh_ratio_gmt->Draw("P same");

    leg = new TLegend(0.15,0.45,.5,0.7,NULL,"brNDC");
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextSize(0.1);
    leg->SetFillStyle(1001);
    leg->SetFillColor(kWhite);
    leg->AddEntry(hh_ratio_gmt, "GEM+CSC/GMT","p");
    leg->AddEntry(hh_ratio,     "GEM+CSC/CSC tight","p");
    leg->Draw("same");

    c->SaveAs(plots + "rates_vs_pt__PU100__def_3s_3s1b_3s1bgem__tight" + ext);
  }

  // EXTRA PLOTS
  {
    result_def_2s__pat8->SetFillColor(kViolet+2);
    result_def_2s1b__pat8->SetFillColor(kAzure+2);
    result_gem_2s1b__pat8->SetFillColor(kGreen-1);

    // GMT; CSCTF 2 stubs; CSCTF 2 stubs + ME1/b; CSCTF 2 stubs + ME1/b + GEM -- LOOSE & TIGHT + GEM-- Absolute + ratio
    TCanvas* c = new TCanvas("c","c",1000,800);
    c->Clear();
    TPad *pad1 = new TPad("pad1","top pad",0.0,0.0,1.0,1.0);
    pad1->Draw();

    pad1->cd();
    pad1->SetLogx(1);
    pad1->SetLogy(1);
    pad1->SetGridx(1);
    pad1->SetGridy(1);
    pad1->SetFrameBorderMode(0);
    pad1->SetFillColor(kWhite);

    result_gmtsing__pat2->Draw("e3");
    result_def_2s__pat2->Draw("same e3");
    result_def_2s1b__pat2->Draw("same e3");
    result_gem_2s1b__pat2->Draw("same e3");
    result_def_2s__pat8->Draw("same e3");
    result_def_2s1b__pat8->Draw("same e3");
    result_gem_2s1b__pat8->Draw("same e3");
    result_gmtsing__pat2->Draw("same e3");
    result_gmtsing__pat2->GetYaxis()->SetRangeUser(0.1, 10000.);
    result_gmtsing__pat2->GetXaxis()->SetTitle("p_{T}^{cut} [GeV/c]");
 
    TLegend *leg = new TLegend(0.5,0.65,.92,0.92,NULL,"brNDC");
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextSize(0.03);
    leg->AddEntry((TObject*)0, "Global Muon Trigger [GMT]:","");
    leg->AddEntry(result_gmtsing__pat2, "default muon selection","f");
    leg->AddEntry((TObject*)0,          "CSCTF tracks with:","");
    leg->AddEntry(result_def_2s__pat2,  "#geq 2 stubs","f");
    leg->AddEntry(result_def_2s1b__pat2,"#geq 2 with ME1/b stub","f");
    leg->AddEntry(result_gem_2s1b__pat2,"#geq 2 with ME1/b stub and GEM pad","f");
    leg->AddEntry(result_def_2s__pat8,  "#geq 2 stubs","f");
    leg->AddEntry(result_def_2s1b__pat8,"#geq 2 with ME1/b stub","f");
    leg->AddEntry(result_gem_2s1b__pat8,"#geq 2 with ME1/b stub and GEM pad","f");
    leg->SetFillStyle(1001);
    leg->SetFillColor(kWhite);
    leg->Draw();
 
    drawLumiLabel(0.17,.3);

    c->SaveAs(plots + "rates_vs_pt__PU100__def_2s_2s1b_2s1bgem" + ext);
  }

  {
    result_def_3s__pat8->SetFillColor(kViolet+2);
    result_def_3s1b__pat8->SetFillColor(kAzure+2);
    result_gem_3s1b__pat8->SetFillColor(kGreen-1);

    // GMT; CSCTF 3 stubs; CSCTF 3 stubs + ME1/b; CSCTF 3 stubs + ME1/b + GEM -- LOOSE & TIGHT + GEM-- Absolute + ratio
    TCanvas* c = new TCanvas("c","c",1000,800);
    c->Clear();
    TPad *pad1 = new TPad("pad1","top pad",0.0,0.0,1.0,1.0);
    pad1->Draw();

    pad1->cd();
    pad1->SetLogx(1);
    pad1->SetLogy(1);
    pad1->SetGridx(1);
    pad1->SetGridy(1);
    pad1->SetFrameBorderMode(0);
    pad1->SetFillColor(kWhite);

    result_gmtsing__pat2->Draw("e3");
    result_def_3s__pat2->Draw("same e3");
    result_def_3s1b__pat2->Draw("same e3");
    result_gem_3s1b__pat2->Draw("same e3");
    result_def_3s__pat8->Draw("same e3");
    result_def_3s1b__pat8->Draw("same e3");
    result_gem_3s1b__pat8->Draw("same e3");
    result_gmtsing__pat2->Draw("same e3");
    result_gmtsing__pat2->GetYaxis()->SetRangeUser(0.01, 10000.);
    result_gmtsing__pat2->GetXaxis()->SetTitle("p_{T}^{cut} [GeV/c]");
 
    TLegend *leg = new TLegend(0.5,0.65,.92,0.92,NULL,"brNDC");
    leg->SetBorderSize(0);
    leg->SetFillStyle(0);
    leg->SetTextSize(0.03);
    leg->AddEntry((TObject*)0, "Global Muon Trigger [GMT]:","");
    leg->AddEntry(result_gmtsing__pat2, "default muon selection","f");
    leg->AddEntry((TObject*)0,          "CSCTF tracks with:","");
    leg->AddEntry(result_def_3s__pat2,  "#geq 3 stubs","f");
    leg->AddEntry(result_def_3s1b__pat2,"#geq 3 with ME1/b stub","f");
    leg->AddEntry(result_gem_3s1b__pat2,"#geq 3 with ME1/b stub and GEM pad","f");
    leg->AddEntry(result_def_3s__pat8,  "#geq 3 stubs","f");
    leg->AddEntry(result_def_3s1b__pat8,"#geq 3 with ME1/b stub","f");
    leg->AddEntry(result_gem_3s1b__pat8,"#geq 3 with ME1/b stub and GEM pad","f");
    leg->SetFillStyle(1001);
    leg->SetFillColor(kWhite);
    leg->Draw();
 
    drawLumiLabel(0.17,.3);

    c->SaveAs(plots + "rates_vs_pt__PU100__def_3s_3s1b_3s1bgem" + ext);
  }
}

void produceRateVsPtPlotsForApproval()
{
  gROOT->ProcessLine(".L drawplot_gmtrt.C");
  gROOT->ProcessLine(".L getPTHistos.C");
  gROOT->SetBatch(true);

  produceRatePlots(".pdf");
  produceRatePlots(".eps");
  produceRatePlots(".png");
}
