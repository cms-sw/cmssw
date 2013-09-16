TH1F* eff_base = 0;

TH1F* draw_eff(TTree *t, TString title, TString h_name, TString h_bins,
               TString to_draw, TCut denom_cut, TCut extra_num_cut, TString opt = "", int color = kBlue, int marker_st = 20)
{
  t->Draw(to_draw + ">>num_" + h_name + h_bins, denom_cut && extra_num_cut, "goff");
  TH1F* eff = (TH1F*) gDirectory->Get("num_" + h_name)->Clone("eff_" + h_name);
  
  t->Draw(to_draw + ">>denom_" + h_name + h_bins, denom_cut, "goff");
  TH1F* den = (TH1F*) gDirectory->Get("denom_" + h_name)->Clone("denom_" + h_name);
  
  eff->SetStats(0);
  eff->SetTitle(title);
  eff->SetLineWidth(2);
  eff->SetLineColor(color);
  eff->Divide(eff,den,1,1,"b");
  eff->Draw(opt);
  eff->SetMarkerStyle(marker_st);
  eff->SetMarkerColor(color);
  eff->SetMarkerSize(1.);
  return eff;
}

TGraphAsymmErrors* draw_geff(TTree *t, TString title, TString h_name, TString h_bins,
               TString to_draw, TCut denom_cut, TCut extra_num_cut, TString opt = "", int color = kBlue, int marker_st = 1, float marker_sz = 1.)
{
  t->Draw(to_draw + ">>num_" + h_name + h_bins, denom_cut && extra_num_cut, "goff");
  TH1F* num = (TH1F*) gDirectory->Get("num_" + h_name)->Clone("eff_" + h_name);
  
  t->Draw(to_draw + ">>denom_" + h_name + h_bins, denom_cut, "goff");
  TH1F* den = (TH1F*) gDirectory->Get("denom_" + h_name)->Clone("denom_" + h_name);
  
  TGraphAsymmErrors *eff = new TGraphAsymmErrors(num, den);
  
  if (!opt.Contains("same")) {
  num->Reset();
  num->GetYaxis()->SetRangeUser(0.0,1.0);
  num->SetStats(0);
  num->SetTitle(title);
  num->Draw();
  eff_base = num;
  }
  eff->SetLineWidth(2);
  eff->SetLineColor(color);
  eff->Draw(opt + " same");
  eff->SetMarkerStyle(marker_st);
  eff->SetMarkerColor(color);
  eff->SetMarkerSize(marker_sz);

  return eff;
}
