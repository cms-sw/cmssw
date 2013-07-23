/*
.L drawplot_gmtrt.C

*/

{
//gem_dir = "gemPT/"; gem_label = "gem98";

gem_dir = "gemPT95/"; gem_label = "gem95";

// ********** PAT2 **********

drawplot_gmtrt("minbias_pt10_pat2");
hh = (TH1D*)result_gem->Clone("gem_new");
hh->SetFillColor(kGreen+4);
for (int b = hh->FindBin(10.01); b <= hh->GetNbinsX(); ++b) hh->SetBinContent(b, 0);
hh_all = (TH1D*)result_gem_eta_all->Clone("gem_new_eta_all");
hh_all->SetFillColor(kGreen+4);
for (int b = hh_all->FindBin(10.01); b <= hh_all->GetNbinsX(); ++b) hh_all->SetBinContent(b, 0);
hh_no1a = (TH1D*)result_gem_eta_no1a->Clone("gem_new_eta_no1a");
hh_no1a->SetFillColor(kGreen+4);
for (int b = hh_no1a->FindBin(10.01); b <= hh_no1a->GetNbinsX(); ++b) hh_no1a->SetBinContent(b, 0);
hh_2s1b = (TH1D*)result_gem_2s1b->Clone("gem_new_2s1b");
hh_2s1b->SetFillColor(kGreen+4);
for (int b = hh_2s1b->FindBin(10.01); b <= hh_2s1b->GetNbinsX(); ++b) hh_2s1b->SetBinContent(b, 0);

drawplot_gmtrt("minbias_pt15_pat2");
h15 = (TH1D*)result_gem->Clone("gem15");
for (int b = h15->FindBin(10.01); b < h15->FindBin(15.01); ++b) hh->SetBinContent(b, h15->GetBinContent(b));
h15_all = (TH1D*)result_gem_eta_all->Clone("gem_new_eta_all_15");
for (int b = h15_all->FindBin(10.01); b < h15_all->FindBin(15.01); ++b) hh_all->SetBinContent(b, h15_all->GetBinContent(b));
h15_no1a = (TH1D*)result_gem_eta_no1a->Clone("gem_new_eta_no1a_15");
for (int b = h15_no1a->FindBin(10.01); b < h15_no1a->FindBin(15.01); ++b) hh_no1a->SetBinContent(b, h15_no1a->GetBinContent(b));
h15_2s1b = (TH1D*)result_gem_2s1b->Clone("gem_new_2s1b_15");
for (int b = h15_2s1b->FindBin(10.01); b < h15_2s1b->FindBin(15.01); ++b) hh_2s1b->SetBinContent(b, h15_2s1b->GetBinContent(b));

drawplot_gmtrt("minbias_pt20_pat2");
h20 = (TH1D*)result_gem->Clone("gem20");
for (int b = h20->FindBin(15.01); b < h20->FindBin(20.01); ++b) hh->SetBinContent(b, h20->GetBinContent(b));
h20_all = (TH1D*)result_gem_eta_all->Clone("gem_new_eta_all_20");
for (int b = h20_all->FindBin(15.01); b < h20_all->FindBin(20.01); ++b) hh_all->SetBinContent(b, h20_all->GetBinContent(b));
h20_no1a = (TH1D*)result_gem_eta_no1a->Clone("gem_new_eta_no1a_20");
for (int b = h20_no1a->FindBin(15.01); b < h20_no1a->FindBin(20.01); ++b) hh_no1a->SetBinContent(b, h20_no1a->GetBinContent(b));
h20_2s1b = (TH1D*)result_gem_2s1b->Clone("gem_new_2s1b_20");
for (int b = h20_2s1b->FindBin(15.01); b < h20_2s1b->FindBin(20.01); ++b) hh_2s1b->SetBinContent(b, h20_2s1b->GetBinContent(b));

drawplot_gmtrt("minbias_pt30_pat2");
h30 = (TH1D*)result_gem->Clone("gem30");
for (int b = h30->FindBin(20.01); b <= h30->FindBin(30.01); ++b) hh->SetBinContent(b, h30->GetBinContent(b));
h30_all = (TH1D*)result_gem_eta_all->Clone("gem_new_eta_all_30");
for (int b = h30_all->FindBin(20.01); b < h30_all->FindBin(30.01); ++b) hh_all->SetBinContent(b, h30_all->GetBinContent(b));
h30_no1a = (TH1D*)result_gem_eta_no1a->Clone("gem_new_eta_no1a_30");
for (int b = h30_no1a->FindBin(20.01); b < h30_no1a->FindBin(30.01); ++b) hh_no1a->SetBinContent(b, h30_no1a->GetBinContent(b));
h30_2s1b = (TH1D*)result_gem_2s1b->Clone("gem_new_2s1b_30");
for (int b = h30_2s1b->FindBin(20.01); b < h30_2s1b->FindBin(30.01); ++b) hh_2s1b->SetBinContent(b, h30_2s1b->GetBinContent(b));

drawplot_gmtrt("minbias_pt40_pat2");
h40 = (TH1D*)result_gem->Clone("gem30");
for (int b = h40->FindBin(30.01); b <= h40->GetNbinsX(); ++b) hh->SetBinContent(b, h40->GetBinContent(b));
h40_all = (TH1D*)result_gem_eta_all->Clone("gem_new_eta_all_40");
for (int b = h40_all->FindBin(30.01); b < h40_all->GetNbinsX(); ++b) hh_all->SetBinContent(b, h40_all->GetBinContent(b));
h40_no1a = (TH1D*)result_gem_eta_no1a->Clone("gem_new_eta_no1a_40");
for (int b = h40_no1a->FindBin(30.01); b < h40_no1a->GetNbinsX(); ++b) hh_no1a->SetBinContent(b, h40_no1a->GetBinContent(b));
h40_2s1b = (TH1D*)result_gem_2s1b->Clone("gem_new_2s1b_40");
for (int b = h40_2s1b->FindBin(30.01); b < h40_2s1b->GetNbinsX(); ++b) hh_2s1b->SetBinContent(b, h40_2s1b->GetBinContent(b));

for (int b = 1; b <= hh->GetNbinsX(); ++b) if (hh->GetBinContent(b)==0) hh->SetBinError(b, 0.);
for (int b = 1; b <= hh_all->GetNbinsX(); ++b) if (hh_all->GetBinContent(b)==0) hh_all->SetBinError(b, 0.);
for (int b = 1; b <= hh_no1a->GetNbinsX(); ++b) if (hh_no1a->GetBinContent(b)==0) hh_no1a->SetBinError(b, 0.);
for (int b = 1; b <= hh_2s1b->GetNbinsX(); ++b) if (hh_2s1b->GetBinContent(b)==0) hh_2s1b->SetBinError(b, 0.);


// --- def-3s   gem-3s-3s1b

((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_def->Draw("e3");
hh->Draw("same e3");

TLegend *leg = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def,"default emulator","f");
leg->AddEntry(result_def,"Tracks: with >=3 stubs in 1.64<|#eta|<2.14","");
leg->AddEntry(hh,"with GEM match","f");
leg->AddEntry(result_def,"Tracks: same, plus req. one stub from ME1/b","");
leg->Draw();

TLatex *  tex = new TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
tex->SetNDC();
tex->Draw();

gPad->Print(gem_dir + "rates__1.6-2.1_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat2.png");


((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
hh_ratio = setHistoRatio(hh, result_def, "", 0.,1.1);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1.6-2.1_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat2__ratio.png");


// --- def-3s-3s1b   gem-3s-3s1b

((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_def_3s1b->Draw("e3");
hh->Draw("same e3");

TLegend *leg = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def_3s1b,"default emulator","f");
leg->AddEntry(hh,"with GEM match","f");
leg->AddEntry(result_def_3s1b,"Tracks req. for both:","");
leg->AddEntry(result_def_3s1b,"with >=3 stubs in 1.64<|#eta|<2.14","");
leg->AddEntry(result_def_3s1b,"and require one stub to be from ME1/b","");
leg->Draw();

TLatex *  tex = new TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
tex->SetNDC();
tex->Draw();

gPad->Print(gem_dir + "rates__1.6-2.1_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat2.png");


((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
hh_ratio = setHistoRatio(hh, result_def_3s1b, "", 0.,1.1);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1.6-2.1_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat2__ratio.png");


// --- def-3s-2s1b   gem-3s-2s1b

((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_def_2s1b->Draw("e3");
hh_2s1b->Draw("same e3");

TLegend *leg = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def_2s1b,"default emulator","f");
leg->AddEntry(hh_2s1b,"with GEM match","f");
leg->AddEntry(result_def_2s1b,"Tracks req. for both:","");
leg->AddEntry(result_def_2s1b,"with >=2 stubs in 1.64<|#eta|<2.14","");
leg->AddEntry(result_def_2s1b,"and require one stub to be from ME1/b","");
leg->Draw();

TLatex *  tex = new TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
tex->SetNDC();
tex->Draw();

gPad->Print(gem_dir + "rates__1.6-2.1_PU100__def-3s-2s1b__gem-3s-2s1b__Frankenstein_pat2.png");


((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
hh_ratio = setHistoRatio(hh_2s1b, result_def_2s1b, "", 0.,1.1);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1.6-2.1_PU100__def-3s-2s1b__gem-3s-2s1b__Frankenstein_pat2__ratio.png");


// --- def-3s   gem-3s-2s1b

((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_def->Draw("e3");
hh_2s1b->Draw("same e3");

TLegend *leg = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def,"default emulator","f");
leg->AddEntry(result_def,"Tracks: with >=3 stubs in 1.64<|#eta|<2.14","");
leg->AddEntry(hh_2s1b,"with GEM match","f");
leg->AddEntry(hh_2s1b,"Tracks: with >=2 stubs in 1.64<|#eta|<2.14","");
leg->AddEntry(hh_2s1b,"and require one stub to be from ME1/b","");
leg->Draw();

TLatex *  tex = new TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
tex->SetNDC();
tex->Draw();

gPad->Print(gem_dir + "rates__1.6-2.1_PU100__def-3s__gem-3s-2s1b__Frankenstein_pat2.png");


((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
hh_ratio = setHistoRatio(hh_2s1b, result_def, "", 0.,1.1);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1.6-2.1_PU100__def-3s__gem-3s-2s1b__Frankenstein_pat2__ratio.png");


// --- def-3s-3s1b   gem-3s-2s1b

((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_def_3s1b->Draw("e3");
hh_2s1b->Draw("same e3");

TLegend *leg = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def_3s1b,"default emulator","f");
leg->AddEntry(result_def_3s1b,"Tracks req. >=3 stubs in 1.64<|#eta|<2.14","");
leg->AddEntry(result_def_3s1b,"and require one stub to be from ME1/b","");
leg->AddEntry(hh_2s1b,"with GEM match","f");
leg->AddEntry(hh_2s1b,"Tracks req. >=2 stubs in 1.64<|#eta|<2.14","");
leg->AddEntry(hh_2s1b,"and require one stub to be from ME1/b","");
leg->Draw();

TLatex *  tex = new TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
tex->SetNDC();
tex->Draw();

gPad->Print(gem_dir + "rates__1.6-2.1_PU100__def-3s-3s1b__gem-3s-2s1b__Frankenstein_pat2.png");


((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
hh_ratio = setHistoRatio(hh_2s1b, result_def_3s1b, "", 0.,3.1);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1.6-2.1_PU100__def-3s-3s1b__gem-3s-2s1b__Frankenstein_pat2__ratio.png");


// --- eta 1-2.4  def-3s   gem-3s-3s1b

((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_def_eta_all->Draw("e3");
hh_all->Draw("same e3");

TLegend *leg = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def_eta_all,"default emulator","f");
leg->AddEntry(result_def_eta_all,"Tracks: with >=3 stubs in 1.<|#eta|<2.4","");
leg->AddEntry(hh_all,"with GEM match","f");
leg->AddEntry(result_def_eta_all,"Tracks: same, plus req. one stub from ME1/b","");
leg->Draw();

TLatex *  tex = new TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
tex->SetNDC();
tex->Draw();

gPad->Print(gem_dir + "rates__1-2.4_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat2.png");


((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
hh_ratio = setHistoRatio(hh_all, result_def_eta_all, "", 0.,1.1);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1-2.4_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat2__ratio.png");


// --- eta 1-2.4  def-3s-3s1b   gem-3s-3s1b

((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_def_eta_all_3s1b->Draw("e3");
hh_all->Draw("same e3");

TLegend *leg = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def_eta_all_3s1b,"default emulator","f");
leg->AddEntry(hh_all,"with GEM match","f");
leg->AddEntry(result_def_eta_all_3s1b,"Tracks req. for both:","");
leg->AddEntry(result_def_eta_all_3s1b,"with >=3 stubs in 1.64<|#eta|<2.14","");
leg->AddEntry(result_def_eta_all_3s1b,"and require one stub to be from ME1/b","");
leg->Draw();

TLatex *  tex = new TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
tex->SetNDC();
tex->Draw();

gPad->Print(gem_dir + "rates__1-2.4_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat2.png");


((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
hh_ratio = setHistoRatio(hh_all, result_def_eta_all_3s1b, "", 0.,1.1);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1-2.4_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat2__ratio.png");


// --- eta 1-2.1 def-3s   gem-3s-3s1b

((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_def_eta_no1a->Draw("e3");
hh_no1a->Draw("same e3");

TLegend *leg = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def_eta_no1a,"default emulator","f");
leg->AddEntry(result_def_eta_no1a,"Tracks: with >=3 stubs in 1.<|#eta|<2.4","");
leg->AddEntry(hh_no1a,"with GEM match","f");
leg->AddEntry(result_def_eta_no1a,"Tracks: same, plus req. one stub from ME1/b","");
leg->Draw();

TLatex *  tex = new TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
tex->SetNDC();
tex->Draw();

gPad->Print(gem_dir + "rates__1-2.1_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat2.png");


((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
hh_ratio = setHistoRatio(hh_no1a, result_def_eta_no1a, "", 0.,1.1);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1-2.1_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat2__ratio.png");


// --- eta 1-2.1  def-3s-3s1b   gem-3s-3s1b

((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_def_eta_no1a_3s1b->Draw("e3");
hh_no1a->Draw("same e3");

TLegend *leg = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def_eta_no1a_3s1b,"default emulator","f");
leg->AddEntry(hh_no1a,"with GEM match","f");
leg->AddEntry(result_def_eta_no1a_3s1b,"Tracks req. for both:","");
leg->AddEntry(result_def_eta_no1a_3s1b,"with >=3 stubs in 1.64<|#eta|<2.14","");
leg->AddEntry(result_def_eta_no1a_3s1b,"and require one stub to be from ME1/b","");
leg->Draw();

TLatex *  tex = new TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
tex->SetNDC();
tex->Draw();

gPad->Print(gem_dir + "rates__1-2.1_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat2.png");


((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
hh_ratio = setHistoRatio(hh_no1a, result_def_eta_no1a_3s1b, "", 0.,1.1);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1-2.1_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat2__ratio.png");







// ********** PAT8 **********

drawplot_gmtrt("minbias_pt10_pat8");
hh = (TH1D*)result_gem->Clone("gem_new");
hh->SetFillColor(kGreen+4);
for (int b = hh->FindBin(10.01); b <= hh->GetNbinsX(); ++b) hh->SetBinContent(b, 0);
hh_all = (TH1D*)result_gem_eta_all->Clone("gem_new_eta_all");
hh_all->SetFillColor(kGreen+4);
for (int b = hh_all->FindBin(10.01); b <= hh_all->GetNbinsX(); ++b) hh_all->SetBinContent(b, 0);
hh_no1a = (TH1D*)result_gem_eta_no1a->Clone("gem_new_eta_no1a");
hh_no1a->SetFillColor(kGreen+4);
for (int b = hh_no1a->FindBin(10.01); b <= hh_no1a->GetNbinsX(); ++b) hh_no1a->SetBinContent(b, 0);
hh_2s1b = (TH1D*)result_gem_2s1b->Clone("gem_new_2s1b");
hh_2s1b->SetFillColor(kGreen+4);
for (int b = hh_2s1b->FindBin(10.01); b <= hh_2s1b->GetNbinsX(); ++b) hh_2s1b->SetBinContent(b, 0);

drawplot_gmtrt("minbias_pt15_pat8");
h15 = (TH1D*)result_gem->Clone("gem15");
for (int b = h15->FindBin(10.01); b < h15->FindBin(15.01); ++b) hh->SetBinContent(b, h15->GetBinContent(b));
h15_all = (TH1D*)result_gem_eta_all->Clone("gem_new_eta_all_15");
for (int b = h15_all->FindBin(10.01); b < h15_all->FindBin(15.01); ++b) hh_all->SetBinContent(b, h15_all->GetBinContent(b));
h15_no1a = (TH1D*)result_gem_eta_no1a->Clone("gem_new_eta_no1a_15");
for (int b = h15_no1a->FindBin(10.01); b < h15_no1a->FindBin(15.01); ++b) hh_no1a->SetBinContent(b, h15_no1a->GetBinContent(b));
h15_2s1b = (TH1D*)result_gem_2s1b->Clone("gem_new_2s1b_15");
for (int b = h15_2s1b->FindBin(10.01); b < h15_2s1b->FindBin(15.01); ++b) hh_2s1b->SetBinContent(b, h15_2s1b->GetBinContent(b));

drawplot_gmtrt("minbias_pt20_pat8");
h20 = (TH1D*)result_gem->Clone("gem20");
for (int b = h20->FindBin(15.01); b < h20->FindBin(20.01); ++b) hh->SetBinContent(b, h20->GetBinContent(b));
h20_all = (TH1D*)result_gem_eta_all->Clone("gem_new_eta_all_20");
for (int b = h20_all->FindBin(15.01); b < h20_all->FindBin(20.01); ++b) hh_all->SetBinContent(b, h20_all->GetBinContent(b));
h20_no1a = (TH1D*)result_gem_eta_no1a->Clone("gem_new_eta_no1a_20");
for (int b = h20_no1a->FindBin(15.01); b < h20_no1a->FindBin(20.01); ++b) hh_no1a->SetBinContent(b, h20_no1a->GetBinContent(b));
h20_2s1b = (TH1D*)result_gem_2s1b->Clone("gem_new_2s1b_20");
for (int b = h20_2s1b->FindBin(15.01); b < h20_2s1b->FindBin(20.01); ++b) hh_2s1b->SetBinContent(b, h20_2s1b->GetBinContent(b));

drawplot_gmtrt("minbias_pt30_pat8");
h30 = (TH1D*)result_gem->Clone("gem30");
for (int b = h30->FindBin(20.01); b <= h30->FindBin(30.01); ++b) hh->SetBinContent(b, h30->GetBinContent(b));
h30_all = (TH1D*)result_gem_eta_all->Clone("gem_new_eta_all_30");
for (int b = h30_all->FindBin(20.01); b < h30_all->FindBin(30.01); ++b) hh_all->SetBinContent(b, h30_all->GetBinContent(b));
h30_no1a = (TH1D*)result_gem_eta_no1a->Clone("gem_new_eta_no1a_30");
for (int b = h30_no1a->FindBin(20.01); b < h30_no1a->FindBin(30.01); ++b) hh_no1a->SetBinContent(b, h30_no1a->GetBinContent(b));
h30_2s1b = (TH1D*)result_gem_2s1b->Clone("gem_new_2s1b_30");
for (int b = h30_2s1b->FindBin(20.01); b < h30_2s1b->FindBin(30.01); ++b) hh_2s1b->SetBinContent(b, h30_2s1b->GetBinContent(b));

drawplot_gmtrt("minbias_pt40_pat8");
h40 = (TH1D*)result_gem->Clone("gem30");
for (int b = h40->FindBin(30.01); b <= h40->GetNbinsX(); ++b) hh->SetBinContent(b, h40->GetBinContent(b));
h40_all = (TH1D*)result_gem_eta_all->Clone("gem_new_eta_all_40");
for (int b = h40_all->FindBin(30.01); b < h40_all->GetNbinsX(); ++b) hh_all->SetBinContent(b, h40_all->GetBinContent(b));
h40_no1a = (TH1D*)result_gem_eta_no1a->Clone("gem_new_eta_no1a_40");
for (int b = h40_no1a->FindBin(30.01); b < h40_no1a->GetNbinsX(); ++b) hh_no1a->SetBinContent(b, h40_no1a->GetBinContent(b));
h40_2s1b = (TH1D*)result_gem_2s1b->Clone("gem_new_2s1b_40");
for (int b = h40_2s1b->FindBin(30.01); b < h40_2s1b->GetNbinsX(); ++b) hh_2s1b->SetBinContent(b, h40_2s1b->GetBinContent(b));

for (int b = 1; b <= hh->GetNbinsX(); ++b) if (hh->GetBinContent(b)==0) hh->SetBinError(b, 0.);
for (int b = 1; b <= hh_all->GetNbinsX(); ++b) if (hh_all->GetBinContent(b)==0) hh_all->SetBinError(b, 0.);
for (int b = 1; b <= hh_no1a->GetNbinsX(); ++b) if (hh_no1a->GetBinContent(b)==0) hh_no1a->SetBinError(b, 0.);
for (int b = 1; b <= hh_2s1b->GetNbinsX(); ++b) if (hh_2s1b->GetBinContent(b)==0) hh_2s1b->SetBinError(b, 0.);


// --- def-3s   gem-3s-3s1b

((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_def->Draw("e3");
hh->Draw("same e3");

TLegend *leg = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def,"default emulator","f");
leg->AddEntry(result_def,"Tracks: with >=3 stubs in 1.64<|#eta|<2.14","");
leg->AddEntry(hh,"with GEM match","f");
leg->AddEntry(result_def,"Tracks: same, plus req. one stub from ME1/b","");
leg->Draw();

TLatex *  tex = new TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
tex->SetNDC();
tex->Draw();

gPad->Print(gem_dir + "rates__1.6-2.1_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat8.png");


((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
hh_ratio = setHistoRatio(hh, result_def, "", 0.,1.1);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1.6-2.1_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat8__ratio.png");


// --- def-3s-3s1b   gem-3s-3s1b

((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_def_3s1b->Draw("e3");
hh->Draw("same e3");

TLegend *leg = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def_3s1b,"default emulator","f");
leg->AddEntry(hh,"with GEM match","f");
leg->AddEntry(result_def_3s1b,"Tracks req. for both:","");
leg->AddEntry(result_def_3s1b,"with >=3 stubs in 1.64<|#eta|<2.14","");
leg->AddEntry(result_def_3s1b,"and require one stub to be from ME1/b","");
leg->Draw();

TLatex *  tex = new TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
tex->SetNDC();
tex->Draw();

gPad->Print(gem_dir + "rates__1.6-2.1_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat8.png");


((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
hh_ratio = setHistoRatio(hh, result_def_3s1b, "", 0.,1.1);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1.6-2.1_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat8__ratio.png");


// --- def-3s-2s1b   gem-3s-2s1b

((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_def_2s1b->Draw("e3");
hh_2s1b->Draw("same e3");

TLegend *leg = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def_2s1b,"default emulator","f");
leg->AddEntry(hh_2s1b,"with GEM match","f");
leg->AddEntry(result_def_2s1b,"Tracks req. for both:","");
leg->AddEntry(result_def_2s1b,"with >=2 stubs in 1.64<|#eta|<2.14","");
leg->AddEntry(result_def_2s1b,"and require one stub to be from ME1/b","");
leg->Draw();

TLatex *  tex = new TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
tex->SetNDC();
tex->Draw();

gPad->Print(gem_dir + "rates__1.6-2.1_PU100__def-3s-2s1b__gem-3s-2s1b__Frankenstein_pat8.png");


((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
hh_ratio = setHistoRatio(hh_2s1b, result_def_2s1b, "", 0.,1.1);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1.6-2.1_PU100__def-3s-2s1b__gem-3s-2s1b__Frankenstein_pat8__ratio.png");


// --- def-3s   gem-3s-2s1b

((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_def->Draw("e3");
hh_2s1b->Draw("same e3");

TLegend *leg = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def,"default emulator","f");
leg->AddEntry(result_def,"Tracks: with >=3 stubs in 1.64<|#eta|<2.14","");
leg->AddEntry(hh_2s1b,"with GEM match","f");
leg->AddEntry(hh_2s1b,"Tracks: with >=2 stubs in 1.64<|#eta|<2.14","");
leg->AddEntry(hh_2s1b,"and require one stub to be from ME1/b","");
leg->Draw();

TLatex *  tex = new TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
tex->SetNDC();
tex->Draw();

gPad->Print(gem_dir + "rates__1.6-2.1_PU100__def-3s__gem-3s-2s1b__Frankenstein_pat8.png");


((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
hh_ratio = setHistoRatio(hh_2s1b, result_def, "", 0.,1.1);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1.6-2.1_PU100__def-3s__gem-3s-2s1b__Frankenstein_pat8__ratio.png");


// --- def-3s-3s1b   gem-3s-2s1b

((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_def_3s1b->Draw("e3");
hh_2s1b->Draw("same e3");

TLegend *leg = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def_3s1b,"default emulator","f");
leg->AddEntry(result_def_3s1b,"Tracks req. >=3 stubs in 1.64<|#eta|<2.14","");
leg->AddEntry(result_def_3s1b,"and require one stub to be from ME1/b","");
leg->AddEntry(hh_2s1b,"with GEM match","f");
leg->AddEntry(hh_2s1b,"Tracks req. >=2 stubs in 1.64<|#eta|<2.14","");
leg->AddEntry(hh_2s1b,"and require one stub to be from ME1/b","");
leg->Draw();

TLatex *  tex = new TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
tex->SetNDC();
tex->Draw();

gPad->Print(gem_dir + "rates__1.6-2.1_PU100__def-3s-3s1b__gem-3s-2s1b__Frankenstein_pat8.png");


((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
hh_ratio = setHistoRatio(hh_2s1b, result_def_3s1b, "", 0.,3.1);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1.6-2.1_PU100__def-3s-3s1b__gem-3s-2s1b__Frankenstein_pat8__ratio.png");


// --- eta 1-2.4 def-3s   gem-3s-3s1b

((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_def_eta_all->Draw("e3");
hh_all->Draw("same e3");

TLegend *leg = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def_eta_all,"default emulator","f");
leg->AddEntry(result_def_eta_all,"Tracks: with >=3 stubs in 1.<|#eta|<2.4","");
leg->AddEntry(hh_all,"with GEM match","f");
leg->AddEntry(result_def_eta_all,"Tracks: same, plus req. one stub from ME1/b","");
leg->Draw();

TLatex *  tex = new TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
tex->SetNDC();
tex->Draw();

gPad->Print(gem_dir + "rates__1-2.4_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat8.png");


((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
hh_ratio = setHistoRatio(hh_all, result_def_eta_all, "", 0.,1.1);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1-2.4_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat8__ratio.png");


// --- eta 1-2.4  def-3s-3s1b   gem-3s-3s1b

((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_def_eta_all_3s1b->Draw("e3");
hh_all->Draw("same e3");

TLegend *leg = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def_eta_all_3s1b,"default emulator","f");
leg->AddEntry(hh_all,"with GEM match","f");
leg->AddEntry(result_def_eta_all_3s1b,"Tracks req. for both:","");
leg->AddEntry(result_def_eta_all_3s1b,"with >=3 stubs in 1.64<|#eta|<2.14","");
leg->AddEntry(result_def_eta_all_3s1b,"and require one stub to be from ME1/b","");
leg->Draw();

TLatex *  tex = new TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
tex->SetNDC();
tex->Draw();

gPad->Print(gem_dir + "rates__1-2.4_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat8.png");


((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
hh_ratio = setHistoRatio(hh_all, result_def_eta_all_3s1b, "", 0.,1.1);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1-2.4_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat8__ratio.png");


// --- eta 1-2.1 def-3s   gem-3s-3s1b

((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_def_eta_no1a->Draw("e3");
hh_no1a->Draw("same e3");

TLegend *leg = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def_eta_no1a,"default emulator","f");
leg->AddEntry(result_def_eta_no1a,"Tracks: with >=3 stubs in 1.<|#eta|<2.4","");
leg->AddEntry(hh_no1a,"with GEM match","f");
leg->AddEntry(result_def_eta_no1a,"Tracks: same, plus req. one stub from ME1/b","");
leg->Draw();

TLatex *  tex = new TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
tex->SetNDC();
tex->Draw();

gPad->Print(gem_dir + "rates__1-2.1_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat8.png");


((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
hh_ratio = setHistoRatio(hh_no1a, result_def_eta_no1a, "", 0.,1.1);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1-2.1_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat8__ratio.png");


// --- eta 1-2.1  def-3s-3s1b   gem-3s-3s1b

((TCanvas*)gROOT->FindObject("cAll100"))->cd();
result_def_eta_no1a_3s1b->Draw("e3");
hh_no1a->Draw("same e3");

TLegend *leg = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->AddEntry(result_def_eta_no1a_3s1b,"default emulator","f");
leg->AddEntry(hh_no1a,"with GEM match","f");
leg->AddEntry(result_def_eta_no1a_3s1b,"Tracks req. for both:","");
leg->AddEntry(result_def_eta_no1a_3s1b,"with >=3 stubs in 1.64<|#eta|<2.14","");
leg->AddEntry(result_def_eta_no1a_3s1b,"and require one stub to be from ME1/b","");
leg->Draw();

TLatex *  tex = new TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
tex->SetNDC();
tex->Draw();

gPad->Print(gem_dir + "rates__1-2.1_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat8.png");


((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
hh_ratio = setHistoRatio(hh_no1a, result_def_eta_no1a_3s1b, "", 0.,1.1);
hh_ratio->Draw("e1");
gPad->Print(gem_dir + "rates__1-2.1_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat8__ratio.png");


}
