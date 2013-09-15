// Design LHC orbit: 2808 filled bickets out of 3564 --->  0.7879 filled BX fraction

TString gem_dir =  "files/";
TString gem_label = "gem98";
char dir[111]="SimMuL1StrictAll";
int gNPU=100;
int gNEvt=238000;
//int gNEvt=128000;
float gdy[2]={0.1,2500};

TLatex* drawLumiLabel2(float x=0.2, float y=0.4)
{
  TLatex *  tex = new TLatex(x, y,"L = 4*10^{34} cm^{-2} s^{-1}");
  tex->SetTextSize(0.05);
  tex->SetNDC();
  tex->Draw();
  return tex;
}

TLatex* drawPULabel(float x=0.17, float y=0.15, float font_size=0.)
{
  TLatex *  tex = new TLatex(x, y,"L=4*10^{34} (25ns PU100)");
  if (font_size > 0.) tex->SetFontSize(font_size);
  tex->SetNDC();
  tex->Draw();
  return tex;
}

TH1D* setHistoEta(TString f_name, char *name, char *cname, char *title, 
		  Color_t lcolor, int lstyle, int lwidth)
{
  cout<<"opening "<<f_name<<endl;
  f = TFile::Open(f_name);

  TH1D* h0 = (TH1D*)getH(f, dir, name);
  TString s_name(name);
  int nb = h0->GetXaxis()->GetNbins();

  TH1D* h = (TH1D*)h0->Clone(s_name+cname);
  h->SetTitle(title);
  h->Sumw2();
  h->Scale(40000./gNEvt/3.*0.795);
  h->SetLineColor(lcolor);
  //h->SetFillColor(lcolor);
  h->SetLineStyle(lstyle);
  h->SetLineWidth(lwidth);
  h->SetTitle(title);
  //h->GetXaxis()->SetRangeUser(1.2, 2.4);
  h->GetYaxis()->SetRangeUser(gdy[0],gdy[1]);
  h->GetXaxis()->SetTitleSize(0.055);
  h->GetXaxis()->SetTitleOffset(1.05);
  h->GetXaxis()->SetLabelSize(0.045);
  h->GetXaxis()->SetLabelOffset(0.003);
  h->GetXaxis()->SetTitleFont(62);
  h->GetXaxis()->SetLabelFont(62);
  h->GetXaxis()->SetMoreLogLabels(1);
  h->GetYaxis()->SetTitleSize(0.055);
  h->GetYaxis()->SetTitleOffset(0.9);
  h->GetYaxis()->SetLabelSize(0.045);
  h->GetYaxis()->SetTitleFont(62);
  h->GetYaxis()->SetLabelFont(62);
  //h->GetYaxis()->SetLabelOffset(0.015);
  return h;
}

TObject* getH(char dir[100], char name[100])
{
  char nm[222];
  sprintf(nm,"%s/%s;1",dir,name);
  return f->Get(nm);
}

TObject* getH(TFile *fi, char dir[100], char name[100])
{
  char nm[222];
  sprintf(nm,"%s/%s;1",dir,name);
  return fi->Get(nm);
}

void Print(TCanvas *c, char nm[200])
{
  if (do_not_print) return;
  if (strcmp(pdir,"")<=0) return;
  gPad->RedrawAxis();
  char dirnm[200];
  sprintf(dirnm,"%s/%s",pdir,nm);
  c->Print(dirnm);
}

TH1D* setHistoRatio(TH1D* num, TH1D* denom, TString title = "", double ymin=0.4, double ymax=1.6, int color = kRed+3)
{
  ratio = (TH1D*) num->Clone(Form("%s--%s_ratio",num->GetName(),denom->GetName()) );
  ratio->Divide(num, denom, 1., 1.);
  ratio->SetTitle(title);
  ratio->GetYaxis()->SetRangeUser(ymin, ymax);
  ratio->GetYaxis()->SetTitle("ratio: (with GEM)/default");
  ratio->GetYaxis()->SetTitle("Ratio");
  //ratio->GetYaxis()->SetTitle("(ME1/b + GEM) / ME1/b");
  ratio->GetYaxis()->SetTitleSize(.14);
  //ratio->GetYaxis()->SetTitleSize(.1);
  ratio->GetYaxis()->SetTitleOffset(0.4);
  ratio->GetYaxis()->SetLabelSize(.11);
  //ratio->GetXaxis()->SetMoreLogLabels(1);
  //ratio->GetXaxis()->SetTitle("track #eta");
  ratio->GetXaxis()->SetLabelSize(.11);
  ratio->GetXaxis()->SetTitleSize(.14);
  ratio->GetXaxis()->SetTitleOffset(1.); 
  ratio->SetLineWidth(2);
  ratio->SetFillColor(color);
  ratio->SetLineColor(color);
  ratio->SetMarkerColor(color);
  ratio->SetMarkerStyle(20);
  ratio->SetLineColor(color);
  ratio->SetMarkerColor(color);
  //ratio->Draw("e3");

  return ratio;
}



void addRatioPlotLegend(TH1* h, TString k)
{
  TLegend* leg = new TLegend(0.17,0.4,.47,0.5,NULL,"brNDC");
  leg->SetMargin(0.1);
  leg->SetBorderSize(0);
  leg->SetTextSize(0.1);
  leg->SetFillStyle(1001);
  leg->SetFillColor(kWhite);
  leg->AddEntry(h, "(GEM+CSC)/CSC #geq" + k + " stubs","P");
  leg->Draw("same");
}

void addRatePlotLegend(TH1* h, TH1* i, TH1* j, TString k, TString l)
{
  TLegend *leg = new TLegend(0.16,0.67,.8,0.9,"L1 Selections (#geq" + k + " stations, L1 candidate p_{T}#geq" + l + " GeV/c):","brNDC");
  leg->SetMargin(0.20);
  leg->SetBorderSize(0);
  leg->SetTextSize(0.04);
  leg->SetFillStyle(1001);
  leg->SetFillColor(kWhite);
  leg->AddEntry(h,"CSC #geq" + k + " stubs (anywhere)","l");
  leg->AddEntry(i,"CSC #geq" + k + " stubs (one in ME1/b)","l");
  leg->AddEntry(j,"GEM+CSC integrated trigger","l");
  leg->Draw("same");
}

void addRatePlots(TH1* h, TH1* i, TH1* j, Color_t col1, Color_t col2, Color_t col3,
		  Style_t sty1, Style_t sty2, Style_t sty3, Style_t sty4, int miny, int maxy)
{
    h->SetFillColor(col1);
    i->SetFillColor(col2);
    j->SetFillColor(col3);

    h->SetFillStyle(sty1);
    i->SetFillStyle(sty2);
    j->SetFillStyle(sty3);

    // Slava's proposal
    h->SetFillStyle(0);
    i->SetFillStyle(0);
    j->SetFillStyle(0);

    h->SetLineStyle(1);
    i->SetLineStyle(4);
    j->SetLineStyle(2);

    h->SetLineWidth(2);
    i->SetLineWidth(2);
    j->SetLineWidth(2);

    h->GetYaxis()->SetRangeUser(miny,maxy);
    i->GetYaxis()->SetRangeUser(miny,maxy);
    j->GetYaxis()->SetRangeUser(miny,maxy);

    TH1* i_clone = i->Clone("i_clone");
    TH1* j_clone = j->Clone("j_clone");
    TH1* i_clone2 = i->Clone("i_clone2");
    TH1* j_clone2 = j->Clone("j_clone2");
    /*
    for (int ii=0; ii<=14; ++ii){
      i_clone2->SetBinContent(ii,0);
      j_clone2->SetBinContent(ii,0);
      i_clone2->SetBinError(ii,0);
      j_clone2->SetBinError(ii,0);
    }
    for (int ii=26; ii<=34; ++ii){
      i_clone2->SetBinContent(ii,0);
      j_clone2->SetBinContent(ii,0);
      i_clone2->SetBinError(ii,0);
      j_clone2->SetBinError(ii,0);
      j_clone2->GetXaxis()->SetRangeUser(1.62,2.12);
      i_clone2->GetXaxis()->SetRangeUser(1.62,2.12);
     }
    
    for (int ii=15; ii<=25; ++ii){
      i_clone->SetBinContent(ii,0);
      j_clone->SetBinContent(ii,0);
      i_clone->SetBinError(ii,0);
      j_clone->SetBinError(ii,0);
    }
    */
    
    // j_clone->SetFillStyle(sty4);

    i_clone->Draw("hist e1 same");
    j_clone->Draw("hist e1 same");
    h->Draw("hist e1 same");
    // i_clone2->Draw("hist e1 same");
    // j_clone2->Draw("hist e1 same");
}

void setPad1Attributes(TPad* pad1)
{
  pad1->SetGridx(1);
  pad1->SetGridy(1);
  pad1->SetFrameBorderMode(0);
  pad1->SetFillColor(kWhite);
  pad1->SetTopMargin(0.06);
  pad1->SetBottomMargin(0.13);
}

void setPad2Attributes(TPad* pad2)
{
  pad2->SetLogy(1);
  pad2->SetGridx(1);
  pad2->SetGridy(1);
  pad2->SetFillColor(kWhite);
  pad2->SetFrameBorderMode(0);
  pad2->SetTopMargin(0.06);
  pad2->SetBottomMargin(0.3);
}

void produceRateVsEtaPlot(TH1D* h, TH1D* i, TH1D* j, Color_t col1, Color_t col2, Color_t col3,
			  Style_t sty1, Style_t sty2, Style_t sty3, Style_t sty4, int miny, int maxy, 
			  TString k, TString l, TString plots, TString ext)
{
  TCanvas* c = new TCanvas("c","c",800,800);
  c->Clear();
  TPad *pad1 = new TPad("pad1","top pad",0.0,0.25,1.0,1.0);
  pad1->Draw();
  TPad *pad2 = new TPad("pad2","bottom pad",0,0.,1.0,.30);
  pad2->Draw();

  pad1->cd();
  setPad1Attributes(pad1);
  addRatePlots(h,i,j,col1,col2,col3,sty1,sty2,sty3,3355,miny,maxy);
  addRatePlotLegend(h, i, j, k,l);
  drawLumiLabel2();

  pad2->cd();
  setPad2Attributes(pad2);
  TH1D* gem_ratio = setHistoRatio(j, i, "", 0.01,2.0, col2);
  gem_ratio->Draw("Pe");
  gem_ratio->GetYaxis()->SetNdivisions(3);
  
  addRatioPlotLegend(gem_ratio, k);
  
  c->SaveAs(plots + "rates_vs_eta__minpt" + l + "__PU100__def_" + k + "s_" + k + "s1b_" + k + "s1bgem" + ext);
}

void produceRateVsEtaPlotsForApproval(TString ext, TString plots)
{
  gStyle->SetOptStat(0);
  gStyle->SetTitleStyle(0);
  // //gStyle->SetPadTopMargin(0.08);
  // gStyle->SetTitleH(0.06);

  // input files
  TString f_def =      gem_dir + "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_def_pat2.root";
  TString f_g98_pt10 = gem_dir + "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_gem98_pt10_pat2.root";
  TString f_g98_pt15 = gem_dir + "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_gem98_pt15_pat2.root";
  TString f_g98_pt20 = gem_dir + "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_gem98_pt20_pat2.root";
  TString f_g98_pt30 = gem_dir + "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_gem98_pt30_pat2.root";
  TString f_g98_pt40 = gem_dir + "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_gem98_pt40_pat2.root";

  // general stuff
  TString hdir = "SimMuL1StrictAll";

  // colors - same colors as for rate vs pt plots!!
  Color_t col1 = kViolet+1;
  Color_t col2 = kAzure+2;
  Color_t col3 = kGreen-2;

  // styles
  Style_t sty1 = 3345;
  Style_t sty2 = 3003;
  Style_t sty3 = 2002;

  // Declaration of histograms
  TString vs_eta_minpt = "10";
  //  TString ttl = "        L1 Single Muon Trigger                   CMS Simulation Preliminary;L1 muon candidate #eta;rate [kHz]";
  TString ttl = "                                              CMS Simulation Preliminary;L1 muon candidate #eta;Trigger rate [kHz]";
  TH1D* h_rt_tf10_2s   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s", "_hAll100", ttl, col1, 1, 2);
  TH1D* h_rt_tf10_2s1b   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s_2s1b", "_hAll100", ttl, col2, 1, 2);
  TH1D* h_rt_tf10_gpt10_2s1b   = setHistoEta(f_g98_pt10, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s_2s1b", "_hAll100", ttl, col3, 1, 2);

  TH1D* h_rt_tf10_3s   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s", "_hAll100", ttl, col1, 1, 2);
  TH1D* h_rt_tf10_3s1b   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1b", "_hAll100", ttl, col2, 1, 2);
  TH1D* h_rt_tf10_gpt10_3s1b   = setHistoEta(f_g98_pt10, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1b", "_hAll100", ttl, col3, 7, 2);
  
  TString vs_eta_minpt = "20";
  TH1D* h_rt_tf20_2s   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s", "_hAll100", ttl, col1, 1, 2);
  TH1D* h_rt_tf20_2s1b   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s_2s1b", "_hAll100", ttl, col2, 1, 2);
  TH1D* h_rt_tf20_gpt20_2s1b   = setHistoEta(f_g98_pt20, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s_2s1b", "_hAll100", ttl, col3, 1, 2);
  TH1D* h_rt_tf20_3s   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s", "_hAll100", ttl, col1, 1, 2);
  TH1D* h_rt_tf20_3s1b   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1b", "_hAll100", ttl, col2, 1, 2);
  TH1D* h_rt_tf20_gpt20_3s1b   = setHistoEta(f_g98_pt20, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1b", "_hAll100", ttl, col3, 1, 2);

  TString vs_eta_minpt = "30";
  TH1D* h_rt_tf30_2s   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s", "_hAll100", ttl, col1, 1, 2);
  TH1D* h_rt_tf30_2s1b   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s_2s1b", "_hAll100", ttl, col2, 1, 2);
  TH1D* h_rt_tf30_gpt30_2s1b   = setHistoEta(f_g98_pt30, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s_2s1b", "_hAll100", ttl, col3, 1, 2);
  TH1D* h_rt_tf30_3s   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s", "_hAll100", ttl, col1, 1, 2);
  TH1D* h_rt_tf30_3s1b   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1b", "_hAll100", ttl, col2, 1, 2);
  TH1D* h_rt_tf30_gpt30_3s1b   = setHistoEta(f_g98_pt30, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1b", "_hAll100", ttl, col3, 1, 2);

  // Style
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

  // producing the histograms 
  float miny = 0.01, maxy;

  // ------------ +2 stubs, L1 candidate muon pt=10GeV ----------------//
  produceRateVsEtaPlot(h_rt_tf10_2s,h_rt_tf10_2s1b,h_rt_tf10_gpt10_2s1b,
		       col1,col2,col3,sty1,sty2,sty3,3355,miny,80,"2","10",plots,ext);
  // ------------ +2 stubs, L1 candidate muon pt=20GeV ----------------//
  produceRateVsEtaPlot(h_rt_tf20_2s,h_rt_tf20_2s1b,h_rt_tf20_gpt20_2s1b,
		       col1,col2,col3,sty1,sty2,sty3,3355,miny,40,"2","20",plots,ext);
  // ------------ +2 stubs, L1 candidate muon pt=30GeV ----------------//
  produceRateVsEtaPlot(h_rt_tf30_2s,h_rt_tf30_2s1b,h_rt_tf30_gpt30_2s1b,
		       col1,col2,col3,sty1,sty2,sty3,3355,miny,30,"2","30",plots,ext);
  // ------------ +3 stubs, L1 candidate muon pt=10GeV ----------------//
  produceRateVsEtaPlot(h_rt_tf10_3s,h_rt_tf10_3s1b,h_rt_tf10_gpt10_3s1b,
		       col1,col2,col3,sty1,sty2,sty3,3355,miny,25,"3","10",plots,ext);
  // ------------ +3 stubs, L1 candidate muon pt=20GeV ----------------//
  produceRateVsEtaPlot(h_rt_tf20_3s,h_rt_tf20_3s1b,h_rt_tf20_gpt20_3s1b,
		       col1,col2,col3,sty1,sty2,sty3,3355,miny,10,"3","20",plots,ext);
  // ------------ +3 stubs, L1 candidate muon pt=30GeV ----------------//
  produceRateVsEtaPlot(h_rt_tf30_3s,h_rt_tf30_3s1b,h_rt_tf30_gpt30_3s1b,
		       col1,col2,col3,sty1,sty2,sty3,3355,miny,6,"3","30",plots,ext);
}

void produceRateVsEtaPlotsForApproval()
{
  //  produceRateVsEtaPlotsForApproval(".C", "plots/rate_vs_eta/");
  //produceRateVsEtaPlotsForApproval(".png", "plots/rate_vs_eta/");
  produceRateVsEtaPlotsForApproval(".png", "plots/rate_vs_eta/");
  //produceRateVsEtaPlotsForApproval(".C", "plots/rate_vs_eta/");
}
/*
void gem_rate_draw()
{
  -- KEEP THIS FRAGMENT FOR THE TIME BEING!! --
  -- NEED TO FIGURE OUT WHAT TO DO WITH ME1A --

  // // Including the region ME1/1a

  //  ((TCanvas*)gROOT->FindObject("cAll100"))->cd();

  //  maxy = 30.;//10;

  //  h_rt_tf20_3s1ab->GetYaxis()->SetRangeUser(miny,maxy);
  //  h_rt_tf20_gpt20_3s1ab->GetYaxis()->SetRangeUser(miny,maxy);


  //  ((TCanvas*)gROOT->FindObject("cAll100"))->cd();
  //  h_rt_tf20_3s->Draw("hist e1");
  //  h_rt_tf20_gpt20_3s1ab->Draw("hist e1 same");
  //  h_rt_tf20_3s->Draw("hist e1 same");
  //  h_rt_tf20_3s1ab->Draw("hist e1 same");

  //  TLegend *leg = new TLegend(0.4,0.63,.98,0.90,NULL,"brNDC");
  //  leg->SetBorderSize(0);
  //  leg->SetFillStyle(0);
  //  leg->AddEntry(h_rt_tf20_3s,"Tracks: p_{T}>=20, 3+ stubs","");
  //  leg->AddEntry(h_rt_tf20_3s,"anywhere","l");
  //  leg->AddEntry(h_rt_tf20_3s1ab,"with ME1 in 1.6<|#eta|","l");
  //  leg->AddEntry(h_rt_tf20_gpt20_3s1ab,"with (ME1+GEM) in 1.6<|#eta|","l");
  //  leg->Draw();

  //  TLatex *  tex = new TLatex(0.17, 0.82,"#splitline{L=4*10^{34}}{(25ns PU100)}");
  //  tex->SetNDC();
  //  tex->Draw();

  //  Print(cAll100, ("rates__vs_eta__minpt"+ vs_eta_minpt +"__PU100__def-3s-3s1ab__gem-3s-3s1ab.png").Data());


  // TCanvas* cAll100r = new TCanvas("cAll100r","cAll100r",800,300) ;
  // gPad->SetGridx(1);gPad->SetGridy(1);

  // gem_ratio = setHistoRatio(h_rt_tf20_gpt20_3s1ab, h_rt_tf20_3s1ab, "", 0.,1.8);
  // gem_ratio->Draw("e1");

  //  Print(cAll100r, ("rates__vs_eta__minpt"+ vs_eta_minpt +"__PU100__def-3s-3s1ab__gem-3s-3s1ab__ratio.png").Data());



  vs_eta_minpt = "30";
  ttl = "CSC L1 trigger rates for p_{T}^{TF}>" + vs_eta_minpt + " GeV/c;track #eta;rate [kHz]";

  
  TH1D* h_rt_tf30_2s   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s", "_hAll100", ttl, kAzure+2, 1, 2);
  TH1D* h_rt_tf30_2s1b   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s_2s1b", "_hAll100", ttl, kAzure+5, 1, 2);
  TH1D* h_rt_tf30_gpt30_2s1b   = setHistoEta(f_g98_pt30, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s_2s1b", "_hAll100", ttl, kGreen+1, 7, 2);
  TH1D* h_rt_tf30_3s   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s", "_hAll100", ttl, kAzure+3, 1, 2);
  TH1D* h_rt_tf30_3s1b   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1b", "_hAll100", ttl, kAzure+6, 1, 2);
  TH1D* h_rt_tf30_3s1ab   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1ab", "_hAll100", ttl, kAzure+2, 1, 2);
  TH1D* h_rt_tf30_gpt30_3s1b   = setHistoEta(f_g98_pt30, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1b", "_hAll100", ttl, kGreen+3, 7, 2);
  TH1D* h_rt_tf30_gpt30_3s1ab   = setHistoEta(f_g98_pt30, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1ab", "_hAll100", ttl, kGreen, 7, 2);





  // //==========================  Including the   ME1a

  // ((TCanvas*)gROOT->FindObject("cAll100"))->cd();

  // h_rt_tf30_3s->GetYaxis()->SetRangeUser(miny,maxy);
  // h_rt_tf30_3s1ab->GetYaxis()->SetRangeUser(miny,maxy);
  h_rt_tf30_gpt30_3s1ab->GetYaxis()->SetRangeUser(miny,maxy);

  // ((TCanvas*)gROOT->FindObject("cAll100"))->cd();
  // h_rt_tf30_3s->Draw("hist e1");
  // h_rt_tf30_gpt30_3s1ab->Draw("hist e1 same");
  // h_rt_tf30_3s->Draw("hist e1 same");
  // h_rt_tf30_3s1ab->Draw("hist e1 same");

  // TLegend *leg = new TLegend(0.4,0.63,.98,0.90,NULL,"brNDC");
  // leg->SetBorderSize(0);
  // leg->SetFillStyle(0);
  // leg->AddEntry(h_rt_tf30_3s,"Tracks: p_{T}>=30, 3+ stubs","");
  // leg->AddEntry(h_rt_tf30_3s,"anywhere","l");
  // leg->AddEntry(h_rt_tf30_3s1ab,"with ME1 in 1.6<|#eta|","l");
  // leg->AddEntry(h_rt_tf30_gpt30_3s1ab,"with (ME1+GEM) in 1.6<|#eta|<2.14","l");
  // leg->Draw();
 
  // TLatex *  tex = new TLatex(0.17, 0.82,"#splitline{L=4*10^{34}}{(25ns PU100)}");
  // tex->SetNDC();
  // tex->Draw();
 
  // Print(cAll100, ("rates__vs_eta__minpt"+ vs_eta_minpt +"__PU100__def-3s-3s1ab__gem-3s-3s1ab.png").Data());

  // TCanvas* cAll100r2 = new TCanvas("cAll100r2","cAll100r2",800,300) ;
  // gPad->SetGridx(1);gPad->SetGridy(1);
 
  // gem_ratio = setHistoRatio(h_rt_tf30_gpt30_3s1ab, h_rt_tf30_3s1ab, "", 0.,1.8);
  // gem_ratio->Draw("e1");

  // Print(cAll100r2, ("rates__vs_eta__minpt"+ vs_eta_minpt +"__PU100__def-3s-3s1ab__gem-3s-3s1ab__ratio.png").Data());



  // //==========================  Comparison with/withous Stub in ME1a ==========================//

  //  ((TCanvas*)gROOT->FindObject("cAll100"))->cd();

  //  h_rt_tf30_3s1b->Draw("hist e1");
  //  h_rt_tf30_3s1ab->Draw("hist e1 same");

  //  TLegend *leg = new TLegend(0.2,0.65,.80,0.90,NULL,"brNDC");
  //  leg->SetBorderSize(0);
  //  leg->SetFillStyle(0);
  //  leg->AddEntry(h_rt_tf30_3s1b,"Tracks: p_{T}>=30, 3+ stubs","");
  //  leg->AddEntry(h_rt_tf30_3s1b,"with ME1 in 1.6<|#eta|<2.14","l");
  //  leg->AddEntry(h_rt_tf30_3s1ab,"with ME1 in 1.6<|#eta|<2.4","l");
  //  leg->Draw();
  
  
  //  Print(cAll100, ("rates__vs_eta__minpt"+ vs_eta_minpt +"__PU100__def-3s-3s1ab__gem-3s-3s1ab_compstubME1a.png").Data());
  
  
  //  TCanvas* cAll100r2 = new TCanvas("cAll100r2","cAll100r2",800,300) ;
  //  gPad->SetGridx(1);gPad->SetGridy(1);
  
  //  gem_ratio = setHistoRatio2(h_rt_tf30_3s1ab, h_rt_tf30_3s1b, "", 0.,1.8);
  //  gem_ratio->Draw("e1");
  
  //  Print(cAll100r2, ("rates__vs_eta__minpt"+ vs_eta_minpt +"__PU100__def-3s-3s1ab__gem-3s-3s1ab__ratio_compstubME1a.png").Data());


  //==========================  Comparison with/withous Stub in ME1a + GEMS ==========================//
  
  ((TCanvas*)gROOT->FindObject("cAll100"))->cd();
  
  h_rt_tf30_gpt30_3s1b->Draw("hist e1");
  h_rt_tf30_gpt30_3s1ab->Draw("hist e1 same");
  
  TLegend *leg = new TLegend(0.2,0.65,.80,0.90,NULL,"brNDC");
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  leg->AddEntry(h_rt_tf30_3s1b,"Tracks: p_{T}>=30, 3+ stubs","");
  leg->AddEntry(h_rt_tf30_3s1b,"with ME1 in 1.6<|#eta|<2.14","l");
  leg->AddEntry(h_rt_tf30_3s1ab,"with ME1 in 1.6<|#eta|<2.4","l");
  leg->Draw();
  
  
  Print(cAll100, ("rates__vs_eta__minpt"+ vs_eta_minpt +"__PU100__def-3s-3s1ab__gem-3s-3s1ab_compstubME1a.png").Data());
}
*/
