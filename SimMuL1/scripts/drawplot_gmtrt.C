
//int kGray=920, kOrange=800, kSpring=820, kTeal=840, kAzure=860, kViolet=880, kPink=900;

// Design LHC orbit: 2808 filled bickets out of 3564 --->  0.7879 filled BX fraction


TFile *f=0;
char pdir[111]="";
char dir[111]="SimMuL1StrictAll";

TObject* NUL;

TH1D *gh;
TH1D *ght;
TLegend *leg;

TH1D* result_gem = 0;
TH1D* result_def = 0;
TH1D* result_def_2s = 0;
TH1D* result_def_3s1b = 0;

TH1D* result_gem_2s1b = 0;
TH1D* result_gem_2s123 = 0;
TH1D* result_gem_2s13 = 0;
TH1D* result_def_2s1b = 0;
TH1D* result_def_2s123 = 0;
TH1D* result_def_2s13 = 0;

TH1D* result_gem_eta_all = 0;
TH1D* result_def_eta_all = 0;
TH1D* result_def_eta_all_3s1b = 0;

TH1D* result_gem_eta_no1a = 0;
TH1D* result_def_eta_no1a = 0;
TH1D* result_def_eta_no1a_3s1b = 0;

TH1D* result_def_gmtsing = 0;
TH1D* result_gmtsing = 0;

TH1D* result_def_gmtsing_no1a = 0;
TH1D* result_gem_gmtsing_no1a = 0;

// for later use
TH1D* result_def_2s__pat2 = 0;
TH1D* result_def_3s__pat2 = 0;
TH1D* result_def_2s1b__pat2 = 0;
TH1D* result_def_3s1b__pat2 = 0;
TH1D* result_def_2s__pat8 = 0;
TH1D* result_def_3s__pat8 = 0;
TH1D* result_def_2s1b__pat8 = 0;
TH1D* result_def_3s1b__pat8 = 0;

TH1D* result_def_gmtsing__pat2 = 0;
TH1D* result_def_gmtsing__pat8 = 0;

TH1D* result_gem_2s1b__pat2 = 0;
TH1D* result_gem_3s1b__pat2 = 0;
TH1D* result_gem_2s1b__pat8 = 0;
TH1D* result_gem_3s1b__pat8 = 0;


int interactive = 1;
bool do_not_print = false;

TString gem_dir = "files/";
TString gem_label = "gem98";

int gNPU=100;
int gNEvt=238000;
//int gNEvt=128000;
float gdy[2]={0.1,2500};

double ptscale[32] = {
  -1.,  0., 1.5,  2., 2.5,  3., 3.5,  4., 4.5,  5.,  6.,  7.,  8.,  10.,  12.,  14.,
  16., 18., 20., 25., 30., 35., 40., 45., 50., 60., 70., 80., 90., 100., 120., 140.
};
const Double_t ptscaleb[31] = {
  1.5,  2., 2.5,  3., 3.5,  4., 4.5,  5.,  6.,  7.,  8.,  10.,  12.,  14.,
  16., 18., 20., 25., 30., 35., 40., 45., 50., 60., 70., 80., 90., 100., 120., 140., 150.
};
const Double_t ptscaleb_[31] = {
  1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75,  5.5, 6.5, 7.5,  9., 11.,  13.,  15.,
  17.,  19., 22.5, 27.5, 32.5, 37.5, 42.5, 47.5,  55.,  65., 75., 85., 95., 110., 130., 150.
};

void dpr(char *prc, int pu, char *suffix="")
//void dpr(char *prc, char *g, int pu, int st, char *suffix="")
{
  gNPU=pu;
  char nm[300], fnm[300],onm[300];
  char spu[20];
  sprintf(spu,"%03d",pu);
  //sprintf(nm,"%s_2_2_6_pu%s_me42_me1a%s_2pi_step_%d_pre3_w3%s",prc,spu,g,st,suffix);
  //sprintf(nm,"%s_3_6_2_pu%s_me42_me1a%s_2pi_step_%d_pre3_w3%s",prc,spu,g,st,suffix);
  //sprintf(nm,"%s_3_6_2_mu_pu%s_me42_me1a%s_2pi_step_%d_pre3_w3%s",prc,spu,g,st,suffix);
  sprintf(nm,"%s_6_0_1_POSTLS161_V12__pu%s_w3%s",prc,spu,suffix);
  sprintf(fnm,"../hp_%s.root",nm);

  sprintf(onm,"rttf_%s",nm);
  drawplot_tfrt(fnm,"StrictChamber",onm);
}


TPaveStats* GetStat(TH1*h)
{
  TPaveStats* stat = (TPaveStats*)h->FindObject("stats");
  return stat;
}

TPaveStats* SetOptStat(TH1*h,int op)
{
  TPaveStats* stat = GetStat(h);
  stat->SetOptStat(op);
  return stat;
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

void myRebin(TH1D* h, int n)
{
  int nb = h->GetNbinsX();
  Double_t entr = h->GetEntries();
  Double_t bin0 = h->GetBinContent(0);
  Double_t binN1 = h->GetBinContent(nb+1);
  if (nb % n) binN1 += h->Integral(nb - nb%n + 1,nb);
  h->Rebin(n);
  nb = h->GetNbinsX();
  h->SetBinContent(0,bin0);
  h->SetBinContent(nb+1,binN1);
  h->SetEntries(entr);
}

//###########################################################################
TH1D* setHistoPt(char *f_name, char *name, char *cname, char *title, 
		 int lcolor, int lstyle, int lwidth)
//double *x_range, double *y_range)
{
  cout<<"opening "<<f_name<<endl;
  f = TFile::Open(f_name);

  TH1D* h0 = (TH1D*)getH(f, dir, name);
  TString s_name(name);
  int nb = h0->GetXaxis()->GetNbins();

  h = new TH1D(s_name+cname,title,30,ptscaleb);
  for (int b=1 ; b<=nb; b++) {
    double bc = h0->GetBinContent(b);
    if (bc==0) continue;
    int bin = h->GetXaxis()->FindFixBin(h0->GetBinCenter(b));
    //cout<<b<<" "<<bc<<" "<<bin<<endl;
    h->SetBinContent(bin, bc);
  }
  //h->Sumw2();
  h->Scale(40000./gNEvt/3.*0.795);

  return h;
}

TH1D* setHistoPtRaw(char *f_name, char *name, char *cname, char *title, 
		    int lcolor, int lstyle, int lwidth)
//double *x_range, double *y_range)
{
  cout<<"opening "<<f_name<<endl;
  f = TFile::Open(f_name);

  TH1D* h0 = (TH1D*)getH(f, dir, name);
  TString s_name(name);
  int nb = h0->GetXaxis()->GetNbins();
  h = (TH1D*)h0->Clone(s_name+cname);
  //h->Sumw2();
  h->Scale(40000./gNEvt/3.*0.795);

  return h;
}


TH1D* setHisto2(char *f_name, char *name1, char *name2, char *cname, char *title, 
		int lcolor, int lstyle, int lwidth)
//double *x_range, double *y_range)
{
  cout<<"opening "<<f_name<<endl;
  f = TFile::Open(f_name);

  TH1D* h1 = (TH1D*)getH(f, dir, name1);
  TH1D* h2 = (TH1D*)getH(f, dir, name2);
  TString s_name(name1);
  h0 = (TH1D*)h1->Clone(s_name+cname);
  h0->Add(h2);
  int nb = h0->GetXaxis()->GetNbins();

  h = new TH1D(s_name+cname,title,30,ptscaleb);
  for (int b=1 ; b<=nb; b++) {
    double bc = h0->GetBinContent(b);
    if (bc==0) continue;
    int bin = h->GetXaxis()->FindFixBin(h0->GetBinCenter(b));
    //cout<<b<<" "<<bc<<" "<<bin<<endl;
    h->SetBinContent(bin, bc);
  }
  for (int b=1 ; b<=30; b++)
    h->SetBinContent(b, h->Integral(b,31));

  /*
    h = (TH1D*)h0->Clone(s_name+cname);
    for (int b=1 ; b<=nb; b++)
    h->SetBinContent(b, h0->Integral(b,nb+1));
  */

  h->Sumw2();
  h->Scale(40000000./gNEvt/3.*0.795/10);

  h->SetLineColor(lcolor);
  h->SetFillColor(lcolor);
  h->SetLineStyle(lstyle);
  h->SetLineWidth(lwidth);

  h->SetTitle(title);
  //h->GetXaxis()->SetRangeUser(x_range[0],x_range[1]);
  //h->GetYaxis()->SetRangeUser(y_range[0],y_range[1]);

  h->GetXaxis()->SetRangeUser(3,130);
  h->GetYaxis()->SetRangeUser(gdy[0],gdy[1]);
  //h->GetYaxis()->SetRangeUser(0.01,3000);

  h->GetXaxis()->SetMoreLogLabels(1);

  h->GetXaxis()->SetTitleSize(0.055);
  h->GetXaxis()->SetTitleOffset(1.05);
  h->GetXaxis()->SetLabelSize(0.045);
  h->GetXaxis()->SetLabelOffset(0.003);
  h->GetXaxis()->SetTitleFont(62);
  h->GetXaxis()->SetLabelFont(62);

  h->GetYaxis()->SetTitleSize(0.055);
  h->GetYaxis()->SetTitleOffset(0.9);
  h->GetYaxis()->SetLabelSize(0.045);
  h->GetYaxis()->SetTitleFont(62);
  h->GetYaxis()->SetLabelFont(62);

  //h->GetYaxis()->SetLabelOffset(0.015);

  return h;
}



TH1D* setHistoEta(TString f_name, char *name, char *cname, char *title, 
		  int lcolor, int lstyle, int lwidth)
//double *x_range, double *y_range)
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

  gh = h;
  return h;
}


TH1D* getPTHisto(TString f_name, TString dir_name, TString h_name, TString clone_suffix = "_cln")
{
  f = TFile::Open(f_name);
  //cout<<dir_name + "/" + h_name<<endl;
  TH1D* h0 = (TH1D*) f->Get(dir_name + "/" + h_name)->Clone(h_name + clone_suffix);
  return h0;
}

TH1D* setPTHisto(TH1D* h0, TString title, int lcolor, int lstyle, int lwidth)
{
  int nb = h0->GetXaxis()->GetNbins();

  h = new TH1D(Form("%s_varpt", h0->GetName()), title, 30, ptscaleb_);
  for (int b=1 ; b<=nb; b++) {
    double bc = h0->GetBinContent(b);
    if (bc==0) continue;
    int bin = h->GetXaxis()->FindFixBin(h0->GetBinCenter(b));
    //cout<<b<<" "<<bc<<" "<<h0->GetBinCenter(b)<<" -> "<<bin<<endl;
    h->SetBinContent(bin, bc);
  }
  for (int b=1 ; b<=30; b++)
    h->SetBinContent(b, h->Integral(b,31));

  h->Sumw2();
  h->Scale(40000./gNEvt/3.*0.795);

  h->SetLineColor(lcolor);
  h->SetFillColor(lcolor);
  h->SetLineStyle(lstyle);
  h->SetLineWidth(lwidth);

  h->SetTitle(title);

  h->GetXaxis()->SetRangeUser(2, 129.);
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

  return h;
}

TH1D* setPTHisto(TString f_name, TString dir_name, TString h_name, TString clone_suffix,
                 TString title, int lcolor, int lstyle, int lwidth)
{
  TH1D* h0 = getPTHisto(f_name, dir_name, h_name, clone_suffix);
  return setPTHisto(h0, title, lcolor, lstyle, lwidth);
}




TH1D* setHisto(TString f_name, char *name, char *cname, char *title, 
               int lcolor, int lstyle, int lwidth)
{
  cout<<"opening "<<f_name<<endl;
  f = TFile::Open(f_name);

  TH1D* h0 = (TH1D*)getH(f, dir, name);
  TString s_name(name);
  int nb = h0->GetXaxis()->GetNbins();

  h = new TH1D(s_name+cname,title,30,ptscaleb_);
  for (int b=1 ; b<=nb; b++) {
    double bc = h0->GetBinContent(b);
    if (bc==0) continue;
    int bin = h->GetXaxis()->FindFixBin(h0->GetBinCenter(b));
    //cout<<b<<" "<<bc<<" "<<h0->GetBinCenter(b)<<" -> "<<bin<<endl;
    h->SetBinContent(bin, bc);
  }
  for (int b=1 ; b<=30; b++)
    h->SetBinContent(b, h->Integral(b,31));

  /*
    h = (TH1D*)h0->Clone(s_name+cname);
    for (int b=1 ; b<=nb; b++)
    h->SetBinContent(b, h0->Integral(b,nb+1));
  */

  h->Sumw2();
  h->Scale(40000./gNEvt/3.*0.795);

  h->SetLineColor(lcolor);
  h->SetFillColor(lcolor);
  h->SetLineStyle(lstyle);
  h->SetLineWidth(lwidth);

  h->SetTitle(title);

  h->GetXaxis()->SetRangeUser(2, 129.);
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

TH1D* setHistoRatio(TH1D* num, TH1D* denom, TString title = "", double ymin=0.4, double ymax=1.6, int color = kRed+3)
{
  ratio = (TH1D*) num->Clone(Form("%s--%s_ratio",num->GetName(),denom->GetName()) );
  ratio->Divide(num, denom, 1., 1.);
  ratio->SetTitle(title);

  ratio->GetYaxis()->SetRangeUser(ymin, ymax);
  ratio->GetYaxis()->SetTitle("ratio: (with GEM)/default");
  ratio->GetYaxis()->SetTitle("Ratio");
  // ratio->GetYaxis()->SetTitle("(ME1/b + GEM) / ME1/b");
  ratio->GetYaxis()->SetTitleSize(.14);
  // ratio->GetYaxis()->SetTitleSize(.1);
  ratio->GetYaxis()->SetTitleOffset(0.4);
  ratio->GetYaxis()->SetLabelSize(.11);

  //ratio->GetXaxis()->SetMoreLogLabels(1);
  ratio->GetXaxis()->SetTitle("p_{T}^{cut} [GeV/c]");
  ratio->GetXaxis()->SetLabelSize(.11);
  ratio->GetXaxis()->SetTitleSize(.14);
  ratio->GetXaxis()->SetTitleOffset(1.3); 

  ratio->SetLineWidth(2);
  ratio->SetFillColor(color);
  ratio->SetLineColor(color);
  ratio->SetMarkerColor(color);
  ratio->SetMarkerStyle(20);
  //ratio->Draw("e3");

  return ratio;
}


TH1D* setHistoRatio2(TH1D* num, TH1D* denom, TString title = "", double ymin=0.4, double ymax=1.6)
{
  ratio = (TH1D*) num->Clone(Form("%s--%s_ratio",num->GetName(),denom->GetName()) );
  ratio->Divide(num, denom, 1., 1.);
  ratio->SetTitle(title);
  ratio->GetYaxis()->SetRangeUser(ymin, ymax);
  ratio->GetYaxis()->SetTitle("ratio: with ME1a stub/without");
  ratio->GetYaxis()->SetLabelSize(0.07);
  ratio->GetYaxis()->SetTitleSize(0.07);
  ratio->GetYaxis()->SetTitleOffset(0.6);
  ratio->GetXaxis()->SetMoreLogLabels(1);
  ratio->SetLineWidth(2);
  ratio->SetFillColor(kRed+3);
  ratio->SetLineColor(kRed+3);
  ratio->SetMarkerColor(kRed+3);
  ratio->SetMarkerStyle(20);
  //ratio->Draw("e3");

  return ratio;
}



TH1D* getEffHisto(TString fname, TString hdir, TString num_name, TString den_name, int nrebin, int lcolor, int lstyle, int lwidth,
		  TString title, double *x_range, double *y_range)
{
  TFile *fh = TFile::Open(fname);

  TH1D* hd0 = (TH1D*)fh->Get(hdir + "/" + den_name);
  TH1D* hn0 = (TH1D*)fh->Get(hdir + "/" + num_name);

  TH1D* hd = (TH1D*)hd0->Clone(den_name+"_cln_"+fname);
  TH1D* hn = (TH1D*)hn0->Clone(num_name+"_cln_"+fname);
  hd->Sumw2();
  hn->Sumw2();

  myRebin(hd, nrebin);
  myRebin(hn, nrebin);

  TH1D* heff = (TH1D*)hn->Clone(num_name+"_eff_"+fname);

  hd->Sumw2();
  heff->Sumw2();

  heff->Divide(heff,hd);

  heff->SetLineColor(lcolor);
  heff->SetLineStyle(lstyle);
  heff->SetLineWidth(lwidth);

  heff->SetTitle(title);
  //heff->GetXaxis()->SetTitle(xtitle);
  //heff->GetYaxis()->SetTitle(ytitle);
  heff->GetXaxis()->SetRangeUser(x_range[0],x_range[1]);
  heff->GetYaxis()->SetRangeUser(y_range[0],y_range[1]);

  heff->GetXaxis()->SetTitleSize(0.07);
  heff->GetXaxis()->SetTitleOffset(0.7);
  heff->GetYaxis()->SetLabelOffset(0.015);

  heff->GetXaxis()->SetLabelSize(0.05);
  heff->GetYaxis()->SetLabelSize(0.05);

  h1 = hn0;
  h2 = hd0;
  he = heff;

  //fh->Close();
  return heff;
}


TLatex* drawPULabel(float x=0.17, float y=0.15, float font_size=0.)
{
  TLatex *  tex = new TLatex(x, y,"L=4*10^{34} (25ns PU100)");
  if (font_size > 0.) tex->SetFontSize(font_size);
  tex->SetNDC();
  tex->Draw();
  return tex;
}


void gem_rate_draw()
{
  do_not_print = false;

  gStyle->SetOptStat(0);
  gStyle->SetTitleStyle(0);
  //gStyle->SetPadTopMargin(0.08);
  gStyle->SetTitleH(0.06);

  int ptreb=2;

  TString hdir = "SimMuL1StrictAll";


  TString f_def = gem_dir + "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_def_pat2.root";
  TString f_g98_pt10 = gem_dir + "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_gem98_pt10_pat2.root";
  TString f_g98_pt15 = gem_dir + "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_gem98_pt15_pat2.root";
  TString f_g98_pt20 = gem_dir + "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_gem98_pt20_pat2.root";
  TString f_g98_pt30 = gem_dir + "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_gem98_pt30_pat2.root";
  TString f_g98_pt40 = gem_dir + "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_gem98_pt40_pat2.root";

  sprintf(pdir,"%s", gem_dir.Data());

  double rpt[2] = {0.,49.99};

  TString htitle = "Efficiency for #mu in 1.6<|#eta|<2.12 to have TF track;p_{T}^{MC}";

  TString hini = "h_pt_initial_1b";
  TString h2s = "h_pt_after_tfcand_eta1b_2s";
  TString h3s = "h_pt_after_tfcand_eta1b_3s";
  TString h2s1b = "h_pt_after_tfcand_eta1b_2s1b";
  TString h3s1b = "h_pt_after_tfcand_eta1b_3s1b";


  //gdy[0]=0; gdy[1]=7.;
  //if (vs_eta_minpt=="20") gdy[1]=10.;
  float miny = 0.01, maxy;

  TString vs_eta_minpt = "20";
  TString ttl = "CSC L1 trigger rates for p_{T}^{TF}>" + vs_eta_minpt + " GeV/c;track #eta;rate/bin [kHz]";


  TH1D* h_rt_tf20_2s   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s", "_hAll100", ttl, kAzure+2, 1, 2);
  TH1D* h_rt_tf20_2s1b   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s_2s1b", "_hAll100", ttl, kAzure+5, 1, 2);
  TH1D* h_rt_tf20_gpt20_2s1b   = setHistoEta(f_g98_pt20, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s_2s1b", "_hAll100", ttl, kGreen+1, 7, 2);

  TH1D* h_rt_tf20_3s   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s", "_hAll100", ttl, kAzure+3, 1, 2);
  TH1D* h_rt_tf20_3s1b   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1b", "_hAll100", ttl, kAzure+6, 1, 2);
  TH1D* h_rt_tf20_gpt20_3s1b   = setHistoEta(f_g98_pt20, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1b", "_hAll100", ttl, kGreen+3, 7, 2);
 
  TH1D* h_rt_tf20_3s1ab   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1ab", "_hAll100", ttl, kAzure+6, 1, 2);
  TH1D* h_rt_tf20_gpt20_3s1ab   = setHistoEta(f_g98_pt20, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1ab", "_hAll100", ttl, kGreen+3, 7, 2);


  // maxy = 300;// 45;
  // h_rt_tf20_2s->GetYaxis()->SetRangeUser(miny,maxy);
  // h_rt_tf20_2s1b->GetYaxis()->SetRangeUser(miny,maxy);
  // h_rt_tf20_gpt20_2s1b->GetYaxis()->SetRangeUser(miny,maxy);


  TCanvas* cAll100 = new TCanvas("cAll100","cAll100",800,600) ;
  cAll100->SetLogy(1);

  // h_rt_tf20_2s->Draw("hist e1");
  // h_rt_tf20_gpt20_2s1b->Draw("hist e1 same");
  // h_rt_tf20_2s->Draw("hist e1 same");
  // h_rt_tf20_2s1b->Draw("hist e1 same");

  // TLegend *leg = new TLegend(0.4,0.63,.98,0.90,NULL,"brNDC");
  // leg->SetBorderSize(0);
  // leg->SetFillStyle(0);
  // leg->AddEntry(h_rt_tf20_2s,"Tracks: p_{T}>=20, 2+ stubs","");
  // leg->AddEntry(h_rt_tf20_2s,"anywhere","l");
  // leg->AddEntry(h_rt_tf20_2s1b,"with ME1 in 1.6<|#eta|<2.14","l");
  // leg->AddEntry(h_rt_tf20_gpt20_2s1b,"with (ME1+GEM) in 1.6<|#eta|<2.14","l");
  // leg->Draw();

  // TLatex *  tex = new TLatex(0.17, 0.82,"#splitline{L=4*10^{34}}{(25ns PU100)}");
  // tex->SetNDC();
  // tex->Draw();

  // Print(cAll100, ("rates__vs_eta__minpt"+ vs_eta_minpt +"__PU100__def-2s-2s1b__gem-2s-2s1b.png").Data());


  // TCanvas* cAll100r = new TCanvas("cAll100r","cAll100r",800,300) ;
  // gPad->SetGridx(1);gPad->SetGridy(1);

  // gem_ratio = setHistoRatio(h_rt_tf20_gpt20_2s1b, h_rt_tf20_2s1b, "", 0.,1.8);
  // gem_ratio->Draw("e1");

  // Print(cAll100r, ("rates__vs_eta__minpt"+ vs_eta_minpt +"__PU100__def-2s-2s1b__gem-2s-2s1b__ratio.png").Data());


  // //========================  3+ Stubs ================================//


  // ((TCanvas*)gROOT->FindObject("cAll100"))->cd();

  maxy = 30.;//10;
  // h_rt_tf20_3s->GetYaxis()->SetRangeUser(miny,maxy);
  // h_rt_tf20_3s1b->GetYaxis()->SetRangeUser(miny,maxy);
  // h_rt_tf20_gpt20_3s1b->GetYaxis()->SetRangeUser(miny,maxy);

  // ((TCanvas*)gROOT->FindObject("cAll100"))->cd();
  // h_rt_tf20_3s->Draw("hist e1");
  // h_rt_tf20_gpt20_3s1b->Draw("hist e1 same");
  // h_rt_tf20_3s->Draw("hist e1 same");
  // h_rt_tf20_3s1b->Draw("hist e1 same");

  // TLegend *leg = new TLegend(0.4,0.63,.98,0.90,NULL,"brNDC");
  // leg->SetBorderSize(0);
  // leg->SetFillStyle(0);
  // leg->AddEntry(h_rt_tf20_3s,"Tracks: p_{T}>=20, 3+ stubs","");
  // leg->AddEntry(h_rt_tf20_3s,"anywhere","l");
  // leg->AddEntry(h_rt_tf20_3s1b,"with ME1 in 1.6<|#eta|<2.14","l");
  // leg->AddEntry(h_rt_tf20_gpt20_3s1b,"with (ME1+GEM) in 1.6<|#eta|<2.14","l");
  // leg->Draw();

  // TLatex *  tex = new TLatex(0.17, 0.82,"#splitline{L=4*10^{34}}{(25ns PU100)}");
  // tex->SetNDC();
  // tex->Draw();

  // Print(cAll100, ("rates__vs_eta__minpt"+ vs_eta_minpt +"__PU100__def-3s-3s1b__gem-3s-3s1b.png").Data());


  // TCanvas* cAll100r = new TCanvas("cAll100r","cAll100r",800,300) ;
  // gPad->SetGridx(1);gPad->SetGridy(1);

  // gem_ratio = setHistoRatio(h_rt_tf20_gpt20_3s1b, h_rt_tf20_3s1b, "", 0.,1.8);
  // gem_ratio->Draw("e1");

  // Print(cAll100r, ("rates__vs_eta__minpt"+ vs_eta_minpt +"__PU100__def-3s-3s1b__gem-3s-3s1b__ratio.png").Data());



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
  ttl = "CSC L1 trigger rates for p_{T}^{TF}>" + vs_eta_minpt + " GeV/c;track #eta;rate/bin [kHz]";


  // TH1D* h_rt_tf30_2s   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s", "_hAll100", ttl, kAzure+2, 1, 2);
  // TH1D* h_rt_tf30_2s1b   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s_2s1b", "_hAll100", ttl, kAzure+5, 1, 2);
  // TH1D* h_rt_tf30_gpt30_2s1b   = setHistoEta(f_g98_pt30, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s_2s1b", "_hAll100", ttl, kGreen+1, 7, 2);

 
  TH1D* h_rt_tf30_3s   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s", "_hAll100", ttl, kAzure+3, 1, 2);
 
  TH1D* h_rt_tf30_3s1b   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1b", "_hAll100", ttl, kAzure+6, 1, 2);
  TH1D* h_rt_tf30_3s1ab   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1ab", "_hAll100", ttl, kAzure+2, 1, 2);
  TH1D* h_rt_tf30_gpt30_3s1b   = setHistoEta(f_g98_pt30, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1b", "_hAll100", ttl, kGreen+3, 7, 2);
  TH1D* h_rt_tf30_gpt30_3s1ab   = setHistoEta(f_g98_pt30, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1ab", "_hAll100", ttl, kGreen, 7, 2);



  // // maxy = 120.;//35.;
  // // h_rt_tf30_2s->GetYaxis()->SetRangeUser(miny,maxy);
  // // h_rt_tf30_2s1b->GetYaxis()->SetRangeUser(miny,maxy);
  // // h_rt_tf30_gpt30_2s1b->GetYaxis()->SetRangeUser(miny,maxy);


  // TCanvas* cAll100 = new TCanvas("cAll100","cAll100",800,600) ;
  // cAll100->SetLogy(1);
  // // h_rt_tf30_2s->Draw("hist e1");
  // // h_rt_tf30_gpt30_2s1b->Draw("hist e1 same");
  // // h_rt_tf30_2s->Draw("hist e1 same");
  // // h_rt_tf30_2s1b->Draw("hist e1 same");

  // // TLegend *leg = new TLegend(0.4,0.63,.98,0.90,NULL,"brNDC");
  // // leg->SetBorderSize(0);
  // // leg->SetFillStyle(0);
  // // leg->AddEntry(h_rt_tf30_2s,"Tracks: p_{T}>=30, 2+ stubs","");
  // // leg->AddEntry(h_rt_tf30_2s,"anywhere","l");
  // // leg->AddEntry(h_rt_tf30_2s1b,"with ME1 in 1.6<|#eta|<2.14","l");
  // // leg->AddEntry(h_rt_tf30_gpt30_2s1b,"with (ME1+GEM) in 1.6<|#eta|<2.14","l");
  // // leg->Draw();

  // // TLatex *  tex = new TLatex(0.17, 0.82,"#splitline{L=4*10^{34}}{(25ns PU100)}");
  // // tex->SetNDC();
  // // tex->Draw();

  // // Print(cAll100, ("rates__vs_eta__minpt"+ vs_eta_minpt +"__PU100__def-2s-2s1b__gem-2s-2s1b.png").Data());


  // // TCanvas* cAll100r = new TCanvas("cAll100r","cAll100r",800,300) ;
  // // gPad->SetGridx(1);gPad->SetGridy(1);

  // // gem_ratio = setHistoRatio(h_rt_tf30_gpt30_2s1b, h_rt_tf30_2s1b, "", 0.,1.8);
  // // gem_ratio->Draw("e1");

  // // Print(cAll100r, ("rates__vs_eta__minpt"+ vs_eta_minpt +"__PU100__def-2s-2s1b__gem-2s-2s1b__ratio.png").Data());



  // ((TCanvas*)gROOT->FindObject("cAll100"))->cd();

  // maxy = 30.;//7.;
  // h_rt_tf30_3s->GetYaxis()->SetRangeUser(miny,maxy);
  // h_rt_tf30_3s1b->GetYaxis()->SetRangeUser(miny,maxy);
  // h_rt_tf30_gpt30_3s1b->GetYaxis()->SetRangeUser(miny,maxy);

  // ((TCanvas*)gROOT->FindObject("cAll100"))->cd();
  // h_rt_tf30_3s->Draw("hist e1");
  // h_rt_tf30_gpt30_3s1b->Draw("hist e1 same");
  // h_rt_tf30_3s->Draw("hist e1 same");
  // h_rt_tf30_3s1b->Draw("hist e1 same");

  // TLegend *leg = new TLegend(0.4,0.63,.98,0.90,NULL,"brNDC");
  // leg->SetBorderSize(0);
  // leg->SetFillStyle(0);
  // leg->AddEntry(h_rt_tf30_3s,"Tracks: p_{T}>=30, 3+ stubs","");
  // leg->AddEntry(h_rt_tf30_3s,"anywhere","l");
  // leg->AddEntry(h_rt_tf30_3s1b,"with ME1 in 1.6<|#eta|<2.14","l");
  // leg->AddEntry(h_rt_tf30_gpt30_3s1b,"with (ME1+GEM) in 1.6<|#eta|<2.14","l");
  // leg->Draw();

  // TLatex *  tex = new TLatex(0.17, 0.82,"#splitline{L=4*10^{34}}{(25ns PU100)}");
  // tex->SetNDC();
  // tex->Draw();

  // Print(cAll100, ("rates__vs_eta__minpt"+ vs_eta_minpt +"__PU100__def-3s-3s1b__gem-3s-3s1b.png").Data());




  // TCanvas* cAll100r = new TCanvas("cAll100r","cAll100r",800,300) ;
  // gPad->SetGridx(1);gPad->SetGridy(1);
 
  // gem_ratio = setHistoRatio(h_rt_tf30_gpt30_3s1b, h_rt_tf30_3s1b, "", 0.,1.8);
  // gem_ratio->Draw("e1");

  // Print(cAll100r, ("rates__vs_eta__minpt"+ vs_eta_minpt +"__PU100__def-3s-3s1b__gem-3s-3s1b__ratio.png").Data());





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
  
  
  TCanvas* cAll100r2 = new TCanvas("cAll100r2","cAll100r2",800,300) ;
  gPad->SetGridx(1);gPad->SetGridy(1);
  
  gem_ratio = setHistoRatio2(h_rt_tf30_3s1ab, h_rt_tf30_3s1b, "", 0.,1.8);
  gem_ratio->Draw("e1");
  
  Print(cAll100r2, ("rates__vs_eta__minpt"+ vs_eta_minpt +"__PU100__def-3s-3s1ab__gem-3s-3s1ab__ratio_compstubME1a.png").Data());
  




}



//###########################################################################
//###########################################################################

//void drawplot_gmtrt(char *fname, char *dname="tf")
void drawplot_gmtrt(TString dname = "", TString vs_eta_minpt = "")
{

  bool vs_eta = false;
  if (vs_eta_minpt.Length() > 0) vs_eta = true;

  // directory for plots is made as 
  // pdir = $PWD/pic[_{dname}]_{pu}

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



  char d1[111]="",d2[111]="";
  if (dname != "") sprintf(d1,"_%s", dname.Data());
  //if (strcmp(pu,"")>0)    sprintf(d2,"_%s",pu);
  //sprintf(pdir,"pic%s%s",d1,d2);

  if (interactive && dname != "") {
    sprintf(pdir,"pic%s%s",d1,d2);
    if( gSystem->AccessPathName(pdir)==0 ) {
      //cout<<"directory "<<pdir<<" exists, removing it!"<<endl;
      char cmd[111];
      //sprintf(cmd,"rm -r %s",pdir);
      //if (gSystem->Exec(cmd) != 0) {cout<<"can't remode directory! exiting..."<<endl; return;};
    }
    else {
      cout<<" creating directory "<<pdir<<endl;
      gSystem->MakeDirectory(pdir);
    }  
  }

  //cout<<"opening "<<fname<<endl;
  //f = TFile::Open(fname);

  // directory inside of root file:
  //char dir[111];
  char pu[15]="StrictChamber";
  if (strcmp(pu,"StrictDirect")==0) sprintf(dir,"SimMuL1Strict");
  if (strcmp(pu,"StrictChamber")==0) sprintf(dir,"SimMuL1StrictAll");
  if (strcmp(pu,"NaturalDirect")==0) sprintf(dir,"SimMuL1");
  if (strcmp(pu,"NaturalChamber")==0) sprintf(dir,"SimMuL1All");
  if (strcmp(pu,"StrictDeltaY")==0) sprintf(dir,"SimMuL1StrictDY");
  if (strcmp(pu,"NaturalDeltaY")==0) sprintf(dir,"SimMuL1DY");
  if (strcmp(pu,"StrictChamber0")==0) sprintf(dir,"SimMuL1StrictAll0");

  char label[200];

  //  = (TH1D*)f->Get("SimMuL1/;1");

  TString  f_pu100_pat8 = gem_dir;
  TString  f_pu100_pat8_gem = gem_dir;

  //TString  f_pu100_pat2 += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_pat2.root";
  //TString  f_pu100_pat8 ="hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_pat8.root";
  //TString  f_pu100_pat2_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_gem_pat2.root";
  //TString  f_pu100_pat8_gem ="hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_gem_pat8.root";

  //TString  f_pu100_pat8 ="hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_pat2.root";
  //TString  f_pu100_pat8_gem ="hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_gem_pat2.root";


  if (dname.Contains("_pat8")) f_pu100_pat8 += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_def_pat8.root";
  if (dname == "minbias_pt05_pat8") f_pu100_pat8_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt05_pat8.root";
  if (dname == "minbias_pt06_pat8") f_pu100_pat8_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt06_pat8.root";
  if (dname == "minbias_pt10_pat8") f_pu100_pat8_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt10_pat8.root";
  if (dname == "minbias_pt15_pat8") f_pu100_pat8_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt15_pat8.root";
  if (dname == "minbias_pt20_pat8") f_pu100_pat8_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt20_pat8.root";
  if (dname == "minbias_pt30_pat8") f_pu100_pat8_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt30_pat8.root";
  if (dname == "minbias_pt40_pat8") f_pu100_pat8_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt40_pat8.root";

  if (dname.Contains("_pat2")) f_pu100_pat8 += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_def_pat2.root";
  if (dname == "minbias_pt05_pat2") f_pu100_pat8_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt05_pat2.root";
  if (dname == "minbias_pt06_pat2") f_pu100_pat8_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt06_pat2.root";
  if (dname == "minbias_pt10_pat2") f_pu100_pat8_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt10_pat2.root";
  if (dname == "minbias_pt15_pat2") f_pu100_pat8_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt15_pat2.root";
  if (dname == "minbias_pt20_pat2") f_pu100_pat8_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt20_pat2.root";
  if (dname == "minbias_pt30_pat2") f_pu100_pat8_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt30_pat2.root";
  if (dname == "minbias_pt40_pat2") f_pu100_pat8_gem += "hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_" + gem_label + "_pt40_pat2.root";

  //TString  f_pu100_pat8_gem ="hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_gemx_pt20_pat2.root";
  //TString  f_pu100_pat8_gem ="hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_gemOldx_pt20_pat2.root";
  //TString  f_pu100_pat8_gem ="hp_minbias_6_0_1_POSTLS161_V12__pu100_w3_gemOld_pt20_pat2.root";


  bool do_return = false;



  //#################################### PU100 ####################################

  // full eta 1. - 2.4    Default: 3station, 2s & 1b    GEM: 3station, 2s & 1b
  if (vs_eta)
    {
      gdy[0]=0; gdy[1]=7.;
      if (vs_eta_minpt=="20") gdy[1]=10.;

      TString ttl = "CSC L1 trigger rates for p_{T}^{TF}>" + vs_eta_minpt + " GeV/c;track #eta;rate/bin [kHz]";
      hAll100    = setHistoEta(f_pu100_pat8,     "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_2s1b", "_hAll100", ttl, kAzure+9, 1, 2);
      hAll100gem = setHistoEta(f_pu100_pat8_gem, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_2s1b", "_hAll100gem", ttl, kGreen+3, 1, 2);

      //gStyle->SetPadTopMargin(0.08);
      gStyle->SetTitleH(0.06);
      TCanvas* cAll100 = new TCanvas("cAll100","cAll100",800,600) ;
      hAll100->Draw("hist e1");
      hAll100gem->Draw("hist e1 same");
      TLegend *leg_cc100 = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
      const TObject obj;
      leg_cc100->SetBorderSize(0);
      //leg_cc100->SetTextSize(0.0368);
      leg_cc100->SetFillStyle(0);
      leg_cc100->AddEntry(hAll100,"default emulator","l");
      leg_cc100->AddEntry(hAll100gem,"with GEM match","l");
      leg_cc100->AddEntry(hAll100,"Tracks req. for both:","");
      leg_cc100->AddEntry(hAll100,">=3 stubs in 1<|#eta|<2.4","");
      leg_cc100->AddEntry(hAll100,"except in ME1/b region require","");
      leg_cc100->AddEntry(hAll100,">=2 stubs and one of them from ME1/b","");
      leg_cc100->Draw();

      TLatex *  tex = new TLatex(0.17, 0.82,"#splitline{L=4*10^{34}}{(25ns PU100)}");
      tex->SetNDC();
      tex->Draw();

      Print(cAll100, ("rates__vs_eta__minpt"+ vs_eta_minpt +"__PU100__def-3s-2s1b__gem-3s-2s1b.png").Data());

      if (do_return) return;
    }



  // full eta 1. - 2.4    Default: 3station, 3s & 1b    GEM: 3station, 3s & 1b
  if (vs_eta)
    {
      gdy[0]=0; gdy[1]=7.;
      if (vs_eta_minpt=="20") gdy[1]=10.;

      TString ttl = "CSC L1 trigger rates for p_{T}^{TF}>" + vs_eta_minpt + " GeV/c;track #eta;rate/bin [kHz]";
      hAll100    = setHistoEta(f_pu100_pat8,     "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1b", "_hAll100", ttl, kAzure+9, 1, 2);
      hAll100gem = setHistoEta(f_pu100_pat8_gem, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1b", "_hAll100gem", ttl, kGreen+3, 1, 2);

      //gStyle->SetPadTopMargin(0.08);
      gStyle->SetTitleH(0.06);
      TCanvas* cAll100 = new TCanvas("cAll100","cAll100",800,600) ;
      hAll100->Draw("hist e1");
      hAll100gem->Draw("hist e1 same");
      TLegend *leg_cc100 = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
      const TObject obj;
      leg_cc100->SetBorderSize(0);
      //leg_cc100->SetTextSize(0.0368);
      leg_cc100->SetFillStyle(0);
      leg_cc100->AddEntry(hAll100,"default emulator","l");
      leg_cc100->AddEntry(hAll100gem,"with GEM match","l");
      leg_cc100->AddEntry(hAll100,"Tracks req. for both:","");
      leg_cc100->AddEntry(hAll100,">=3 stubs in 1<|#eta|<2.4","");
      leg_cc100->AddEntry(hAll100,"in ME1/b region also require","");
      leg_cc100->AddEntry(hAll100,"one stub to be from ME1/b","");
      leg_cc100->Draw();

      TLatex *  tex = new TLatex(0.17, 0.82,"#splitline{L=4*10^{34}}{(25ns PU100)}");
      tex->SetNDC();
      tex->Draw();

      Print(cAll100, ("rates__vs_eta__minpt"+ vs_eta_minpt +"__PU100__def-3s-3s1b__gem-3s-3s1b.png").Data());

      if (do_return) return;
    }

  // full eta 1. - 2.4    Default: 3station    GEM: 3station, 2s & 1b
  if (vs_eta)
    {
      gdy[0]=0; gdy[1]=7.;
      if (vs_eta_minpt=="20") gdy[1]=10.;

      TString ttl = "CSC L1 trigger rates for p_{T}^{TF}>" + vs_eta_minpt + " GeV/c;track #eta;rate/bin [kHz]";
      hAll100    = setHistoEta(f_pu100_pat8,     "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s", "_hAll100", ttl, kAzure+9, 1, 2);
      hAll100gem = setHistoEta(f_pu100_pat8_gem, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_2s1b", "_hAll100gem", ttl, kGreen+3, 1, 2);

      //gStyle->SetPadTopMargin(0.08);
      gStyle->SetTitleH(0.06);
      TCanvas* cAll100 = new TCanvas("cAll100","cAll100",800,600) ;
      hAll100->Draw("hist e1");
      hAll100gem->Draw("hist e1 same");
      TLegend *leg_cc100 = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
      const TObject obj;
      leg_cc100->SetBorderSize(0);
      //leg_cc100->SetTextSize(0.0368);
      leg_cc100->SetFillStyle(0);
      leg_cc100->AddEntry(hAll100,"default emulator","l");
      leg_cc100->AddEntry(hAll100,">=3 stubs in 1<|#eta|<2.4","");
      leg_cc100->AddEntry(hAll100gem,"with GEM match","l");
      leg_cc100->AddEntry(hAll100,">=3 stubs in 1<|#eta|<2.4","");
      leg_cc100->AddEntry(hAll100,"except in ME1/b region require","");
      leg_cc100->AddEntry(hAll100,">=2 stubs and one of them from ME1/b","");
      leg_cc100->Draw();

      TLatex *  tex = new TLatex(0.17, 0.82,"#splitline{L=4*10^{34}}{(25ns PU100)}");
      tex->SetNDC();
      tex->Draw();

      Print(cAll100, ("rates__vs_eta__minpt"+ vs_eta_minpt +"__PU100__def-3s__gem-3s-2s1b.png").Data());

      if (do_return) return;
    }


  // full eta 1. - 2.4    Default: 3station    GEM: 3station, 3s & 1b
  if (vs_eta)
    {
      gdy[0]=0; gdy[1]=7.;
      if (vs_eta_minpt=="20") gdy[1]=10.;

      TString ttl = "CSC L1 trigger rates for p_{T}^{TF}>" + vs_eta_minpt + " GeV/c;track #eta;rate/bin [kHz]";
      hAll100    = setHistoEta(f_pu100_pat8,     "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s", "_hAll100", ttl, kAzure+9, 1, 2);
      hAll100gem = setHistoEta(f_pu100_pat8_gem, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1b", "_hAll100gem", ttl, kGreen+3, 1, 2);

      //gStyle->SetPadTopMargin(0.08);
      gStyle->SetTitleH(0.06);
      TCanvas* cAll100 = new TCanvas("cAll100","cAll100",800,600) ;
      hAll100->Draw("hist e1");
      hAll100gem->Draw("hist e1 same");
      TLegend *leg_cc100 = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
      const TObject obj;
      leg_cc100->SetBorderSize(0);
      //leg_cc100->SetTextSize(0.0368);
      leg_cc100->SetFillStyle(0);
      leg_cc100->AddEntry(hAll100,"default emulator","l");
      leg_cc100->AddEntry(hAll100,">=3 stubs in 1<|#eta|<2.4","");
      leg_cc100->AddEntry(hAll100gem,"with GEM match","l");
      leg_cc100->AddEntry(hAll100,">=3 stubs in 1<|#eta|<2.4","");
      leg_cc100->AddEntry(hAll100,"in ME1/b region also require","");
      leg_cc100->AddEntry(hAll100,"one stub to be from ME1/b","");
      leg_cc100->Draw();

      TLatex *  tex = new TLatex(0.17, 0.82,"#splitline{L=4*10^{34}}{(25ns PU100)}");
      tex->SetNDC();
      tex->Draw();

      Print(cAll100, ("rates__vs_eta__minpt"+ vs_eta_minpt +"__PU100__def-3s__gem-3s-3s1b.png").Data());

      if (do_return) return;
    }


  if (vs_eta) return;



  // full eta 1. - 2.4    Default: 3station, 2s & 1b    GEM: 3station, 2s & 1b
  if (1)
    {
      gdy[0]=2; gdy[1]=2000.;

      hAll100    = setHisto(f_pu100_pat8,     "h_rt_gmt_csc_ptmax_3s_2s1b", "_hAll100", "CSC L1 trigger rates;p_{T}^{cut} [GeV/c];rate [kHz]", kAzure+9, 1, 1);
      hAll100gem = setHisto(f_pu100_pat8_gem, "h_rt_gmt_csc_ptmax_3s_2s1b", "_hAll100gem", "CSC L1 trigger rates;p_{T}^{cut} [GeV/c];rate [kHz]", kGreen+3, 1, 1);

      TCanvas* cAll100 = new TCanvas("cAll100","cAll100",800,600) ;
      gPad->SetLogx(1);gPad->SetLogy(1);
      gPad->SetGridx(1);gPad->SetGridy(1);
      hAll100->Draw("e3");
      hAll100gem->Draw("e3 same");
      TLegend *leg_cc100 = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
      const TObject obj;
      leg_cc100->SetBorderSize(0);
      //leg_cc100->SetTextSize(0.0368);
      leg_cc100->SetFillStyle(0);
      leg_cc100->AddEntry(hAll100,"default emulator","f");
      leg_cc100->AddEntry(hAll100gem,"with GEM match","f");
      leg_cc100->AddEntry(hAll100,"Tracks req. for both:","");
      leg_cc100->AddEntry(hAll100,">=3 stubs in 1<|#eta|<2.4","");
      leg_cc100->AddEntry(hAll100,"except in ME1/b region require","");
      leg_cc100->AddEntry(hAll100,">=2 stubs and one of them from ME1/b","");
      leg_cc100->Draw();

      TLatex *  tex = new TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
      tex->SetNDC();
      tex->Draw();

      Print(cAll100, "rates__1-2.4__PU100__def-3s-2s1b__gem-3s-2s1b.png");


      TCanvas* cAll100r = new TCanvas("cAll100r","cAll100r",800,300) ;
      gPad->SetLogx(1);//gPad->SetLogy(1);
      gPad->SetGridx(1);gPad->SetGridy(1);

      hAll100gem_ratio = setHistoRatio(hAll100gem, hAll100, "", 0.4,1.4);
      hAll100gem_ratio->Draw("e1");

      Print(cAll100r, "rates__1-2.4__PU100__def-3s-2s1b__gem-3s-2s1b__ratio.png");

      if (do_return) return;
    }



  // full eta 1. - 2.4    Default: 3station   GEM: 3station, 2s & 1b
  if (1)
    {
      gdy[0]=2; gdy[1]=2000.;

      hAll100    = setHisto(f_pu100_pat8,     "h_rt_gmt_csc_ptmax_3s", "_hAll100s3", "CSC L1 trigger rates;p_{T}^{cut} [GeV/c];rate [kHz]", kAzure+9, 1, 1);
      hAll100gem = setHisto(f_pu100_pat8_gem, "h_rt_gmt_csc_ptmax_3s_2s1b", "_hAll100gem", "CSC L1 trigger rates;p_{T}^{cut} [GeV/c];rate [kHz]", kGreen+3, 1, 1);

      TCanvas* cAll100 = new TCanvas("cAll100","cAll100",800,600) ;
      gPad->SetLogx(1);gPad->SetLogy(1);
      gPad->SetGridx(1);gPad->SetGridy(1);
      hAll100->Draw("e3");
      hAll100gem->Draw("e3 same");
      TLegend *leg_cc100 = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
      const TObject obj;
      leg_cc100->SetBorderSize(0);
      //leg_cc100->SetTextSize(0.0368);
      leg_cc100->SetFillStyle(0);
      leg_cc100->AddEntry(hAll100,"default emulator","f");
      leg_cc100->AddEntry(hAll100,"Tracks: with >=3 stubs in 1<|#eta|<2.4","");
      leg_cc100->AddEntry(hAll100gem,"with GEM match","f");
      leg_cc100->AddEntry(hAll100,"Tracks: same, except in ME1/b region req.","");
      leg_cc100->AddEntry(hAll100,">=2 stubs and one of them from ME1/b","");
      leg_cc100->Draw();

      TLatex *  tex = new TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
      tex->SetNDC();
      tex->Draw();

      Print(cAll100, "rates__1-2.4__PU100__def-3s__gem-3s-2s1b.png");


      TCanvas* cAll100r = new TCanvas("cAll100r","cAll100r",800,300) ;
      gPad->SetLogx(1);//gPad->SetLogy(1);
      gPad->SetGridx(1);gPad->SetGridy(1);

      hAll100gem_ratio = setHistoRatio(hAll100gem, hAll100, "", 0.4,1.2);
      hAll100gem_ratio->Draw("e1");

      Print(cAll100r, "rates__1-2.4__PU100__def-3s__gem-3s-2s1b__ratio.png");

      if (do_return) return;
    }



  // no ME1/a eta 1. - 2.1    Default: 3station   GEM: 3station, 3s & 1b
  if (1)
    {
      gdy[0]=2; gdy[1]=2000.;

      hAll100    = setHisto(f_pu100_pat8,     "h_rt_gmt_csc_ptmax_3s_no1a", "_hAll100s3", "CSC L1 trigger rates;p_{T}^{cut} [GeV/c];rate [kHz]", kAzure+9, 1, 1);
      hAll100gem = setHisto(f_pu100_pat8_gem, "h_rt_gmt_csc_ptmax_3s_3s1b_no1a", "_hAll100gem", "CSC L1 trigger rates;p_{T}^{cut} [GeV/c];rate [kHz]", kGreen+3, 1, 1);

      TCanvas* cAll100 = new TCanvas("cAll100","cAll100",800,600) ;
      gPad->SetLogx(1);gPad->SetLogy(1);
      gPad->SetGridx(1);gPad->SetGridy(1);
      hAll100->Draw("e3");
      hAll100gem->Draw("e3 same");
      TLegend *leg_cc100 = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
      const TObject obj;
      leg_cc100->SetBorderSize(0);
      //leg_cc100->SetTextSize(0.0368);
      leg_cc100->SetFillStyle(0);
      leg_cc100->AddEntry(hAll100,"default emulator","f");
      leg_cc100->AddEntry(hAll100,"Tracks: with >=3 stubs in 1<|#eta|<2.1","");
      leg_cc100->AddEntry(hAll100gem,"with GEM match","f");
      leg_cc100->AddEntry(hAll100,"Tracks: same, except in ME1/b region req.","");
      leg_cc100->AddEntry(hAll100,">=3 stubs and one of them from ME1/b","");
      leg_cc100->Draw();

      TLatex *  tex = new TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
      tex->SetNDC();
      tex->Draw();

      Print(cAll100, "rates__1-2.1__PU100__def-3s__gem-3s-3s1b.png");


      TCanvas* cAll100r = new TCanvas("cAll100r","cAll100r",800,300) ;
      gPad->SetLogx(1);
      gPad->SetGridx(1);gPad->SetGridy(1);

      hAll100gem_ratio = setHistoRatio(hAll100gem, hAll100, "", 0.4,1.2);
      hAll100gem_ratio->Draw("e1");

      Print(cAll100r, "rates__1-2.1__PU100__def-3s__gem-3s-3s1b__ratio.png");


      result_gem_eta_no1a = hAll100gem;
      result_def_eta_no1a = hAll100;

      if (do_return) return;
    }


  // no ME1/a eta 1. - 2.1    Default: 3station, 3s & 1b   GEM: 3station, 3s & 1b
  if (1)
    {
      gdy[0]=2; gdy[1]=2000.;

      hAll100    = setHisto(f_pu100_pat8,     "h_rt_gmt_csc_ptmax_3s_3s1b_no1a", "_hAll100s3", "CSC L1 trigger rates;p_{T}^{cut} [GeV/c];rate [kHz]", kAzure+9, 1, 1);
      hAll100gem = setHisto(f_pu100_pat8_gem, "h_rt_gmt_csc_ptmax_3s_3s1b_no1a", "_hAll100gem", "CSC L1 trigger rates;p_{T}^{cut} [GeV/c];rate [kHz]", kGreen+3, 1, 1);

      TCanvas* cAll100 = new TCanvas("cAll100","cAll100",800,600) ;
      gPad->SetLogx(1);gPad->SetLogy(1);
      gPad->SetGridx(1);gPad->SetGridy(1);
      hAll100->Draw("e3");
      hAll100gem->Draw("e3 same");
      TLegend *leg_cc100 = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
      const TObject obj;
      leg_cc100->SetBorderSize(0);
      //leg_cc100->SetTextSize(0.0368);
      leg_cc100->SetFillStyle(0);
      leg_cc100->AddEntry(hAll100,"default emulator","f");
      leg_cc100->AddEntry(hAll100gem,"with GEM match","f");
      leg_cc100->AddEntry(hAll100,"Tracks req. for both:","");
      leg_cc100->AddEntry(hAll100,"with >=3 stubs in 1<|#eta|<2.1","");
      leg_cc100->AddEntry(hAll100,"for ME1/b etas require one stub from ME1/b","");
      leg_cc100->Draw();

      TLatex *  tex = new TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
      tex->SetNDC();
      tex->Draw();

      Print(cAll100, "rates__1-2.1__PU100__def-3s-3s1b__gem-3s-3s1b.png");


      TCanvas* cAll100r = new TCanvas("cAll100r","cAll100r",800,300) ;
      gPad->SetLogx(1);
      gPad->SetGridx(1);gPad->SetGridy(1);

      hAll100gem_ratio = setHistoRatio(hAll100gem, hAll100, "", 0.4,1.2);
      hAll100gem_ratio->Draw("e1");

      Print(cAll100r, "rates__1-2.1__PU100__def-3s-3s1b__gem-3s-3s1b__ratio.png");

      //result_gem_eta_no1a = hAll100gem;
      result_def_eta_no1a_3s1b = hAll100;

      if (do_return) return;
    }


  // Full eta 1. - 2.4    Default: 3station   GEM: 3station, 3s & 1b
  if (1)
    {
      gdy[0]=2; gdy[1]=2000.;

      hAll100    = setHisto(f_pu100_pat8,     "h_rt_gmt_csc_ptmax_3s", "_hAll100s3", "CSC L1 trigger rates;p_{T}^{cut} [GeV/c];rate [kHz]", kAzure+9, 1, 1);
      hAll100gem = setHisto(f_pu100_pat8_gem, "h_rt_gmt_csc_ptmax_3s_3s1b", "_hAll100gem", "CSC L1 trigger rates;p_{T}^{cut} [GeV/c];rate [kHz]", kGreen+3, 1, 1);

      TCanvas* cAll100 = new TCanvas("cAll100","cAll100",800,600) ;
      gPad->SetLogx(1);gPad->SetLogy(1);
      gPad->SetGridx(1);gPad->SetGridy(1);
      hAll100->Draw("e3");
      hAll100gem->Draw("e3 same");
      TLegend *leg_cc100 = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
      const TObject obj;
      leg_cc100->SetBorderSize(0);
      //leg_cc100->SetTextSize(0.0368);
      leg_cc100->SetFillStyle(0);
      leg_cc100->AddEntry(hAll100,"default emulator","f");
      leg_cc100->AddEntry(hAll100,"Tracks: with >=3 stubs in 1<|#eta|<2.4","");
      leg_cc100->AddEntry(hAll100gem,"with GEM match","f");
      leg_cc100->AddEntry(hAll100,"Tracks: same, except","");
      leg_cc100->AddEntry(hAll100,"for ME1/b etas require one stub from ME1/b","");
      leg_cc100->Draw();

      TLatex *  tex = new TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
      tex->SetNDC();
      tex->Draw();

      Print(cAll100, "rates__1-2.4__PU100__def-3s__gem-3s-3s1b.png");


      TCanvas* cAll100r = new TCanvas("cAll100r","cAll100r",800,300) ;
      gPad->SetLogx(1);
      gPad->SetGridx(1);gPad->SetGridy(1);

      hAll100gem_ratio = setHistoRatio(hAll100gem, hAll100, "", 0.4,1.2);
      hAll100gem_ratio->Draw("e1");

      Print(cAll100r, "rates__1-2.4__PU100__def-3s__gem-3s-3s1b__ratio.png");

      result_gem_eta_all = hAll100gem;
      result_def_eta_all = hAll100;

      if (do_return) return;
    }



  // Full eta 1. - 2.4    Default: 3station, 3s & 1b   GEM: 3station, 3s & 1b
  if (1)
    {
      gdy[0]=2; gdy[1]=2000.;

      hAll100    = setHisto(f_pu100_pat8,     "h_rt_gmt_csc_ptmax_3s_3s1b", "_hAll100s3", "CSC L1 trigger rates;p_{T}^{cut} [GeV/c];rate [kHz]", kAzure+9, 1, 1);
      hAll100gem = setHisto(f_pu100_pat8_gem, "h_rt_gmt_csc_ptmax_3s_3s1b", "_hAll100gem", "CSC L1 trigger rates;p_{T}^{cut} [GeV/c];rate [kHz]", kGreen+3, 1, 1);

      TCanvas* cAll100 = new TCanvas("cAll100","cAll100",800,600) ;
      gPad->SetLogx(1);gPad->SetLogy(1);
      gPad->SetGridx(1);gPad->SetGridy(1);
      hAll100->Draw("e3");
      hAll100gem->Draw("e3 same");
      TLegend *leg_cc100 = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
      const TObject obj;
      leg_cc100->SetBorderSize(0);
      //leg_cc100->SetTextSize(0.0368);
      leg_cc100->SetFillStyle(0);
      leg_cc100->AddEntry(hAll100,"default emulator","f");
      leg_cc100->AddEntry(hAll100gem,"with GEM match","f");
      leg_cc100->AddEntry(hAll100,"Tracks req. for both:","");
      leg_cc100->AddEntry(hAll100,"with >=3 stubs in 1<|#eta|<2.4","");
      leg_cc100->AddEntry(hAll100,"for ME1/b etas require one stub from ME1/b","");
      leg_cc100->Draw();

      TLatex *  tex = new TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
      tex->SetNDC();
      tex->Draw();

      Print(cAll100, "rates__1-2.4__PU100__def-3s-3s1b__gem-3s-3s1b.png");


      TCanvas* cAll100r = new TCanvas("cAll100r","cAll100r",800,300) ;
      gPad->SetLogx(1);
      gPad->SetGridx(1);gPad->SetGridy(1);

      hAll100gem_ratio = setHistoRatio(hAll100gem, hAll100, "", 0.4,1.6);
      hAll100gem_ratio->Draw();

      Print(cAll100r, "rates__1-2.4__PU100__def-3s-3s1b__gem-3s-3s1b__ratio.png");

      //result_gem_eta_all = hAll100gem;
      result_def_eta_all_3s1b = hAll100;

      if (do_return) return;
    }



  // ME1b eta 1.64 - 2.14    Default: 3station, 3s & 1b   GEM: 3station, 3s & 1b
  if (1)
    {
      gdy[0]=0.02; gdy[1]=1000.;

      hAll100    = setHisto(f_pu100_pat8,     "h_rt_gmt_csc_ptmax_3s_3s1b_1b", "_hAll100s3", "CSC L1 trigger rates in ME1/b eta region;p_{T}^{cut} [GeV/c];rate [kHz]", kAzure+9, 1, 1);
      hAll100gem = setHisto(f_pu100_pat8_gem, "h_rt_gmt_csc_ptmax_3s_3s1b_1b", "_hAll100gem", "CSC L1 trigger rates in ME1/b eta region;p_{T}^{cut} [GeV/c];rate [kHz]", kGreen+3, 1, 1);

      TCanvas* cAll100 = new TCanvas("cAll100","cAll100",800,600) ;
      gPad->SetLogx(1);gPad->SetLogy(1);
      gPad->SetGridx(1);gPad->SetGridy(1);
      hAll100->Draw("e3");
      hAll100gem->Draw("e3 same");
      TLegend *leg_cc100 = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
      const TObject obj;
      leg_cc100->SetBorderSize(0);
      //leg_cc100->SetTextSize(0.0368);
      leg_cc100->SetFillStyle(0);
      leg_cc100->AddEntry(hAll100,"default emulator","f");
      leg_cc100->AddEntry(hAll100gem,"with GEM match","f");
      leg_cc100->AddEntry(hAll100,"Tracks req. for both:","");
      leg_cc100->AddEntry(hAll100,"with >=3 stubs in 1.64<|#eta|<2.14","");
      leg_cc100->AddEntry(hAll100,"and require one stub to be from ME1/b","");
      leg_cc100->Draw();

      TLatex *  tex = new TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
      tex->SetNDC();
      tex->Draw();

      Print(cAll100, "rates__1.6-2.1_PU100__def-3s-3s1b__gem-3s-3s1b.png");


      TCanvas* cAll100r = new TCanvas("cAll100r","cAll100r",800,300) ;
      gPad->SetLogx(1);
      gPad->SetGridx(1);gPad->SetGridy(1);

      hAll100gem_ratio = setHistoRatio(hAll100gem, hAll100, "", 0.,2.1);
      hAll100gem_ratio->Draw("e1");

      Print(cAll100r, "rates__1.6-2.1_PU100__def-3s-3s1b__gem-3s-3s1b__ratio.png");

      result_gem = hAll100gem;
      result_def_3s1b = hAll100;

      if (do_return) return;
    }



  // ME1b eta 1.64 - 2.14    Default: 3station, 3s   GEM: 3station, 3s & 1b
  if (1)
    {
      gdy[0]=0.02; gdy[1]=1000.;

      hAll100    = setHisto(f_pu100_pat8,     "h_rt_gmt_csc_ptmax_3s_1b", "_hAll100s3", "CSC L1 trigger rates in ME1/b eta region;p_{T}^{cut} [GeV/c];rate [kHz]", kAzure+9, 1, 1);
      hAll100gem = setHisto(f_pu100_pat8_gem, "h_rt_gmt_csc_ptmax_3s_3s1b_1b", "_hAll100gem", "CSC L1 trigger rates in ME1/b eta region;p_{T}^{cut} [GeV/c];rate [kHz]", kGreen+3, 1, 1);

      TCanvas* cAll100 = new TCanvas("cAll100","cAll100",800,600) ;
      gPad->SetLogx(1);gPad->SetLogy(1);
      gPad->SetGridx(1);gPad->SetGridy(1);
      hAll100->Draw("e3");
      hAll100gem->Draw("e3 same");
      TLegend *leg_cc100 = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
      const TObject obj;
      leg_cc100->SetBorderSize(0);
      //leg_cc100->SetTextSize(0.0368);
      leg_cc100->SetFillStyle(0);
      leg_cc100->AddEntry(hAll100,"default emulator","f");
      leg_cc100->AddEntry(hAll100,"Tracks: with >=3 stubs in 1.64<|#eta|<2.14","");
      leg_cc100->AddEntry(hAll100gem,"with GEM match","f");
      leg_cc100->AddEntry(hAll100,"Tracks: same, plus req. one stub from ME1/b","");
      leg_cc100->Draw();

      TLatex *  tex = new TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
      tex->SetNDC();
      tex->Draw();

      Print(cAll100, "rates__1.6-2.1_PU100__def-3s__gem-3s-3s1b.png");


      TCanvas* cAll100r = new TCanvas("cAll100r","cAll100r",800,300) ;
      gPad->SetLogx(1);
      gPad->SetGridx(1);gPad->SetGridy(1);

      hAll100gem_ratio = setHistoRatio(hAll100gem, hAll100, "", 0.,1.1);
      hAll100gem_ratio->Draw("e1");

      Print(cAll100r, "rates__1.6-2.1_PU100__def-3s__gem-3s-3s1b__ratio.png");

      //result_gem = hAll100gem;
      result_def = hAll100;

      hAll100    = setHisto(f_pu100_pat8,     "h_rt_gmt_csc_ptmax_2s_1b", "_hAll100s2", "CSC L1 trigger rates in ME1/b eta region;p_{T}^{cut} [GeV/c];rate [kHz]", kAzure+9, 1, 1);
      result_def_2s = hAll100;

      hAll100    = setHisto(f_pu100_pat8,     "h_rt_gmt_csc_ptmax_2s_1b", "_hAll100s2", "CSC L1 trigger rates in ME1/b eta region;p_{T}^{cut} [GeV/c];rate [kHz]", kAzure+9, 1, 1);
      result_def_2s = hAll100;

      if (do_return) return;
    }



  // ME1b eta 1.64 - 2.14    Default: 3station, 3s   GMT single trigg
  if (1)
    {
      gdy[0]=0.02; gdy[1]=1000.;

      hAll100    = setHisto(f_pu100_pat8,     "h_rt_gmt_csc_ptmax_3s_1b", "_hAll100s3", "CSC L1 trigger rates in ME1/b eta region;p_{T}^{cut} [GeV/c];rate [kHz]", kAzure+9, 1, 1);
      hAll100gem = setHisto(f_pu100_pat8_gem, "h_rt_gmt_ptmax_sing_1b", "_hAll100gem", "CSC L1 trigger rates in ME1/b eta region;p_{T}^{cut} [GeV/c];rate [kHz]", kAzure+1, 1, 1);

      TCanvas* cAll100 = new TCanvas("cAll100","cAll100",800,600) ;
      gPad->SetLogx(1);gPad->SetLogy(1);
      gPad->SetGridx(1);gPad->SetGridy(1);
      hAll100->Draw("e3");
      hAll100gem->Draw("e3 same");
      TLegend *leg_cc100 = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
      const TObject obj;
      leg_cc100->SetBorderSize(0);
      //leg_cc100->SetTextSize(0.0368);
      leg_cc100->SetFillStyle(0);
      leg_cc100->AddEntry(hAll100,"Tracks: with >=3 stubs in 1.64<|#eta|<2.14","");
      leg_cc100->AddEntry(hAll100gem,"GMT selection for Single Trigger","f");
      leg_cc100->Draw();

      TLatex *  tex = new TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
      tex->SetNDC();
      tex->Draw();

      Print(cAll100, "rates__1.6-2.1_PU100__def-3s__gmtsing.png");

      result_def_gmtsing = hAll100gem;

      if (do_return) return;
    }


  // ME1b eta 1.64 - 2.14    Default: 3station, 2s & 1b   GEM: 3station, 2s & 1b
  if (1)
    {
      gdy[0]=0.02; gdy[1]=1000.;

      hAll100    = setHisto(f_pu100_pat8,     "h_rt_gmt_csc_ptmax_3s_2s1b_1b", "_hAll100s3", "CSC L1 trigger rates in ME1/b eta region;p_{T}^{cut} [GeV/c];rate [kHz]", kAzure+9, 1, 1);
      hAll100gem = setHisto(f_pu100_pat8_gem, "h_rt_gmt_csc_ptmax_3s_2s1b_1b", "_hAll100gem", "CSC L1 trigger rates in ME1/b eta region;p_{T}^{cut} [GeV/c];rate [kHz]", kGreen+3, 1, 1);

      TCanvas* cAll100 = new TCanvas("cAll100","cAll100",800,600) ;
      gPad->SetLogx(1);gPad->SetLogy(1);
      gPad->SetGridx(1);gPad->SetGridy(1);
      hAll100->Draw("e3");
      hAll100gem->Draw("e3 same");
      TLegend *leg_cc100 = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
      const TObject obj;
      leg_cc100->SetBorderSize(0);
      //leg_cc100->SetTextSize(0.0368);
      leg_cc100->SetFillStyle(0);
      leg_cc100->AddEntry(hAll100,"default emulator","f");
      leg_cc100->AddEntry(hAll100gem,"with GEM match","f");
      leg_cc100->AddEntry(hAll100,"Tracks req. for both:","");
      leg_cc100->AddEntry(hAll100,"with >=2 stubs in 1.64<|#eta|<2.14","");
      leg_cc100->AddEntry(hAll100,"and require one stub to be from ME1/b","");
      leg_cc100->Draw();

      TLatex *  tex = new TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
      tex->SetNDC();
      tex->Draw();

      Print(cAll100, "rates__1.6-2.1_PU100__def-3s-2s1b__gem-3s-2s1b.png");


      TCanvas* cAll100r = new TCanvas("cAll100r","cAll100r",800,300) ;
      gPad->SetLogx(1);
      gPad->SetGridx(1);gPad->SetGridy(1);

      hAll100gem_ratio = setHistoRatio(hAll100gem, hAll100, "", 0.,2.1);
      hAll100gem_ratio->Draw("e1");

      Print(cAll100r, "rates__1.6-2.1_PU100__def-3s-2s1b__gem-3s-2s1b__ratio.png");

      result_gem_2s1b = hAll100gem;
      result_def_2s1b = hAll100;

      if (do_return) return;
    }



  /*

  // ME1b eta 1.64 - 2.14    Default: 3station, 3s   GEM: 3station, 2s & 1b
  if (1)
  {
  gdy[0]=0.02; gdy[1]=1000.;

  hAll100    = setHisto(f_pu100_pat8,     "h_rt_gmt_csc_ptmax_3s_1b", "_hAll100s3", "CSC L1 trigger rates in ME1/b eta region;p_{T}^{cut} [GeV/c];rate [kHz]", kAzure+9, 1, 1);
  hAll100gem = setHisto(f_pu100_pat8_gem, "h_rt_gmt_csc_ptmax_3s_2s1b_1b", "_hAll100gem", "CSC L1 trigger rates in ME1/b eta region;p_{T}^{cut} [GeV/c];rate [kHz]", kGreen+3, 1, 1);

  TCanvas* cAll100 = new TCanvas("cAll100","cAll100",800,600) ;
  gPad->SetLogx(1);gPad->SetLogy(1);
  gPad->SetGridx(1);gPad->SetGridy(1);
  hAll100->Draw("e3");
  hAll100gem->Draw("e3 same");
  TLegend *leg_cc100 = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
  const TObject obj;
  leg_cc100->SetBorderSize(0);
  //leg_cc100->SetTextSize(0.0368);
  leg_cc100->SetFillStyle(0);
  leg_cc100->AddEntry(hAll100,"default emulator","f");
  leg_cc100->AddEntry(hAll100,"Tracks: with >=3 stubs in 1.64<|#eta|<2.14","");
  leg_cc100->AddEntry(hAll100gem,"with GEM match","f");
  leg_cc100->AddEntry(hAll100gem,"Tracks: with >=2 stubs in 1.64<|#eta|<2.14","");
  leg_cc100->AddEntry(hAll100gem,"plus req. one stub from ME1/b","");
  leg_cc100->Draw();

  TLatex *  tex = new TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
  tex->SetNDC();
  tex->Draw();

  Print(cAll100, "rates__1.6-2.1_PU100__def-3s__gem-3s-2s1b.png");


  TCanvas* cAll100r = new TCanvas("cAll100r","cAll100r",800,300) ;
  gPad->SetLogx(1);
  gPad->SetGridx(1);gPad->SetGridy(1);

  hAll100gem_ratio = setHistoRatio(hAll100gem, hAll100, "", 0.,1.1);
  hAll100gem_ratio->Draw("e1");

  Print(cAll100r, "rates__1.6-2.1_PU100__def-3s__gem-3s-2s1b__ratio.png");

  result_gem_2s1b = hAll100gem;
  //result_def = hAll100;

  if (do_return) return;
  }

  */



  return;
}

/*

  .L drawplot_gmtrt.C
  drawplot_gmtrt("minbias_pt10_pat2")
  hh = (TH1D*)result_gem->Clone("gem_new")
  hh->SetFillColor(kGreen+4)
  for (int b = hh->FindBin(15); b <= hh->GetNbinsX(); ++b) hh->SetBinContent(b, 0);
  drawplot_gmtrt("minbias_pt15_pat2")
  h15 = (TH1D*)result_gem->Clone("gem15")
  for (int b = h15->FindBin(15); b < h15->FindBin(20); ++b) hh->SetBinContent(b, h15->GetBinContent(b));
  drawplot_gmtrt("minbias_pt20_pat2")
  h20 = (TH1D*)result_gem->Clone("gem20")
  for (int b = h20->FindBin(20); b < h20->FindBin(30); ++b) hh->SetBinContent(b, h20->GetBinContent(b));
  drawplot_gmtrt("minbias_pt30_pat2")
  h30 = (TH1D*)result_gem->Clone("gem30")
  for (int b = h30->FindBin(30); b <= h30->GetNbinsX(); ++b) hh->SetBinContent(b, h30->GetBinContent(b));
  for (int b = 1; b <= hh->GetNbinsX(); ++b) if (hh->GetBinContent(b)==0) hh->SetBinError(b, 0.);

  ((TCanvas*)gROOT->FindObject("cAll100"))->cd();
  result_def->Draw("e3");
  hh->Draw("same e3");

  TLegend *leg_cc100 = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
  const TObject obj;
  leg_cc100->SetBorderSize(0);
  leg_cc100->SetFillStyle(0);
  leg_cc100->AddEntry(result_def,"default emulator","f");
  leg_cc100->AddEntry(result_def,"Tracks: with >=3 stubs in 1.14<|#eta|<2.14","");
  leg_cc100->AddEntry(hh,"with GEM match","f");
  leg_cc100->AddEntry(result_def,"Tracks: same, plus req. one stub from ME1/b","");
  leg_cc100->Draw();

  TLatex *  tex = new TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
  tex->SetNDC();
  tex->Draw();

  gPad->Print("gem/rates__1.6-2.1_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat2.png")


  ((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
  hh_ratio = setHistoRatio(hh, result_def, "", 0.,1.1);
  hh_ratio->Draw("e1");
  gPad->Print("gem/rates__1.6-2.1_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat2__ratio.png")


  ((TCanvas*)gROOT->FindObject("cAll100"))->cd();
  result_def_3s1b->Draw("e3")
  hh->Draw("same e3")

  TLegend *leg_cc100 = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
  const TObject obj;
  leg_cc100->SetBorderSize(0);
  leg_cc100->SetFillStyle(0);
  leg_cc100->AddEntry(result_def_3s1b,"default emulator","f");
  leg_cc100->AddEntry(hh,"with GEM match","f");
  leg_cc100->AddEntry(result_def_3s1b,"Tracks req. for both:","");
  leg_cc100->AddEntry(result_def_3s1b,"with >=3 stubs in 1.64<|#eta|<2.14","");
  leg_cc100->AddEntry(result_def_3s1b,"and require one stub to be from ME1/b","");
  leg_cc100->Draw();

  TLatex *  tex = new TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
  tex->SetNDC();
  tex->Draw();

  gPad->Print("gem/rates__1.6-2.1_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat2.png")


  ((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
  hh_ratio = setHistoRatio(hh, result_def_3s1b, "", 0.,1.1);
  hh_ratio->Draw("e1");
  gPad->Print("gem/rates__1.6-2.1_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat2__ratio.png")




  .L drawplot_gmtrt.C
  drawplot_gmtrt("minbias_pt10_pat8")
  hh = (TH1D*)result_gem->Clone("gem_new")
  hh->SetFillColor(kGreen+4)
  for (int b = hh->FindBin(15); b <= hh->GetNbinsX(); ++b) hh->SetBinContent(b, 0);
  drawplot_gmtrt("minbias_pt15_pat8")
  h15 = (TH1D*)result_gem->Clone("gem15")
  for (int b = h15->FindBin(15); b < h15->FindBin(20); ++b) hh->SetBinContent(b, h15->GetBinContent(b));
  drawplot_gmtrt("minbias_pt20_pat8")
  h20 = (TH1D*)result_gem->Clone("gem20")
  for (int b = h20->FindBin(20); b < h20->FindBin(30); ++b) hh->SetBinContent(b, h20->GetBinContent(b));
  drawplot_gmtrt("minbias_pt30_pat8")
  h30 = (TH1D*)result_gem->Clone("gem30")
  for (int b = h30->FindBin(30); b <= h30->GetNbinsX(); ++b) hh->SetBinContent(b, h30->GetBinContent(b));
  for (int b = 1; b <= hh->GetNbinsX(); ++b) if (hh->GetBinContent(b)==0) hh->SetBinError(b, 0.);


  ((TCanvas*)gROOT->FindObject("cAll100"))->cd();
  result_def->Draw("e3")
  hh->Draw("same e3")

  TLegend *leg_cc100 = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
  const TObject obj;
  leg_cc100->SetBorderSize(0);
  leg_cc100->SetFillStyle(0);
  leg_cc100->AddEntry(result_def,"default emulator","f");
  leg_cc100->AddEntry(result_def,"Tracks: with >=3 stubs in 1.14<|#eta|<2.14","");
  leg_cc100->AddEntry(hh,"with GEM match","f");
  leg_cc100->AddEntry(result_def,"Tracks: same, plus req. one stub from ME1/b","");
  leg_cc100->Draw();

  TLatex *  tex = new TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
  tex->SetNDC();
  tex->Draw();

  gPad->Print("gem/rates__1.6-2.1_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat8.png")


  ((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
  hh_ratio = setHistoRatio(hh, result_def, "", 0.,1.1);
  hh_ratio->Draw("e1");
  gPad->Print("gem/rates__1.6-2.1_PU100__def-3s__gem-3s-3s1b__Frankenstein_pat8__ratio.png")



  ((TCanvas*)gROOT->FindObject("cAll100"))->cd();
  result_def_3s1b->Draw("e3")
  hh->Draw("same e3")

  TLegend *leg_cc100 = new TLegend(0.4,0.65,.98,0.92,NULL,"brNDC");
  const TObject obj;
  leg_cc100->SetBorderSize(0);
  leg_cc100->SetFillStyle(0);
  leg_cc100->AddEntry(result_def_3s1b,"default emulator","f");
  leg_cc100->AddEntry(hh,"with GEM match","f");
  leg_cc100->AddEntry(result_def_3s1b,"Tracks req. for both:","");
  leg_cc100->AddEntry(result_def_3s1b,"with >=3 stubs in 1.64<|#eta|<2.14","");
  leg_cc100->AddEntry(result_def_3s1b,"and require one stub to be from ME1/b","");
  leg_cc100->Draw();

  TLatex *  tex = new TLatex(0.17, 0.15,"L=4*10^{34} (25ns PU100)");
  tex->SetNDC();
  tex->Draw();

  gPad->Print("gem/rates__1.6-2.1_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat8.png")


  ((TCanvas*)gROOT->FindObject("cAll100r"))->cd();
  hh_ratio = setHistoRatio(hh, result_def_3s1b, "", 0.,1.1);
  hh_ratio->Draw("e1");
  gPad->Print("gem/rates__1.6-2.1_PU100__def-3s-3s1b__gem-3s-3s1b__Frankenstein_pat8__ratio.png")


*/
