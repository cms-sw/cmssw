
//int kGray=920, kOrange=800, kSpring=820, kTeal=840, kAzure=860, kViolet=880, kPink=900;

TFile *f;
char pdir[111] = "plots/";

TObject* NUL;

int interactive = 1;
TH1F *gh;

TH1D *h1, *h2, *he;

TString filesDir = "files/";
TString plotDir = "plots/efficiency/";
TString ext = ".pdf";

double yrange[2]={0.,1.04};
double yrange05[2]={0.5,1.04};
double yrange06[2]={0.6,1.04};
double yrange07[2]={0.7,1.04};
//double yrange[2]={0.5,1.02};
//double yrange[2]={0.8,1.02};
double xrange[2]={0.86,2.5};
double xrangept[2]={0.,100.};

bool do_h_eff_eta_steps_xchk1 = false;
bool do_h_eff_eta_me1_after_alct_okAlct = false;
bool h_eff_eta_steps_full10 = true;

//do_h_eff_eta_me1_after_alct_okAlct = true;

void dp(char *prc, char *g, char *pu, int st, char *suffix="", char *neu="")
{
char nm[300], fnm[300],onm[300];
//sprintf(nm,"%s_2_2_6_pu%s_me42_me1a%s_2pi_step_%d_pre3_w3",prc,pu,g,st);
sprintf(nm,"%s_3_6_2_%spu%s_me42_me1a%s_2pi_step_%d_pre3_w3%s",prc,neu,pu,g,st,suffix);
//sprintf(nm,"%s_3_7_0_pu%s_me42_me1a%s_2pi_step_%d_pre3_w3",prc,pu,g,st);
sprintf(fnm,"../hp_%s.root",nm);
sprintf(onm,"st_%s",nm);
drawplot_etastep(fnm,"StrictChamber",onm);
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


void Print(TCanvas *c, char nm[200])
{
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

TH1D* setEffHisto(char *num_name, char *den_name, char *dir, int nrebin, 
                  int lcolor, int lstyle, int lwidth,
		  char *htitle, char *xtitle, char *ytitle, 
		  double *x_range, double *y_range)
{
TH1D* hd0 = (TH1D*)getH(dir,den_name);
TH1D* hn0 = (TH1D*)getH(dir,num_name);

TString sden_name(den_name);
TString snum_name(num_name);

TH1D* hd = (TH1D*)hd0->Clone(sden_name+"_cln");
TH1D* hn = (TH1D*)hn0->Clone(snum_name+"_cln");
hd->Sumw2();
hn->Sumw2();

myRebin(hd, nrebin);
myRebin(hn, nrebin);

TH1D* heff = (TH1D*)hn->Clone(snum_name+"_eff");

hd->Sumw2();
heff->Sumw2();

heff->Divide(heff,hd);

heff->SetLineColor(lcolor);
heff->SetLineStyle(lstyle);
heff->SetLineWidth(lwidth);

heff->SetTitle(htitle);
heff->GetXaxis()->SetTitle(xtitle);
heff->GetYaxis()->SetTitle(ytitle);
heff->GetXaxis()->SetRangeUser(x_range[0],x_range[1]);
heff->GetYaxis()->SetRangeUser(y_range[0],y_range[1]);

heff->GetXaxis()->SetTitleSize(0.07);
heff->GetXaxis()->SetTitleOffset(0.7);
heff->GetYaxis()->SetLabelOffset(0.015);

heff->GetXaxis()->SetLabelSize(0.05);
heff->GetYaxis()->SetLabelSize(0.05);

return heff;
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

void gem_eff_draw()
{
gStyle->SetOptStat(0);
gStyle->SetTitleStyle(0);

int ptreb=2;

TString hdir = "SimMuL1StrictAll";

//TString f_def = filesDir + "hp_dimu_6_0_1_POSTLS161_V12__pu000_w3_def_pat2.root";
TString f_def = filesDir + "hp_dimu_6_0_1_POSTLS161_V12__pu000_w3_gem_dphi0_pat2.root";
TString f_g98_pt10 = filesDir + "hp_dimu_6_0_1_POSTLS161_V12__pu000_w3_gem98_pt10_pat2.root";
TString f_g98_pt15 = filesDir + "hp_dimu_6_0_1_POSTLS161_V12__pu000_w3_gem98_pt15_pat2.root";
TString f_g98_pt20 = filesDir + "hp_dimu_6_0_1_POSTLS161_V12__pu000_w3_gem98_pt20_pat2.root";
TString f_g98_pt30 = filesDir + "hp_dimu_6_0_1_POSTLS161_V12__pu000_w3_gem98_pt30_pat2.root";
TString f_g98_pt40 = filesDir + "hp_dimu_6_0_1_POSTLS161_V12__pu000_w3_gem98_pt40_pat2.root";

TString f_g95_pt10 = filesDir + "hp_dimu_6_0_1_POSTLS161_V12__pu000_w3_gem95_pt10_pat2.root";
TString f_g95_pt20 = filesDir + "hp_dimu_6_0_1_POSTLS161_V12__pu000_w3_gem95_pt20_pat2.root";
TString f_g95_pt30 = filesDir + "hp_dimu_6_0_1_POSTLS161_V12__pu000_w3_gem95_pt30_pat2.root";
TString f_g95_pt40 = filesDir + "hp_dimu_6_0_1_POSTLS161_V12__pu000_w3_gem95_pt40_pat2.root";

double rpt[2] = {0.,49.99};

TString htitle = "Efficiency for #mu in 1.6<|#eta|<2.12 to have TF track;p_{T}^{MC}";

TString hini = "h_pt_initial_1b";
TString h2s = "h_pt_after_tfcand_eta1b_2s";
TString h3s = "h_pt_after_tfcand_eta1b_3s";
TString h2s1b = "h_pt_after_tfcand_eta1b_2s1b";
TString h3s1b = "h_pt_after_tfcand_eta1b_3s1b";


TH1D* h_eff_tf0_2s  = getEffHisto(f_def, hdir, h2s, hini, ptreb, kBlack, 1, 2, htitle, rpt, yrange);
TH1D* h_eff_tf0_3s  = getEffHisto(f_def, hdir, h3s, hini, ptreb, kBlack, 1, 2, htitle, rpt, yrange);
TH1D* h_eff_tf0_2s1b  = getEffHisto(f_def, hdir, h2s1b, hini, ptreb, kBlack, 1, 2, htitle, rpt, yrange);
TH1D* h_eff_tf0_3s1b  = getEffHisto(f_def, hdir, h3s1b, hini, ptreb, kBlack, 1, 2, htitle, rpt, yrange);


TH1D* h_eff_tf10_2s  = getEffHisto(f_def, hdir, h2s + "_pt10", hini, ptreb, kGreen+4, 1, 2, htitle, rpt,yrange);
TH1D* h_eff_tf10_2s1b  = getEffHisto(f_def, hdir, h2s1b + "_pt10", hini, ptreb, kGreen+2, 1, 2, htitle, rpt,yrange);
TH1D* h_eff_tf10_3s  = getEffHisto(f_def, hdir, h3s + "_pt10", hini, ptreb, kGreen+2, 1, 2, htitle, rpt,yrange);
TH1D* h_eff_tf10_3s1b  = getEffHisto(f_def, hdir, h3s1b + "_pt10", hini, ptreb, kGreen+2, 1, 2, htitle, rpt,yrange);

//TH1D* h_eff_tf15_2s1b  = getEffHisto(f_def, hdir, h2s1b + "_pt15", hini, ptreb, kBlue, 1, 2, htitle, rpt,yrange);
//TH1D* h_eff_tf15_3s  = getEffHisto(f_def, hdir, h3s + "_pt15", hini, ptreb, kBlue, 1, 2, htitle, rpt,yrange);
//TH1D* h_eff_tf15_3s1b  = getEffHisto(f_def, hdir, h3s1b + "_pt15", hini, ptreb, kBlue, 1, 2, htitle, rpt,yrange);

TH1D* h_eff_tf20_2s  = getEffHisto(f_def, hdir, h2s + "_pt20", hini, ptreb, kOrange+4, 1, 2, htitle, rpt,yrange);
TH1D* h_eff_tf20_2s1b  = getEffHisto(f_def, hdir, h2s1b + "_pt20", hini, ptreb, kOrange, 1, 2, htitle, rpt,yrange);
TH1D* h_eff_tf20_3s  = getEffHisto(f_def, hdir, h3s + "_pt20", hini, ptreb, kOrange, 1, 2, htitle, rpt,yrange);
TH1D* h_eff_tf20_3s1b  = getEffHisto(f_def, hdir, h3s1b + "_pt20", hini, ptreb, kOrange, 1, 2, htitle, rpt,yrange);

TH1D* h_eff_tf30_2s  = getEffHisto(f_def, hdir, h2s + "_pt30", hini, ptreb, kRed+4, 1, 2, htitle, rpt,yrange);
TH1D* h_eff_tf30_2s1b  = getEffHisto(f_def, hdir, h2s1b + "_pt30", hini, ptreb, kRed, 1, 2, htitle, rpt,yrange);
TH1D* h_eff_tf30_3s  = getEffHisto(f_def, hdir, h3s + "_pt30", hini, ptreb, kRed, 1, 2, htitle, rpt,yrange);
TH1D* h_eff_tf30_3s1b  = getEffHisto(f_def, hdir, h3s1b + "_pt30", hini, ptreb, kRed, 1, 2, htitle, rpt,yrange);

TH1D* h_eff_tf40_2s  = getEffHisto(f_def, hdir, h2s + "_pt40", hini, ptreb, kViolet+4, 1, 2, htitle, rpt,yrange);
TH1D* h_eff_tf40_2s1b  = getEffHisto(f_def, hdir, h2s1b + "_pt40", hini, ptreb, kViolet, 1, 2, htitle, rpt,yrange);
TH1D* h_eff_tf40_3s  = getEffHisto(f_def, hdir, h3s + "_pt40", hini, ptreb, kViolet, 1, 2, htitle, rpt,yrange);
TH1D* h_eff_tf40_3s1b  = getEffHisto(f_def, hdir, h3s1b + "_pt40", hini, ptreb, kViolet, 1, 2, htitle, rpt,yrange);



TH1D* h_eff_tf10_gpt10_2s1b  = getEffHisto(f_g98_pt10, hdir, h2s1b + "_pt10", hini, ptreb, kGreen+2, 7, 2, htitle, rpt,yrange);
TH1D* h_eff_tf10_gpt10_3s1b  = getEffHisto(f_g98_pt10, hdir, h3s1b + "_pt10", hini, ptreb, kGreen+2, 7, 2, htitle, rpt,yrange);

TH1D* h_eff_tf15_gpt15_2s1b  = getEffHisto(f_g98_pt15, hdir, h2s1b + "_pt15", hini, ptreb, kBlue, 7, 2, htitle, rpt,yrange);
TH1D* h_eff_tf15_gpt15_3s1b  = getEffHisto(f_g98_pt15, hdir, h3s1b + "_pt15", hini, ptreb, kBlue, 7, 2, htitle, rpt,yrange);

TH1D* h_eff_tf20_gpt20_2s1b  = getEffHisto(f_g98_pt20, hdir, h2s1b + "_pt20", hini, ptreb, kOrange, 7, 2, htitle, rpt,yrange);
TH1D* h_eff_tf20_gpt20_3s1b  = getEffHisto(f_g98_pt20, hdir, h3s1b + "_pt20", hini, ptreb, kOrange, 7, 2, htitle, rpt,yrange);

TH1D* h_eff_tf30_gpt30_2s1b  = getEffHisto(f_g98_pt30, hdir, h2s1b + "_pt30", hini, ptreb, kRed, 7, 2, htitle, rpt,yrange);
TH1D* h_eff_tf30_gpt30_3s1b  = getEffHisto(f_g98_pt30, hdir, h3s1b + "_pt30", hini, ptreb, kRed, 7, 2, htitle, rpt,yrange);

TH1D* h_eff_tf40_gpt40_2s1b  = getEffHisto(f_g98_pt40, hdir, h2s1b + "_pt40", hini, ptreb, kViolet, 7, 2, htitle, rpt,yrange);
TH1D* h_eff_tf40_gpt40_3s1b  = getEffHisto(f_g98_pt40, hdir, h3s1b + "_pt40", hini, ptreb, kViolet, 7, 2, htitle, rpt,yrange);


TCanvas* c2s1b = new TCanvas("c2s1b","c2s1b",800,600) ;

/*
TH1D* h_eff_gmt20_1b  = getEffHisto(f_def, hdir, "h_pt_after_gmt_eta1b_1mu_pt20", hini, ptreb, kBlack, 1, 2, htitle, rpt, yrange);
TH1D* h_eff_gmt30_1b  = getEffHisto(f_def, hdir, "h_pt_after_gmt_eta1b_1mu_pt30", hini, ptreb, kBlack-1, 1, 2, htitle, rpt, yrange);
TH1D* h_eff_gmt40_1b  = getEffHisto(f_def, hdir, "h_pt_after_gmt_eta1b_1mu_pt40", hini, ptreb, kBlack-2, 1, 2, htitle, rpt, yrange);
h_eff_gmt20_1b->Draw("hist");
h_eff_gmt30_1b->Draw("hist same");
h_eff_gmt40_1b->Draw("hist same");
return;

h_eff_tf40_3s->Draw("hist");
h_eff_tf40_3s1b->Draw("hist same");
h_eff_tf40_gpt40_3s1b->Draw("hist same");
return;
*/


TH1D* h_eff_tf10_gpt15_2s1b  = getEffHisto(f_g98_pt15, hdir, h2s1b + "_pt10", hini, ptreb, kGreen+2, 3, 2, htitle, rpt,yrange);
TH1D* h_eff_tf10_gpt15_3s1b  = getEffHisto(f_g98_pt15, hdir, h3s1b + "_pt10", hini, ptreb, kGreen+2, 3, 2, htitle, rpt,yrange);

TH1D* h_eff_tf15_gpt20_2s1b  = getEffHisto(f_g98_pt20, hdir, h2s1b + "_pt15", hini, ptreb, kViolet, 3, 2, htitle, rpt,yrange);
TH1D* h_eff_tf15_gpt20_3s1b  = getEffHisto(f_g98_pt20, hdir, h3s1b + "_pt15", hini, ptreb, kViolet, 3, 2, htitle, rpt,yrange);

TH1D* h_eff_tf20_gpt30_2s1b  = getEffHisto(f_g98_pt30, hdir, h2s1b + "_pt20", hini, ptreb, kOrange, 3, 2, htitle, rpt,yrange);
TH1D* h_eff_tf20_gpt30_3s1b  = getEffHisto(f_g98_pt30, hdir, h3s1b + "_pt20", hini, ptreb, kOrange, 3, 2, htitle, rpt,yrange);

TH1D* h_eff_tf30_gpt40_2s1b  = getEffHisto(f_g98_pt40, hdir, h2s1b + "_pt30", hini, ptreb, kRed, 3, 2, htitle, rpt,yrange);
TH1D* h_eff_tf30_gpt40_3s1b  = getEffHisto(f_g98_pt40, hdir, h3s1b + "_pt30", hini, ptreb, kRed, 3, 2, htitle, rpt,yrange);


TH1D* h_eff_tf10_gpt20_2s1b  = getEffHisto(f_g98_pt20, hdir, h2s1b + "_pt10", hini, ptreb, kGreen+2, 3, 2, htitle, rpt,yrange);
TH1D* h_eff_tf10_gpt20_3s1b  = getEffHisto(f_g98_pt20, hdir, h3s1b + "_pt10", hini, ptreb, kGreen+2, 3, 2, htitle, rpt,yrange);

TH1D* h_eff_tf15_gpt30_2s1b  = getEffHisto(f_g98_pt30, hdir, h2s1b + "_pt15", hini, ptreb, kViolet, 3, 2, htitle, rpt,yrange);
TH1D* h_eff_tf15_gpt30_3s1b  = getEffHisto(f_g98_pt30, hdir, h3s1b + "_pt15", hini, ptreb, kViolet, 3, 2, htitle, rpt,yrange);

TH1D* h_eff_tf20_gpt40_2s1b  = getEffHisto(f_g98_pt40, hdir, h2s1b + "_pt20", hini, ptreb, kOrange, 3, 2, htitle, rpt,yrange);
TH1D* h_eff_tf20_gpt40_3s1b  = getEffHisto(f_g98_pt40, hdir, h3s1b + "_pt20", hini, ptreb, kOrange, 3, 2, htitle, rpt,yrange);



TCanvas* c2s1b = new TCanvas("c2s1b","c2s1b",800,600) ;

//h_eff_tf0_2s1b->Draw("hist");
h_eff_tf10_2s1b->Draw("hist");
//h_eff_tf15_2s1b->Draw("hist same");
h_eff_tf20_2s1b->Draw("hist same");
h_eff_tf30_2s1b->Draw("hist same");
//h_eff_tf40_2s1b->Draw("hist same");

h_eff_tf10_gpt10_2s1b->Draw("hist same");
h_eff_tf20_gpt20_2s1b->Draw("hist same");
h_eff_tf30_gpt30_2s1b->Draw("hist same");

TLegend *leg = new TLegend(0.50,0.17,.999,0.57, NULL, "brNDC");
leg->SetNColumns(2);
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->SetHeader("TF track requires 2+ stubs, one from ME1");
leg->AddEntry(h_eff_tf10_2s1b, "Trigger p_{T}:", "");
leg->AddEntry(h_eff_tf10_gpt10_2s1b, "with GEM:", "");
leg->AddEntry(h_eff_tf10_2s1b, "p_{T}^{TF}>=10", "l");
leg->AddEntry(h_eff_tf10_gpt10_2s1b, "#Delta#phi for p_{T}=10", "l");
leg->AddEntry(h_eff_tf20_2s1b, "p_{T}^{TF}>=20", "l");
leg->AddEntry(h_eff_tf20_gpt20_2s1b, "#Delta#phi for p_{T}=20", "l");
leg->AddEntry(h_eff_tf30_2s1b, "p_{T}^{TF}>=30", "l");
leg->AddEntry(h_eff_tf30_gpt30_2s1b, "#Delta#phi for p_{T}=30", "l");
leg->Draw();

c2s1b->Print(plotDir + "eff_2s1b" + ext);



TCanvas* c3s1b = new TCanvas("c3s1b","c3s1b",800,600) ;

h_eff_tf10_3s1b->Draw("hist");
h_eff_tf20_3s1b->Draw("hist same");
h_eff_tf30_3s1b->Draw("hist same");

h_eff_tf10_gpt10_3s1b->Draw("hist same");
h_eff_tf20_gpt20_3s1b->Draw("hist same");
h_eff_tf30_gpt30_3s1b->Draw("hist same");

TLegend *leg = new TLegend(0.50,0.17,.999,0.57, NULL, "brNDC");
leg->SetNColumns(2);
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->SetHeader("TF track requires 3+ stubs, one from ME1");
leg->AddEntry(h_eff_tf10_3s1b, "Trigger p_{T}:", "");
leg->AddEntry(h_eff_tf10_gpt10_3s1b, "with GEM:", "");
leg->AddEntry(h_eff_tf10_3s1b, "p_{T}^{TF}>=10", "l");
leg->AddEntry(h_eff_tf10_gpt10_3s1b, "#Delta#phi for p_{T}=10", "l");
leg->AddEntry(h_eff_tf20_3s1b, "p_{T}^{TF}>=20", "l");
leg->AddEntry(h_eff_tf20_gpt20_3s1b, "#Delta#phi for p_{T}=20", "l");
leg->AddEntry(h_eff_tf30_3s1b, "p_{T}^{TF}>=30", "l");
leg->AddEntry(h_eff_tf30_gpt30_3s1b, "#Delta#phi for p_{T}=30", "l");
leg->Draw();

c3s1b->Print(plotDir + "eff_3s1b" + ext);



TCanvas* c3s_2s1b = new TCanvas("c3s_2s1b","c3s_2s1b",800,600);

h_eff_tf10_3s->Draw("hist");
h_eff_tf20_3s->Draw("hist same");
h_eff_tf30_3s->Draw("hist same");

h_eff_tf10_gpt10_2s1b->Draw("hist same");
h_eff_tf20_gpt20_2s1b->Draw("hist same");
h_eff_tf30_gpt30_2s1b->Draw("hist same");

TLegend *leg = new TLegend(0.50,0.17,.999,0.57, NULL, "brNDC");
leg->SetNColumns(2);
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->SetHeader("TF track requires");
leg->AddEntry(h_eff_tf10_3s, "3+ stubs", "");
leg->AddEntry(h_eff_tf10_gpt10_2s1b, "2+ stubs with GEM in ME1", "");
leg->AddEntry(h_eff_tf10_3s, "p_{T}^{TF}>=10", "l");
leg->AddEntry(h_eff_tf10_gpt10_2s1b, "#Delta#phi for p_{T}=10", "l");
leg->AddEntry(h_eff_tf20_3s, "p_{T}^{TF}>=20", "l");
leg->AddEntry(h_eff_tf20_gpt20_2s1b, "#Delta#phi for p_{T}=20", "l");
leg->AddEntry(h_eff_tf30_3s, "p_{T}^{TF}>=30", "l");
leg->AddEntry(h_eff_tf30_gpt30_2s1b, "#Delta#phi for p_{T}=30", "l");
leg->Draw();

c3s_2s1b->Print(plotDir + "eff_3s_2s1b" + ext);




TCanvas* c3s_def = new TCanvas("c3s_def","c3s_def",800,600);

h_eff_tf10_3s->Draw("hist");
h_eff_tf20_3s->Draw("hist same");
h_eff_tf30_3s->Draw("hist same");

TLegend *leg = new TLegend(0.55,0.17,.999,0.57, NULL, "brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->SetHeader("TF track requires 3+ stubs and");
leg->AddEntry(h_eff_tf10_3s, "p_{T}^{TF}>=10", "l");
leg->AddEntry(h_eff_tf20_3s, "p_{T}^{TF}>=20", "l");
leg->AddEntry(h_eff_tf30_3s, "p_{T}^{TF}>=30", "l");
leg->Draw();

c3s_def->Print(plotDir + "eff_3s_def" + ext);


TCanvas* c3s1b_def = new TCanvas("c3s1b_def","c3s1b_def",800,600);

h_eff_tf10_3s1b->Draw("hist");
h_eff_tf20_3s1b->Draw("hist same");
h_eff_tf30_3s1b->Draw("hist same");

TLegend *leg = new TLegend(0.55,0.17,.999,0.57, NULL, "brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->SetHeader("TF track requires 3+ stubs with ME1 and");
leg->AddEntry(h_eff_tf10_3s, "p_{T}^{TF}>=10", "l");
leg->AddEntry(h_eff_tf20_3s, "p_{T}^{TF}>=20", "l");
leg->AddEntry(h_eff_tf30_3s, "p_{T}^{TF}>=30", "l");
leg->Draw();

c3s1b_def->Print(plotDir + "eff_3s1b_def" + ext);



h_eff_tf10_2s->SetLineColor(kAzure+2);
h_eff_tf10_2s1b->SetLineColor(kAzure+6);
h_eff_tf10_3s->SetLineColor(kAzure+3);
h_eff_tf10_3s1b->SetLineColor(kAzure+7);
h_eff_tf10_gpt10_2s1b->SetLineColor(kAzure+6);
h_eff_tf10_gpt10_3s1b->SetLineColor(kAzure+7);

h_eff_tf20_2s->SetLineColor(kAzure+2);
h_eff_tf20_2s1b->SetLineColor(kAzure+6);
h_eff_tf20_3s->SetLineColor(kAzure+3);
h_eff_tf20_3s1b->SetLineColor(kAzure+7);
h_eff_tf20_gpt20_2s1b->SetLineColor(kAzure+6);
h_eff_tf20_gpt20_3s1b->SetLineColor(kAzure+7);

h_eff_tf30_2s->SetLineColor(kAzure+2);
h_eff_tf30_2s1b->SetLineColor(kAzure+6);
h_eff_tf30_3s->SetLineColor(kAzure+3);
h_eff_tf30_3s1b->SetLineColor(kAzure+7);
h_eff_tf30_gpt30_2s1b->SetLineColor(kAzure+6);
h_eff_tf30_gpt30_3s1b->SetLineColor(kAzure+7);

h_eff_tf40_2s->SetLineColor(kAzure+2);
h_eff_tf40_2s1b->SetLineColor(kAzure+6);
h_eff_tf40_3s->SetLineColor(kAzure+3);
h_eff_tf40_3s1b->SetLineColor(kAzure+7);
h_eff_tf40_gpt40_2s1b->SetLineColor(kAzure+6);
h_eff_tf40_gpt40_3s1b->SetLineColor(kAzure+7);


TCanvas* c2s_pt10_def = new TCanvas("c2s_pt10_def","c2s_pt10_def",800,600);

h_eff_tf10_2s->Draw("hist");
h_eff_tf10_2s1b->Draw("hist same");

TLegend *leg = new TLegend(0.55,0.17,.999,0.57, NULL, "brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->SetHeader("TF track: p_{T}^{TF}>=10 and 2+ stubs");
leg->AddEntry(h_eff_tf10_2s, "anywhere", "l");
leg->AddEntry(h_eff_tf10_2s1b, "with ME1", "l");
leg->Draw();

c2s_pt10_def->Print(plotDir + "eff_2s_pt10_def" + ext);

h_eff_tf10_gpt10_2s1b->Draw("hist same");
leg->AddEntry(h_eff_tf10_gpt10_2s1b, "with (ME1 + GEM)", "l");
c2s_pt10_def->Print(plotDir + "eff_2s_pt10_gem" + ext);



TCanvas* c3s_pt10_def = new TCanvas("c3s_pt10_def","c3s_pt10_def",800,600);

h_eff_tf10_3s->Draw("hist");
h_eff_tf10_3s1b->Draw("hist same");

TLegend *leg = new TLegend(0.55,0.17,.999,0.57, NULL, "brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->SetHeader("TF track: p_{T}^{TF}>=10 and 3+ stubs");
leg->AddEntry(h_eff_tf10_3s, "anywhere", "l");
leg->AddEntry(h_eff_tf10_3s1b, "with ME1", "l");
leg->Draw();

c3s_pt10_def->Print(plotDir + "eff_3s_pt10_def" + ext);

h_eff_tf10_gpt10_3s1b->Draw("hist same");
leg->AddEntry(h_eff_tf10_gpt10_3s1b, "with (ME1 + GEM)", "l");
c3s_pt10_def->Print(plotDir + "eff_3s_pt10_gem" + ext);




TCanvas* c2s_pt20_def = new TCanvas("c2s_pt20_def","c2s_pt20_def",800,600);

h_eff_tf20_2s->Draw("hist");
h_eff_tf20_2s1b->Draw("hist same");

TLegend *leg = new TLegend(0.55,0.17,.999,0.57, NULL, "brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->SetHeader("TF track: p_{T}^{TF}>=20 and 2+ stubs");
leg->AddEntry(h_eff_tf20_2s, "anywhere", "l");
leg->AddEntry(h_eff_tf20_2s1b, "with ME1", "l");
leg->Draw();

c2s_pt20_def->Print(plotDir + "eff_2s_pt20_def" + ext);

h_eff_tf20_gpt20_2s1b->Draw("hist same");
leg->AddEntry(h_eff_tf20_gpt20_2s1b, "with (ME1 + GEM)", "l");
c2s_pt20_def->Print(plotDir + "eff_2s_pt20_gem" + ext);



TCanvas* c3s_pt20_def = new TCanvas("c3s_pt20_def","c3s_pt20_def",800,600);

h_eff_tf20_3s->Draw("hist");
h_eff_tf20_3s1b->Draw("hist same");

TLegend *leg = new TLegend(0.55,0.17,.999,0.57, NULL, "brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->SetHeader("TF track: p_{T}^{TF}>=20 and 3+ stubs");
leg->AddEntry(h_eff_tf20_3s, "anywhere", "l");
leg->AddEntry(h_eff_tf20_3s1b, "with ME1", "l");
leg->Draw();

c3s_pt20_def->Print(plotDir + "eff_3s_pt20_def" + ext);

h_eff_tf20_gpt20_3s1b->Draw("hist same");
leg->AddEntry(h_eff_tf20_gpt20_3s1b, "with (ME1 + GEM)", "l");
c3s_pt20_def->Print(plotDir + "eff_3s_pt20_gem" + ext);



TCanvas* c2s_pt30_def = new TCanvas("c2s_pt30_def","c2s_pt30_def",800,600);

h_eff_tf30_2s->Draw("hist");
h_eff_tf30_2s1b->Draw("hist same");

TLegend *leg = new TLegend(0.55,0.17,.999,0.57, NULL, "brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->SetHeader("TF track: p_{T}^{TF}>=30 and 2+ stubs");
leg->AddEntry(h_eff_tf30_2s, "anywhere", "l");
leg->AddEntry(h_eff_tf30_2s1b, "with ME1", "l");
leg->Draw();

c2s_pt30_def->Print(plotDir + "eff_2s_pt30_def" + ext);

h_eff_tf30_gpt30_2s1b->Draw("hist same");
leg->AddEntry(h_eff_tf30_gpt30_2s1b, "with (ME1 + GEM)", "l");
c2s_pt30_def->Print(plotDir + "eff_2s_pt30_gem" + ext);



TCanvas* c3s_pt30_def = new TCanvas("c3s_pt30_def","c3s_pt30_def",800,600);

h_eff_tf30_3s->Draw("hist");
h_eff_tf30_3s1b->Draw("hist same");

TLegend *leg = new TLegend(0.55,0.17,.999,0.57, NULL, "brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->SetHeader("TF track: p_{T}^{TF}>=30 and 3+ stubs");
leg->AddEntry(h_eff_tf30_3s, "anywhere", "l");
leg->AddEntry(h_eff_tf30_3s1b, "with ME1", "l");
leg->Draw();

c3s_pt30_def->Print(plotDir + "eff_3s_pt30_def" + ext);

h_eff_tf30_gpt30_3s1b->Draw("hist same");
leg->AddEntry(h_eff_tf30_gpt30_3s1b, "with (ME1 + GEM)", "l");
c3s_pt30_def->Print(plotDir + "eff_3s_pt30_gem" + ext);



TCanvas* c2s_pt40_def = new TCanvas("c2s_pt40_def","c2s_pt40_def",800,600);

h_eff_tf40_2s->Draw("hist");
h_eff_tf40_2s1b->Draw("hist same");

TLegend *leg = new TLegend(0.55,0.17,.999,0.57, NULL, "brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->SetHeader("TF track: p_{T}^{TF}>=40 and 2+ stubs");
leg->AddEntry(h_eff_tf40_2s, "anywhere", "l");
leg->AddEntry(h_eff_tf40_2s1b, "with ME1", "l");
leg->Draw();

c2s_pt40_def->Print(plotDir + "eff_2s_pt40_def" + ext);

h_eff_tf40_gpt40_2s1b->Draw("hist same");
leg->AddEntry(h_eff_tf40_gpt40_2s1b, "with (ME1 + GEM)", "l");
c2s_pt40_def->Print(plotDir + "eff_2s_pt40_gem" + ext);



TCanvas* c3s_pt40_def = new TCanvas("c3s_pt40_def","c3s_pt40_def",800,600);

h_eff_tf40_3s->Draw("hist");
h_eff_tf40_3s1b->Draw("hist same");

TLegend *leg = new TLegend(0.55,0.17,.999,0.57, NULL, "brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->SetHeader("TF track: p_{T}^{TF}>=40 and 3+ stubs");
leg->AddEntry(h_eff_tf40_3s, "anywhere", "l");
leg->AddEntry(h_eff_tf40_3s1b, "with ME1", "l");
leg->Draw();

c3s_pt40_def->Print(plotDir + "eff_3s_pt40_def" + ext);

h_eff_tf40_gpt40_3s1b->Draw("hist same");
leg->AddEntry(h_eff_tf40_gpt40_3s1b, "with (ME1 + GEM)", "l");
c3s_pt40_def->Print(plotDir + "eff_3s_pt40_gem" + ext);



//return;

h_eff_tf10_gpt10_3s1b->SetLineColor(kBlue);
h_eff_tf10_gpt15_3s1b->SetLineColor(kMagenta);
h_eff_tf20_gpt20_3s1b->SetLineColor(kBlue+2);
h_eff_tf20_gpt30_3s1b->SetLineColor(kMagenta+2);
h_eff_tf30_gpt30_3s1b->SetLineColor(kBlue+4);
h_eff_tf30_gpt40_3s1b->SetLineColor(kMagenta+4);

TCanvas* c3s_tight = new TCanvas("c3s_tight","c3s_tight",800,600);

h_eff_tf10_gpt10_3s1b->Draw("hist");
h_eff_tf10_gpt15_3s1b->Draw("hist same");

//h_eff_tf15_gpt15_3s1b->Draw("hist same");
//h_eff_tf15_gpt20_3s1b->Draw("hist same");

h_eff_tf20_gpt20_3s1b->Draw("hist same");
h_eff_tf20_gpt30_3s1b->Draw("hist same");

h_eff_tf30_gpt30_3s1b->Draw("hist same");
h_eff_tf30_gpt40_3s1b->Draw("hist same");

TLegend *leg = new TLegend(0.55,0.17,.999,0.57, NULL, "brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->SetNColumns(2);
leg->SetHeader("TF track: 3+ stubs with ME1");
leg->AddEntry(h_eff_tf10_gpt10_3s1b, "p_{T}^{TF} cut and", "");
leg->AddEntry(h_eff_tf10_gpt10_3s1b, "p_{T}^{TF} cut and", "");
leg->AddEntry(h_eff_tf10_gpt10_3s1b, "p_{T} for #Delta#phi(GEM,LCT)", "");
leg->AddEntry(h_eff_tf10_gpt10_3s1b, "p_{T} for #Delta#phi(GEM,LCT)", "");
leg->AddEntry(h_eff_tf10_gpt10_3s1b, "#geq10 and 10", "l");
leg->AddEntry(h_eff_tf10_gpt15_3s1b, "#geq10 and 15", "l");
leg->AddEntry(h_eff_tf20_gpt20_3s1b, "#geq20 and 20", "l");
leg->AddEntry(h_eff_tf20_gpt30_3s1b, "#geq20 and 30", "l");
leg->AddEntry(h_eff_tf30_gpt30_3s1b, "#geq30 and 30", "l");
leg->AddEntry(h_eff_tf30_gpt40_3s1b, "#geq30 and 40", "l");
leg->Draw();

c3s_tight->Print(plotDir + "eff_3s_gemtight" + ext);



h_eff_tf10_gpt10_3s1b->SetLineColor(kBlue);
h_eff_tf10_gpt20_3s1b->SetLineColor(kMagenta);
h_eff_tf15_gpt15_3s1b->SetLineColor(kBlue+2);
h_eff_tf15_gpt30_3s1b->SetLineColor(kMagenta+2);
h_eff_tf20_gpt20_3s1b->SetLineColor(kBlue+4);
h_eff_tf20_gpt40_3s1b->SetLineColor(kMagenta+4);

TCanvas* c3s_tight = new TCanvas("c3s_tight","c3s_tight",800,600);

h_eff_tf10_gpt10_3s1b->Draw("hist");
h_eff_tf10_gpt20_3s1b->Draw("hist same");

h_eff_tf15_gpt15_3s1b->Draw("hist same");
h_eff_tf15_gpt30_3s1b->Draw("hist same");

h_eff_tf20_gpt20_3s1b->Draw("hist same");
h_eff_tf20_gpt40_3s1b->Draw("hist same");

TLegend *leg = new TLegend(0.55,0.17,.999,0.57, NULL, "brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->SetNColumns(2);
leg->SetHeader("TF track: 3+ stubs with ME1");
leg->AddEntry(h_eff_tf10_gpt10_3s1b, "p_{T}^{TF} cut and", "");
leg->AddEntry(h_eff_tf10_gpt10_3s1b, "p_{T}^{TF} cut and", "");
leg->AddEntry(h_eff_tf10_gpt10_3s1b, "p_{T} for #Delta#phi(GEM,LCT)", "");
leg->AddEntry(h_eff_tf10_gpt10_3s1b, "p_{T} for #Delta#phi(GEM,LCT)", "");
leg->AddEntry(h_eff_tf10_gpt10_3s1b, "#geq10 and 10", "l");
leg->AddEntry(h_eff_tf10_gpt20_3s1b, "#geq10 and 20", "l");
leg->AddEntry(h_eff_tf15_gpt15_3s1b, "#geq15 and 15", "l");
leg->AddEntry(h_eff_tf15_gpt30_3s1b, "#geq15 and 30", "l");
leg->AddEntry(h_eff_tf20_gpt20_3s1b, "#geq20 and 20", "l");
leg->AddEntry(h_eff_tf20_gpt40_3s1b, "#geq20 and 40", "l");
leg->Draw();

c3s_tight->Print(plotDir + "eff_3s_gemtightX" + ext);

}




void gem_eff_draw_gem1b()
{
gStyle->SetOptStat(0);
gStyle->SetTitleStyle(0);

int ptreb=2;

TString hdir = "SimMuL1StrictAll";

//TString f_def = filesDir + "hp_dimu_6_0_1_POSTLS161_V12__pu000_w3_def_pat2.root";
TString f_def = filesDir + "hp_dimu_6_0_1_POSTLS161_V12__pu000_w3_gem_dphi0_pat2.root";

double rpt[2] = {0.,49.99};



TString hini = "h_pt_initial_gem_1b";
TString hini_g = "h_pt_gem_1b";
TString hini_gl = "h_pt_lctgem_1b";

TString h2g_00 = "h_pt_after_tfcand_gem1b_2s1b";
TString h2g_00_123 = "h_pt_after_tfcand_gem1b_2s123";
TString h2g_00_13  = "h_pt_after_tfcand_gem1b_2s13";
TString h3g_00 = "h_pt_after_tfcand_gem1b_3s1b";
TString h2p_00 = "h_pt_after_tfcand_dphigem1b_2s1b";
TString h2p_00_123 = "h_pt_after_tfcand_dphigem1b_2s123";
TString h2p_00_13 = "h_pt_after_tfcand_dphigem1b_2s13";
TString h3p_00 = "h_pt_after_tfcand_dphigem1b_3s1b";

TString h2g_15 = "h_pt_after_tfcand_gem1b_2s1b_pt15";
TString h2g_15_123 = "h_pt_after_tfcand_gem1b_2s123_pt15";
TString h2g_15_13 = "h_pt_after_tfcand_gem1b_2s13_pt15";
TString h3g_15 = "h_pt_after_tfcand_gem1b_3s1b_pt15";
TString h2p_15 = "h_pt_after_tfcand_dphigem1b_2s1b_pt15";
TString h2p_15_123 = "h_pt_after_tfcand_dphigem1b_2s123_pt15";
TString h2p_15_13 = "h_pt_after_tfcand_dphigem1b_2s13_pt15";
TString h3p_15 = "h_pt_after_tfcand_dphigem1b_3s1b_pt15";

TString h2g_20 = "h_pt_after_tfcand_gem1b_2s1b_pt20";
TString h2g_20_123 = "h_pt_after_tfcand_gem1b_2s123_pt20";
TString h2g_20_13 = "h_pt_after_tfcand_gem1b_2s13_pt20";
TString h3g_20 = "h_pt_after_tfcand_gem1b_3s1b_pt20";
TString h2p_20 = "h_pt_after_tfcand_dphigem1b_2s1b_pt20";
TString h2p_20_123 = "h_pt_after_tfcand_dphigem1b_2s123_pt20";
TString h2p_20_13 = "h_pt_after_tfcand_dphigem1b_2s13_pt20";
TString h3p_20 = "h_pt_after_tfcand_dphigem1b_3s1b_pt20";

TString h2g_30 = "h_pt_after_tfcand_gem1b_2s1b_pt30";
TString h2g_30_123 = "h_pt_after_tfcand_gem1b_2s123_pt30";
TString h2g_30_13 = "h_pt_after_tfcand_gem1b_2s13_pt30";
TString h3g_30 = "h_pt_after_tfcand_gem1b_3s1b_pt30";
TString h2p_30 = "h_pt_after_tfcand_dphigem1b_2s1b_pt30";
TString h2p_30_123 = "h_pt_after_tfcand_dphigem1b_2s123_pt30";
TString h2p_30_13 = "h_pt_after_tfcand_dphigem1b_2s13_pt30";
TString h3p_30 = "h_pt_after_tfcand_dphigem1b_3s1b_pt30";


TCanvas* c2 = new TCanvas("c2","c2",800,600) ;
gPad->SetGridx(1);
gPad->SetGridy(1);


TString htitle = "Efficiency for #mu (GEM) in 1.64<|#eta|<2.05 to have TF track with ME1/b stub;p_{T}^{MC}";

hel = getEffHisto(f_def, hdir, hini_gl, hini_g, ptreb, kBlack, 1, 2, htitle, rpt, yrange07);
hel->Draw("hist");
het2 = getEffHisto(f_def, hdir, h2g_00, hini_g, ptreb, kGreen+2, 1, 2, htitle, rpt, yrange07);
het2->Draw("same hist");
het3 = getEffHisto(f_def, hdir, h3g_00, hini_g, ptreb, kGreen+2, 2, 2, htitle, rpt, yrange07);
het3->Draw("same hist");
het2pt20 = getEffHisto(f_def, hdir, h2g_20, hini_g, ptreb, kBlue, 1, 2, htitle, rpt, yrange07);
het2pt20->Draw("same hist");
het3pt20 = getEffHisto(f_def, hdir, h3g_20, hini_g, ptreb, kBlue, 2, 2, htitle, rpt, yrange07);
het3pt20->Draw("same hist");

TLegend *leg = new TLegend(0.55,0.17,.999,0.57, NULL, "brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->SetNColumns(2);
//leg->SetHeader("TF track: 3+ stubs with ME1");
leg->AddEntry(hel, "ME1/b LCT stub", "l");
leg->AddEntry(hel, " ", "");
leg->AddEntry(het2, "any p_{T}^{TF}, 2+ stubs", "l");
leg->AddEntry(het2pt20, "p_{T}^{TF}#geq20, 2+ stubs", "l");
leg->AddEntry(het3, "any p_{T}^{TF}, 3+ stubs", "l");
leg->AddEntry(het3pt20, "p_{T}^{TF}#geq20, 3+ stubs", "l");
leg->Draw();

c2->Print(plotDir + "eff_gem1b_basegem" + ext);



TString htitle = "Efficiency for #mu (GEM+LCT) in 1.64<|#eta|<2.05 to have TF track with ME1/b stub;p_{T}^{MC}";

helt2pt20 = getEffHisto(f_def, hdir, h2g_20, hini_gl, ptreb, kMagenta-3, 1, 2, htitle, rpt, yrange07);
helt2pt20->Draw("hist");
helt3pt20 = getEffHisto(f_def, hdir, h3g_20, hini_gl, ptreb, kMagenta-3, 2, 2, htitle, rpt, yrange07);
helt3pt20->Draw("same hist");
het2pt20->Draw("same hist");
het3pt20->Draw("same hist");

TLegend *leg = new TLegend(0.55,0.17,.999,0.57, NULL, "brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->SetNColumns(2);
leg->SetHeader("TF track p_{T}^{TF}#geq20 with ME1");
leg->AddEntry(het2pt20, "GEM baseline", "");
leg->AddEntry(helt2pt20, "GEM+LCT baseline", "");
leg->AddEntry(het2pt20, "2+ stubs", "l");
leg->AddEntry(helt2pt20, "2+ stubs", "l");
leg->AddEntry(het3pt20, "3+ stubs", "l");
leg->AddEntry(helt3pt20, "3+ stubs", "l");
leg->Draw();
c2->Print(plotDir + "eff_gem1b_baselctgem" + ext);

//return;

TString htitle = "Efficiency for #mu (GEM) in 1.64<|#eta|<2.05 to have TF track with ME1/b stub;p_{T}^{MC}";

het2pt20->Draw("hist");
het3pt20->Draw("same hist");
het2pt20p = getEffHisto(f_def, hdir, h2p_20, hini_g, ptreb, kGray+2, 1, 2, htitle, rpt, yrange07);
het2pt20p->Draw("same hist");
het3pt20p = getEffHisto(f_def, hdir, h3p_20, hini_g, ptreb, kGray+2, 2, 2, htitle, rpt, yrange07);
het3pt20p->Draw("same hist");

TLegend *leg = new TLegend(0.55,0.17,.999,0.57, NULL, "brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->SetNColumns(2);
leg->SetHeader("TF track p_{T}^{TF}#geq20 with ME1");
leg->AddEntry(het2pt20, "no GEM #Delta#phi", "");
leg->AddEntry(het2pt20p, "with GEM #Delta#phi", "l");
leg->AddEntry(het2pt20, "2+ stubs", "l");
leg->AddEntry(het2pt20p, "2+ stubs", "l");
leg->AddEntry(het3pt20, "3+ stubs", "l");
leg->AddEntry(het3pt20p, "3+ stubs", "l");
leg->Draw();
c2->Print(plotDir + "eff_gem1b_basegem_dphi" + ext);


htitle = "Efficiency for #mu (GEM+LCT) in 1.64<|#eta|<2.05 to have TF track with ME1/b stub;p_{T}^{MC}";

helt2pt20->Draw("hist");
helt3pt20->Draw("same hist");
helt2pt20p = getEffHisto(f_def, hdir, h2p_20, hini_gl, ptreb, kGray+2, 1, 2, htitle, rpt, yrange07);
helt2pt20p->Draw("same hist");
helt3pt20p = getEffHisto(f_def, hdir, h3p_20, hini_gl, ptreb, kGray+2, 2, 2, htitle, rpt, yrange07);
helt3pt20p->Draw("same hist");

TLegend *leg = new TLegend(0.55,0.17,.999,0.57, NULL, "brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->SetNColumns(2);
leg->SetHeader("TF track p_{T}^{TF}#geq20 with ME1");
leg->AddEntry(helt2pt20, "no GEM #Delta#phi", "");
leg->AddEntry(helt2pt20p, "with GEM #Delta#phi", "l");
leg->AddEntry(helt2pt20, "2+ stubs", "l");
leg->AddEntry(helt2pt20p, "2+ stubs", "l");
leg->AddEntry(helt3pt20, "3+ stubs", "l");
leg->AddEntry(helt3pt20p, "3+ stubs", "l");
leg->Draw();
c2->Print(plotDir + "eff_gem1b_baselpcgem_dphi" + ext);


helt2pt20->Draw("hist");
helt3pt20->Draw("same hist");
helt2pt20_123 = getEffHisto(f_def, hdir, h2g_20_123, hini_gl, ptreb, kMagenta-3, 9, 2, htitle, rpt, yrange07);
helt2pt20_123->Draw("same hist");
helt3pt20_13 = getEffHisto(f_def, hdir, h2g_20_13, hini_gl, ptreb, kMagenta-3, 7, 2, htitle, rpt, yrange07);
helt3pt20_13->Draw("same hist");

TLegend *leg = new TLegend(0.5,0.17,.999,0.55, NULL, "brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
//leg->SetNColumns(2);
leg->SetHeader("TF track p_{T}^{TF}#geq20 with ME1");
//leg->AddEntry(helt2pt20, "no GEM #Delta#phi", "");
//leg->AddEntry(helt2pt20p, "with GEM #Delta#phi", "");
leg->AddEntry(helt2pt20, "2+ stubs", "l");
leg->AddEntry(helt2pt20_123, "2+ stubs (no ME1-4 tracks)", "l");
leg->AddEntry(helt3pt20_13, "2+ stubs (no ME1-2 and ME1-4)", "l");
leg->AddEntry(helt3pt20, "3+ stubs", "l");
leg->Draw();
c2->Print(plotDir + "eff_gem1b_baselpcgem_123" + ext);

return;

hegl = getEffHisto(f_def, hdir, hgl, hini, ptreb, kRed, 1, 2, htitle, rpt, yrange);
hegl->Draw("same hist")
heg = getEffHisto(f_def, hdir, hg, hini, ptreb, kBlack, 1, 2, htitle, rpt, yrange);
heg->Draw("same hist")




TString hini = "h_pt_initial_1b";
TString h2s = "h_pt_after_tfcand_eta1b_2s";
TString h3s = "h_pt_after_tfcand_eta1b_3s";
TString h2s1b = "h_pt_after_tfcand_eta1b_2s1b";
TString h3s1b = "h_pt_after_tfcand_eta1b_3s1b";


TH1D* h_eff_tf0_2s  = getEffHisto(f_def, hdir, h2s, hini, ptreb, kBlack, 1, 2, htitle, rpt, yrange);
TH1D* h_eff_tf0_3s  = getEffHisto(f_def, hdir, h3s, hini, ptreb, kBlack, 1, 2, htitle, rpt, yrange);
TH1D* h_eff_tf0_2s1b  = getEffHisto(f_def, hdir, h2s1b, hini, ptreb, kBlack, 1, 2, htitle, rpt, yrange);
TH1D* h_eff_tf0_3s1b  = getEffHisto(f_def, hdir, h3s1b, hini, ptreb, kBlack, 1, 2, htitle, rpt, yrange);


TH1D* h_eff_tf10_2s  = getEffHisto(f_def, hdir, h2s + "_pt10", hini, ptreb, kGreen+4, 1, 2, htitle, rpt,yrange);
TH1D* h_eff_tf10_2s1b  = getEffHisto(f_def, hdir, h2s1b + "_pt10", hini, ptreb, kGreen+2, 1, 2, htitle, rpt,yrange);
TH1D* h_eff_tf10_3s  = getEffHisto(f_def, hdir, h3s + "_pt10", hini, ptreb, kGreen+2, 1, 2, htitle, rpt,yrange);
TH1D* h_eff_tf10_3s1b  = getEffHisto(f_def, hdir, h3s1b + "_pt10", hini, ptreb, kGreen+2, 1, 2, htitle, rpt,yrange);
}







void drawplot_etastep(char *fname, char* pu, char *dname="tf")
{

// directory for plots is made as 
// pdir = $PWD/pic[_{dname}]_{pu}

//gStyle->SetStatW(0.13);
//gStyle->SetStatH(0.08);
gStyle->SetStatW(0.07);
gStyle->SetStatH(0.06);

gStyle->SetOptStat(0);

gStyle->SetTitleStyle(0);

char d1[111]="",d2[111]="";
if (strcmp(dname,"")>0) sprintf(d1,"_%s",dname);
if (strcmp(pu,"")>0)    sprintf(d2,"_%s",pu);
sprintf(pdir,"pic%s%s",d1,d2);

if (interactive){
  if( gSystem->AccessPathName(pdir)==0 ) {
    cout<<"directory "<<pdir<<" exists, removing it!"<<endl;
    char cmd[111];
    sprintf(cmd,"rm -r %s",pdir);
    if (gSystem->Exec(cmd) != 0) {cout<<"can't remode directory! exiting..."<<endl; return;};
  }
  gSystem->MakeDirectory(pdir);
}

//TFile *f = TFile::Open("piluphisto.root");

cout<<"opening "<<fname<<endl;

f = TFile::Open(fname);

// directory inside of root file:
char dir[111];
if (strcmp(pu,"StrictDirect")==0) sprintf(dir,"SimMuL1Strict");
if (strcmp(pu,"StrictChamber")==0) sprintf(dir,"SimMuL1StrictAll");
if (strcmp(pu,"NaturalDirect")==0) sprintf(dir,"SimMuL1");
if (strcmp(pu,"NaturalChamber")==0) sprintf(dir,"SimMuL1All");
if (strcmp(pu,"StrictDeltaY")==0) sprintf(dir,"SimMuL1StrictDY");
if (strcmp(pu,"NaturalDeltaY")==0) sprintf(dir,"SimMuL1DY");

if (strcmp(pu,"StrictChamber0")==0) sprintf(dir,"SimMuL1StrictAll0");

char label[200];

//  = (TH1D*)f->Get("SimMuL1/;1");

int etareb=1;
int ptreb=2;


//TH1D* eff_eta_after_mpc_ok_plus = setEffHisto("h_eta_after_mpc_ok_plus","h_eta_initial",dir, etareb, kRed, 1,2, "asdasd","xxx","yyy",xrange,yrange);
//eff_eta_after_mpc_ok_plus ->Draw("hist");


//################################################################################################
if (interactive) {

TCanvas* c_eff_eta_simh_by_st = new TCanvas("c_eff_eta_simh_by_st","c_eff_eta_simh_by_st",900,800 ) ;
c_eff_eta_simh_by_st->Divide(2,2,0.0001,0.0001);

TH1D* h_eta_me1_initial = setEffHisto("h_eta_me1_initial","h_eta_initial0",dir, etareb, kBlue, 1, 2, "eff(#eta): SimHits#geq4 in ME1 (MC 20<p_{T}<100)","MC #eta","",xrange,yrange);
TH1D* h_eta_me2_initial = setEffHisto("h_eta_me2_initial","h_eta_initial0",dir, etareb, kBlue, 1, 2, "eff(#eta): SimHits#geq4 in ME2 (MC 20<p_{T}<100)","MC #eta","",xrange,yrange);
TH1D* h_eta_me3_initial = setEffHisto("h_eta_me3_initial","h_eta_initial0",dir, etareb, kBlue, 1, 2, "eff(#eta): SimHits#geq4 in ME3 (MC 20<p_{T}<100)","MC #eta","",xrange,yrange);
TH1D* h_eta_me4_initial = setEffHisto("h_eta_me4_initial","h_eta_initial0",dir, etareb, kBlue, 1, 2, "eff(#eta): SimHits#geq4 in ME4 (MC 20<p_{T}<100)","MC #eta","",xrange,yrange);

c_eff_eta_simh_by_st->cd(1);
h_eta_me1_initial->Draw("hist");
c_eff_eta_simh_by_st->cd(2);
h_eta_me2_initial->Draw("hist");
c_eff_eta_simh_by_st->cd(3);
h_eta_me3_initial->Draw("hist");
c_eff_eta_simh_by_st->cd(4);
h_eta_me4_initial->Draw("hist");

Print(c_eff_eta_simh_by_st,"c_eff_eta_simh_by_st" + ext);


TCanvas* c_eff_eta_simh_by_st = new TCanvas("c_eff_eta_mpc_by_st","c_eff_eta_mpc_by_st",900,800 ) ;
c_eff_eta_mpc_by_st->Divide(2,2,0.0001,0.0001);

TH1D* h_eta_me1_mpc = setEffHisto("h_eta_me1_mpc","h_eta_initial0",dir, etareb, kBlack, 1, 2, "","MC #eta","",xrange,yrange);
TH1D* h_eta_me2_mpc = setEffHisto("h_eta_me2_mpc","h_eta_initial0",dir, etareb, kBlack, 1, 2, "","MC #eta","",xrange,yrange);
TH1D* h_eta_me3_mpc = setEffHisto("h_eta_me3_mpc","h_eta_initial0",dir, etareb, kBlack, 1, 2, "","MC #eta","",xrange,yrange);
TH1D* h_eta_me4_mpc = setEffHisto("h_eta_me4_mpc","h_eta_initial0",dir, etareb, kBlack, 1, 2, "","MC #eta","",xrange,yrange);

h_eta_me1_initial->SetTitle("eff(#eta): matched MPC in ME1 (MC 20<p_{T}<100)");
h_eta_me2_initial->SetTitle("eff(#eta): matched MPC in ME2 (MC 20<p_{T}<100)");
h_eta_me3_initial->SetTitle("eff(#eta): matched MPC in ME3 (MC 20<p_{T}<100)");
h_eta_me4_initial->SetTitle("eff(#eta): matched MPC in ME4 (MC 20<p_{T}<100)");

c_eff_eta_mpc_by_st->cd(1);
h_eta_me1_initial->Draw("hist");
h_eta_me1_mpc->Draw("hist same");
c_eff_eta_mpc_by_st->cd(2);
h_eta_me2_initial->Draw("hist");
h_eta_me2_mpc->Draw("hist same");
c_eff_eta_mpc_by_st->cd(3);
h_eta_me3_initial->Draw("hist");
h_eta_me3_mpc->Draw("hist same");
c_eff_eta_mpc_by_st->cd(4);
h_eta_me4_initial->Draw("hist");
h_eta_me4_mpc->Draw("hist same");

Print(c_eff_eta_mpc_by_st,"c_eff_eta_mpc_by_st" + ext);


}
//################################################################################################
if (interactive) {

TCanvas* c_eff_eta_simh = new TCanvas("c_eff_eta_simh","c_eff_eta_simh",1000,600 ) ;

TH1D* h_eta_initial_1st = setEffHisto("h_eta_initial_1st","h_eta_initial0",dir, etareb, kBlue, 1, 2, "eff(#eta): SimHits#geq4 in #geqN CSC stations","MC #eta","",xrange,yrange);
TH1D* h_eta_initial_2st = setEffHisto("h_eta_initial_2st","h_eta_initial0",dir, etareb, kBlue, 9, 2, "","MC #eta","",xrange,yrange);
TH1D* h_eta_initial_3st = setEffHisto("h_eta_initial_3st","h_eta_initial0",dir, etareb, kBlue, 2, 2, "","MC #eta","",xrange,yrange);

h_eta_initial_1st->Draw("hist");
h_eta_initial_2st->Draw("hist same");
h_eta_initial_3st->Draw("hist same");

TLegend *l_eff_eta_simh = new TLegend(0.347,0.222,0.926,0.535,NULL,"brNDC");
l_eff_eta_simh->SetBorderSize(0);
l_eff_eta_simh->SetFillStyle(0);
l_eff_eta_simh->SetHeader("Efficiency for #mu with p_{T}>20 to leave #geq4 SimHits in");
l_eff_eta_simh->AddEntry(h_eta_initial_1st,"#geq1 CSC stations","pl");
l_eff_eta_simh->AddEntry(h_eta_initial_2st,"#geq2 CSC stations","pl");
l_eff_eta_simh->AddEntry(h_eta_initial_3st,"#geq3 CSC stations","pl");
l_eff_eta_simh->Draw();

Print(c_eff_eta_simh,"c_eff_eta_simh" + ext);


TCanvas* c_eff_eta_mpc = new TCanvas("c_eff_eta_mpc","c_eff_eta_mpc",1000,600 ) ;

TH1D* h_eta_mpc_1st = setEffHisto("h_eta_mpc_1st","h_eta_initial0",dir, etareb, kBlue, 1, 2, "eff(#eta): matched MPC in #geqN CSC stations","MC #eta","",xrange,yrange);
TH1D* h_eta_mpc_2st = setEffHisto("h_eta_mpc_2st","h_eta_initial0",dir, etareb, kBlue, 9, 2, "","MC #eta","",xrange,yrange);
TH1D* h_eta_mpc_3st = setEffHisto("h_eta_mpc_3st","h_eta_initial0",dir, etareb, kBlue, 2, 2, "","MC #eta","",xrange,yrange);

h_eta_mpc_1st->Draw("hist");
h_eta_mpc_2st->Draw("hist same");
h_eta_mpc_3st->Draw("hist same");

TLegend *l_eff_eta_mpc = new TLegend(0.347,0.222,0.926,0.535,NULL,"brNDC");
l_eff_eta_mpc->SetBorderSize(0);
l_eff_eta_mpc->SetFillStyle(0);
l_eff_eta_mpc->SetHeader("Efficiency for #mu with p_{T}>20 to have matched MPCs in");
l_eff_eta_mpc->AddEntry(h_eta_mpc_1st,"#geq1 CSC stations","pl");
l_eff_eta_mpc->AddEntry(h_eta_mpc_2st,"#geq2 CSC stations","pl");
l_eff_eta_mpc->AddEntry(h_eta_mpc_3st,"#geq3 CSC stations","pl");
l_eff_eta_mpc->Draw();

Print(c_eff_eta_mpc,"c_eff_eta_mpc" + ext);


TCanvas* c_eff_eta_mpc_relative = new TCanvas("c_eff_eta_mpc_relative","c_eff_eta_mpc_relative",1000,600 ) ;

TH1D* h_eta_mpc_1st_r = setEffHisto("h_eta_mpc_1st","h_eta_initial_1st",dir, etareb, kBlue, 1, 2, "eff(#eta): matched MPC if SimHits#geq4 in #geqN CSC stations","MC #eta","",xrange,yrange);
TH1D* h_eta_mpc_2st_r = setEffHisto("h_eta_mpc_2st","h_eta_initial_2st",dir, etareb, kBlue, 9, 2, "","MC #eta","",xrange,yrange);
TH1D* h_eta_mpc_3st_r = setEffHisto("h_eta_mpc_3st","h_eta_initial_3st",dir, etareb, kBlue, 2, 2, "","MC #eta","",xrange,yrange);

h_eta_mpc_1st_r->Draw("hist");
h_eta_mpc_2st_r->Draw("hist same");
h_eta_mpc_3st_r->Draw("hist same");

TLegend *l_eff_eta_mpc_relative = new TLegend(0.347,0.222,0.926,0.535,NULL,"brNDC");
l_eff_eta_mpc_relative->SetBorderSize(0);
l_eff_eta_mpc_relative->SetFillStyle(0);
l_eff_eta_mpc_relative->SetHeader("Efficiency for #mu with p_{T}>20 to have matched MPCs if it has SimHits#geq4 in");
l_eff_eta_mpc_relative->AddEntry(h_eta_mpc_1st_r,"#geq1 CSC stations","pl");
l_eff_eta_mpc_relative->AddEntry(h_eta_mpc_2st_r,"#geq2 CSC stations","pl");
l_eff_eta_mpc_relative->AddEntry(h_eta_mpc_3st_r,"#geq3 CSC stations","pl");
l_eff_eta_mpc_relative->Draw();

Print(c_eff_eta_mpc_relative,"c_eff_eta_mpc_relative" + ext);

}

//################################################################################################
if (interactive) {
TCanvas* c_eff_eta_simh_me1 = new TCanvas("c_eff_eta_simh_me1","c_eff_eta_simh_me1",1000,600 ) ;

TH1D* h_eta_me1_initial_ = setEffHisto("h_eta_me1_initial","h_eta_initial0",dir, etareb, kBlue, 1, 2, "eff(#eta): SimHits#geq4 in ME1 plus more stations","MC #eta","",xrange,yrange);
TH1D* h_eta_me1_initial_2st = setEffHisto("h_eta_me1_initial_2st","h_eta_initial0",dir, etareb, kBlue, 9, 2, "","MC #eta","",xrange,yrange);
TH1D* h_eta_me1_initial_3st = setEffHisto("h_eta_me1_initial_3st","h_eta_initial0",dir, etareb, kBlue, 2, 2, "","MC #eta","",xrange,yrange);

h_eta_me1_initial_->Draw("hist");
h_eta_me1_initial_2st->Draw("hist same");
h_eta_me1_initial_3st->Draw("hist same");

TLegend *l_eff_eta_simh_me1 = new TLegend(0.347,0.222,0.926,0.535,NULL,"brNDC");
l_eff_eta_simh_me1->SetBorderSize(0);
l_eff_eta_simh_me1->SetFillStyle(0);
l_eff_eta_simh_me1->SetHeader("Efficiency for #mu with p_{T}>20 to leave #geq4 SimHits in");
l_eff_eta_simh_me1->AddEntry(h_eta_me1_initial_,"ME1","pl");
l_eff_eta_simh_me1->AddEntry(h_eta_me1_initial_2st,"ME1 + #geq1 stations","pl");
l_eff_eta_simh_me1->AddEntry(h_eta_me1_initial_3st,"ME1 + #geq2 stations","pl");
l_eff_eta_simh_me1->Draw();

Print(c_eff_eta_simh_me1,"c_eff_eta_simh_me1" + ext);
}
//################################################################################################

TCanvas* c_eff_eta_me1_stubs = new TCanvas("c_eff_eta_me1_stubs","c_eff_eta_me1_stubs",1000,600 ) ;

TH1D* h_eff_eta_me1_after_lct = setEffHisto("h_eta_me1_after_lct","h_eta_initial",dir, etareb, kRed, 2, 2, "eff(#eta): ME1 stub studies","#eta","",xrange,yrange);
TH1D* h_eff_eta_me1_after_alct = setEffHisto("h_eta_me1_after_alct","h_eta_initial",dir, etareb, kBlue+1, 2, 2, "","","",xrange,yrange);
TH1D* h_eff_eta_me1_after_clct = setEffHisto("h_eta_me1_after_clct","h_eta_initial",dir, etareb, kGreen+1, 2, 2, "","","",xrange,yrange);
TH1D* h_eff_eta_me1_after_alct_okAlct = setEffHisto("h_eta_me1_after_alct_okAlct","h_eta_initial",dir, etareb, kBlue+1, 1, 2, "","","",xrange,yrange);
TH1D* h_eff_eta_me1_after_clct_okClct = setEffHisto("h_eta_me1_after_clct_okClct","h_eta_initial",dir, etareb, kGreen+1, 1, 2, "","","",xrange,yrange);
TH1D* h_eff_eta_me1_after_alctclct = setEffHisto("h_eta_me1_after_alctclct","h_eta_initial",dir, etareb, kYellow+2, 2, 2, "","","",xrange,yrange);
TH1D* h_eff_eta_me1_after_alctclct_okAlctClct = setEffHisto("h_eta_me1_after_alctclct_okAlctClct","h_eta_initial",dir, etareb, kYellow+2, 1,2, "","","",xrange,yrange);
TH1D* h_eff_eta_me1_after_lct_okClctAlct = setEffHisto("h_eta_me1_after_lct_okAlctClct","h_eta_initial",dir, etareb, kRed, 1,2, "","","",xrange,yrange);

h_eff_eta_me1_after_lct->Draw("hist");
h_eff_eta_me1_after_alct->Draw("same hist");
h_eff_eta_me1_after_alct_okAlct->Draw("same hist");
h_eff_eta_me1_after_clct->Draw("same hist");
h_eff_eta_me1_after_clct_okClct->Draw("same hist");
h_eff_eta_me1_after_alctclct->Draw("same hist");
h_eff_eta_me1_after_alctclct_okAlctClct->Draw("same hist");
h_eff_eta_me1_after_lct->Draw("same hist");
h_eff_eta_me1_after_lct_okClctAlct->Draw("same hist");

h_eff_eta_me1_after_lct_okClctAlct->Fit("pol0","R0","",1.63,2.38);
eff11 = (h_eff_eta_me1_after_lct_okClctAlct->GetFunction("pol0"))->GetParameter(0);
cout<<eff11<<endl;
h_eff_eta_me1_after_lct_okClctAlct->Fit("pol0","R0","",1.63,2.05);
eff1b = (h_eff_eta_me1_after_lct_okClctAlct->GetFunction("pol0"))->GetParameter(0);
h_eff_eta_me1_after_lct_okClctAlct->Fit("pol0","R0","",2.05,2.38);
eff1a = (h_eff_eta_me1_after_lct_okClctAlct->GetFunction("pol0"))->GetParameter(0);
cout<<eff11<<"  "<<eff1b<<"  "<<eff1a<<endl;


TLegend *leg = new TLegend(0.347,0.222,0.926,0.535,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->SetNColumns(2);
leg->SetHeader("Efficiency for #mu with p_{T}>20 crossing a ME1 chamber with");
leg->AddEntry(h_eff_eta_me1_after_alct,"any ALCT","pl");
leg->AddEntry(h_eff_eta_me1_after_alct_okAlct,"correct ALCT","pl");
leg->AddEntry(h_eff_eta_me1_after_clct,"any CLCT","pl");
leg->AddEntry(h_eff_eta_me1_after_clct_okClct,"correct CLCT","pl");
leg->AddEntry(h_eff_eta_me1_after_alctclct,"any ALCT and CLCT","pl");
leg->AddEntry(h_eff_eta_me1_after_alctclct_okAlctClct,"correct ALCT and CLCT","pl");
leg->AddEntry(h_eff_eta_me1_after_lct,"any LCT","pl");
leg->AddEntry(h_eff_eta_me1_after_lct_okClctAlct,"correct LCT","pl");
leg->Draw();

if (!interactive) {
  char ghn[111];
  sprintf(ghn,"h_eff_eta_me1_after_lct_okClctAlct_%s",dname);
  gh = (TH1F*)h_eff_eta_me1_after_lct_okClctAlct->Clone(ghn);
  gh->GetYaxis()->SetRangeUser(0.8,1.02);

  gh->SetTitle("LCT finding efficiency in ME1 for #mu with p_{T}>10");
  gh->GetXaxis()->SetRangeUser(0.8,2.5);
  gh->GetYaxis()->SetRangeUser(0.,1.05);
  gh->GetXaxis()->SetTitle("#eta");
  gh->GetYaxis()->SetTitle("Eff.");
  gh->GetXaxis()->SetTitleSize(0.07);
  gh->GetXaxis()->SetTitleOffset(0.7);
  gh->GetYaxis()->SetLabelOffset(0.015);
  return;
}

Print(c_eff_eta_me1_stubs,"h_eff_eta_me1_steps_stubs" + ext);

//################################################################################################

TCanvas* c_eff_eta_me1_tf = new TCanvas("c_eff_eta_me1_tf","c_eff_eta_me1_tf",1000,600 ) ;

TH1D* h_eff_eta_me1_after_lct_okClctAlct         = setEffHisto("h_eta_me1_after_lct_okAlctClct","h_eta_initial",dir, etareb, kRed, 1,2, "eff(#eta): ME1 TF studies", "#eta","",xrange,yrange);
TH1D* h_eff_eta_me1_after_mplct_okClctAlct_plus = setEffHisto("h_eta_me1_after_mplct_okAlctClct_plus","h_eta_initial",dir, etareb, kBlack, 1,2, "","","",xrange,yrange);
//TH1D* h_eff_eta_after_tfcand_ok_plus      = setEffHisto("h_eta_after_tfcand_ok_plus","h_eta_initial",dir, etareb, kBlue, 1,2, "","","",xrange,yrange);
//TH1D* h_eff_eta_after_tfcand_ok_plus_pt10 = setEffHisto("h_eta_after_tfcand_ok_plus_pt10","h_eta_initial",dir, etareb, kBlue, 2,2, "","","",xrange,yrange);
TH1D* h_eff_eta_me1_after_tf_ok_plus	  = setEffHisto("h_eta_me1_after_tf_ok_plus","h_eta_initial",dir, etareb, kBlue, 1,2, "","","",xrange,yrange);
TH1D* h_eff_eta_me1_after_tf_ok_plus_pt10 = setEffHisto("h_eta_me1_after_tf_ok_plus_pt10","h_eta_initial",dir, etareb, kBlue, 2,2, "","","",xrange,yrange);

h_eff_eta_me1_after_lct_okClctAlct->Draw("hist");
h_eff_eta_me1_after_mplct_okClctAlct_plus->Draw("same hist");
h_eff_eta_me1_after_tf_ok_plus->Draw("same hist");
h_eff_eta_me1_after_tf_ok_plus_pt10->Draw("same hist");

leg = new TLegend(0.347,0.222,0.926,0.535,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->SetHeader("Eff. for #mu with p_{T}>20 crossing ME1+one more station with");
leg->AddEntry(h_eff_eta_me1_after_lct_okClctAlct,"LCT matched in ME1","pl");
leg->AddEntry(h_eff_eta_me1_after_mplct_okClctAlct_plus,"MPC matched in ME1+one","pl");
leg->AddEntry(h_eff_eta_me1_after_tf_ok_plus,"TF track with matched stubs in ME1+one","pl");
leg->AddEntry(h_eff_eta_me1_after_tf_ok_plus_pt10,"p_{T}^{TF}>10 TF track with matched stubs in ME1+one","pl");
leg->Draw();

Print(c_eff_eta_me1_tf,"h_eff_eta_me1_tf" + ext);


//################################################################################################

TCanvas* c_eff_eta_tf = new TCanvas("c_eff_eta_tf","c_eff_eta_tf",1000,600 ) ;

TH1D* h_eff_eta_after_mpc_ok_plus = setEffHisto("h_eta_after_mpc_ok_plus","h_eta_initial",dir, etareb, kBlack, 1,2, "eff(#eta): TF studies","#eta","",xrange,yrange);
TH1D* h_eff_eta_after_tfcand_ok_plus      = setEffHisto("h_eta_after_tfcand_ok_plus","h_eta_initial",dir, etareb, kBlue, 1,2, "","","",xrange,yrange);
TH1D* h_eff_eta_after_tfcand_ok_plus_pt10 = setEffHisto("h_eta_after_tfcand_ok_plus_pt10","h_eta_initial",dir, etareb, kBlue, 2,2, "","","",xrange,yrange);

h_eff_eta_after_mpc_ok_plus->Draw("hist");
h_eff_eta_after_tfcand_ok_plus->Draw("same hist");
h_eff_eta_after_tfcand_ok_plus_pt10->Draw("same hist");

leg = new TLegend(0.347,0.19,0.926,0.45,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->SetHeader("Eff. for #mu with p_{T}>20 crossing #geq2 stations with");
leg->AddEntry(h_eff_eta_after_mpc_ok_plus,"MPC matched in 2stations","pl");
leg->AddEntry(h_eff_eta_after_tfcand_ok_plus,"TF track with matched stubs in 2st","pl");
leg->AddEntry(h_eff_eta_after_tfcand_ok_plus_pt10,"p_{T}^{TF}>10 TF track with matched stubs in 2st","pl");
leg->Draw();

Print(c_eff_eta_tf,"h_eff_eta_tf" + ext);

//################################################################################################

TCanvas* c_eff_eta_tf_3st1a = new TCanvas("c_eff_eta_tf_3st1a","c_eff_eta_tf_3st1a",1000,600 ) ;

TH1D* h_eff_eta_after_mpc_ok_plus = setEffHisto("h_eta_after_mpc_ok_plus","h_eta_initial",dir, etareb, kBlack, 1,2, "eff(#eta): TF studies (3TF stubs in ME1a)","#eta","",xrange,yrange);
TH1D* h_eff_eta_after_mpc_ok_plus_3st = setEffHisto("h_eta_after_mpc_ok_plus_3st","h_eta_initial",dir, etareb, kOrange+2, 1,2, "","","",xrange,yrange);
//TH1D* h_eff_eta_after_mpc_ok_plus_3st1a = setEffHisto("h_eta_after_mpc_ok_plus_3st1a","h_eta_initial",dir, etareb, kBlack-4, 1,2, "","","",xrange,yrange);
TH1D* h_eff_eta_after_tfcand_ok_plus_3st1a      = setEffHisto("h_eta_after_tfcand_ok_plus_3st1a","h_eta_initial",dir, etareb, kBlue, 1,2, "","","",xrange,yrange);
TH1D* h_eff_eta_after_tfcand_ok_plus_pt10_3st1a = setEffHisto("h_eta_after_tfcand_ok_plus_pt10_3st1a","h_eta_initial",dir, etareb, kBlue, 2,2, "","","",xrange,yrange);

h_eff_eta_after_mpc_ok_plus->Draw("hist");
h_eff_eta_after_mpc_ok_plus_3st->Draw("same hist");
//h_eff_eta_after_mpc_ok_plus_3st1a->Draw("same hist");
h_eff_eta_after_tfcand_ok_plus_3st1a->Draw("same hist");
h_eff_eta_after_tfcand_ok_plus_pt10_3st1a->Draw("same hist");

leg = new TLegend(0.347,0.19,0.926,0.45,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->SetHeader("Eff. for #mu with p_{T}>20 crossing #geq2 stations with");
leg->AddEntry(h_eff_eta_after_mpc_ok_plus,"MPC matched in 2stations","pl");
leg->AddEntry(h_eff_eta_after_mpc_ok_plus_3st,"MPC matched in 3stations","pl");
leg->AddEntry(h_eff_eta_after_tfcand_ok_plus_3st1a,"TF track with matched stubs in 2st","pl");
leg->AddEntry(h_eff_eta_after_tfcand_ok_plus_pt10_3st1a,"p_{T}^{TF}>10 TF track with matched stubs in 2st","pl");
leg->Draw();

Print(c_eff_eta_tf_3st1a,"h_eff_eta_tf_3st1a" + ext);


//################################################################################################

TCanvas* c_eff_eta_tf_q = new TCanvas("c_eff_eta_tf_q","c_eff_eta_tf_q",1000,600 ) ;

TH1D* h_eff_eta_after_tfcand_ok_plus_q1   = setEffHisto("h_eta_after_tfcand_ok_plus_q1","h_eta_after_mpc_ok_plus",dir, etareb, kBlue, 1,1, "eff(#eta): TF quality studies (denom: 2MPCs)","#eta","",xrange,yrange);
TH1D* h_eff_eta_after_tfcand_ok_plus_q2   = setEffHisto("h_eta_after_tfcand_ok_plus_q2","h_eta_after_mpc_ok_plus",dir, etareb, kCyan+2, 1,1, "","","",xrange,yrange);
TH1D* h_eff_eta_after_tfcand_ok_plus_q3   = setEffHisto("h_eta_after_tfcand_ok_plus_q3","h_eta_after_mpc_ok_plus",dir, etareb, kMagenta+1, 1,1, "","","",xrange,yrange);

TH1D* h_eff_eta_after_tfcand_ok_plus_pt10_q1   = setEffHisto("h_eta_after_tfcand_ok_plus_pt10_q1","h_eta_after_mpc_ok_plus",dir, etareb, kBlue, 2,2, "eff(#eta): TF quality studies (denom: 2MPCs)","#eta","",xrange,yrange);
TH1D* h_eff_eta_after_tfcand_ok_plus_pt10_q2   = setEffHisto("h_eta_after_tfcand_ok_plus_pt10_q2","h_eta_after_mpc_ok_plus",dir, etareb, kCyan+2, 2,2, "","","",xrange,yrange);
TH1D* h_eff_eta_after_tfcand_ok_plus_pt10_q3   = setEffHisto("h_eta_after_tfcand_ok_plus_pt10_q3","h_eta_after_mpc_ok_plus",dir, etareb, kMagenta+1, 2,2, "","","",xrange,yrange);

h_eff_eta_after_tfcand_ok_plus_q1->Draw("hist");
h_eff_eta_after_tfcand_ok_plus_pt10_q1->Draw("same hist");
h_eff_eta_after_tfcand_ok_plus_q2->Draw("same hist");
h_eff_eta_after_tfcand_ok_plus_pt10_q2->Draw("same hist");
h_eff_eta_after_tfcand_ok_plus_q3->Draw("same hist");
h_eff_eta_after_tfcand_ok_plus_pt10_q3->Draw("same hist");

leg = new TLegend(0.347,0.19,0.926,0.45,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->SetNColumns(2);
leg->SetHeader("TF track with matched stubs in 2st and ");
leg->AddEntry(h_eff_eta_after_tfcand_ok_plus_q1,"Q#geq1","pl");
leg->AddEntry(h_eff_eta_after_tfcand_ok_plus_pt10_q1,"Q#geq1, p_{T}^{TF}>10","pl");
leg->AddEntry(h_eff_eta_after_tfcand_ok_plus_q2,"Q#geq2","pl");
leg->AddEntry(h_eff_eta_after_tfcand_ok_plus_pt10_q2,"Q#geq2, p_{T}^{TF}>10","pl");
leg->AddEntry(h_eff_eta_after_tfcand_ok_plus_q3,"Q=3","pl");
leg->AddEntry(h_eff_eta_after_tfcand_ok_plus_pt10_q3,"Q=3, p_{T}^{TF}>10","pl");
leg->Draw();

Print(c_eff_eta_tf_q,"h_eff_eta_tf_q" + ext);


//################################################################################################

TCanvas* c_eff_pt_tf = new TCanvas("c_eff_pt_tf","c_eff_pt_tf",1000,600 ) ;

TH1D* h_eff_pt_after_mpc_ok_plus = setEffHisto("h_pt_after_mpc_ok_plus","h_pt_initial",dir, ptreb, kBlack, 1,2, "eff(p_{T}^{MC}): TF studies (1.2<#eta<2.1)","p_{T}","",xrangept,yrange);
TH1D* h_eff_pt_after_tfcand_ok_plus      = setEffHisto("h_pt_after_tfcand_ok_plus","h_pt_initial",dir, ptreb, kBlue, 1,2, "","","",xrangept,yrange);
TH1D* h_eff_pt_after_tfcand_ok_plus_pt10 = setEffHisto("h_pt_after_tfcand_ok_plus_pt10","h_pt_initial",dir, ptreb, kBlue, 2,2, "","","",xrangept,yrange);

h_eff_pt_after_mpc_ok_plus->Draw("hist");
h_eff_pt_after_tfcand_ok_plus->Draw("same hist");
h_eff_pt_after_tfcand_ok_plus_pt10->Draw("same hist");

leg1 = new TLegend(0.347,0.19,0.926,0.45,NULL,"brNDC");
leg1->SetBorderSize(0);
leg1->SetFillStyle(0);
leg1->SetHeader("Eff. for #mu crossing ME1+one more station in 1.2<#eta<2.1 with");
leg1->AddEntry(h_eff_pt_after_mpc_ok_plus,"MPC matched in 2stations","pl");
leg1->AddEntry(h_eff_pt_after_tfcand_ok_plus,"TF track with matched stubs in 2st","pl");
leg1->AddEntry(h_eff_pt_after_tfcand_ok_plus_pt10,"p_{T}^{TF}>10 TF track with matched stubs in 2st","pl");
leg1->Draw();

Print(c_eff_pt_tf,"h_eff_pt_tf" + ext);


//################################################################################################

TCanvas* c_eff_pt_tf_eta1b_2s = new TCanvas("c_eff_pt_tf_eta1b_2s","c_eff_pt_tf_eta1b_2s",1000,600 ) ;

TH1D* h_eff_pt_after_tfcand_eta1b_2s  = setEffHisto("h_pt_after_tfcand_eta1b_2s","h_pt_initial_1b",dir, ptreb, kBlack, 1,2, "eff(p_{T}^{MC}): TF studies (1.64<#eta<2.14)","p_{T}^{MC}","",xrangept,yrange);
TH1D* h_eff_pt_after_tfcand_eta1b_2s_pt10 = setEffHisto("h_pt_after_tfcand_eta1b_2s_pt10","h_pt_initial_1b",dir, ptreb, kGreen+2, 1,2, "","","",xrangept,yrange);
TH1D* h_eff_pt_after_tfcand_eta1b_2s_pt20 = setEffHisto("h_pt_after_tfcand_eta1b_2s_pt20","h_pt_initial_1b",dir, ptreb, kBlue, 1,2, "","","",xrangept,yrange);
TH1D* h_eff_pt_after_tfcand_eta1b_2s_pt25 = setEffHisto("h_pt_after_tfcand_eta1b_2s_pt25","h_pt_initial_1b",dir, ptreb, kOrange, 1,2, "","","",xrangept,yrange);
TH1D* h_eff_pt_after_tfcand_eta1b_2s_pt30 = setEffHisto("h_pt_after_tfcand_eta1b_2s_pt30","h_pt_initial_1b",dir, ptreb, kRed, 1,2, "","","",xrangept,yrange);

h_eff_pt_after_tfcand_eta1b_2s->GetXaxis()->SetRangeUser(0.,49.99);

h_eff_pt_after_tfcand_eta1b_2s->Draw("hist");
h_eff_pt_after_tfcand_eta1b_2s_pt10->Draw("same hist");
h_eff_pt_after_tfcand_eta1b_2s_pt20->Draw("same hist");
h_eff_pt_after_tfcand_eta1b_2s_pt25->Draw("same hist");
h_eff_pt_after_tfcand_eta1b_2s_pt30->Draw("same hist");

leg1 = new TLegend(0.5,0.15,0.99,0.5,NULL,"brNDC");
leg1->SetBorderSize(0);
leg1->SetFillStyle(0);
leg1->SetHeader("Eff. for #mu in 1.64<#eta<2.14 to have TF track with");
leg1->AddEntry(h_eff_pt_after_tfcand_eta1b_2s,"stubs in (2+)st","pl");
leg1->AddEntry(h_eff_pt_after_tfcand_eta1b_2s_pt10,"p_{T}^{TF}>=10, stubs in (2+)st","pl");
leg1->AddEntry(h_eff_pt_after_tfcand_eta1b_2s_pt20,"p_{T}^{TF}>=20, stubs in (2+)st","pl");
leg1->AddEntry(h_eff_pt_after_tfcand_eta1b_2s_pt25,"p_{T}^{TF}>=25, stubs in (2+)st","pl");
leg1->AddEntry(h_eff_pt_after_tfcand_eta1b_2s_pt30,"p_{T}^{TF}>=30, stubs in (2+)st","pl");
leg1->Draw();

Print(c_eff_pt_tf_eta1b_2s, "h_eff_pt_tf_eta1b_2s" + ext);

//################################################################################################

TCanvas* c_eff_pt_tf_eta1b_2s1b = new TCanvas("c_eff_pt_tf_eta1b_2s1b","c_eff_pt_tf_eta1b_2s1b",1000,600 ) ;

TH1D* h_eff_pt_after_tfcand_eta1b_2s1b  = setEffHisto("h_pt_after_tfcand_eta1b_2s1b","h_pt_initial_1b",dir, ptreb, kBlack, 1,2, "eff(p_{T}^{MC}): TF studies (1.64<#eta<2.14)","p_{T}^{MC}","",xrangept,yrange);
TH1D* h_eff_pt_after_tfcand_eta1b_2s1b_pt10 = setEffHisto("h_pt_after_tfcand_eta1b_2s1b_pt10","h_pt_initial_1b",dir, ptreb, kGreen+2, 1,2, "","","",xrangept,yrange);
TH1D* h_eff_pt_after_tfcand_eta1b_2s1b_pt20 = setEffHisto("h_pt_after_tfcand_eta1b_2s1b_pt20","h_pt_initial_1b",dir, ptreb, kBlue, 1,2, "","","",xrangept,yrange);
TH1D* h_eff_pt_after_tfcand_eta1b_2s1b_pt25 = setEffHisto("h_pt_after_tfcand_eta1b_2s1b_pt25","h_pt_initial_1b",dir, ptreb, kOrange, 1,2, "","","",xrangept,yrange);
TH1D* h_eff_pt_after_tfcand_eta1b_2s1b_pt30 = setEffHisto("h_pt_after_tfcand_eta1b_2s1b_pt30","h_pt_initial_1b",dir, ptreb, kRed, 1,2, "","","",xrangept,yrange);

h_eff_pt_after_tfcand_eta1b_2s1b->GetXaxis()->SetRangeUser(0.,49.99);

h_eff_pt_after_tfcand_eta1b_2s1b->Draw("hist");
h_eff_pt_after_tfcand_eta1b_2s1b_pt10->Draw("same hist");
h_eff_pt_after_tfcand_eta1b_2s1b_pt20->Draw("same hist");
h_eff_pt_after_tfcand_eta1b_2s1b_pt25->Draw("same hist");
h_eff_pt_after_tfcand_eta1b_2s1b_pt30->Draw("same hist");

leg1 = new TLegend(0.5,0.15,0.99,0.5,NULL,"brNDC");
leg1->SetBorderSize(0);
leg1->SetFillStyle(0);
leg1->SetHeader("Eff. for #mu in 1.64<#eta<2.14 to have TF track with");
leg1->AddEntry(h_eff_pt_after_tfcand_eta1b_2s1b,"stubs in ME1+(1+)st","pl");
leg1->AddEntry(h_eff_pt_after_tfcand_eta1b_2s1b_pt10,"p_{T}^{TF}>=10, stubs in ME1+(1+)st","pl");
leg1->AddEntry(h_eff_pt_after_tfcand_eta1b_2s1b_pt20,"p_{T}^{TF}>=20, stubs in ME1+(1+)st","pl");
leg1->AddEntry(h_eff_pt_after_tfcand_eta1b_2s1b_pt25,"p_{T}^{TF}>=25, stubs in ME1+(1+)st","pl");
leg1->AddEntry(h_eff_pt_after_tfcand_eta1b_2s1b_pt30,"p_{T}^{TF}>=30, stubs in ME1+(1+)st","pl");
leg1->Draw();

Print(c_eff_pt_tf_eta1b_2s1b, "h_eff_pt_tf_eta1b_2s1b" + ext);


//################################################################################################

TCanvas* c_eff_pt_tf_eta1b_3s = new TCanvas("c_eff_pt_tf_eta1b_3s","c_eff_pt_tf_eta1b_3s",1000,600 ) ;

TH1D* h_eff_pt_after_tfcand_eta1b_3s  = setEffHisto("h_pt_after_tfcand_eta1b_3s","h_pt_initial_1b",dir, ptreb, kBlack, 1,2, "eff(p_{T}^{MC}): TF studies (1.64<#eta<2.14)","p_{T}^{MC}","",xrangept,yrange);
TH1D* h_eff_pt_after_tfcand_eta1b_3s_pt10 = setEffHisto("h_pt_after_tfcand_eta1b_3s_pt10","h_pt_initial_1b",dir, ptreb, kGreen+2, 1,2, "","","",xrangept,yrange);
TH1D* h_eff_pt_after_tfcand_eta1b_3s_pt20 = setEffHisto("h_pt_after_tfcand_eta1b_3s_pt20","h_pt_initial_1b",dir, ptreb, kBlue, 1,2, "","","",xrangept,yrange);
TH1D* h_eff_pt_after_tfcand_eta1b_3s_pt25 = setEffHisto("h_pt_after_tfcand_eta1b_3s_pt25","h_pt_initial_1b",dir, ptreb, kOrange, 1,2, "","","",xrangept,yrange);
TH1D* h_eff_pt_after_tfcand_eta1b_3s_pt30 = setEffHisto("h_pt_after_tfcand_eta1b_3s_pt30","h_pt_initial_1b",dir, ptreb, kRed, 1,2, "","","",xrangept,yrange);

h_eff_pt_after_tfcand_eta1b_3s->GetXaxis()->SetRangeUser(0.,49.99);

h_eff_pt_after_tfcand_eta1b_3s->Draw("hist");
h_eff_pt_after_tfcand_eta1b_3s_pt10->Draw("same hist");
h_eff_pt_after_tfcand_eta1b_3s_pt20->Draw("same hist");
h_eff_pt_after_tfcand_eta1b_3s_pt25->Draw("same hist");
h_eff_pt_after_tfcand_eta1b_3s_pt30->Draw("same hist");

leg1 = new TLegend(0.5,0.15,0.99,0.5,NULL,"brNDC");
leg1->SetBorderSize(0);
leg1->SetFillStyle(0);
leg1->SetHeader("Eff. for #mu in 1.64<#eta<2.14 to have TF track with");
leg1->AddEntry(h_eff_pt_after_tfcand_eta1b_3s,"stubs in (3+)st","pl");
leg1->AddEntry(h_eff_pt_after_tfcand_eta1b_3s_pt10,"p_{T}^{TF}>=10, stubs in (3+)st","pl");
leg1->AddEntry(h_eff_pt_after_tfcand_eta1b_3s_pt20,"p_{T}^{TF}>=20, stubs in (3+)st","pl");
leg1->AddEntry(h_eff_pt_after_tfcand_eta1b_3s_pt25,"p_{T}^{TF}>=25, stubs in (3+)st","pl");
leg1->AddEntry(h_eff_pt_after_tfcand_eta1b_3s_pt30,"p_{T}^{TF}>=30, stubs in (3+)st","pl");
leg1->Draw();

Print(c_eff_pt_tf_eta1b_3s, "h_eff_pt_tf_eta1b_3s" + ext);

//################################################################################################

TCanvas* c_eff_pt_tf_eta1b_3s1b = new TCanvas("c_eff_pt_tf_eta1b_3s1b","c_eff_pt_tf_eta1b_3s1b",1000,600 ) ;

TH1D* h_eff_pt_after_tfcand_eta1b_3s1b  = setEffHisto("h_pt_after_tfcand_eta1b_3s1b","h_pt_initial_1b",dir, ptreb, kBlack, 1,2, "eff(p_{T}^{MC}): TF studies (1.64<#eta<2.14)","p_{T}^{MC}","",xrangept,yrange);
TH1D* h_eff_pt_after_tfcand_eta1b_3s1b_pt10 = setEffHisto("h_pt_after_tfcand_eta1b_3s1b_pt10","h_pt_initial_1b",dir, ptreb, kGreen+2, 1,2, "","","",xrangept,yrange);
TH1D* h_eff_pt_after_tfcand_eta1b_3s1b_pt20 = setEffHisto("h_pt_after_tfcand_eta1b_3s1b_pt20","h_pt_initial_1b",dir, ptreb, kBlue, 1,2, "","","",xrangept,yrange);
TH1D* h_eff_pt_after_tfcand_eta1b_3s1b_pt25 = setEffHisto("h_pt_after_tfcand_eta1b_3s1b_pt25","h_pt_initial_1b",dir, ptreb, kOrange, 1,2, "","","",xrangept,yrange);
TH1D* h_eff_pt_after_tfcand_eta1b_3s1b_pt30 = setEffHisto("h_pt_after_tfcand_eta1b_3s1b_pt30","h_pt_initial_1b",dir, ptreb, kRed, 1,2, "","","",xrangept,yrange);

h_eff_pt_after_tfcand_eta1b_3s1b->GetXaxis()->SetRangeUser(0.,49.99);

h_eff_pt_after_tfcand_eta1b_3s1b->Draw("hist");
h_eff_pt_after_tfcand_eta1b_3s1b_pt10->Draw("same hist");
h_eff_pt_after_tfcand_eta1b_3s1b_pt20->Draw("same hist");
h_eff_pt_after_tfcand_eta1b_3s1b_pt25->Draw("same hist");
h_eff_pt_after_tfcand_eta1b_3s1b_pt30->Draw("same hist");

leg1 = new TLegend(0.5,0.15,0.99,0.5,NULL,"brNDC");
leg1->SetBorderSize(0);
leg1->SetFillStyle(0);
leg1->SetHeader("Eff. for #mu in 1.64<#eta<2.14 to have TF track with");
leg1->AddEntry(h_eff_pt_after_tfcand_eta1b_3s1b,"stubs in ME1+(2+)st","pl");
leg1->AddEntry(h_eff_pt_after_tfcand_eta1b_3s1b_pt10,"p_{T}^{TF}>=10, stubs in ME1+(2+)st","pl");
leg1->AddEntry(h_eff_pt_after_tfcand_eta1b_3s1b_pt20,"p_{T}^{TF}>=20, stubs in ME1+(2+)st","pl");
leg1->AddEntry(h_eff_pt_after_tfcand_eta1b_3s1b_pt25,"p_{T}^{TF}>=25, stubs in ME1+(2+)st","pl");
leg1->AddEntry(h_eff_pt_after_tfcand_eta1b_3s1b_pt30,"p_{T}^{TF}>=30, stubs in ME1+(2+)st","pl");
leg1->Draw();

Print(c_eff_pt_tf_eta1b_3s1b, "h_eff_pt_tf_eta1b_3s1b" + ext);



//################################################################################################

/*
TCanvas* c_eff_pth_tf = new TCanvas("c_eff_pth_tf","c_eff_pth_tf",1000,600 ) ;

TH1D* h_eff_pth_after_mpc_ok_plus = setEffHisto("h_pth_after_mpc_ok_plus","h_pth_initial",dir, ptreb, kBlack, 1,2, "eff(p_{T}^{MC}): TF studies (2.1<#eta<2.4)","p_{T}","",xrangept,yrange);
TH1D* h_eff_pth_after_tfcand_ok_plus      = setEffHisto("h_pth_after_tfcand_ok_plus","h_pth_initial",dir, ptreb, kBlue, 1,2, "","","",xrangept,yrange);
TH1D* h_eff_pth_after_tfcand_ok_plus_pt10 = setEffHisto("h_pth_after_tfcand_ok_plus_pt10","h_pth_initial",dir, ptreb, kBlue, 2,2, "","","",xrangept,yrange);

h_eff_pth_after_mpc_ok_plus->Draw("hist");
h_eff_pth_after_tfcand_ok_plus->Draw("same hist");
h_eff_pth_after_tfcand_ok_plus_pt10->Draw("same hist");

leg1 = new TLegend(0.347,0.19,0.926,0.45,NULL,"brNDC");
leg1->SetBorderSize(0);
leg1->SetFillStyle(0);
leg1->SetHeader("Eff. for #mu crossing ME1+one more station in 2.1<#eta<2.4 with");
leg1->AddEntry(h_eff_pth_after_mpc_ok_plus,"MPC matched in 2stations","pl");
leg1->AddEntry(h_eff_pth_after_tfcand_ok_plus,"TF track with matched stubs in 2st","pl");
leg1->AddEntry(h_eff_pth_after_tfcand_ok_plus_pt10,"p_{T}^{TF}>10 TF track with matched stubs in 2st","pl");
leg1->Draw();

Print(c_eff_pth_tf,"h_eff_pth_tf" + ext);
*/

//################################################################################################

/*
TCanvas* c_eff_pth_tf_3st1a = new TCanvas("c_eff_pth_tf_3st1a","c_eff_pth_tf_3st1a",1000,600 ) ;

TH1D* h_eff_pth_after_mpc_ok_plus = setEffHisto("h_pth_after_mpc_ok_plus","h_pth_initial",dir, ptreb, kBlack, 1,2, "eff(p_{T}^{MC}): TF studies (2.1<#eta<2.4)","p_{T}","",xrangept,yrange);
TH1D* h_eff_pth_after_tfcand_ok_plus_3st1a      = setEffHisto("h_pth_after_tfcand_ok_plus_3st1a","h_pth_initial",dir, ptreb, kBlue, 1,2, "eff(p_{T}^{MC}): TF studies (denom: 2MPCs, 2.1<#eta<2.4)","p_{T}","",xrangept,yrange);
TH1D* h_eff_pth_after_tfcand_ok_plus_pt10_3st1a = setEffHisto("h_pth_after_tfcand_ok_plus_pt10_3st1a","h_pth_initial",dir, ptreb, kBlue, 2,2, "","","",xrangept,yrange);

h_eff_pth_after_mpc_ok_plus->Draw("hist");
h_eff_pth_after_tfcand_ok_plus_3st1a->Draw("same hist");
h_eff_pth_after_tfcand_ok_plus_pt10_3st1a->Draw("same hist");

leg1 = new TLegend(0.347,0.19,0.926,0.45,NULL,"brNDC");
leg1->SetBorderSize(0);
leg1->SetFillStyle(0);
leg1->SetHeader("Eff. for #mu crossing ME1+one more station in 2.1<#eta<2.4 with");
leg1->AddEntry(h_eff_pth_after_mpc_ok_plus,"MPC matched in 2stations","pl");
leg1->AddEntry(h_eff_pth_after_tfcand_ok_plus_3st1a,"TF track with matched stubs in 3st","pl");
leg1->AddEntry(h_eff_pth_after_tfcand_ok_plus_pt10_3st1a,"p_{T}^{TF}>10 TF track with matched stubs in 3st","pl");
leg1->Draw();

Print(c_eff_pth_tf_3st1a,"h_eff_pth_tf_3st1a" + ext);

*/
//################################################################################################

TCanvas* c_eff_pt_tf_q = new TCanvas("c_eff_pt_tf_q","c_eff_pt_tf_q",1000,600 ) ;

TH1D* h_eff_pt_after_tfcand_ok_plus_q1   = setEffHisto("h_pt_after_tfcand_ok_plus_q1","h_pt_after_mpc_ok_plus",dir, etareb, kBlue, 1,1, "eff(p_{T}^{MC}): TF quality studies (denom: 2MPCs, 1.2<#eta<2.1)","p_{T}^{MC}","",xrangept,yrange);
TH1D* h_eff_pt_after_tfcand_ok_plus_q2   = setEffHisto("h_pt_after_tfcand_ok_plus_q2","h_pt_after_mpc_ok_plus",dir, etareb, kCyan+2, 1,1, "eff(p_{T}^{MC}): TF quality studies (denom: 2MPCs, 1.2<#eta<2.1)","p_{T}^{MC}","",xrangept,yrange);
TH1D* h_eff_pt_after_tfcand_ok_plus_q3   = setEffHisto("h_pt_after_tfcand_ok_plus_q3","h_pt_after_mpc_ok_plus",dir, etareb, kMagenta+1, 1,1, "","","",xrangept,yrange);

//TH1D* h_eff_pt_after_tfcand_ok_plus_pt10_q1   = setEffHisto("h_pt_after_tfcand_ok_plus_pt10_q1","h_pt_after_mpc_ok_plus",dir, etareb, kBlue, 2,2, "","","",xrangept,yrange);
TH1D* h_eff_pt_after_tfcand_ok_plus_pt10_q2   = setEffHisto("h_pt_after_tfcand_ok_plus_pt10_q2","h_pt_after_mpc_ok_plus",dir, etareb, kCyan+2, 2,2, "","","",xrangept,yrange);
TH1D* h_eff_pt_after_tfcand_ok_plus_pt10_q3   = setEffHisto("h_pt_after_tfcand_ok_plus_pt10_q3","h_pt_after_mpc_ok_plus",dir, etareb, kMagenta+1, 2,2, "","","",xrangept,yrange);

h_eff_pt_after_tfcand_ok_plus_q1->Draw("hist");
//h_eff_pt_after_tfcand_ok_plus_pt10_q1->Draw("same hist");
h_eff_pt_after_tfcand_ok_plus_q2->Draw("same hist");
//h_eff_pt_after_tfcand_ok_plus_pt10_q2->Draw("same hist");
h_eff_pt_after_tfcand_ok_plus_q3->Draw("same hist");
h_eff_pt_after_tfcand_ok_plus_pt10_q3->Draw("same hist");

leg1 = new TLegend(0.347,0.19,0.926,0.45,NULL,"brNDC");
leg1->SetBorderSize(0);
leg1->SetFillStyle(0);
leg1->SetNColumns(2);
leg1->SetHeader("TF track with matched stubs in 2st and ");
leg1->AddEntry(h_eff_pt_after_tfcand_ok_plus_q1,"Q#geq1","pl");
//leg1->AddEntry(h_eff_pt_after_tfcand_ok_plus_pt10_q1,"Q#geq1, p_{T}^{TF}>10","pl");
leg1->AddEntry(h_eff_pt_after_tfcand_ok_plus_q2,"Q#geq2","pl");
//leg1->AddEntry(h_eff_pt_after_tfcand_ok_plus_pt10_q2,"Q#geq2, p_{T}^{TF}>10","pl");
leg1->AddEntry(h_eff_pt_after_tfcand_ok_plus_q3,"Q=3","pl");
leg1->AddEntry(h_eff_pt_after_tfcand_ok_plus_pt10_q3,"Q=3, p_{T}^{TF}>10","pl");
leg1->Draw();

Print(c_eff_pt_tf_q,"h_eff_pt_tf_q" + ext);

//################################################################################################

TCanvas* c_eff_ptres_tf = new TCanvas("c_eff_ptres_tf","c_eff_ptres_tf",1000,600 ) ;

TH1D* h_eff_pt_after_tfcand_ok_plus_pt10 = setEffHisto("h_pt_after_tfcand_ok_plus_pt10","h_pt_after_tfcand_ok_plus",dir, ptreb, kBlue, 1,2, "p_{T}^{TF}>10 assignment eff(p_{T}^{MC}) studies (denom: any p_{T}^{TF}, 1.2<#eta<2.1)","p_{T}^{MC}","",xrangept,yrange);
TH1D* h_eff_pt_after_tfcand_ok_plus_pt10_q2 = setEffHisto("h_pt_after_tfcand_ok_plus_pt10_q2","h_pt_after_tfcand_ok_plus_q2",dir, ptreb, kCyan+2, 1,2, "","","",xrangept,yrange);
TH1D* h_eff_pt_after_tfcand_ok_plus_pt10_q3 = setEffHisto("h_pt_after_tfcand_ok_plus_pt10_q3","h_pt_after_tfcand_ok_plus_q3",dir, ptreb, kMagenta+1, 1,2, "","","",xrangept,yrange);

h_eff_pt_after_tfcand_ok_plus_pt10->Draw("hist");
h_eff_pt_after_tfcand_ok_plus_pt10_q2->Draw("same hist");
h_eff_pt_after_tfcand_ok_plus_pt10_q3->Draw("same hist");

leg1 = new TLegend(0.347,0.19,0.926,0.45,NULL,"brNDC");
leg1->SetBorderSize(0);
leg1->SetFillStyle(0);
leg1->SetHeader("for #mu with p_{T}>20 crossing ME1+one station to pass p_{T}^{TF}>10 with");
leg1->AddEntry(h_eff_pt_after_tfcand_ok_plus_pt10,"any Q","pl");
leg1->AddEntry(h_eff_pt_after_tfcand_ok_plus_pt10_q2,"Q#geq2","pl");
leg1->AddEntry(h_eff_pt_after_tfcand_ok_plus_pt10_q3,"Q=3","pl");
leg1->Draw();

Print(c_eff_ptres_tf,"h_eff_ptres_tf" + ext);


//################################################################################################

/*
TCanvas* c_eff_DR2_tf = new TCanvas("c_eff_DR2_tf","c_eff_DR2_tf",1000,600 ) ;

double xrangeDR[2]={0.,3.5};

TH1D* h_eff_DR_2SimTr_after_tfcand_ok_plus = setEffHisto("h_DR_2SimTr_after_tfcand_ok_plus","h_DR_2SimTr_after_mpc_ok_plus",dir, 3, kBlue, 1,2, "eff(#DeltaR(Tr1,Tr2)): for Tr1: p_{T}^{MC}>20, denom: 2 MPCs, 1.2<#eta<2.1","#DeltaR(Tr1,Tr2)","",xrangeDR,yrange);

h_eff_DR_2SimTr_after_tfcand_ok_plus->SetLineColor(0);
h_eff_DR_2SimTr_after_tfcand_ok_plus->Draw("hist");


TGraphAsymmErrors *gg = new TGraphAsymmErrors();
gg->BayesDivide((const TH1*)h1,(const TH1*)h2);
gg->Draw("p");

Print(c_eff_DR2_tf,"h_eff_DR_2SimTr_tf" + ext);
*/


return;return;










h_eta_initial = (TH1D*)getH(dir,"h_eta_initial");
h_eta_after_mpc = (TH1D*)getH(dir,"h_eta_after_mpc");
h_eta_after_mpc_st1 = (TH1D*)getH(dir,"h_eta_after_mpc_st1");
h_eta_after_mpc_st1_good = (TH1D*)getH(dir,"h_eta_after_mpc_st1_good");
h_eta_after_mpc_ok = (TH1D*)getH(dir,"h_eta_after_mpc_ok");
h_eta_after_mpc_ok_plus = (TH1D*)getH(dir,"h_eta_after_mpc_ok_plus");

h_eta_after_tftrack = (TH1D*)getH(dir,"h_eta_after_tftrack");

h_eta_after_tfcand = (TH1D*)getH(dir,"h_eta_after_tfcand");
h_eta_after_tfcand_q1 = (TH1D*)getH(dir,"h_eta_after_tfcand_q1");
h_eta_after_tfcand_q2 = (TH1D*)getH(dir,"h_eta_after_tfcand_q2");
h_eta_after_tfcand_q3 = (TH1D*)getH(dir,"h_eta_after_tfcand_q3");
h_eta_after_tfcand_ok = (TH1D*)getH(dir,"h_eta_after_tfcand_ok");
h_eta_after_tfcand_ok_plus = (TH1D*)getH(dir,"h_eta_after_tfcand_ok_plus");
h_eta_after_tfcand_ok_pt10 = (TH1D*)getH(dir,"h_eta_after_tfcand_ok_pt10");
h_eta_after_tfcand_ok_plus_pt10 = (TH1D*)getH(dir,"h_eta_after_tfcand_ok_plus_pt10");

h_eta_after_tfcand_all = (TH1D*)getH(dir,"h_eta_after_tfcand_all");
h_eta_after_tfcand_all_pt10 = (TH1D*)getH(dir,"h_eta_after_tfcand_all_pt10");

h_eta_after_gmtreg     = (TH1D*)getH(dir,"h_eta_after_gmtreg");
h_eta_after_gmtreg_all = (TH1D*)getH(dir,"h_eta_after_gmtreg_all");
h_eta_after_gmtreg_dr  = (TH1D*)getH(dir,"h_eta_after_gmtreg_dr");
h_eta_after_gmt        = (TH1D*)getH(dir,"h_eta_after_gmt");
h_eta_after_gmt_all    = (TH1D*)getH(dir,"h_eta_after_gmt_all");
h_eta_after_xtra       = (TH1D*)getH(dir,"h_eta_after_xtra");
h_eta_after_xtra_all   = (TH1D*)getH(dir,"h_eta_after_xtra_all");
h_eta_after_xtra_dr    = (TH1D*)getH(dir,"h_eta_after_xtra_dr");

h_eta_after_tfcand_pt10 = (TH1D*)getH(dir,"h_eta_after_tfcand_pt10");
h_eta_after_tfcand_my_st1 = (TH1D*)getH(dir,"h_eta_after_tfcand_my_st1");
h_eta_after_tfcand_org_st1 = (TH1D*)getH(dir,"h_eta_after_tfcand_org_st1");
h_eta_after_tfcand_comm_st1 = (TH1D*)getH(dir,"h_eta_after_tfcand_comm_st1");
h_eta_after_tfcand_my_st1_pt10 = (TH1D*)getH(dir,"h_eta_after_tfcand_my_st1_pt10");
h_eta_after_gmtreg_dr_pt10  = (TH1D*)getH(dir,"h_eta_after_gmtreg_dr_pt10");
h_eta_after_gmtreg_pt10= (TH1D*)getH(dir,"h_eta_after_gmtreg_pt10");
h_eta_after_gmt_pt10        = (TH1D*)getH(dir,"h_eta_after_gmt_pt10");
h_eta_after_xtra_dr_pt10    = (TH1D*)getH(dir,"h_eta_after_xtra_dr_pt10");

// = (TH1D*)getH(dir,"");

h_eta_me1_after_alct = (TH1D*)getH(dir,"h_eta_me1_after_alct");
h_eta_me1_after_alct_okAlct = (TH1D*)getH(dir,"h_eta_me1_after_alct_okAlct");
h_eta_me1_after_clct = (TH1D*)getH(dir,"h_eta_me1_after_clct");
h_eta_me1_after_clct_okClct = (TH1D*)getH(dir,"h_eta_me1_after_clct_okClct");
h_eta_me1_after_alctclct = (TH1D*)getH(dir,"h_eta_me1_after_alctclct");
h_eta_me1_after_alctclct_okAlct = (TH1D*)getH(dir,"h_eta_me1_after_alctclct_okAlct");
h_eta_me1_after_alctclct_okClct = (TH1D*)getH(dir,"h_eta_me1_after_alctclct_okClct");
h_eta_me1_after_alctclct_okAlctClct = (TH1D*)getH(dir,"h_eta_me1_after_alctclct_okAlctClct");

h_eta_me1_after_lct = (TH1D*)getH(dir,"h_eta_me1_after_lct");
h_eta_me1_after_lct_okAlct = (TH1D*)getH(dir,"h_eta_me1_after_lct_okAlct");
h_eta_me1_after_lct_okAlctClct = (TH1D*)getH(dir,"h_eta_me1_after_lct_okAlctClct");
h_eta_me1_after_lct_okClct = (TH1D*)getH(dir,"h_eta_me1_after_lct_okClct");
h_eta_me1_after_lct_okClctAlct = (TH1D*)getH(dir,"h_eta_me1_after_lct_okClctAlct");
h_eta_me1_after_mplct_okAlctClct = (TH1D*)getH(dir,"h_eta_me1_after_mplct_okAlctClct");
h_eta_me1_after_mplct_okAlctClct_plus = (TH1D*)getH(dir,"h_eta_me1_after_mplct_okAlctClct_plus");
h_eta_me1_after_tf_ok = (TH1D*)getH(dir,"h_eta_me1_after_tf_ok");
h_eta_me1_after_tf_ok_pt10 = (TH1D*)getH(dir,"h_eta_me1_after_tf_ok_pt10");
h_eta_me1_after_tf_ok_plus = (TH1D*)getH(dir,"h_eta_me1_after_tf_ok_plus");
h_eta_me1_after_tf_ok_plus_pt10 = (TH1D*)getH(dir,"h_eta_me1_after_tf_ok_plus_pt10");



myRebin(h_eta_initial,etareb);
myRebin(h_eta_after_mpc,etareb);
myRebin(h_eta_after_mpc_st1,etareb);
myRebin(h_eta_after_mpc_st1_good,etareb);
myRebin(h_eta_after_mpc_ok,etareb);
myRebin(h_eta_after_mpc_ok_plus,etareb);
myRebin(h_eta_after_tftrack,etareb);

myRebin(h_eta_after_tfcand,etareb);
myRebin(h_eta_after_tfcand_q1,etareb);
myRebin(h_eta_after_tfcand_q2,etareb);
myRebin(h_eta_after_tfcand_q3,etareb);
myRebin(h_eta_after_tfcand_ok,etareb);
myRebin(h_eta_after_tfcand_ok_plus,etareb);
myRebin(h_eta_after_tfcand_ok_pt10,etareb);
myRebin(h_eta_after_tfcand_ok_plus_pt10,etareb);

myRebin(h_eta_after_tfcand_all,etareb);
myRebin(h_eta_after_tfcand_all_pt10,etareb);

myRebin(h_eta_after_gmtreg    ,etareb);
myRebin(h_eta_after_gmtreg_all,etareb);
myRebin(h_eta_after_gmtreg_dr ,etareb);
myRebin(h_eta_after_gmt       ,etareb);
myRebin(h_eta_after_gmt_all   ,etareb);
myRebin(h_eta_after_xtra      ,etareb);
myRebin(h_eta_after_xtra_all  ,etareb);
myRebin(h_eta_after_xtra_dr   ,etareb);

myRebin(h_eta_after_tfcand_pt10,etareb);
myRebin(h_eta_after_tfcand_my_st1,etareb);
myRebin(h_eta_after_tfcand_org_st1,etareb);
myRebin(h_eta_after_tfcand_comm_st1,etareb);
myRebin(h_eta_after_tfcand_my_st1_pt10,etareb);
myRebin(h_eta_after_gmtreg_pt10    ,etareb);
myRebin(h_eta_after_gmtreg_dr_pt10 ,etareb);
myRebin(h_eta_after_gmt_pt10       ,etareb);
myRebin(h_eta_after_xtra_dr_pt10   ,etareb);

myRebin(h_eta_me1_after_alct,etareb);
myRebin(h_eta_me1_after_alct_okAlct,etareb);
myRebin(h_eta_me1_after_clct,etareb);
myRebin(h_eta_me1_after_clct_okClct,etareb);
myRebin(h_eta_me1_after_alctclct,etareb);
myRebin(h_eta_me1_after_alctclct_okAlct,etareb);
myRebin(h_eta_me1_after_alctclct_okClct,etareb);
myRebin(h_eta_me1_after_alctclct_okAlctClct,etareb);

myRebin(h_eta_me1_after_lct ,etareb);
myRebin(h_eta_me1_after_lct_okAlct ,etareb);
myRebin(h_eta_me1_after_lct_okAlctClct ,etareb);
myRebin(h_eta_me1_after_lct_okClct ,etareb);
myRebin(h_eta_me1_after_lct_okClctAlct ,etareb);
myRebin(h_eta_me1_after_mplct_okAlctClct ,etareb);
myRebin(h_eta_me1_after_mplct_okAlctClct_plus ,etareb);
myRebin(h_eta_me1_after_tf_ok ,etareb);
myRebin(h_eta_me1_after_tf_ok_pt10 ,etareb);
myRebin(h_eta_me1_after_tf_ok_plus ,etareb);
myRebin(h_eta_me1_after_tf_ok_plus_pt10 ,etareb);

h_eff_eta_after_mpc = (TH1D*) h_eta_after_mpc->Clone("h_eff_eta_after_mpc");
h_eff_eta_after_mpc_st1 = (TH1D*) h_eta_after_mpc_st1->Clone("h_eff_eta_after_mpc_st1");
h_eff_eta_after_mpc_st1_good = (TH1D*) h_eta_after_mpc_st1_good->Clone("h_eff_eta_after_mpc_st1_good");
h_eff_eta_after_mpc_ok = (TH1D*) h_eta_after_mpc_ok->Clone("h_eff_eta_after_mpc_ok");
h_eff_eta_after_mpc_ok_plus = (TH1D*) h_eta_after_mpc_ok_plus->Clone("h_eff_eta_after_mpc_ok_plus");
h_eff_eta_after_tftrack = (TH1D*) h_eta_after_tftrack->Clone("h_eff_eta_after_tftrack");

h_eff_eta_after_tfcand = (TH1D*) h_eta_after_tfcand->Clone("h_eff_eta_after_tfcand");
h_eff_eta_after_tfcand_q1 = (TH1D*) h_eta_after_tfcand_q1->Clone("h_eff_eta_after_tfcand_q1");
h_eff_eta_after_tfcand_q2 = (TH1D*) h_eta_after_tfcand_q2->Clone("h_eff_eta_after_tfcand_q2");
h_eff_eta_after_tfcand_q3 = (TH1D*) h_eta_after_tfcand_q3->Clone("h_eff_eta_after_tfcand_q3");
h_eff_eta_after_tfcand_ok = (TH1D*) h_eta_after_tfcand_ok->Clone("h_eff_eta_after_tfcand_ok");
h_eff_eta_after_tfcand_ok_plus = (TH1D*) h_eta_after_tfcand_ok_plus->Clone("h_eff_eta_after_tfcand_ok_plus");
h_eff_eta_after_tfcand_ok_pt10 = (TH1D*) h_eta_after_tfcand_ok_pt10->Clone("h_eff_eta_after_tfcand_ok_pt10");
h_eff_eta_after_tfcand_ok_plus_pt10 = (TH1D*) h_eta_after_tfcand_ok_plus_pt10->Clone("h_eff_eta_after_tfcand_ok_plus_pt10");

h_eff_eta_after_tfcand_all = (TH1D*) h_eta_after_tfcand_all->Clone("h_eff_eta_after_tfcand_all");
h_eff_eta_after_tfcand_all_pt10 = (TH1D*) h_eta_after_tfcand_all_pt10->Clone("h_eff_eta_after_tfcand_all_pt10");

h_eff_eta_after_gmtreg = (TH1D*) h_eta_after_gmtreg->Clone("h_eff_eta_after_gmtreg");
h_eff_eta_after_gmtreg_all = (TH1D*) h_eta_after_gmtreg_all->Clone("h_eff_eta_after_gmtreg_all");
h_eff_eta_after_gmtreg_dr = (TH1D*) h_eta_after_gmtreg_dr->Clone("h_eff_eta_after_gmtreg_dr");
h_eff_eta_after_gmt = (TH1D*) h_eta_after_gmt->Clone("h_eff_eta_after_gmt");
h_eff_eta_after_gmt_all = (TH1D*) h_eta_after_gmt_all->Clone("h_eff_eta_after_gmt_all");
h_eff_eta_after_xtra = (TH1D*) h_eta_after_xtra->Clone("h_eff_eta_after_xtra");
h_eff_eta_after_xtra_all = (TH1D*) h_eta_after_xtra_all->Clone("h_eff_eta_after_xtra_all");
h_eff_eta_after_xtra_dr = (TH1D*) h_eta_after_xtra_dr->Clone("h_eff_eta_after_xtra_dr");

h_eff_eta_after_tfcand_pt10 = (TH1D*) h_eta_after_tfcand_pt10->Clone("h_eff_eta_after_tfcand_pt10");
h_eff_eta_after_tfcand_my_st1 = (TH1D*) h_eta_after_tfcand_my_st1->Clone("h_eff_eta_after_tfcand_my_st1");
h_eff_eta_after_tfcand_org_st1 = (TH1D*) h_eta_after_tfcand_org_st1->Clone("h_eff_eta_after_tfcand_org_st1");
h_eff_eta_after_tfcand_comm_st1 = (TH1D*) h_eta_after_tfcand_comm_st1->Clone("h_eff_eta_after_tfcand_comm_st1");
h_eff_eta_after_tfcand_my_st1_pt10 = (TH1D*) h_eta_after_tfcand_my_st1_pt10->Clone("h_eff_eta_after_tfcand_my_st1_pt10");
h_eff_eta_after_gmtreg_pt10 = (TH1D*) h_eta_after_gmtreg_pt10->Clone("h_eff_eta_after_gmtreg_pt10");
h_eff_eta_after_gmtreg_dr_pt10 = (TH1D*) h_eta_after_gmtreg_dr_pt10->Clone("h_eff_eta_after_gmtreg_dr_pt10");
h_eff_eta_after_gmt_pt10 = (TH1D*) h_eta_after_gmt_pt10->Clone("h_eff_eta_after_gmt_pt10");
h_eff_eta_after_xtra_dr_pt10 = (TH1D*) h_eta_after_xtra_dr_pt10->Clone("h_eff_eta_after_xtra_dr_pt10");

h_eff_eta_me1_after_alct = (TH1D*) h_eta_me1_after_alct->Clone("h_eff_eta_me1_after_alct");
h_eff_eta_me1_after_alct_okAlct = (TH1D*) h_eta_me1_after_alct_okAlct->Clone("h_eff_eta_me1_after_alct_okAlct");
h_eff_eta_me1_after_clct = (TH1D*) h_eta_me1_after_clct->Clone("h_eff_eta_me1_after_clct");
h_eff_eta_me1_after_clct_okClct = (TH1D*) h_eta_me1_after_clct_okClct->Clone("h_eff_eta_me1_after_clct_okClct");
h_eff_eta_me1_after_alctclct = (TH1D*) h_eta_me1_after_alctclct->Clone("h_eff_eta_me1_after_alctclct");
h_eff_eta_me1_after_alctclct_okAlct = (TH1D*) h_eta_me1_after_alctclct_okAlct->Clone("h_eff_eta_me1_after_alctclct_okAlct");
h_eff_eta_me1_after_alctclct_okClct = (TH1D*) h_eta_me1_after_alctclct_okClct->Clone("h_eff_eta_me1_after_alctclct_okClct");
h_eff_eta_me1_after_alctclct_okAlctClct = (TH1D*) h_eta_me1_after_alctclct_okAlctClct->Clone("h_eff_eta_me1_after_alctclct_okAlctClct");

h_eff_eta_me1_after_lct = (TH1D*) h_eta_me1_after_lct->Clone("h_eff_eta_me1_after_lct");
h_eff_eta_me1_after_lct_okAlct = (TH1D*) h_eta_me1_after_lct_okAlct->Clone("h_eff_eta_me1_after_lct_okAlct");
h_eff_eta_me1_after_lct_okAlctClct = (TH1D*) h_eta_me1_after_lct_okAlctClct->Clone("h_eff_eta_me1_after_lct_okAlctClct");
h_eff_eta_me1_after_lct_okClct = (TH1D*) h_eta_me1_after_lct_okClct->Clone("h_eff_eta_me1_after_lct_okClct");
h_eff_eta_me1_after_lct_okClctAlct = (TH1D*) h_eta_me1_after_lct_okClctAlct->Clone("h_eff_eta_me1_after_lct_okClctAlct");
h_eff_eta_me1_after_mplct_okAlctClct = (TH1D*) h_eta_me1_after_mplct_okAlctClct->Clone("h_eff_eta_me1_after_mplct_okAlctClct");
h_eff_eta_me1_after_mplct_okAlctClct_plus = (TH1D*) h_eta_me1_after_mplct_okAlctClct_plus->Clone("h_eff_eta_me1_after_mplct_okAlctClct_plus");
h_eff_eta_me1_after_tf_ok = (TH1D*) h_eta_me1_after_tf_ok->Clone("h_eff_eta_me1_after_tf_ok");
h_eff_eta_me1_after_tf_ok_pt10 = (TH1D*) h_eta_me1_after_tf_ok_pt10->Clone("h_eff_eta_me1_after_tf_ok_pt10");
h_eff_eta_me1_after_tf_ok_plus = (TH1D*) h_eta_me1_after_tf_ok_plus->Clone("h_eff_eta_me1_after_tf_ok_plus");
h_eff_eta_me1_after_tf_ok_plus_pt10 = (TH1D*) h_eta_me1_after_tf_ok_plus_pt10->Clone("h_eff_eta_me1_after_tf_ok_plus_pt10");


h_eta_initial->Sumw2();
h_eff_eta_after_mpc_st1->Sumw2();
h_eff_eta_after_tfcand_my_st1->Sumw2();
h_eff_eta_me1_after_alctclct_okAlctClct->Sumw2();
h_eff_eta_me1_after_lct_okClctAlct->Sumw2();

h_eff_eta_after_mpc->Divide(h_eff_eta_after_mpc,h_eta_initial);
h_eff_eta_after_mpc_st1->Divide(h_eff_eta_after_mpc_st1,h_eta_initial,1,1,"B");
h_eff_eta_after_mpc_st1_good->Divide(h_eff_eta_after_mpc_st1_good,h_eta_initial,1,1,"B");
h_eff_eta_after_mpc_ok->Divide(h_eff_eta_after_mpc_ok,h_eta_initial);
h_eff_eta_after_mpc_ok_plus->Divide(h_eff_eta_after_mpc_ok_plus,h_eta_initial);

h_eff_eta_after_tftrack->Divide(h_eff_eta_after_tftrack,h_eta_initial);
h_eff_eta_after_tfcand->Divide(h_eff_eta_after_tfcand,h_eta_initial);
h_eff_eta_after_tfcand_q1->Divide(h_eff_eta_after_tfcand_q1,h_eta_initial);
h_eff_eta_after_tfcand_q2->Divide(h_eff_eta_after_tfcand_q2,h_eta_initial);
h_eff_eta_after_tfcand_q3->Divide(h_eff_eta_after_tfcand_q3,h_eta_initial);
h_eff_eta_after_tfcand_ok->Divide(h_eff_eta_after_tfcand_ok,h_eta_initial);
h_eff_eta_after_tfcand_ok_plus->Divide(h_eff_eta_after_tfcand_ok_plus,h_eta_initial);
h_eff_eta_after_tfcand_ok_pt10->Divide(h_eff_eta_after_tfcand_ok_pt10,h_eta_initial);
h_eff_eta_after_tfcand_ok_plus_pt10->Divide(h_eff_eta_after_tfcand_ok_plus_pt10,h_eta_initial);
h_eff_eta_after_tfcand_all     ->Divide(h_eff_eta_after_tfcand_all,h_eta_initial);
h_eff_eta_after_tfcand_all_pt10->Divide(h_eff_eta_after_tfcand_all_pt10,h_eta_initial);

h_eff_eta_after_gmtreg    ->Divide(h_eff_eta_after_gmtreg,h_eta_initial);
h_eff_eta_after_gmtreg_all->Divide(h_eff_eta_after_gmtreg_all,h_eta_initial);
h_eff_eta_after_gmtreg_dr ->Divide(h_eff_eta_after_gmtreg_dr,h_eta_initial);
h_eff_eta_after_gmt       ->Divide(h_eff_eta_after_gmt,h_eta_initial);
h_eff_eta_after_gmt_all   ->Divide(h_eff_eta_after_gmt_all,h_eta_initial);
h_eff_eta_after_xtra      ->Divide(h_eff_eta_after_xtra,h_eta_initial);
h_eff_eta_after_xtra_all  ->Divide(h_eff_eta_after_xtra_all,h_eta_initial);
h_eff_eta_after_xtra_dr   ->Divide(h_eff_eta_after_xtra_dr,h_eta_initial);

h_eff_eta_after_tfcand_pt10    ->Divide(h_eff_eta_after_tfcand_pt10,h_eta_initial,1,1,"B");
h_eff_eta_after_tfcand_my_st1 ->Divide(h_eff_eta_after_tfcand_my_st1,h_eta_initial,1,1,"B");
h_eff_eta_after_tfcand_org_st1 ->Divide(h_eff_eta_after_tfcand_org_st1,h_eta_initial,1,1,"B");
h_eff_eta_after_tfcand_comm_st1 ->Divide(h_eff_eta_after_tfcand_comm_st1,h_eta_initial,1,1,"B");
h_eff_eta_after_tfcand_my_st1_pt10 ->Divide(h_eff_eta_after_tfcand_my_st1_pt10,h_eta_initial,1,1,"B");
h_eff_eta_after_gmtreg_pt10    ->Divide(h_eff_eta_after_gmtreg_pt10,h_eta_initial);
h_eff_eta_after_gmtreg_dr_pt10 ->Divide(h_eff_eta_after_gmtreg_dr_pt10,h_eta_initial);
h_eff_eta_after_gmt_pt10       ->Divide(h_eff_eta_after_gmt_pt10,h_eta_initial);
h_eff_eta_after_xtra_dr_pt10   ->Divide(h_eff_eta_after_xtra_dr_pt10,h_eta_initial);


h_eff_eta_me1_after_alct->Divide(h_eff_eta_me1_after_alct,h_eta_initial);
h_eff_eta_me1_after_alct_okAlct->Divide(h_eff_eta_me1_after_alct_okAlct,h_eta_initial);
h_eff_eta_me1_after_clct->Divide(h_eff_eta_me1_after_clct,h_eta_initial);
h_eff_eta_me1_after_clct_okClct->Divide(h_eff_eta_me1_after_clct_okClct,h_eta_initial);
h_eff_eta_me1_after_alctclct->Divide(h_eff_eta_me1_after_alctclct,h_eta_initial);
h_eff_eta_me1_after_alctclct_okAlct->Divide(h_eff_eta_me1_after_alctclct_okAlct,h_eta_initial);
h_eff_eta_me1_after_alctclct_okClct->Divide(h_eff_eta_me1_after_alctclct_okClct,h_eta_initial);
h_eff_eta_me1_after_alctclct_okAlctClct->Divide(h_eff_eta_me1_after_alctclct_okAlctClct,h_eta_initial);

h_eff_eta_me1_after_lct ->Divide(h_eff_eta_me1_after_lct,h_eta_initial);
h_eff_eta_me1_after_lct_okAlct ->Divide(h_eff_eta_me1_after_lct_okAlct,h_eta_initial);
h_eff_eta_me1_after_lct_okAlctClct->Divide(h_eff_eta_me1_after_lct_okAlctClct,h_eta_initial);
h_eff_eta_me1_after_lct_okClct ->Divide(h_eff_eta_me1_after_lct_okClct,h_eta_initial);
h_eff_eta_me1_after_lct_okClctAlct->Divide(h_eff_eta_me1_after_lct_okClctAlct,h_eta_initial);
h_eff_eta_me1_after_mplct_okAlctClct->Divide(h_eff_eta_me1_after_mplct_okAlctClct,h_eta_initial);
h_eff_eta_me1_after_mplct_okAlctClct_plus->Divide(h_eff_eta_me1_after_mplct_okAlctClct_plus,h_eta_initial);
h_eff_eta_me1_after_tf_ok      ->Divide(h_eff_eta_me1_after_tf_ok,h_eta_initial);
h_eff_eta_me1_after_tf_ok_pt10 ->Divide(h_eff_eta_me1_after_tf_ok_pt10,h_eta_initial);
h_eff_eta_me1_after_tf_ok_plus      ->Divide(h_eff_eta_me1_after_tf_ok_plus,h_eta_initial);
h_eff_eta_me1_after_tf_ok_plus_pt10 ->Divide(h_eff_eta_me1_after_tf_ok_plus_pt10,h_eta_initial);



//h_eff_eta_after_mpc->SetFillColor(7);
//h_eff_eta_after_tftrack->SetFillColor(8);
//
//h_eff_eta_after_tfcand     ->SetFillColor(1);
//h_eff_eta_after_tfcand_q1->SetFillColor(2);
//h_eff_eta_after_tfcand_q2->SetFillColor(3);
//h_eff_eta_after_tfcand_q3->SetFillColor(4);
//
//h_eff_eta_after_tfcand_all     ->SetFillColor(5);


TCanvas* c2 = new TCanvas("h_eff_eta","h_eff_eta",900,900 ) ;
c2->Divide(2,2);
c2->cd(1);
h_eff_eta_after_mpc->GetXaxis()->SetRangeUser(0.9,2.5);
h_eff_eta_after_mpc->Draw("hist");
//h_eff_eta_after_mpc_ok->Draw("same hist");
h_eff_eta_after_mpc_ok_plus->Draw("same hist");
//h_eff_eta_after_tftrack->Draw("same hist");
h_eff_eta_after_tfcand->Draw("same hist");
//h_eff_eta_after_tfcand_ok->Draw("same hist");
h_eff_eta_after_tfcand_ok_plus->Draw("same hist");
h_eff_eta_after_tfcand_ok_plus_pt10->Draw("same hist");

c2->cd(2);
h_eff_eta_after_tfcand->GetXaxis()->SetRangeUser(0.9,2.5);
h_eff_eta_after_tfcand     ->Draw("hist");
h_eff_eta_after_tfcand_q1->Draw("same hist");
h_eff_eta_after_tfcand_q2->Draw("same hist");
h_eff_eta_after_tfcand_q3->Draw("same hist");
c2->cd(3);
h_eff_eta_after_tfcand_all     ->GetXaxis()->SetRangeUser(0.9,2.5);
h_eff_eta_after_tfcand_all     ->Draw("hist");
h_eff_eta_after_tfcand->Draw("same hist");
c2->cd(2);
Print(c2,"h_eff_eta" + ext);


h_eff_eta_after_mpc->SetLineColor(kBlack);
h_eff_eta_after_mpc_st1->SetLineColor(kBlack+2);
h_eff_eta_after_mpc_st1_good->SetLineColor(kBlack+4);
h_eff_eta_after_tftrack->SetLineColor(kViolet-2);
h_eff_eta_after_tfcand->SetLineColor(kBlue);
h_eff_eta_after_gmtreg->SetLineColor(kMagenta-2);
h_eff_eta_after_gmtreg_dr->SetLineColor(kOrange-3);
h_eff_eta_after_gmt->SetLineColor(kGreen+1);
h_eff_eta_after_xtra->SetLineColor(kPink-4);
h_eff_eta_after_xtra_dr->SetLineColor(kRed);

h_eff_eta_after_tfcand_all     ->SetLineColor(kBlue);
h_eff_eta_after_gmtreg_all->SetLineColor(kMagenta-2);
h_eff_eta_after_gmt_all->SetLineColor(kGreen+1);
h_eff_eta_after_xtra_all->SetLineColor(kPink-4);


h_eff_eta_after_tfcand_pt10    ->SetLineColor(kBlue);
h_eff_eta_after_tfcand_my_st1 ->SetLineColor(kBlue);
h_eff_eta_after_tfcand_org_st1 ->SetLineColor(kBlue);
h_eff_eta_after_tfcand_comm_st1 ->SetLineColor(kBlue);
h_eff_eta_after_tfcand_my_st1_pt10 ->SetLineColor(30);
h_eff_eta_after_gmtreg_pt10    ->SetLineColor(kMagenta-2);
h_eff_eta_after_gmtreg_dr_pt10 ->SetLineColor(kOrange-3);
h_eff_eta_after_gmt_pt10       ->SetLineColor(kGreen+1);
h_eff_eta_after_xtra_dr_pt10   ->SetLineColor(kRed);

h_eff_eta_after_mpc_st1_good->SetLineStyle(7);

h_eff_eta_after_tfcand_pt10    ->SetLineStyle(7);
h_eff_eta_after_tfcand_my_st1  ->SetLineStyle(7);
h_eff_eta_after_tfcand_org_st1 ->SetLineStyle(3);
h_eff_eta_after_tfcand_comm_st1->SetLineStyle(2);
h_eff_eta_after_tfcand_my_st1_pt10->SetLineStyle(7);
h_eff_eta_after_gmtreg_pt10    ->SetLineStyle(7);
h_eff_eta_after_gmtreg_dr_pt10 ->SetLineStyle(7);
h_eff_eta_after_gmt_pt10       ->SetLineStyle(7);
h_eff_eta_after_xtra_dr_pt10   ->SetLineStyle(7);

h_eff_eta_after_mpc->SetTitle("L1 CSC trigger efficiency dependence on #eta");
h_eff_eta_after_mpc->GetXaxis()->SetRangeUser(0.85,2.5);
h_eff_eta_after_mpc->GetXaxis()->SetTitle("#eta");
h_eff_eta_after_mpc->SetMinimum(0);



/*
TCanvas* c22 = new TCanvas("h_eff_eta_steps","h_eff_eta_steps",1200,900 ) ;
c22->Divide(1,2);
c22->cd(1);
h_eff_eta_after_mpc->Draw("hist");
//h_eff_eta_after_tftrack->Draw("same");
h_eff_eta_after_tfcand->Draw("same");
//h_eff_eta_after_gmtreg->Draw("same");
h_eff_eta_after_gmt->Draw("same");
//h_eff_eta_after_xtra->Draw("same");
h_eff_eta_after_xtra_dr->Draw("same");
leg = new TLegend(0.7815507,0.1702982,0.9846086,0.5740635,NULL,"brNDC");
//leg->SetTextFont(12);
leg->SetBorderSize(0);
leg->SetTextFont(12);
leg->SetLineColor(1);
leg->SetLineStyle(1);
leg->SetLineWidth(1);
leg->SetFillColor(19);
leg->SetFillStyle(0);
leg->AddEntry(h_eff_eta_after_mpc,"h_eff_eta_after_mpc","pl");
//leg->AddEntry(h_eff_eta_after_tftrack,"h_eff_eta_after_tftrack","pl");
leg->AddEntry(h_eff_eta_after_tfcand,"h_eff_eta_after_tfcand","pl");
//leg->AddEntry(h_eff_eta_after_gmtreg,"h_eff_eta_after_gmtreg","pl");
leg->AddEntry(h_eff_eta_after_gmt,"h_eff_eta_after_gmt","pl");
//leg->AddEntry(h_eff_eta_after_xtra,"h_eff_eta_after_xtra","pl");
leg->AddEntry(h_eff_eta_after_xtra_dr,"h_eff_eta_after_xtra_dr","pl");
leg->Draw();
c22->cd(2);
h_eff_eta_after_mpc->GetXaxis()->SetRangeUser(0.9,2.5);
h_eff_eta_after_mpc->Draw("hist");
//h_eff_eta_after_tftrack->Draw("same");
h_eff_eta_after_tfcand_all->Draw("same");
//h_eff_eta_after_gmtreg_all->Draw("same");
h_eff_eta_after_gmt_all->Draw("same");
//h_eff_eta_after_xtra_all->Draw("same");
h_eff_eta_after_xtra_dr->Draw("same");
leg = new TLegend(0.7815507,0.1702982,0.9846086,0.5740635,NULL,"brNDC");
//leg->SetTextFont(12);
leg->SetBorderSize(0);
leg->SetTextFont(12);
leg->SetLineColor(1);
leg->SetLineStyle(1);
leg->SetLineWidth(1);
leg->SetFillColor(19);
leg->SetFillStyle(0);
leg->AddEntry(h_eff_eta_after_mpc,"h_eff_eta_after_mpc","pl");
leg->AddEntry(h_eff_eta_after_tfcand_all,"h_eff_eta_after_tfcand_all","pl");
//leg->AddEntry(h_eff_eta_after_gmtreg_all,"h_eff_eta_after_gmtreg_all","pl");
leg->AddEntry(h_eff_eta_after_gmt_all,"h_eff_eta_after_gmt_all","pl");
//leg->AddEntry(h_eff_eta_after_xtra_all,"h_eff_eta_after_xtra_all","pl");
leg->AddEntry(h_eff_eta_after_xtra_dr,"h_eff_eta_after_xtra_dr","pl");
leg->Draw("hist");
Print(c22,"h_eff_eta_step.eps");
*/

/*  // commented on 10/21/09
TCanvas* c22x = new TCanvas("h_eff_eta_steps_full","h_eff_eta_steps_full",1000,600 ) ;

h_eff_eta_after_mpc->Draw("hist");
////h_eff_eta_after_tftrack->Draw("same");
h_eff_eta_after_tfcand->Draw("same");
h_eff_eta_after_gmtreg->Draw("same");
//h_eff_eta_after_gmtreg_dr->Draw("same");
h_eff_eta_after_gmt->Draw("same");
////h_eff_eta_after_xtra->Draw("same");
//h_eff_eta_after_xtra_dr->Draw("same");

//h_eff_eta_after_tfcand_pt10    ->Draw("same");
//h_eff_eta_after_gmtreg_pt10    ->Draw("same");
////h_eff_eta_after_gmtreg_dr_pt10 ->Draw("same");
//h_eff_eta_after_gmt_pt10       ->Draw("same");
//h_eff_eta_after_xtra_dr_pt10   ->Draw("same");

TLegend *leg = new TLegend(0.2518248,0.263986,0.830292,0.5332168,NULL,"brNDC");
//leg->SetTextFont(12);
leg->SetBorderSize(0);
//leg->SetTextFont(12);
leg->SetFillStyle(0);

leg->SetHeader("Efficiency after");
leg->AddEntry(h_eff_eta_after_mpc,"match to MPC","pl");
//leg->AddEntry(h_eff_eta_after_tftrack,"h_eff_eta_after_tftrack","pl");
leg->AddEntry(h_eff_eta_after_tfcand,"match to MPC & TF track","pl");
leg->AddEntry(h_eff_eta_after_gmtreg,"match to MPC & TF trk & CSC GMT trk","pl");
//leg->AddEntry(h_eff_eta_after_gmtreg_dr,"#Delta R match to CSC GMT trk","pl");
leg->AddEntry(h_eff_eta_after_gmt,"match to MPC & TF trk & GMT trk","pl");
//leg->AddEntry(h_eff_eta_after_xtra,"h_eff_eta_after_xtra","pl");
//leg->AddEntry(h_eff_eta_after_xtra_dr,"#Delta R match to GMT","pl");
//leg->AddEntry(h_eff_eta_after_xtra_dr,"(dashed lines correspond to p_{T}>10)","");

leg->Draw();
Print(c22x,"h_eff_eta_steps_full.eps");
Print(c22x,"h_eff_eta_steps_full" + ext);
*/

if (h_eff_eta_steps_full10) {

TCanvas* c22x10 = new TCanvas("h_eff_eta_steps_full10","h_eff_eta_steps_full10",1000,600 ) ;

h_eff_eta_after_mpc->Draw("hist");
h_eff_eta_after_tfcand_pt10    ->Draw("same hist");
h_eff_eta_after_gmtreg_pt10    ->Draw("same hist");
h_eff_eta_after_gmt_pt10       ->Draw("same hist");
//h_eff_eta_after_xtra_dr_pt10   ->Draw("same hist");

TLegend *leg = new TLegend(0.2518248,0.263986,0.830292,0.5332168,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);

leg->SetHeader("With additional TF cut p_{T}>10 Efficiency after");
leg->AddEntry(h_eff_eta_after_mpc,"match to MPC","pl");
leg->AddEntry(h_eff_eta_after_tfcand_pt10,"match to MPC & TF track","pl");
leg->AddEntry(h_eff_eta_after_gmtreg_pt10,"match to MPC & TF trk & CSC GMT trk","pl");
leg->AddEntry(h_eff_eta_after_gmt_pt10,"match to MPC & TF trk & GMT trk","pl");
//leg->AddEntry(h_eff_eta_after_xtra_dr,"(dashed lines correspond to p_{T}>10)","");

leg->Draw();
Print(c22x10,"h_eff_eta_steps_full10.eps");
Print(c22x10,"h_eff_eta_steps_full10" + ext);
Print(c22x10,"h_eff_eta_steps_full10.pdf");

}


/*

TCanvas* c22x10tf = new TCanvas("h_eff_eta_steps_full10tf","h_eff_eta_steps_full10tf",1000,600 ) ;

h_eff_eta_after_tfcand_pt10    ->SetLineStyle(1);

h_eff_eta_after_mpc->Draw("hist");
h_eff_eta_after_mpc_st1->Draw("same hist");
//h_eff_eta_after_mpc_st1_good->Draw("same hist");
h_eff_eta_after_tfcand_org_st1 ->Draw("same hist");
//h_eff_eta_after_tfcand_comm_st1 ->Draw("same hist");
h_eff_eta_after_tfcand_pt10    ->Draw("same hist");
h_eff_eta_after_tfcand_my_st1 ->Draw("same hist");
h_eff_eta_after_tfcand_my_st1_pt10 ->Draw("same hist");
//h_eff_eta_after_xtra_dr_pt10   ->Draw("same hist");

TLegend *leg = new TLegend(0.2518248,0.263986,0.830292,0.5332168,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);

leg->SetHeader("Efficiency after");
leg->AddEntry(h_eff_eta_after_mpc,"match to MPC","pl");
leg->AddEntry(h_eff_eta_after_mpc_st1,"match to MPC (at least one in ME1)","pl");
//leg->AddEntry(h_eff_eta_after_mpc_st1_good,"match to MPC (at least one good in ME1)","pl");
leg->AddEntry(NUL,"match to MPC & TF track:","");
leg->AddEntry(h_eff_eta_after_tfcand_pt10," (a) with additional TF cut p_{T}>10","pl");
leg->AddEntry(h_eff_eta_after_tfcand_org_st1," (b) at least 1 original TF stub in ME1","pl");
leg->AddEntry(h_eff_eta_after_tfcand_my_st1," (c) at least 1 matched TF stub in ME1","pl");
leg->AddEntry(h_eff_eta_after_tfcand_my_st1_pt10," (d) at least 1 matched TF stub in ME1 and TF p_{T}>10","pl");
//leg->AddEntry(h_eff_eta_after_tfcand_comm_st1," at least 1 my=original stub in St1","pl");
//leg->AddEntry(h_eff_eta_after_xtra_dr,"(dashed lines correspond to p_{T}>10)","");

leg->Draw();
Print(c22x10tf,"h_eff_eta_steps_full10_tf.eps");
Print(c22x10tf,"h_eff_eta_steps_full10_tf" + ext);

*/

/*  // commented on 10/21/09
TCanvas* c22x10tfs = new TCanvas("h_eff_eta_steps_full10tfs","h_eff_eta_steps_full10tfs",1000,600 ) ;

h_eff_eta_after_tfcand_pt10    ->SetLineStyle(1);

h_eff_eta_after_mpc->Draw("hist");
h_eff_eta_after_mpc_st1->Draw("same hist");
//h_eff_eta_after_mpc_st1_good->Draw("same hist");
//h_eff_eta_after_tfcand_org_st1 ->Draw("same hist");
//h_eff_eta_after_tfcand_comm_st1 ->Draw("same hist");
h_eff_eta_after_tfcand_pt10    ->Draw("same hist");
h_eff_eta_after_tfcand_my_st1 ->Draw("same hist");
//h_eff_eta_after_tfcand_my_st1_pt10 ->Draw("same hist");
//h_eff_eta_after_xtra_dr_pt10   ->Draw("same hist");

TLegend *leg = new TLegend(0.2518248,0.263986,0.830292,0.5332168,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);

leg->SetHeader("Efficiency after");
leg->AddEntry(NUL,"match to MPC:","");
leg->AddEntry(h_eff_eta_after_mpc," at least one","pl");
leg->AddEntry(h_eff_eta_after_mpc_st1," at least one in ME1","pl");
//leg->AddEntry(h_eff_eta_after_mpc_st1_good,"match to MPC (at least one good in ME1)","pl");
leg->AddEntry(NUL,"match to MPC & TF track:","");
leg->AddEntry(h_eff_eta_after_tfcand_pt10," with additional TF cut p_{T}>10","pl");
//leg->AddEntry(h_eff_eta_after_tfcand_org_st1," (b) at least 1 original TF stub in ME1","pl");
leg->AddEntry(h_eff_eta_after_tfcand_my_st1," at least 1 matched TF stub in ME1","pl");
//leg->AddEntry(h_eff_eta_after_tfcand_my_st1_pt10," (d) at least 1 matched TF stub in ME1 and TF p_{T}>10","pl");
//leg->AddEntry(h_eff_eta_after_tfcand_comm_st1," at least 1 my=original stub in St1","pl");
//leg->AddEntry(h_eff_eta_after_xtra_dr,"(dashed lines correspond to p_{T}>10)","");

leg->Draw();
Print(c22x10tfs,"h_eff_eta_steps_full10_tfs.eps");
Print(c22x10tfs,"h_eff_eta_steps_full10_tfs" + ext);
Print(c22x10tfs,"h_eff_eta_steps_full10_tfs.pdf");
*/



if (do_h_eff_eta_steps_xchk1) {

TCanvas* c222xx1 = new TCanvas("h_eff_eta_steps_xchk1","h_eff_eta_steps_xchk1",1000,600 ) ;

h_eff_eta_after_gmtreg->SetTitle("CSC GMT efficiency dependence on #eta (cross-check)");
h_eff_eta_after_gmtreg->GetXaxis()->SetRangeUser(0.9,2.5);
h_eff_eta_after_gmtreg->GetXaxis()->SetTitle("#eta");
h_eff_eta_after_gmtreg->SetMinimum(0);
h_eff_eta_after_gmtreg->Draw("hist");
h_eff_eta_after_gmtreg_dr->Draw("same  hist");

//h_eff_eta_after_gmtreg_pt10    ->Draw("same hist");
//h_eff_eta_after_gmtreg_dr_pt10 ->Draw("same hist");

TLegend *leg = new TLegend(0.2518248,0.263986,0.830292,0.5332168,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);

leg->SetHeader("Efficiency after");
leg->AddEntry(h_eff_eta_after_gmtreg,"match to MPC & TF trk & CSC GMT trk","pl");
leg->AddEntry(h_eff_eta_after_gmtreg_dr,"#Delta R match to CSC GMT trk","pl");
//leg->AddEntry(h_eff_eta_after_xtra_dr,"(dashed lines correspond to p_{T}>10)","");

leg->Draw();
Print(c222xx1,"h_eff_eta_steps_xchk1.eps");
Print(c222xx1,"h_eff_eta_steps_xchk1" + ext);

}



/*
TCanvas* c222xx2 = new TCanvas("h_eff_eta_steps_xchk2","h_eff_eta_steps_xchk2",1000,600 ) ;

h_eff_eta_after_gmt->SetTitle("GMT efficiency dependence on #eta (cross-check)");
h_eff_eta_after_gmt->GetXaxis()->SetRangeUser(0.9,2.5);
h_eff_eta_after_gmt->GetXaxis()->SetTitle("#eta");
h_eff_eta_after_gmt->SetMinimum(0);
h_eff_eta_after_gmt->SetMaximum(1.05);

h_eff_eta_after_gmt->Draw();
h_eff_eta_after_xtra_dr->Draw("same hist");

//h_eff_eta_after_gmt_pt10       ->Draw("same hist");
//h_eff_eta_after_xtra_dr_pt10   ->Draw("same hist");

TLegend *leg = new TLegend(0.2518248,0.263986,0.830292,0.5332168,NULL,"brNDC");
//leg->SetTextFont(12);
leg->SetBorderSize(0);
//leg->SetTextFont(12);
leg->SetFillStyle(0);

leg->SetHeader("Efficiency after");
leg->AddEntry(h_eff_eta_after_gmt,"match to MPC & TF trk & GMT trk","pl");
leg->AddEntry(h_eff_eta_after_xtra_dr,"#Delta R match to GMT","pl");
//leg->AddEntry(h_eff_eta_after_xtra_dr,"(dashed lines correspond to p_{T}>10)","");

leg->Draw();
Print(c222xx2,"h_eff_eta_steps_xchk2.eps");
Print(c222xx2,"h_eff_eta_steps_xchk2" + ext);
return;

*/



h_eff_eta_me1_after_lct ->SetLineColor(kRed);
h_eff_eta_me1_after_lct_okAlct ->SetLineColor(kRed-6);
h_eff_eta_me1_after_lct_okAlctClct->SetLineColor(kMagenta-6);
h_eff_eta_me1_after_lct_okClct ->SetLineColor(kRed-6);
h_eff_eta_me1_after_lct_okClctAlct->SetLineColor(kMagenta-6);
h_eff_eta_me1_after_mplct_okAlctClct->SetLineColor(kBlack);
h_eff_eta_me1_after_mplct_okAlctClct_plus->SetLineColor(kBlack);
h_eff_eta_me1_after_tf_ok   ->SetLineColor(kBlue);   
h_eff_eta_me1_after_tf_ok_pt10 ->SetLineColor(kBlue+2);
h_eff_eta_me1_after_tf_ok_plus   ->SetLineColor(kBlue);   
h_eff_eta_me1_after_tf_ok_plus_pt10 ->SetLineColor(kBlue+2);

h_eff_eta_after_tfcand_all_pt10->SetLineColor(kCyan+2);

h_eff_eta_me1_after_lct->SetLineWidth(2);
h_eff_eta_me1_after_lct_okAlct->SetLineWidth(2);
h_eff_eta_me1_after_lct_okAlctClct->SetLineWidth(2);
h_eff_eta_me1_after_lct_okClct->SetLineWidth(2);
h_eff_eta_me1_after_lct_okClctAlct->SetLineWidth(2);

//h_eff_eta_me1_after_lct->SetLineStyle(7);
h_eff_eta_me1_after_lct_okAlct->SetLineStyle(3);
h_eff_eta_me1_after_lct_okAlctClct->SetLineStyle(7);
h_eff_eta_me1_after_lct_okClct->SetLineStyle(3);
h_eff_eta_me1_after_lct_okClctAlct->SetLineStyle(7);
h_eff_eta_me1_after_mplct_okAlctClct_plus->SetLineStyle(7);

h_eff_eta_me1_after_tf_ok_pt10->SetLineStyle(7);
h_eff_eta_me1_after_tf_ok_plus_pt10->SetLineStyle(7);
h_eff_eta_after_tfcand_all_pt10->SetLineStyle(3);

h_eff_eta_me1_after_lct->SetTitle("efficiency dependence on #eta: ME1 studies");
h_eff_eta_me1_after_lct->GetXaxis()->SetRangeUser(0.86,2.5);
h_eff_eta_me1_after_lct->GetXaxis()->SetTitle("#eta");
h_eff_eta_me1_after_lct->SetMinimum(0);
h_eff_eta_me1_after_lct->GetYaxis()->SetRangeUser(0.,1.05);



TCanvas* cme1n1 = new TCanvas("h_eff_eta_me1_after_lct","h_eff_eta_me1_after_lct",1000,600 ) ;


h_eff_eta_me1_after_lct->Draw("hist");
//h_eff_eta_me1_after_lct_okAlct->Draw("same hist");
//h_eff_eta_me1_after_lct_okAlctClct->Draw("same hist");
h_eff_eta_me1_after_lct_okClct->Draw("same hist");
h_eff_eta_me1_after_lct_okClctAlct->Draw("same hist");
h_eff_eta_me1_after_mplct_okAlctClct->Draw("same hist");
h_eff_eta_me1_after_mplct_okAlctClct_plus->Draw("same hist");
h_eff_eta_me1_after_tf_ok      ->Draw("same hist");
h_eff_eta_me1_after_tf_ok_pt10 ->Draw("same hist");
h_eff_eta_after_tfcand_all_pt10->Draw("same hist");

TLegend *leg = new TLegend(0.347,0.222,0.926,0.535,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);

leg->SetHeader("Efficiency for muon with p_{T}>10");
//leg->AddEntry(NUL,"match to LCT:","");
leg->AddEntry(h_eff_eta_me1_after_lct,"any LCT in ME1 chamber crossed by #mu","pl");
//leg->AddEntry(h_eff_eta_me1_after_lct_okAlct,"  + track's ALCT","pl");
//leg->AddEntry(h_eff_eta_me1_after_lct_okAlctClct,"  + track's CLCT","pl");
leg->AddEntry(h_eff_eta_me1_after_lct_okClct,"  + correct CLCT picked","pl");
leg->AddEntry(h_eff_eta_me1_after_lct_okClctAlct,"  + correct ALCT mached to CLCT","pl");
leg->AddEntry(h_eff_eta_me1_after_mplct_okAlctClct,"  + pass MPC selection","pl");
leg->AddEntry(h_eff_eta_me1_after_mplct_okAlctClct,"  + has two MPC matched","pl");
leg->AddEntry(h_eff_eta_me1_after_tf_ok,"  + stub used in any TF track","pl");
leg->AddEntry(h_eff_eta_me1_after_tf_ok_pt10,"  + stub used in TF track with p_{T}>10 GeV","pl");
leg->AddEntry(h_eff_eta_after_tfcand_all_pt10,"there is TF track p_{T}>10 in #Delta R<0.3","pl");
//leg->AddEntry(," ","pl");
leg->Draw();
Print(cme1n1,"h_eff_eta_me1_steps_all.eps");
Print(cme1n1,"h_eff_eta_me1_steps_all" + ext);
Print(cme1n1,"h_eff_eta_me1_steps_all.pdf");




TCanvas* cme1n2 = new TCanvas("h_eff_eta_me1_after_lct2","h_eff_eta_me1_after_lct2",800,600 ) ;
h_eff_eta_me1_after_mplct_okAlctClct->GetXaxis()->SetRangeUser(0.86,2.5);
h_eff_eta_me1_after_mplct_okAlctClct->GetXaxis()->SetTitle("#eta");
h_eff_eta_me1_after_mplct_okAlctClct->SetMinimum(0);
h_eff_eta_me1_after_mplct_okAlctClct->GetYaxis()->SetRangeUser(0.,1.05);
h_eff_eta_me1_after_mplct_okAlctClct->SetTitle("Efficiency in ME1 for #mu with p_{T}>10");
h_eff_eta_me1_after_mplct_okAlctClct->GetYaxis()->SetTitle("Eff.");
h_eff_eta_me1_after_mplct_okAlctClct->GetXaxis()->SetTitleSize(0.07);
h_eff_eta_me1_after_mplct_okAlctClct->GetXaxis()->SetTitleOffset(0.7);
h_eff_eta_me1_after_mplct_okAlctClct->GetYaxis()->SetLabelOffset(0.015);
gStyle->SetTitleW(1);
gStyle->SetTitleH(0.08);
gStyle->SetTitleStyle(0);
h_eff_eta_me1_after_mplct_okAlctClct->Draw("hist");
h_eff_eta_me1_after_mplct_okAlctClct_plus->Draw("same hist");
h_eff_eta_me1_after_tf_ok      ->Draw("same hist");
h_eff_eta_me1_after_tf_ok_pt10 ->Draw("same hist");


TLegend *leg = new TLegend(0.34,0.16,0.87,0.39,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->SetHeader("SimMuon has matched");
leg->AddEntry(h_eff_eta_me1_after_mplct_okAlctClct,"MPC LCT stub in ME1","pl");
leg->AddEntry(h_eff_eta_me1_after_mplct_okAlctClct_plus," + one more MPC LCT stub in ME234 + ","pl");
leg->AddEntry(h_eff_eta_me1_after_tf_ok," + this stub was used by a TF track","pl");
leg->AddEntry(h_eff_eta_me1_after_tf_ok_pt10,"  + this TF track has p_{T}(TF)>10 GeV","pl");
leg->Draw();
Print(cme1n2,"h_eff_eta_me1_tf" + ext);



TCanvas* cme1n2_plus = new TCanvas("h_eff_eta_me1_after_tfplus","h_eff_eta_me1_after_tfplus",800,600 ) ;
h_eff_eta_me1_after_mplct_okAlctClct_plus->GetXaxis()->SetRangeUser(0.86,2.5);
h_eff_eta_me1_after_mplct_okAlctClct_plus->GetXaxis()->SetTitle("#eta");
h_eff_eta_me1_after_mplct_okAlctClct_plus->SetMinimum(0);
h_eff_eta_me1_after_mplct_okAlctClct_plus->GetYaxis()->SetRangeUser(0.,1.05);
h_eff_eta_me1_after_mplct_okAlctClct_plus->SetTitle("Efficiency in ME1 for #mu with p_{T}>10");
h_eff_eta_me1_after_mplct_okAlctClct_plus->GetYaxis()->SetTitle("Eff.");
h_eff_eta_me1_after_mplct_okAlctClct_plus->GetXaxis()->SetTitleSize(0.07);
h_eff_eta_me1_after_mplct_okAlctClct_plus->GetXaxis()->SetTitleOffset(0.7);
h_eff_eta_me1_after_mplct_okAlctClct_plus->GetYaxis()->SetLabelOffset(0.015);
gStyle->SetTitleW(1);
gStyle->SetTitleH(0.08);
gStyle->SetTitleStyle(0);
h_eff_eta_me1_after_mplct_okAlctClct_plus->Draw("hist");
h_eff_eta_me1_after_tf_ok_plus      ->Draw("same hist");
h_eff_eta_me1_after_tf_ok_plus_pt10 ->Draw("same hist");
h_eff_eta_me1_after_lct_okClctAlct->Draw("same hist");

leg = new TLegend(0.34,0.16,0.87,0.39,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);
leg->SetHeader("SimMuon has matched");
leg->AddEntry(h_eff_eta_me1_after_mplct_okAlctClct_plus,"MPC LCT stubs in ME1 and some other station","pl");
leg->AddEntry(h_eff_eta_me1_after_tf_ok_plus," + these stubs were used by a TF track","pl");
leg->AddEntry(h_eff_eta_me1_after_tf_ok_plus_pt10,"  + this TF track has p_{T}(TF)>10 GeV","pl");
leg->Draw();
Print(cme1n2_plus,"h_eff_eta_me1_tfplus" + ext);







h_eff_eta_me1_after_alct->SetLineColor(kBlue+1);
h_eff_eta_me1_after_alct_okAlct->SetLineColor(kBlue+1);
h_eff_eta_me1_after_clct->SetLineColor(kGreen+1);
h_eff_eta_me1_after_clct_okClct->SetLineColor(kGreen+1);
h_eff_eta_me1_after_alctclct->SetLineColor(kYellow+2);
h_eff_eta_me1_after_alctclct_okAlct->SetLineColor(kMagenta+2);
h_eff_eta_me1_after_alctclct_okClct->SetLineColor(kMagenta+3);
h_eff_eta_me1_after_alctclct_okAlctClct->SetLineColor(kYellow+2);

/*
h_eff_eta_me1_after_alct
h_eff_eta_me1_after_alct_okAlct
h_eff_eta_me1_after_clct
h_eff_eta_me1_after_clct_okClct
h_eff_eta_me1_after_alctclct
h_eff_eta_me1_after_alctclct_okAlct
h_eff_eta_me1_after_alctclct_okClct
h_eff_eta_me1_after_alctclct_okAlctClct
*/

h_eff_eta_me1_after_alct->SetLineStyle(2);
h_eff_eta_me1_after_clct->SetLineStyle(2);
h_eff_eta_me1_after_alctclct->SetLineStyle(2);

h_eff_eta_me1_after_alct->SetLineWidth(2);
h_eff_eta_me1_after_alct_okAlct->SetLineWidth(2);
h_eff_eta_me1_after_clct->SetLineWidth(2);
h_eff_eta_me1_after_clct_okClct->SetLineWidth(2);
h_eff_eta_me1_after_alctclct->SetLineWidth(2);
h_eff_eta_me1_after_alctclct_okAlct->SetLineWidth(2);
h_eff_eta_me1_after_alctclct_okClct->SetLineWidth(2);
h_eff_eta_me1_after_alctclct_okAlctClct->SetLineWidth(2);


h_eff_eta_me1_after_lct->SetLineStyle(2);
h_eff_eta_me1_after_lct_okClctAlct->SetLineStyle(1);
h_eff_eta_me1_after_lct_okClctAlct->SetLineColor(kRed);









TCanvas* cme1na = new TCanvas("h_eff_eta_me1_after_alct","h_eff_eta_me1_after_alct",1000,600 ) ;

//h_eff_eta_me1_after_lct->GetYaxis()->SetRangeUser(0.,1.05);
//h_eff_eta_me1_after_lct->GetYaxis()->SetRangeUser(0.5,1.05);
//h_eff_eta_me1_after_lct->GetYaxis()->SetRangeUser(0.8,1.02);
h_eff_eta_me1_after_lct->GetYaxis()->SetRangeUser(yrange[0],yrange[1]);
h_eff_eta_me1_after_lct->Draw("hist");
//h_eff_eta_me1_after_lct_okClctAlct->Draw("same hist");

h_eff_eta_me1_after_alct->Draw("same hist");
h_eff_eta_me1_after_alct_okAlct->Draw("same hist");
h_eff_eta_me1_after_clct->Draw("same hist");
h_eff_eta_me1_after_clct_okClct->Draw("same hist");
h_eff_eta_me1_after_alctclct->Draw("same hist");
//h_eff_eta_me1_after_alctclct_okAlct->Draw("same hist");
//h_eff_eta_me1_after_alctclct_okClct->Draw("same hist");
h_eff_eta_me1_after_alctclct_okAlctClct->Draw("same hist");

h_eff_eta_me1_after_lct->Draw("same hist");
h_eff_eta_me1_after_lct_okClctAlct->Draw("same hist");

h_eff_eta_me1_after_lct_okClctAlct->Fit("pol0","R0","",1.63,2.38);
eff11 = (h_eff_eta_me1_after_lct_okClctAlct->GetFunction("pol0"))->GetParameter(0);
cout<<eff11<<endl;
h_eff_eta_me1_after_lct_okClctAlct->Fit("pol0","R0","",1.63,2.05);
eff1b = (h_eff_eta_me1_after_lct_okClctAlct->GetFunction("pol0"))->GetParameter(0);
h_eff_eta_me1_after_lct_okClctAlct->Fit("pol0","R0","",2.05,2.38);
eff1a = (h_eff_eta_me1_after_lct_okClctAlct->GetFunction("pol0"))->GetParameter(0);

cout<<eff11<<"  "<<eff1b<<"  "<<eff1a<<endl;




TLegend *leg = new TLegend(0.347,0.222,0.926,0.535,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);

leg->SetNColumns(2);
leg->SetHeader("Efficiency for #mu with p_{T}>10 crossing a ME1 chamber with");

leg->AddEntry(h_eff_eta_me1_after_alct,"any ALCT","pl");
leg->AddEntry(h_eff_eta_me1_after_alct_okAlct,"correct ALCT","pl");
leg->AddEntry(h_eff_eta_me1_after_clct,"any CLCT","pl");
leg->AddEntry(h_eff_eta_me1_after_clct_okClct,"correct CLCT","pl");

leg->AddEntry(h_eff_eta_me1_after_alctclct,"any ALCT and CLCT","pl");
//leg->AddEntry(h_eff_eta_me1_after_alctclct_okAlct,"correct ALCT and any CLCT","pl");
//leg->AddEntry(h_eff_eta_me1_after_alctclct_okClct,"any ALCT and correct CLCT","pl");
leg->AddEntry(h_eff_eta_me1_after_alctclct_okAlctClct,"correct ALCT and CLCT","pl");

leg->AddEntry(h_eff_eta_me1_after_lct,"any LCT","pl");
leg->AddEntry(h_eff_eta_me1_after_lct_okClctAlct,"correct LCT","pl");

leg->Draw();
Print(cme1na,"h_eff_eta_me1_steps_stubs.eps");
Print(cme1na,"h_eff_eta_me1_steps_stubs" + ext);
Print(cme1na,"h_eff_eta_me1_steps_stubs.pdf");




char ghn[111];
sprintf(ghn,"h_eff_eta_me1_after_lct_okClctAlct_%s",dname);
gh = (TH1F*)h_eff_eta_me1_after_lct_okClctAlct->Clone(ghn);
gh->GetYaxis()->SetRangeUser(0.8,1.02);

gh->SetTitle("LCT finding efficiency in ME1 for #mu with p_{T}>10");
gh->GetXaxis()->SetRangeUser(0.8,2.5);
gh->GetYaxis()->SetRangeUser(0.,1.05);
gh->GetXaxis()->SetTitle("#eta");
gh->GetYaxis()->SetTitle("Eff.");
gh->GetXaxis()->SetTitleSize(0.07);
gh->GetXaxis()->SetTitleOffset(0.7);
gh->GetYaxis()->SetLabelOffset(0.015);



/*

h_eff_eta_me1_after_clct->SetTitle("CLCT efficiency dependence on #eta: ME1 studies");
h_eff_eta_me1_after_clct->GetXaxis()->SetRangeUser(0.86,2.5);
h_eff_eta_me1_after_clct->GetXaxis()->SetTitle("#eta");
h_eff_eta_me1_after_clct->SetMinimum(0);
h_eff_eta_me1_after_clct->GetYaxis()->SetRangeUser(0.,1.05);

h_eff_eta_me1_after_clct_okClct->SetTitle("Correct CLCT efficiency dependence on #eta: ME1 studies");
h_eff_eta_me1_after_clct_okClct->GetXaxis()->SetRangeUser(0.86,2.5);
h_eff_eta_me1_after_clct_okClct->GetXaxis()->SetTitle("#eta");
h_eff_eta_me1_after_clct_okClct->SetMinimum(0);
h_eff_eta_me1_after_clct_okClct->GetYaxis()->SetRangeUser(0.,1.05);

TCanvas* cme1nc = new TCanvas("h_eff_eta_me1_after_clct","h_eff_eta_me1_after_clct",1000,600 ) ;

h_eff_eta_me1_after_clct_okClct->GetYaxis()->SetRangeUser(0.5,1.02);

//h_eff_eta_me1_after_clct->Draw();
//h_eff_eta_me1_after_clct_okClct->Draw("same hist");
h_eff_eta_me1_after_clct_okClct->Draw();


TLegend *leg = new TLegend(0.347,0.222,0.926,0.535,NULL,"brNDC");
leg->SetBorderSize(0);
leg->SetFillStyle(0);

leg->SetNColumns(2);
leg->SetHeader("Efficiency for #mu with p_{T}>10 crossing a ME1 chamber with");

leg->AddEntry(h_eff_eta_me1_after_alct,"any ALCT","pl");
leg->AddEntry(h_eff_eta_me1_after_alct_okAlct,"correct ALCT","pl");
leg->AddEntry(h_eff_eta_me1_after_clct,"any CLCT","pl");
leg->AddEntry(h_eff_eta_me1_after_clct_okClct,"correct CLCT","pl");

leg->AddEntry(h_eff_eta_me1_after_alctclct,"any ALCT and CLCT","pl");
//leg->AddEntry(h_eff_eta_me1_after_alctclct_okAlct,"correct ALCT and any CLCT","pl");
//leg->AddEntry(h_eff_eta_me1_after_alctclct_okClct,"any ALCT and correct CLCT","pl");
leg->AddEntry(h_eff_eta_me1_after_alctclct_okAlctClct,"correct ALCT and CLCT","pl");

leg->AddEntry(h_eff_eta_me1_after_lct,"any LCT","pl");
leg->AddEntry(h_eff_eta_me1_after_lct_okClctAlct,"correct LCT","pl");

leg->Draw();

Print(cme1na,"h_eff_eta_me1_steps_stubs.eps");
Print(cme1na,"h_eff_eta_me1_steps_stubs" + ext);
Print(cme1na,"h_eff_eta_me1_steps_stubs.pdf");
*/

//return;



if (do_h_eff_eta_me1_after_alct_okAlct) {

h_eff_eta_me1_after_alct_okAlct->SetTitle("Correct ALCT efficiency dependence on #eta: ME1 studies");
h_eff_eta_me1_after_alct_okAlct->GetXaxis()->SetRangeUser(0.86,2.5);
h_eff_eta_me1_after_alct_okAlct->GetXaxis()->SetTitle("#eta");
h_eff_eta_me1_after_alct_okAlct->SetMinimum(0);
h_eff_eta_me1_after_alct_okAlct->GetYaxis()->SetRangeUser(0.,1.05);

TCanvas* cme1nc = new TCanvas("h_eff_eta_me1_after_alctt","h_eff_eta_me1_after_alctt",1000,600 ) ;

h_eff_eta_me1_after_alct_okAlct->GetYaxis()->SetRangeUser(yrange[0],yrange[1]);

//h_eff_eta_me1_after_alct->Draw();
//h_eff_eta_me1_after_alct_okAlct->Draw("same hist");
h_eff_eta_me1_after_alct_okAlct->Draw("hist");

}



return;

/// EFFICIENCY dep. on WIREGROUP

h_wg_me11_initial = (TH1D*)getH(dir,"h_wg_me11_initial");
h_wg_me11_after_alct_okAlct = (TH1D*)getH(dir,"h_wg_me11_after_alct_okAlct");
h_wg_me11_after_alctclct_okAlctClct = (TH1D*)getH(dir,"h_wg_me11_after_alctclct_okAlctClct");
h_wg_me11_after_lct_okAlctClct = (TH1D*)getH(dir,"h_wg_me11_after_lct_okAlctClct");

h_eff_wg_me11_after_alct_okAlct = (TH1D*) h_wg_me11_after_alct_okAlct->Clone("h_eff_wg_me11_after_alct_okAlct");
h_eff_wg_me11_after_alctclct_okAlctClct = (TH1D*) h_wg_me11_after_alctclct_okAlctClct->Clone("h_eff_wg_me11_after_alctclct_okAlctClct");
h_eff_wg_me11_after_lct_okAlctClct = (TH1D*) h_wg_me11_after_lct_okAlctClct->Clone("h_eff_wg_me11_after_lct_okAlctClct");


h_eff_wg_me11_after_alct_okAlct->Divide(h_eff_wg_me11_after_alct_okAlct,h_wg_me11_initial);
h_eff_wg_me11_after_alctclct_okAlctClct->Divide(h_eff_wg_me11_after_alctclct_okAlctClct,h_wg_me11_initial);
h_eff_wg_me11_after_lct_okAlctClct->Divide(h_eff_wg_me11_after_lct_okAlctClct,h_wg_me11_initial);


h_eff_wg_me11_after_alct_okAlct->SetLineWidth(2);
h_eff_wg_me11_after_alctclct_okAlctClct->SetLineWidth(2);
h_eff_wg_me11_after_lct_okAlctClct->SetLineWidth(2);

h_eff_wg_me11_after_alct_okAlct->SetLineColor(kBlue+1);
h_eff_wg_me11_after_alctclct_okAlctClct->SetLineColor(kYellow+2);
h_eff_wg_me11_after_lct_okAlctClct->SetLineColor(kRed);

h_eff_wg_me11_after_alct_okAlct->SetTitle("Correct ALCT efficiency dependence on WG in ME11");
h_eff_wg_me11_after_alct_okAlct->SetTitle("wire group");
h_eff_wg_me11_after_alct_okAlct->SetMinimum(0);
h_eff_wg_me11_after_alct_okAlct->GetYaxis()->SetRangeUser(0.,1.05);

TCanvas* cme11wgv= new TCanvas("h_eff_wg_me11_after_alct","h_eff_wg_me11_after_alct",1000,600 ) ;
h_eff_wg_me11_after_alct_okAlct->Draw("hist");
h_eff_wg_me11_after_alctclct_okAlctClct->Draw("same hist");
h_eff_wg_me11_after_lct_okAlctClct->Draw("same hist");



}
