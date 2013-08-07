// Design LHC orbit: 2808 filled bickets out of 3564 --->  0.7879 filled BX fraction

TString gem_dir =  "files/";
TString gem_label = "gem98";
char dir[111]="SimMuL1StrictAll";
int gNPU=100;
int gNEvt=238000;
//int gNEvt=128000;
float gdy[2]={0.1,2500};

TLatex* drawLumiLabel2(float x=0.17, float y=0.35)
{
  TLatex *  tex = new TLatex(x, y,"L = 4*10^{34} cm^{-2} s^{-1}");
  tex->SetTextSize(0.04);
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
  ratio->GetYaxis()->SetTitle("ratio");
  // ratio->GetYaxis()->SetTitle("(ME1/b + GEM) / ME1/b");
  ratio->GetYaxis()->SetTitleSize(.14);
  // ratio->GetYaxis()->SetTitleSize(.1);
  ratio->GetYaxis()->SetTitleOffset(0.4);
  ratio->GetYaxis()->SetLabelSize(.11);

  //ratio->GetXaxis()->SetMoreLogLabels(1);
  ratio->GetXaxis()->SetTitle("track #eta");
  ratio->GetXaxis()->SetLabelSize(.11);
  ratio->GetXaxis()->SetTitleSize(.14);
  ratio->GetXaxis()->SetTitleOffset(1.); 

  ratio->SetLineWidth(2);
  ratio->SetFillColor(color);
  ratio->SetLineColor(color);
  ratio->SetMarkerColor(color);
  ratio->SetMarkerStyle(20);
  //ratio->Draw("e3");

  ratio->SetLineColor(color);
  ratio->SetMarkerColor(color);


  return ratio;
}


/*
void gem_rate_draw()
{

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
  
  return;
}
*/

void produceRateVsEtaPlotsForApproval()
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

  TString hdir = "SimMuL1StrictAll";

  // general stuff

  TString ext = ".pdf";
  TString plots = "plots/rate_vs_eta/";

  // colors - same colors as for rate vs eta plots!!
  Color_t col1 = kViolet+1;
  Color_t col2 = kAzure+1;
  Color_t col3 = kGreen-2;

  // Declaration of histograms
  TString vs_eta_minpt = "10";
  TString ttl = "         L1 trigger rates versus track #eta        CMS Simulation;;rate/bin [kHz]";
  TH1D* h_rt_tf10_2s   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s", "_hAll100", ttl, col1, 1, 2);
  TH1D* h_rt_tf10_2s1b   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s_2s1b", "_hAll100", ttl, col2, 1, 2);
  TH1D* h_rt_tf10_gpt10_2s1b   = setHistoEta(f_g98_pt10, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s_2s1b", "_hAll100", ttl, col3, 1, 2);
  TH1D* h_rt_tf10_3s   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s", "_hAll100", ttl, col1, 1, 2);
  TH1D* h_rt_tf10_3s1b   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1b", "_hAll100", ttl, col2, 1, 2);
  TH1D* h_rt_tf10_gpt10_3s1b   = setHistoEta(f_g98_pt10, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1b", "_hAll100", ttl, col3, 7, 2);
  
  /*
  TString vs_eta_minpt = "15";
  TString ttl = "CSC L1 trigger rates for p_{T}^{TF}>" + vs_eta_minpt + " GeV/c;;rate/bin [kHz]";
  TH1D* h_rt_tf15_2s   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s", "_hAll100", ttl, col1, 1, 2);
  TH1D* h_rt_tf15_2s1b   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s_2s1b", "_hAll100", ttl, col2, 1, 2);
  TH1D* h_rt_tf15_gpt15_2s1b   = setHistoEta(f_g98_pt15, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s_2s1b", "_hAll100", ttl, col3, 1, 2);
  TH1D* h_rt_tf15_3s   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s", "_hAll100", ttl, col1, 1, 2);
  TH1D* h_rt_tf15_3s1b   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1b", "_hAll100", ttl, col2, 1, 2);
  TH1D* h_rt_tf15_gpt15_3s1b   = setHistoEta(f_g98_pt15, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1b", "_hAll100", ttl, col3, 1, 2);
  */

  TString vs_eta_minpt = "20";
  TString ttl = "         L1 trigger rates versus track #eta        CMS Simulation;;rate/bin [kHz]";
  TH1D* h_rt_tf20_2s   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s", "_hAll100", ttl, col1, 1, 2);
  TH1D* h_rt_tf20_2s1b   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s_2s1b", "_hAll100", ttl, col2, 1, 2);
  TH1D* h_rt_tf20_gpt20_2s1b   = setHistoEta(f_g98_pt20, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s_2s1b", "_hAll100", ttl, col3, 1, 2);
  TH1D* h_rt_tf20_3s   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s", "_hAll100", ttl, col1, 1, 2);
  TH1D* h_rt_tf20_3s1b   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1b", "_hAll100", ttl, col2, 1, 2);
  TH1D* h_rt_tf20_gpt20_3s1b   = setHistoEta(f_g98_pt20, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1b", "_hAll100", ttl, col3, 1, 2);

  TString vs_eta_minpt = "30";
  TString ttl = "         L1 trigger rates versus track #eta        CMS Simulation;;rate/bin [kHz]";
  TH1D* h_rt_tf30_2s   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s", "_hAll100", ttl, col1, 1, 2);
  TH1D* h_rt_tf30_2s1b   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s_2s1b", "_hAll100", ttl, col2, 1, 2);
  TH1D* h_rt_tf30_gpt30_2s1b   = setHistoEta(f_g98_pt30, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s_2s1b", "_hAll100", ttl, col3, 1, 2);
  TH1D* h_rt_tf30_3s   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s", "_hAll100", ttl, col1, 1, 2);
  TH1D* h_rt_tf30_3s1b   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1b", "_hAll100", ttl, col2, 1, 2);
  TH1D* h_rt_tf30_gpt30_3s1b   = setHistoEta(f_g98_pt30, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1b", "_hAll100", ttl, col3, 1, 2);

  /*
  TString vs_eta_minpt = "40";
  TString ttl = "CSC L1 trigger rates for p_{T}^{TF}>" + vs_eta_minpt + " GeV/c;;rate/bin [kHz]";
  TH1D* h_rt_tf40_2s   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s", "_hAll100", ttl, col1, 1, 2);
  TH1D* h_rt_tf40_2s1b   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s_2s1b", "_hAll100", ttl, col2, 1, 2);
  TH1D* h_rt_tf40_gpt40_2s1b   = setHistoEta(f_g98_pt40, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_2s_2s1b", "_hAll100", ttl, col3, 1, 2);
  TH1D* h_rt_tf40_3s   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s", "_hAll100", ttl, col1, 1, 2);
  TH1D* h_rt_tf40_3s1b   = setHistoEta(f_def, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1b", "_hAll100", ttl, col2, 1, 2);
  TH1D* h_rt_tf40_gpt40_3s1b   = setHistoEta(f_g98_pt40, "h_rt_gmt_csc_ptmax" + vs_eta_minpt + "_eta_3s_3s1b", "_hAll100", ttl, col3, 1, 2);
  */


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

  // ------------ +2 stubs, track pt=10GeV ----------------//
  vs_eta_minpt = "10";

  {
    TCanvas* c = new TCanvas("c","c",800,800);
    c->Clear();
    TPad *pad1 = new TPad("pad1","top pad",0.0,0.25,1.0,1.0);
    pad1->Draw();
    TPad *pad2 = new TPad("pad2","bottom pad",0,0.,1.0,.30);
    pad2->Draw();

    pad1->cd();
    //pad1->SetLogy(1);
    pad1->SetGridx(1);
    pad1->SetGridy(1);
    pad1->SetFrameBorderMode(0);
    pad1->SetFillColor(kWhite);
    pad1->SetTopMargin(0.06);
    pad1->SetBottomMargin(0.13);

    h_rt_tf10_2s->Draw("hist e1");
    h_rt_tf10_2s1b->Draw("hist e1 same");
    h_rt_tf10_gpt10_2s1b->Draw("hist e1 same");
    //h_rt_tf10_2s1b->Draw("hist e1 same");
    //    h_rt_tf10_2s->Draw("hist e1 same ");

    h_rt_tf10_2s->SetFillColor(col1);
    h_rt_tf10_2s1b->SetFillColor(col2);
    h_rt_tf10_gpt10_2s1b->SetFillColor(col3);

    h_rt_tf10_2s->SetFillStyle(3345);
    h_rt_tf10_2s1b->SetFillStyle(3354);
    h_rt_tf10_gpt10_2s1b->SetFillStyle(3344);

    maxy = 80;
    h_rt_tf10_2s->GetYaxis()->SetRangeUser(miny,maxy);
    h_rt_tf10_2s1b->GetYaxis()->SetRangeUser(miny,maxy);
    h_rt_tf10_gpt10_2s1b->GetYaxis()->SetRangeUser(miny,maxy);
    
    TLegend *leg = new TLegend(0.17,0.65,.8,0.90,NULL,"brNDC");
    leg->SetBorderSize(0);
    leg->SetTextSize(0.04);
    leg->SetFillStyle(1001);
    leg->SetFillColor(kWhite);
    leg->AddEntry(h_rt_tf10_2s,"L1 Selections (#geq2 stations, track p_{T}#geq10):","");
    leg->AddEntry(h_rt_tf10_2s,"CSC, loose","f");
    leg->AddEntry(h_rt_tf10_2s1b,"CSC, tight","f");
    leg->AddEntry(h_rt_tf10_gpt10_2s1b,"GEM+CSC Integrated Trigger","f");
    leg->Draw();

    drawLumiLabel2();
    
    pad2->cd();
    pad2->SetLogy(1);
    pad2->SetGridx(1);
    pad2->SetGridy(1);
    pad2->SetFillColor(kWhite);
    pad2->SetFrameBorderMode(0);
    pad2->SetTopMargin(0.06);
    pad2->SetBottomMargin(0.3);
    
    gem_ratio = setHistoRatio(h_rt_tf10_gpt10_2s1b, h_rt_tf10_2s1b, "", 0.01,2.0, col2);
    gem_ratio->Draw("Pe");
    
    leg = new TLegend(0.16,0.33,.5,0.45,NULL,"brNDC");
    leg->SetBorderSize(0);
    leg->SetTextSize(0.1);
    leg->SetFillStyle(1001);
    leg->SetFillColor(kWhite);
    leg->AddEntry(gem_ratio, "GEM+CSC/CSC tight","P");
    leg->Draw("same");
    
    c->SaveAs(plots + "rates_vs_eta__minpt" + vs_eta_minpt + "__PU100__def_2s_2s1b_2s1bgem" + ext);
  }


  // ------------ +2 stubs, track pt=20GeV ----------------//
  vs_eta_minpt = "20";

  {
    TCanvas* c = new TCanvas("c","c",800,800);
    c->Clear();
    TPad *pad1 = new TPad("pad1","top pad",0.0,0.25,1.0,1.0);
    pad1->Draw();
    TPad *pad2 = new TPad("pad2","bottom pad",0,0.,1.0,.30);
    pad2->Draw();

    pad1->cd();
    //pad1->SetLogy(1);
    pad1->SetGridx(1);
    pad1->SetGridy(1);
    pad1->SetFrameBorderMode(0);
    pad1->SetFillColor(kWhite);
    pad1->SetTopMargin(0.06);
    pad1->SetBottomMargin(0.13);

    h_rt_tf20_2s->Draw("hist e1");
    h_rt_tf20_2s1b->Draw("hist e1 same");
    h_rt_tf20_gpt20_2s1b->Draw("hist e1 same");
    //h_rt_tf20_2s1b->Draw("hist e1 same");
    //    h_rt_tf20_2s->Draw("hist e1 same ");

    h_rt_tf20_2s->SetFillColor(col1);
    h_rt_tf20_2s1b->SetFillColor(col2);
    h_rt_tf20_gpt20_2s1b->SetFillColor(col3);

    h_rt_tf20_2s->SetFillStyle(3345);
    h_rt_tf20_2s1b->SetFillStyle(3354);
    h_rt_tf20_gpt20_2s1b->SetFillStyle(3344);

    maxy = 35;
    h_rt_tf20_2s->GetYaxis()->SetRangeUser(miny,maxy);
    h_rt_tf20_2s1b->GetYaxis()->SetRangeUser(miny,maxy);
    h_rt_tf20_gpt20_2s1b->GetYaxis()->SetRangeUser(miny,maxy);
    
    TLegend *leg = new TLegend(0.17,0.65,.8,0.90,NULL,"brNDC");
    leg->SetBorderSize(0);
    leg->SetTextSize(0.04);
    leg->SetFillStyle(1001);
    leg->SetFillColor(kWhite);
    leg->AddEntry(h_rt_tf20_2s,"L1 Selections (#geq2 stations, track p_{T}#geq20):","");
    leg->AddEntry(h_rt_tf20_2s,"CSC, loose","f");
    leg->AddEntry(h_rt_tf20_2s1b,"CSC, tight","f");
    leg->AddEntry(h_rt_tf20_gpt20_2s1b,"GEM+CSC Integrated Trigger","f");
    leg->Draw();

    drawLumiLabel2();
    
    pad2->cd();
    pad2->SetLogy(1);
    pad2->SetGridx(1);
    pad2->SetGridy(1);
    pad2->SetFillColor(kWhite);
    pad2->SetFrameBorderMode(0);
    pad2->SetTopMargin(0.06);
    pad2->SetBottomMargin(0.3);
    
    gem_ratio = setHistoRatio(h_rt_tf20_gpt20_2s1b, h_rt_tf20_2s1b, "", 0.01,2.0, col2);
    gem_ratio->Draw("Pe");
    
    leg = new TLegend(0.16,0.33,.5,0.45,NULL,"brNDC");
    leg->SetBorderSize(0);
    leg->SetTextSize(0.1);
    leg->SetFillStyle(1001);
    leg->SetFillColor(kWhite);
    leg->AddEntry(gem_ratio, "GEM+CSC/CSC tight","P");
    leg->Draw("same");
    
    c->SaveAs(plots + "rates_vs_eta__minpt" + vs_eta_minpt + "__PU100__def_2s_2s1b_2s1bgem" + ext);
  }

  // ------------ +2 stubs, track pt=30GeV ----------------//
  vs_eta_minpt = "30";
    
  {
    TCanvas* c = new TCanvas("c","c",800,800);
    c->Clear();
    TPad *pad1 = new TPad("pad1","top pad",0.0,0.25,1.0,1.0);
    pad1->Draw();
    TPad *pad2 = new TPad("pad2","bottom pad",0,0.,1.0,.30);
    pad2->Draw();

    pad1->cd();
    //pad1->SetLogy(1);
    pad1->SetGridx(1);
    pad1->SetGridy(1);
    pad1->SetFrameBorderMode(0);
    pad1->SetFillColor(kWhite);
    pad1->SetTopMargin(0.06);
    pad1->SetBottomMargin(0.13);

    h_rt_tf30_2s->Draw("hist e1");
    h_rt_tf30_2s1b->Draw("hist e1 same");
    h_rt_tf30_gpt30_2s1b->Draw("hist e1 same");
    //h_rt_tf30_2s1b->Draw("hist e1 same");
    //    h_rt_tf30_2s->Draw("hist e1 same ");

    h_rt_tf30_2s->SetFillColor(col1);
    h_rt_tf30_2s1b->SetFillColor(col2);
    h_rt_tf30_gpt30_2s1b->SetFillColor(col3);

    h_rt_tf30_2s->SetFillStyle(3345);
    h_rt_tf30_2s1b->SetFillStyle(3354);
    h_rt_tf30_gpt30_2s1b->SetFillStyle(3344);

    maxy = 30;
    h_rt_tf30_2s->GetYaxis()->SetRangeUser(miny,maxy);
    h_rt_tf30_2s1b->GetYaxis()->SetRangeUser(miny,maxy);
    h_rt_tf30_gpt30_2s1b->GetYaxis()->SetRangeUser(miny,maxy);
    
    TLegend *leg = new TLegend(0.17,0.65,.8,0.90,NULL,"brNDC");
    leg->SetBorderSize(0);
    leg->SetTextSize(0.04);
    leg->SetFillStyle(1001);
    leg->SetFillColor(kWhite);
    leg->AddEntry(h_rt_tf30_2s,"L1 Selections (#geq2 stations, track p_{T}#geq30):","");
    leg->AddEntry(h_rt_tf30_2s,"CSC, loose","f");
    leg->AddEntry(h_rt_tf30_2s1b,"CSC, tight","f");
    leg->AddEntry(h_rt_tf30_gpt30_2s1b,"GEM+CSC Integrated Trigger","f");
    leg->Draw();

    drawLumiLabel2();
    
    pad2->cd();
    pad2->SetLogy(1);
    pad2->SetGridx(1);
    pad2->SetGridy(1);
    pad2->SetFillColor(kWhite);
    pad2->SetFrameBorderMode(0);
    pad2->SetTopMargin(0.06);
    pad2->SetBottomMargin(0.3);
    
    gem_ratio = setHistoRatio(h_rt_tf30_gpt30_2s1b, h_rt_tf30_2s1b, "", 0.01,2.0, col2);
    gem_ratio->Draw("Pe");
    
    leg = new TLegend(0.16,0.33,.5,0.45,NULL,"brNDC");
    leg->SetBorderSize(0);
    leg->SetTextSize(0.1);
    leg->SetFillStyle(1001);
    leg->SetFillColor(kWhite);
    leg->AddEntry(gem_ratio, "GEM+CSC/CSC tight","P");
    leg->Draw("same");
    
    c->SaveAs(plots + "rates_vs_eta__minpt" + vs_eta_minpt + "__PU100__def_2s_2s1b_2s1bgem" + ext);
  }


  //// +3 Stub plots



  // ------------ +3 stubs, track pt=10GeV ----------------//
  vs_eta_minpt = "10";

  {
    TCanvas* c = new TCanvas("c","c",800,800);
    c->Clear();
    TPad *pad1 = new TPad("pad1","top pad",0.0,0.25,1.0,1.0);
    pad1->Draw();
    TPad *pad2 = new TPad("pad2","bottom pad",0,0.,1.0,.30);
    pad2->Draw();

    pad1->cd();
    //pad1->SetLogy(1);
    pad1->SetGridx(1);
    pad1->SetGridy(1);
    pad1->SetFrameBorderMode(0);
    pad1->SetFillColor(kWhite);
    pad1->SetTopMargin(0.06);
    pad1->SetBottomMargin(0.13);

    h_rt_tf10_3s->Draw("hist e1");
    h_rt_tf10_3s1b->Draw("hist e1 same");
    h_rt_tf10_gpt10_3s1b->Draw("hist e1 same");
    //h_rt_tf10_3s1b->Draw("hist e1 same");
    //    h_rt_tf10_3s->Draw("hist e1 same ");

    h_rt_tf10_3s->SetFillColor(col1);
    h_rt_tf10_3s1b->SetFillColor(col2);
    h_rt_tf10_gpt10_3s1b->SetFillColor(col3);

    h_rt_tf10_3s->SetFillStyle(3345);
    h_rt_tf10_3s1b->SetFillStyle(3354);
    h_rt_tf10_gpt10_3s1b->SetFillStyle(3344);

    maxy = 23;
    h_rt_tf10_3s->GetYaxis()->SetRangeUser(miny,maxy);
    h_rt_tf10_3s1b->GetYaxis()->SetRangeUser(miny,maxy);
    h_rt_tf10_gpt10_3s1b->GetYaxis()->SetRangeUser(miny,maxy);
    
    TLegend *leg = new TLegend(0.17,0.65,.8,0.90,NULL,"brNDC");
    leg->SetBorderSize(0);
    leg->SetTextSize(0.04);
    leg->SetFillStyle(1001);
    leg->SetFillColor(kWhite);
    leg->AddEntry(h_rt_tf10_3s,"L1 Selections (#geq3 stations, track p_{T}#geq10):","");
    leg->AddEntry(h_rt_tf10_3s,"CSC, loose","f");
    leg->AddEntry(h_rt_tf10_3s1b,"CSC, tight","f");
    leg->AddEntry(h_rt_tf10_gpt10_3s1b,"GEM+CSC Integrated Trigger","f");
    leg->Draw();

    drawLumiLabel2();
    
    pad2->cd();
    pad2->SetLogy(1);
    pad2->SetGridx(1);
    pad2->SetGridy(1);
    pad2->SetFillColor(kWhite);
    pad2->SetFrameBorderMode(0);
    pad2->SetTopMargin(0.06);
    pad2->SetBottomMargin(0.3);
    
    gem_ratio = setHistoRatio(h_rt_tf10_gpt10_3s1b, h_rt_tf10_3s1b, "", 0.01,2.0, col2);
    gem_ratio->Draw("Pe");
    
    leg = new TLegend(0.16,0.33,.5,0.45,NULL,"brNDC");
    leg->SetBorderSize(0);
    leg->SetTextSize(0.1);
    leg->SetFillStyle(1001);
    leg->SetFillColor(kWhite);
    leg->AddEntry(gem_ratio, "GEM+CSC/CSC tight","P");
    leg->Draw("same");
    
    c->SaveAs(plots + "rates_vs_eta__minpt" + vs_eta_minpt + "__PU100__def_3s_3s1b_3s1bgem" + ext);
  }


  // ------------ +3 stubs, track pt=20GeV ----------------//
  vs_eta_minpt = "20";

  {
    TCanvas* c = new TCanvas("c","c",800,800);
    c->Clear();
    TPad *pad1 = new TPad("pad1","top pad",0.0,0.25,1.0,1.0);
    pad1->Draw();
    TPad *pad2 = new TPad("pad2","bottom pad",0,0.,1.0,.30);
    pad2->Draw();

    pad1->cd();
    //pad1->SetLogy(1);
    pad1->SetGridx(1);
    pad1->SetGridy(1);
    pad1->SetFrameBorderMode(0);
    pad1->SetFillColor(kWhite);
    pad1->SetTopMargin(0.06);
    pad1->SetBottomMargin(0.13);

    h_rt_tf20_3s->Draw("hist e1");
    h_rt_tf20_3s1b->Draw("hist e1 same");
    h_rt_tf20_gpt20_3s1b->Draw("hist e1 same");
    //h_rt_tf20_3s1b->Draw("hist e1 same");
    //    h_rt_tf20_3s->Draw("hist e1 same ");

    h_rt_tf20_3s->SetFillColor(col1);
    h_rt_tf20_3s1b->SetFillColor(col2);
    h_rt_tf20_gpt20_3s1b->SetFillColor(col3);

    h_rt_tf20_3s->SetFillStyle(3345);
    h_rt_tf20_3s1b->SetFillStyle(3354);
    h_rt_tf20_gpt20_3s1b->SetFillStyle(3344);

    maxy = 10;
    h_rt_tf20_3s->GetYaxis()->SetRangeUser(miny,maxy);
    h_rt_tf20_3s1b->GetYaxis()->SetRangeUser(miny,maxy);
    h_rt_tf20_gpt20_3s1b->GetYaxis()->SetRangeUser(miny,maxy);
    
    TLegend *leg = new TLegend(0.17,0.65,.8,0.90,NULL,"brNDC");
    leg->SetBorderSize(0);
    leg->SetTextSize(0.04);
    leg->SetFillStyle(1001);
    leg->SetFillColor(kWhite);
    leg->AddEntry(h_rt_tf20_3s,"L1 Selections (#geq3 stations, track p_{T}#geq20):","");
    leg->AddEntry(h_rt_tf20_3s,"CSC, loose","f");
    leg->AddEntry(h_rt_tf20_3s1b,"CSC, tight","f");
    leg->AddEntry(h_rt_tf20_gpt20_3s1b,"GEM+CSC Integrated Trigger","f");
    leg->Draw();

    drawLumiLabel2();
    
    pad2->cd();
    pad2->SetLogy(1);
    pad2->SetGridx(1);
    pad2->SetGridy(1);
    pad2->SetFillColor(kWhite);
    pad2->SetFrameBorderMode(0);
    pad2->SetTopMargin(0.06);
    pad2->SetBottomMargin(0.3);
    
    gem_ratio = setHistoRatio(h_rt_tf20_gpt20_3s1b, h_rt_tf20_3s1b, "", 0.01,2.0, col2);
    gem_ratio->Draw("Pe");
    
    leg = new TLegend(0.16,0.33,.5,0.45,NULL,"brNDC");
    leg->SetBorderSize(0);
    leg->SetTextSize(0.1);
    leg->SetFillStyle(1001);
    leg->SetFillColor(kWhite);
    leg->AddEntry(gem_ratio, "GEM+CSC/CSC tight","P");
    leg->Draw("same");
    
    c->SaveAs(plots + "rates_vs_eta__minpt" + vs_eta_minpt + "__PU100__def_3s_3s1b_3s1bgem" + ext);
  }

  // ------------ +3 stubs, track pt=30GeV ----------------//
  vs_eta_minpt = "30";
    
  {
    TCanvas* c = new TCanvas("c","c",800,800);
    c->Clear();
    TPad *pad1 = new TPad("pad1","top pad",0.0,0.25,1.0,1.0);
    pad1->Draw();
    TPad *pad2 = new TPad("pad2","bottom pad",0,0.,1.0,.30);
    pad2->Draw();

    pad1->cd();
    //pad1->SetLogy(1);
    pad1->SetGridx(1);
    pad1->SetGridy(1);
    pad1->SetFrameBorderMode(0);
    pad1->SetFillColor(kWhite);
    pad1->SetTopMargin(0.06);
    pad1->SetBottomMargin(0.13);

    h_rt_tf30_3s->Draw("hist e1");
    h_rt_tf30_3s1b->Draw("hist e1 same");
    h_rt_tf30_gpt30_3s1b->Draw("hist e1 same");
    //h_rt_tf30_3s1b->Draw("hist e1 same");
    //    h_rt_tf30_3s->Draw("hist e1 same ");

    h_rt_tf30_3s->SetFillColor(col1);
    h_rt_tf30_3s1b->SetFillColor(col2);
    h_rt_tf30_gpt30_3s1b->SetFillColor(col3);

    h_rt_tf30_3s->SetFillStyle(3345);
    h_rt_tf30_3s1b->SetFillStyle(3354);
    h_rt_tf30_gpt30_3s1b->SetFillStyle(3344);

    maxy = 6;
    h_rt_tf30_3s->GetYaxis()->SetRangeUser(miny,maxy);
    h_rt_tf30_3s1b->GetYaxis()->SetRangeUser(miny,maxy);
    h_rt_tf30_gpt30_3s1b->GetYaxis()->SetRangeUser(miny,maxy);
    
    TLegend *leg = new TLegend(0.17,0.65,.8,0.90,NULL,"brNDC");
    leg->SetBorderSize(0);
    leg->SetTextSize(0.04);
    leg->SetFillStyle(1001);
    leg->SetFillColor(kWhite);
    leg->AddEntry(h_rt_tf30_3s,"L1 Selections (#geq3 stations, track p_{T}#geq30):","");
    leg->AddEntry(h_rt_tf30_3s,"CSC, loose","f");
    leg->AddEntry(h_rt_tf30_3s1b,"CSC, tight","f");
    leg->AddEntry(h_rt_tf30_gpt30_3s1b,"GEM+CSC Integrated Trigger","f");
    leg->Draw();

    drawLumiLabel2();
    
    pad2->cd();
    pad2->SetLogy(1);
    pad2->SetGridx(1);
    pad2->SetGridy(1);
    pad2->SetFillColor(kWhite);
    pad2->SetFrameBorderMode(0);
    pad2->SetTopMargin(0.06);
    pad2->SetBottomMargin(0.3);
    
    gem_ratio = setHistoRatio(h_rt_tf30_gpt30_3s1b, h_rt_tf30_3s1b, "", 0.01,2.0, col2);
    gem_ratio->Draw("Pe");
    
    leg = new TLegend(0.16,0.33,.5,0.45,NULL,"brNDC");
    leg->SetBorderSize(0);
    leg->SetTextSize(0.1);
    leg->SetFillStyle(1001);
    leg->SetFillColor(kWhite);
    leg->AddEntry(gem_ratio, "GEM+CSC/CSC tight","P");
    leg->Draw("same");
    
    c->SaveAs(plots + "rates_vs_eta__minpt" + vs_eta_minpt + "__PU100__def_3s_3s1b_3s1bgem" + ext);
  }

}
