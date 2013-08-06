gROOT->ProcessLine(".L effFunctions.C");

/*

  .L 

  "gem_csc_eff_pt2pt50_pad4.root"

*/

TString filesDir = "files/";
TString plotDir = "plots/efficiency/";
TString ext = ".png";

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



TCut ok_dphi1 = "dphi_pad_odd < 10.";
TCut ok_dphi2 = "dphi_pad_even < 10.";

enum {GEM_EFF95 = 0, GEM_EFF98, GEM_EFF99};
enum {DPHI_PT10 = 0, DPHI_PT15, DPHI_PT20, DPHI_PT30, DPHI_PT40};

void setDPhi(int label_pt, int label_eff = GEM_EFF98)
{
  float dphi_odd[3][20]  = {
    {0.009887, 0.006685, 0.005194, 0.003849, 0.003133},
    {0.01076 , 0.007313, 0.005713, 0.004263, 0.003513},
    {0.011434, 0.007860, 0.006162, 0.004615, 0.003891} };
  float dphi_even[3][20] = {
    {0.004418, 0.003274, 0.002751, 0.002259, 0.002008},
    {0.004863, 0.003638, 0.003063, 0.002563, 0.002313},
    {0.005238, 0.003931, 0.003354, 0.002809, 0.002574} };
  ok_dphi1 = Form("TMath::Abs(dphi_pad_odd) < %f", dphi_odd[label_eff][label_pt]);
  ok_dphi2 = Form("TMath::Abs(dphi_pad_even) < %f", dphi_even[label_eff][label_pt]);
}


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


void gemTurnOns(int label_eff = GEM_EFF98)
{
  //  gROOT->ProcessLine(".L effFunctions.C");

  TCut ok_eta = "TMath::Abs(eta)>1.64 && TMath::Abs(eta)<2.14";

  const int N = 5;

  int pt_lbl[N] = {DPHI_PT10, DPHI_PT15, DPHI_PT20, DPHI_PT30, DPHI_PT40};
  TString pt[N] = {"pt10","pt15","pt20","pt30","pt40"};
  int marker_styles[5] = {24, 28, 22 , 21, 20};

  TCanvas* gEff = new TCanvas("gEff","gEff",700,500);
  gEff->SetGridx(1);  gEff->SetGridy(1);
  TCanvas* gEff_odd = new TCanvas("gEff_odd","gEff_odd",700,500);
  gEff_odd->SetGridx(1);  gEff_odd->SetGridy(1);
  TCanvas* gEff_even = new TCanvas("gEff_even","gEff_even",700,500);
  gEff_even->SetGridx(1);  gEff_even->SetGridy(1);

  TTree *t = getTree(filesDir + "gem_csc_eff_pt2pt50_pad4.root");
  TH1F *ho[N], *he[N];
  for (int n=0; n<N; ++n) {
    if (n==1) continue;
    TString opt = "p";
    if (n>0) opt = "same p";
    setDPhi(n, label_eff);
    gEff->cd();
    ho[n] = draw_eff(gt, "Eff. for track with LCT to have matched GEM pad;p_{T}, GeV/c;Eff.", Form("h_odd%d",n), "(50,0.,50.)", "pt", ok_lct1 && ok_eta && ok_pad1, ok_dphi1, opt, kRed, marker_styles[n]);
    he[n] = draw_eff(gt, "Eff. for track with LCT to have matched GEM pad;p_{T}, GeV/c;Eff.", Form("h_evn%d",n), "(50,0.,50.)", "pt", ok_lct2 && ok_eta && ok_pad2, ok_dphi2, "same p", kBlue, marker_styles[n]);
    gEff_odd->cd();
    ho[n]->Draw(opt);
    gEff_even->cd();
    he[n]->Draw(opt);
  }

  TString pts[N] = {"10","15","20","30","40"};
  TString efs[5] = {"95", "98", "99"};
  TString leg_header =  "    #Delta#phi(LCT,GEM) is " + efs[label_eff] + "% efficient for";

  gEff->cd();
  TLegend *leg = new TLegend(0.50,0.17,.99,0.57, NULL, "brNDC");
  leg->SetNColumns(2);
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  leg->SetHeader(leg_header);
  leg->AddEntry(ho[0], "odd chambers", "");
  leg->AddEntry(he[0], "even chambers", "");
  leg->AddEntry(ho[0], "at pt", "");
  leg->AddEntry(he[0], "at pt", "");
  for (int n=0; n<N; ++n) {
    if (n==1) continue;
    leg->AddEntry(ho[n], pts[n], "p");
    leg->AddEntry(he[n], pts[n], "p");
  }
  leg->Draw();
  gEff->Print(plotDir + "gem_pad_eff_for_LCT_gemEff" + efs[label_eff] +"" + ext);

  gEff_odd->cd();
  TLegend *leg = new TLegend(0.50,0.17,.99,0.57, NULL, "brNDC");
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  leg->SetHeader(leg_header);
  leg->AddEntry(ho[0], "odd chambers at pt", "");
  for (int n=0; n<N; ++n) {
    if (n==1) continue;
    leg->AddEntry(ho[n], pts[n], "p");
  }
  leg->Draw();
  gEff_odd->Print(plotDir + "gem_pad_eff_for_LCT_gemEff" + efs[label_eff] +"_odd" + ext);

  gEff_even->cd();
  TLegend *leg = new TLegend(0.50,0.17,.99,0.57, NULL, "brNDC");
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  leg->SetHeader(leg_header);
  leg->AddEntry(ho[0], "even chambers at pt", "");
  for (int n=0; n<N; ++n) {
    if (n==1) continue;
    leg->AddEntry(he[n], pts[n], "p");
  }
  leg->Draw();
  gEff_even->Print(plotDir + "gem_pad_eff_for_LCT_gemEff" + efs[label_eff] +"_even" + ext);
}


float dphiCut(TH1* h, float fractionToKeep)
{
  TAxis *ax = h->GetXaxis();
  double total = h->Integral();
  int bin=1;
  for(int b=1; b<ax->GetNbins(); ++b) {
    //cout<<b<<" "<<ax->GetBinUpEdge(b)<<" "<<h->Integral(0,b)/total<<endl;
    if (h->Integral(0,b)/total > fractionToKeep) { bin = b - 1; break; }
  }
  // interpolate
  float x1 = ax->GetBinUpEdge(bin), x2 = ax->GetBinUpEdge(bin + 1);
  float y1 = h->Integral(0, bin)/total, y2 = h->Integral(0, bin + 1)/total;
  float x = x1 + (fractionToKeep - y1)/(y2-y1)*(x2-x1);
  cout<<"fraction "<<fractionToKeep<<": dphi < "<<x<<endl;
  return x;
}


void getDphis()
{

  float dphis[6][3][2] = {{{0.}}};
  TString pt[6] = {"pt5","pt10","pt15","pt20","pt30","pt40"};
  float fr[3] = {0.95, 0.98, 0.99};

  for (int n=0; n<6; ++n) {
    t = getTree(filesDir + TString("gem_csc_delta_")+pt[n] + "_pad4.root");
    t->Draw("TMath::Abs(dphi_pad_odd)>>dphi_odd(600,0.,0.03)" , ok_pad1 && ok_lct1);
    t->Draw("TMath::Abs(dphi_pad_even)>>dphi_even(600,0.,0.03)" , ok_pad2 && ok_lct2);
    for (int f=0; f<3; ++f) {
      dphis[n][f][0] = dphiCut(dphi_odd, fr[f]);
      dphis[n][f][1] = dphiCut(dphi_even, fr[f]);
    }
  }

  cout<<setprecision(4);
  cout<<endl;
  for (int i=0; i<3; ++i) {
    cout<<"efficiency "<<fr[i]*100<<"%"<<endl;
    cout<<"       odd     even"<<endl;
    for (int n=0; n<6; ++n) {
      cout<<pt[n]<<" "<<setw(7)<<dphis[n][i][0]*1000.<<" "<<setw(7)<<dphis[n][i][1]*1000.<<endl;
    }
  }
  //TF1 fu("[0]*exp([1]*log(x))");
  //dphi_lct_pad95:
  //odd  {  ,0.02023  ,0.009775  ,0.006611  ,0.005124  ,0.003744  ,0.003077}
  //even {  ,0.008336  ,0.004356  ,0.003241  ,0.002721  ,0.00222  ,0.001982}

  cout<<endl;
  for (int i=0; i<3; ++i) {
    cout<<"dphi_lct_pad"<<fr[i]*100<<":"<<endl<<"odd  {";
    for (int n=0; n<6; ++n) { cout<<"  ,"<<setw(7)<<dphis[n][i][0]; }
    cout<<"}"<<endl<<"even {";
    for (int n=0; n<6; ++n) { cout<<"  ,"<<setw(7)<<dphis[n][i][1]; }
    cout<<"}"<<endl;
  }

  cout<<setprecision(6);
  cout<<endl;
  for (int i=0; i<3; ++i) {
    cout<<"dphi_lct_pad"<<fr[i]*100<<" = {"<<endl;
    for (int n=0; n<6; ++n) {
      cout<<"  '"<<pt[n]<<"' : { 'odd' : "<<setw(11)<<dphis[n][i][0]<<" , 'even' : "<<setw(11)<<dphis[n][i][1]<<" },"<<endl;
    }
    cout<<"}"<<endl;
  }

}



double mymod(double x, double y) {return fmod(x,y);}

void eff_hs_dphi(TString f_name, TString p_name)
{
  // efficiency vs half-strip  - separate odd-even
  TCut ok_eta = "TMath::Abs(eta)>1.64 && TMath::Abs(eta)<2.12";
  //TCut ok_eta = "TMath::Abs(eta)>1.64 && TMath::Abs(eta)<1.74";
  //TCut ok_eta = "TMath::Abs(eta)>1.94 && TMath::Abs(eta)<2.12";

  TTree *t = getTree(f_name);
  setDPhi(DPHI_PT40, GEM_EFF98);

  TH1F* ho = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip;Eff.", "h_odd", "(130,0.5,130.5)", "hs_lct_odd", ok_lct1 && ok_eta && ok_gsh1 , ok_pad1, "", kRed);
  TH1F* he = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip;Eff.", "h_evn", "(130,0.5,130.5)", "hs_lct_even", ok_lct2 && ok_eta && ok_gsh2, ok_pad2, "same");

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


void eff_hs(TString f_name, TString p_name)
{
  // efficiency vs half-strip  - separate odd-even
  TCut ok_eta = "TMath::Abs(eta)>1.64 && TMath::Abs(eta)<2.12";

  TTree *t = getTree(f_name);
  TH1F* ho = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip;Eff.", "h_odd", "(130,0.5,130.5)", "hs_lct_odd", ok_lct1 && ok_eta , ok_pad1, "", kRed);
  TH1F* he = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip;Eff.", "h_evn", "(130,0.5,130.5)", "hs_lct_even", ok_lct2 && ok_eta , ok_pad2, "same");

  TLegend *leg = new TLegend(0.50,0.23,.9,0.4, NULL, "brNDC");
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  leg->AddEntry(ho,"odd chambers","l");
  leg->AddEntry(he,"even chambers","l");
  leg->Draw();

  gPad->Print(p_name);
}


void eff_hs_overlap(TString f_name, TString p_name)
{
  // efficiency vs half-strip  - including overlaps in odd&even
  TCut ok_eta = "TMath::Abs(eta)>1.64 && TMath::Abs(eta)<2.12";

  TTree *t = getTree(f_name);
  TH1F* ho = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip;Eff.", "h_odd", "(130,0.5,130.5)", "hs_lct_odd", ok_lct1 && ok_eta , ok_pad1_overlap, "", kRed);
  TH1F* he = draw_eff(t, "Eff. for track with LCT to have GEM pad in chamber;LCT half-strip;Eff.", "h_evn", "(130,0.5,130.5)", "hs_lct_even", ok_lct2 && ok_eta , ok_pad2_overlap, "same");

  TF1 fo("fo", "pol0", 6., 123.);
  ho->Fit("fo","RN");
  TF1 fe("fe", "pol0", 6., 123.);
  he->Fit("fe","RN");

  TLegend *leg = new TLegend(0.42,0.23,.96,0.4, NULL, "brNDC");
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  //leg->AddEntry(ho, Form("odd chambers (%0.2f #pm %0.2f)%%", 100.*fo.GetParameter(0), 100.*fo.GetParError(0)),"l");
  //leg->AddEntry(he, Form("even chambers (%0.1f #pm %0.1f)%%", 100.*fe.GetParameter(0), 100.*fe.GetParError(0)),"l");
  leg->AddEntry(ho, "odd chambers","l");
  leg->AddEntry(he, "even chambers","l");
  leg->Draw();

  gPad->Print(p_name);
}


void drawplot_eff_eta()
{

  gROOT->ProcessLine(".L effFunctions.C");

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
  cEff->Print("lct_eff_for_Trk_vsTrkEta_pt40" + ext);


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
  cEff->Print("gem_pad_and_lct_eff_for_Trk_vsTrkEta_pt40" + ext);

  return;

  h1 = draw_geff(gt, "Eff. for a SimTrack to have an associated GEM Pad;SimTrack |#eta|;Eff.", "h_odd", "(70,1.54,2.2)", "TMath::Abs(eta)", "", ok_pad1 || ok_pad2, "P", kViolet);
  eff_base->GetYaxis()->SetRangeUser(0.6,1.05);
  TLatex *  tex = new TLatex(0.17, 0.16,"No Pile-Up");
  tex->SetNDC();
  tex->Draw();
  cEff->Print("gem_pad0_eff_for_Trk_vsTrkEta_pt40" + ext);




  TTree *gt15 = getTree(filesDir + "gem_csc_delta_pt15_pad4.root");
  h1 = draw_geff(gt15, "Eff. for a SimTrack to have an associated LCT;SimTrack |#eta|;Eff.", "h_odd", "(140,1.54,2.2)", "TMath::Abs(eta)", "", ok_lct1 || ok_lct2, "P", kViolet+2);
  cEff->Print("lct_eff_for_Trk_vsTrkEta_pt15" + ext);


  ho = draw_geff(gt, "Eff. for track with LCT to have GEM pad in chamber;LCT |#eta|;Eff.", "h_odd", "(140,1.5,2.2)", "TMath::Abs(eta_lct_odd)", ok_lct1, ok_pad1, "P", kRed);
  he = draw_geff(gt, "Eff. for track with LCT to have GEM pad in chamber;LCT |#eta|;Eff.", "h_evn", "(140,1.5,2.2)", "TMath::Abs(eta_lct_even)", ok_lct2, ok_pad2, "P same");
  TLegend *leg = new TLegend(0.42,0.23,.96,0.4, NULL, "brNDC");
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  leg->AddEntry(ho, "odd chambers","l");
  leg->AddEntry(he, "even chambers","l");
  leg->Draw();
  cEff->Print("gem_pad_eff_for_LCT_vsLCTEta_pt40" + ext);

  draw_geff(gt, "Eff. for track with LCT to have GEM pad in chamber;SimTrack |#eta|;Eff.", "h_odd", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_lct1, ok_pad1, "P", kRed);
  draw_geff(gt, "Eff. for track with LCT to have GEM pad in chamber;SimTrack |#eta|;Eff.", "h_evn", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_lct2, ok_pad2, "P same");
  leg->Draw();
  cEff->Print("gem_pad_eff_for_LCT_vsTrkEta_pt40" + ext);

  draw_geff(gt, "Eff. for track with LCT to have GEM pad in chamber;LCT |#eta|;Eff.", "h_odd", "(140,1.5,2.2)", "TMath::Abs(eta_lct_odd)", ok_lct1, ok_pad1_overlap, "P", kRed);
  draw_geff(gt, "Eff. for track with LCT to have GEM pad in chamber;LCT |#eta|;Eff.", "h_evn", "(140,1.5,2.2)", "TMath::Abs(eta_lct_even)", ok_lct2, ok_pad2_overlap, "P same");
  leg->Draw();
  cEff->Print("gem_pad_eff_for_LCT_vsLCTEta_pt40_overlap" + ext);

  draw_geff(gt, "Eff. for track with LCT to have GEM pad in chamber;SimTrack |#eta|;Eff.", "h_odd", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_lct1, ok_pad1_overlap, "P", kRed);
  draw_geff(gt, "Eff. for track with LCT to have GEM pad in chamber;SimTrack |#eta|;Eff.", "h_evn", "(140,1.5,2.2)", "TMath::Abs(eta)", ok_lct2, ok_pad2_overlap, "P same");
  leg->Draw();
  cEff->Print("gem_pad_eff_for_LCT_vsTrkEta_pt40_overlap" + ext);

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
  cEff->Print("gem_pad_eff_for_Trk_vsTrkEta_pt40" + ext);

  return;
  draw_geff(gt, "Eff. for a SimTrack to have an associated GEM pad;SimTrack |#eta|;Eff.", "h_odd", "(140,1.54,2.2)", "TMath::Abs(eta)", "", ok_gsh1 || ok_gsh2, "P", kViolet);
  draw_geff(gt, "Eff. for a SimTrack to have an associated GEM pad;SimTrack |#eta|;Eff.", "h_odd", "(140,1.54,2.2)", "TMath::Abs(eta)", "", ok_g2sh1 || ok_g2sh2 , "P", kOrange);
  draw_geff(gt, "Eff. for a SimTrack to have an associated GEM pad;SimTrack |#eta|;Eff.", "h_odd", "(140,1.54,2.2)", "TMath::Abs(eta)", "", ok_copad1 || ok_copad2 , "P same", kRed);


}


void drawplot_eff()
{
  gROOT->ProcessLine(".L effFunctions.C");

  gf_name = "";

  TCanvas* cEff = new TCanvas("cEff","cEff",700,500);

  eff_hs(filesDir + "gem_csc_delta_pt5_pad4.root",  plotDir + "gem_pad_eff_for_LCT_vsHS_pt05" + ext);
  eff_hs(filesDir + "gem_csc_delta_pt10_pad4.root", plotDir + "gem_pad_eff_for_LCT_vsHS_pt10" + ext);
  eff_hs(filesDir + "gem_csc_delta_pt15_pad4.root", plotDir + "gem_pad_eff_for_LCT_vsHS_pt15" + ext);
  eff_hs(filesDir + "gem_csc_delta_pt20_pad4.root", plotDir + "gem_pad_eff_for_LCT_vsHS_pt20" + ext);
  eff_hs(filesDir + "gem_csc_delta_pt30_pad4.root", plotDir + "gem_pad_eff_for_LCT_vsHS_pt30" + ext);
  eff_hs(filesDir + "gem_csc_delta_pt40_pad4.root", plotDir + "gem_pad_eff_for_LCT_vsHS_pt40" + ext);

  eff_hs_overlap(filesDir + "gem_csc_delta_pt5_pad4.root",  plotDir + "gem_pad_eff_for_LCT_vsHS_pt05_overlap" + ext);
  eff_hs_overlap(filesDir + "gem_csc_delta_pt10_pad4.root", plotDir + "gem_pad_eff_for_LCT_vsHS_pt10_overlap" + ext);
  eff_hs_overlap(filesDir + "gem_csc_delta_pt15_pad4.root", plotDir + "gem_pad_eff_for_LCT_vsHS_pt15_overlap" + ext);
  eff_hs_overlap(filesDir + "gem_csc_delta_pt20_pad4.root", plotDir + "gem_pad_eff_for_LCT_vsHS_pt20_overlap" + ext);
  eff_hs_overlap(filesDir + "gem_csc_delta_pt30_pad4.root", plotDir + "gem_pad_eff_for_LCT_vsHS_pt30_overlap" + ext);
  eff_hs_overlap(filesDir + "gem_csc_delta_pt40_pad4.root", plotDir + "gem_pad_eff_for_LCT_vsHS_pt40_overlap" + ext);

  gemTurnOns();

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




























  --------------------------- 2012  Oct 30 -----------------------

  .L shFunctions.C

  //fname="shtree_std_pt100.root";  suff = "std_pt100";
  fname="shtree_POSTLS161_pt100.root";  suff = "postls1_pt100";


  TFile *f = TFile::Open(fname)
  tree = (TTree *) f->Get("neutronAna/CSCSimHitsTree");

  TCanvas c1("c1","c1",1200,600);
  c1->SetBorderSize(0);
  c1->SetLeftMargin(0.084);
  c1->SetRightMargin(0.033);
  c1->SetTopMargin(0.089);
  c1->SetBottomMargin(0.086);
  c1->SetGridy();
  c1->SetTickx(1);
  c1->SetTicky(1);
  gStyle->SetOptStat(0);




  TH2D *hc = new TH2D("hc","Strip readout channels closest to SimHit",21,-10.5,10.5,82,0,82);
  hc->Draw()
  setupH2DType(hc);
  hc->GetYaxis()->SetTitle("readout channel #");

  tree->Draw("sh.chan:(3-2*id.e)*id.t","","same", 10000000, 0);
  g = (TGraph*)gPad->FindObject("Graph");
  g->SetMarkerColor(kBlue);
  g->SetMarkerStyle(2);
  g->SetMarkerSize(0.4);
  gPad->Modified()

  c1.Print((string("chan_")+suff+"" + ext).c_str())




  TH2D *hc = new TH2D("hc","Strips closest to SimHit",21,-10.5,10.5,82,0,82)
  for(int b=1; b<=hc->GetXaxis()->GetNbins(); ++b) hc->GetXaxis()->SetBinLabel(b, types2[b].c_str())
  hc->Draw()
  setupH2DType(hc);
  hc->GetYaxis()->SetTitle("strip #");

  tree->Draw("sh.s:(3-2*id.e)*id.t","","same", 10000000, 0);
  g = (TGraph*)gPad->FindObject("Graph");
  g->SetMarkerColor(kBlue);
  g->SetMarkerStyle(2);
  g->SetMarkerSize(0.4);
  gPad->Modified()

  c1.Print((string("strip_")+suff+"" + ext).c_str())



  TH2D *hc = new TH2D("hc","Wire groups closest to SimHit",21,-10.5,10.5,115,-1,114)
  for(int b=1; b<=hc->GetXaxis()->GetNbins(); ++b) hc->GetXaxis()->SetBinLabel(b, types2[b].c_str())
  hc->Draw()
  setupH2DType(hc);
  hc->GetYaxis()->SetTitle("wiregroup #");
  hc->GetYaxis()->SetNdivisions(1020);


  tree->Draw("sh.w:(3-2*id.e)*id.t","","same", 10000000, 0);
  g = (TGraph*)gPad->FindObject("Graph");
  g->SetMarkerColor(kBlue);
  g->SetMarkerStyle(2);
  g->SetMarkerSize(0.4);
  gPad->Modified()

  c1.Print((string("wg_")+suff+"" + ext).c_str())





  .L shFunctions.C
  //fname="shtree_POSTLS161_pt100.root";  suff = "postls1_pt100";
  //fname="shtree_POSTLS161_pt10.root";  suff = "postls1_pt10";
  fname="shtree_POSTLS161_pt1000.root";  suff = "postls1_pt1000";
  TFile *f = TFile::Open(fname)

  globalPosfromTree("+ME1", f, 1, 1, "rphi_+ME1_postls1" + ext)
  globalPosfromTree("+ME2", f, 1, 2, "rphi_+ME2_postls1" + ext)
  globalPosfromTree("+ME3", f, 1, 3, "rphi_+ME3_postls1" + ext)
  globalPosfromTree("+ME4", f, 1, 4, "rphi_+ME4_postls1" + ext)
  globalPosfromTree("-ME1", f, 2, 1, "rphi_-ME1_postls1" + ext)
  globalPosfromTree("-ME2", f, 2, 2, "rphi_-ME2_postls1" + ext)
  globalPosfromTree("-ME3", f, 2, 3, "rphi_-ME3_postls1" + ext)
  globalPosfromTree("-ME4", f, 2, 4, "rphi_-ME4_postls1" + ext)

  .L shFunctions.C
  fname="shtree_std_pt100.root";  suff = "std_pt100";
  TFile *f = TFile::Open(fname)
  globalPosfromTree("+ME1", f, 1, 1, "rphi_+ME1_std" + ext)
  globalPosfromTree("+ME2", f, 1, 2, "rphi_+ME2_std" + ext)
  globalPosfromTree("+ME3", f, 1, 3, "rphi_+ME3_std" + ext)
  globalPosfromTree("+ME4", f, 1, 4, "rphi_+ME4_std" + ext)
  globalPosfromTree("-ME1", f, 2, 1, "rphi_-ME1_std" + ext)
  globalPosfromTree("-ME2", f, 2, 2, "rphi_-ME2_std" + ext)
  globalPosfromTree("-ME3", f, 2, 3, "rphi_-ME3_std" + ext)
  globalPosfromTree("-ME4", f, 2, 4, "rphi_-ME4_std" + ext)






  fname="shtree_std_pt100.root";  suff = "std_pt100";
  //fname="shtree_POSTLS161_pt100.root";  suff = "postls1_pt100";


  TFile *f = TFile::Open(fname)
  tree = (TTree *) f->Get("neutronAna/CSCSimHitsTree");

  TCanvas c1("c1","c1",900,900);
  c1->SetBorderSize(0);
  c1->SetLeftMargin(0.13);
  c1->SetRightMargin(0.012);
  c1->SetTopMargin(0.022);
  c1->SetBottomMargin(0.111);
  c1->SetGridy();
  c1->SetTickx(1);
  c1->SetTicky(1);
  gStyle->SetOptStat(0);



  TH2D *hc = new TH2D("hc",";z, cm;r, cm",2,560,1070,82,70,710);
  hc->Draw()
  h->GetXaxis()->SetTickLength(0.02);
  h->GetYaxis()->SetTickLength(0.02);

  tree->Draw("sh.r:sh.gz","","same", 10000000, 0);
  g = (TGraph*)gPad->FindObject("Graph");
  g->SetMarkerColor(kBlue);
  g->SetMarkerStyle(1);
  g->SetMarkerSize(0.1);
  gPad->Modified()

  c1.Print((string("rz_+ME_")+suff+"" + ext).c_str())



  TH2D *hc = new TH2D("hc",";z, cm;r, cm",2,-1070,-560,82,70,710);
  hc->Draw()
  h->GetXaxis()->SetTickLength(0.02);
  h->GetYaxis()->SetTickLength(0.02);

  tree->Draw("sh.r:sh.gz","","same", 10000000, 0);
  g = (TGraph*)gPad->FindObject("Graph");
  g->SetMarkerColor(kBlue);
  g->SetMarkerStyle(1);
  g->SetMarkerSize(0.1);
  gPad->Modified()

  c1.Print((string("rz_-ME_")+suff+"" + ext).c_str())







  TFile *f1 = TFile::Open("shtree_std_pt100.root")
  t1 = (TTree *) f1->Get("neutronAna/CSCSimHitsTree");
  TFile *f2 = TFile::Open("shtree_POSTLS161_pt100.root")
  t2 = (TTree *) f2->Get("neutronAna/CSCSimHitsTree");

  TCanvas c1("c1","c1",900,900);
  c1->SetBorderSize(0);
  c1->SetLeftMargin(0.13);
  c1->SetRightMargin(0.023);
  c1->SetTopMargin(0.081);
  c1->SetBottomMargin(0.13);
  c1->SetGridy();
  c1->SetTickx(1);
  c1->SetTicky(1);
  c1->SetLogy(1);
  gStyle->SetOptStat(0);

  t1->Draw("TMath::Log10(sh.e*1000000)>>htmp(100,-2.5,2.5)","","", 10000000, 0);
  htmp->SetTitle("SimHit energy loss");
  htmp->SetXTitle("log_{10}E_{loss}, eV");
  htmp->SetYTitle("entries");
  htmp->SetLineWidth(2);
  htmp->GetXaxis()->SetNdivisions(509);
  t2->Draw("TMath::Log10(sh.e*1000000)>>htmp2(100,-2.5,2.5)","","same", 10000000, 0);
  htmp2->SetLineWidth(2);
  htmp2->SetLineStyle(7);
  htmp2->SetLineColor(kRed);
  scale = htmp->GetEntries()/(1.*htmp2->GetEntries());
  htmp2->Scale(scale);
  gPad->Modified();

  TLegend *leg1 = new TLegend(0.17,0.7,0.47,0.85,NULL,"brNDC");
  leg1->SetBorderSize(0);
  leg1->SetFillStyle(0);
  leg1->AddEntry(htmp,"6_0_0_patch1","l");
  leg1->AddEntry(htmp2,"POSTLS161","l");
  leg1->Draw();

  c1.Print("sh_eloss" + ext)





  TFile *f1 = TFile::Open("shtree_std_pt100.root")
  t1 = (TTree *) f1->Get("neutronAna/CSCSimHitsTree");
  TFile *f2 = TFile::Open("shtree_POSTLS161_pt100.root")
  t2 = (TTree *) f2->Get("neutronAna/CSCSimHitsTree");

  TCanvas c1("c1","c1",900,900);
  c1->SetBorderSize(0);
  c1->SetLeftMargin(0.13);
  c1->SetRightMargin(0.023);
  c1->SetTopMargin(0.081);
  c1->SetBottomMargin(0.13);
  c1->SetGridy();
  c1->SetTickx(1);
  c1->SetTicky(1);
  //c1->SetLogy(1);
  gStyle->SetOptStat(0);

  t1->Draw("sh.t>>htmp(200,0,50.)","","", 10000000, 0);
  htmp->SetTitle("SimHit TOF");
  htmp->SetXTitle("TOF, ns");
  htmp->SetYTitle("entries");
  htmp->SetLineWidth(2);
  t2->Draw("sh.t>>htmp2(200,0,50)","","same", 10000000, 0);
  htmp2->SetLineWidth(2);
  htmp2->SetLineStyle(7);
  htmp2->SetLineColor(kRed);
  scale = htmp->GetEntries()/(1.*htmp2->GetEntries());
  htmp2->Scale(scale);
  gPad->Modified();

  TLegend *leg1 = new TLegend(0.17,0.7,0.47,0.85,NULL,"brNDC");
  leg1->SetBorderSize(0);
  leg1->SetFillStyle(0);
  leg1->AddEntry(htmp,"6_0_0_patch1","l");
  leg1->AddEntry(htmp2,"POSTLS161","l");
  leg1->Draw();

  c1.Print("sh_tof" + ext)











  cvs co -r V50-00-00      CondFormats/CSCObjects
  V01-21-03-03   Configuration/StandardSequences
  V50-00-00      DataFormats/MuonDetId
  V50-00-00      L1Trigger/CSCTrackFinder
  V50-00-00      L1Trigger/CSCTriggerPrimitives
  V00-00-02      SLHCUpgradeSimulations/Configuration
  V00-00-04      SLHCUpgradeSimulations/Geometry


*/
