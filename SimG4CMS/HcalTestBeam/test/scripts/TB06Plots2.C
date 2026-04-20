
///////////////////////////////////////////////////////////////////////////////
//
// Analysis script to compare energy distribution of TB06 data with MC
//
// TB06Plots.C        Class to run over histograms created by TB06Analysis
//                    within the framewok of CMSSW.
//
//
///////////////////////////////////////////////////////////////////////////////

#include "TCanvas.h"
#include "TChain.h"
#include "TDirectory.h"
#include "TF1.h"
#include "TFile.h"
#include "TFitResult.h"
#include "TFitResultPtr.h"
#include "TGraphErrors.h"
#include "TGraphAsymmErrors.h"
#include "TH1D.h"
#include "TH2.h"
#include "THStack.h"
#include "TLegend.h"
#include "TMinuit.h"
#include "TMath.h"
#include "TPaveStats.h"
#include "TPaveText.h"
#include "TProfile.h"
#include "TROOT.h"
#include "TStyle.h"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// results
const int nnmin = -1;
const int verb = 2;
const double marsize = 1.5;
std::string pltype = ".png";
//std::string pltype = ".pdf";
//--------
const int nn = 2;
std::string rdir[nn] = {"CMSSW_16_1_0_pre4/FTFP_BERT_EMH_", "CMSSW_17_0_G4VECGEOM_X_2026-04-05-2300/FTFP_BERT_EMH_"};
std::string res[nn] = {"G4 11.2.2", "G4 11.4.1"};
//--------
int col[5] = {2, 4, 3, 6, 1};
int mar[5] = {21, 22, 23, 20, 28};

std::string partsF[6] = {"pi-", "p", "pi+", "kaon+", "kaon-", "pbar"};
std::string partsN[6] = {"#pi^{-}", "p", "#pi^{+}", "K^{+}", "K^{-}", "pbar"};

std::string tpin[15] = {"2", "3", "4", "5", "6", "7", "8", "9", "20", "30", "50", "100", "150", "200", "300"};
std::string tpip[9] = {"2", "3", "4", "5", "6", "7", "8", "9", "20"};
std::string tp[11] = {"2", "3", "4", "5", "6", "7", "8", "9", "20", "30", "350"};
std::string tpbar[9] = {"2", "2.5", "3", "4", "5", "6", "7", "8", "9"};
std::string tkp[8] = {"2", "3", "4", "5", "6", "7", "8", "9"};
std::string tkn[8] = {"2.5", "3", "4", "5", "6", "7", "8", "9"};

double ppin[15] = {2, 3, 4, 5, 6, 7, 8, 9, 20, 30, 50, 100, 150, 200, 300};
double ppip[9] = {2, 3, 4, 5, 6, 7, 8, 9, 20};
double pp[11] = {2, 3, 4, 5, 6, 7, 8, 9, 20, 30, 350};
double ppbar[9] = {2, 2.5, 3, 4, 5, 6, 7, 8, 9};
double pkp[8] = {2, 3, 4, 5, 6, 7, 8, 9};
double pkn[8] = {2.5, 3, 4, 5, 6, 7, 8, 9};

double spin[15] = {
    0.53, 0.57, 0.574, 0.615, 0.644, 0.647, 0.642, 0.666, 0.733, 0.761, 0.775, 0.798, 0.802, 0.808, 0.827};
double spip[9] = {0.58, 0.6, 0.62, 0.62, 0.64, 0.65, 0.66, 0.67, 0.74};
double sp[11] = {0.35, 0.413, 0.444, 0.468, 0.504, 0.517, 0.534, 0.578, 0.64, 0.687, 0.814};
double spbar[9] = {0.738, 0.738, 0.73, 0.7, 0.695, 0.69, 0.685, 0.68, 0.675};
double skp[8] = {0.36, 0.475, 0.58, 0.59, 0.63, 0.65, 0.62, 0.635};
double skn[8] = {0.63, 0.44, 0.59, 0.58, 0.61, 0.63, 0.63, 0.66};

double ww[15] = {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};
double wss[15] = {0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02};
double wpbar[9] = {0.07, 0.03, 0.025, 0.02, 0.02, 0.015, 0.015, 0.015, 0.015};
double wpip[9] = {0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01};
double wkp[8] = {0.12, 0.13, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02};
double wkn[8] = {0.1, 0.06, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02};

double rpin[15] = {
    0.987, 0.756, 0.63, 0.546, 0.491, 0.447, 0.421, 0.393, 0.274, 0.236, 0.195, 0.164, 0.151, 0.146, 0.132};
double rp[11] = {1.354, 0.908, 0.776, 0.642, 0.528, 0.483, 0.466, 0.45, 0.304, 0.235, 0.123};

double hpin[15] = {0.46, 0.585, 0.618, 0.7, 0.747, 0.746, 0.733, 0.773, 0.819, 0.835, 0.825, 0.832, 0.82, 0.827, 0.846};
double hp[11] = {0.298, 0.37, 0.44, 0.489, 0.569, 0.579, 0.607, 0.682, 0.748, 0.784, 0.834};

double whpin[15] = {
    1.186, 0.931, 0.701, 0.569, 0.492, 0.449, 0.424, 0.383, 0.251, 0.221, 0.202, 0.188, 0.184, 0.184, 0.166};
double whp[11] = {1.551, 1.038, 0.88, 0.722, 0.569, 0.504, 0.457, 0.444, 0.267, 0.226, 0.17};

double mpin[15] = {.68, .5, .42, .375, .36, .34, .345, .33, .328, .326, .323, .33, .32, .29, .26};
double mp[11] = {.835, .635, .49, 0.438, .39, 0.387, 0.375, 0.37, 0.3, 0.3, 0.285};

double xmin[6] = {1, 1, 1, 1.8, 1.8, 1.8};
double xmax[6] = {350, 400, 30, 11, 11, 11};
double ymin[6] = {0.4, 0.3, 0.4, 0.2, 0.2, 0.5};
double ymax[6] = {1, 0.9, 1, 0.82, 0.82, 0.9};
double rmin[6] = {0.9, 0.9, 0.94, 0.6, 0.5, 0.85};
double rmax[6] = {1.1, 1.1, 1.06, 1.4, 1.3, 1.15};

TFile* ff[1000];
TH1F* hh[1000];
TLegend* leg[1000];
TGraphErrors* gr[1000];
TGraph* grr[1000];

double vx[15];
double vmean[15];
double vrms[15];
double verr[15];
double rmean[15];
double rrms[15];
double rerr[15];
double rerrr[15];

double hmean[15];
double herr[15];
double hrmean[15];
double hverr[15];
double hrms[15];
double hrrms[15];
double hvrms[15];
double hrerr[15];

double mip[15];
double mipr[15];
double rmip[15];
double rmipr[15];

double qx[40];
double q1[40];
double q2[40];
double q3[40];
double q4[40];

double sn = std::sqrt(5000);
double sq2 = std::sqrt(2);

void PlotRMSArea(int np, double* mom, double* sig, double* err, double x1, double x2) {
  for (int j = 0; j < np; ++j) {
    qx[j + 1] = mom[j];
    double x = err[j] / sig[j];
    q2[j + 1] = 1. - x;
    q1[j + 1] = 1. + x;
  }
  qx[0] = x1;
  qx[np + 1] = x2;
  q1[0] = q1[1];
  q2[0] = q2[1];
  q1[np + 1] = q1[np];
  q2[np + 1] = q2[np];

  int np1 = 2 * np + 4;
  for (int i = 0; i < np + 2; ++i) {
    q3[i] = qx[i];
    q4[i] = q1[i];
    q3[np1 - i - 1] = qx[i];
    q4[np1 - i - 1] = q2[i];
  }
  q3[np1] = q3[0];
  q4[np1] = q4[0];
  TPolyLine* pline = new TPolyLine(np1 + 1, q3, q4);
  pline->SetFillColorAlpha(5, 0.5);
  pline->SetLineColor(6);
  pline->SetLineWidth(2);
  pline->Draw("f L SAME");

  TLine* line = new TLine(x1, 1.0, x2, 1.0);
  line->SetLineStyle(2);
  line->SetLineWidth(1);
  line->SetLineColor(kBlack);
  line->Draw("SAME 9");
}

void PlotMeanTotal(int ip,
                   int np,
                   double* mom,
                   double* sig,
                   double* err,
                   std::string* ttt,
                   double* x1,
                   double* x2,
                   double* x3,
                   double* x4) {
  TPad* pad = new TCanvas("c1", "c1", 700, 900);

  gStyle->SetPadBottomMargin(0.15);
  gStyle->SetPadTopMargin(0.10);
  gStyle->SetLabelSize(0.035, "x");
  gStyle->SetLabelSize(0.035, "y");
  gStyle->SetTickLength(0.06, "x");
  gStyle->SetTitleOffset(1.4, "x");
  gStyle->SetTitleOffset(1.6, "y");
  gStyle->SetTitleSize(0.03, "x");
  gStyle->SetTitleSize(0.03, "y");

  TPad* pad1 = new TPad(partsF[ip].c_str(), "pad1", 0, 0.3, 1, 1);
  pad1->SetBottomMargin(0.15);
  pad1->SetTopMargin(0.1);
  pad1->Update();
  pad1->Modified();
  pad1->Draw();
  pad1->cd();

  std::string tleg = "2006 Test Beam Data (" + partsN[ip] + ")";
  std::string title = "";
  gPad->SetLogx();
  hh[ip] = pad1->DrawFrame(xmin[ip], ymin[ip], xmax[ip], ymax[ip], title.c_str());
  hh[ip]->GetXaxis()->SetTitle("p_{Beam} (GeV/c)");
  hh[ip]->GetYaxis()->SetTitle("Mean of E_{Measured}/p_{Beam}");

  leg[999] = new TLegend(0.71, 0.75, 0.95, 0.895, "CMS Preliminary");
  leg[999]->SetBorderSize(0);
  leg[999]->SetMargin(0.01);

  leg[996] = new TLegend(0.20, 0.75, 0.60, 0.895, "Yellow band #pm 1 #sigma of data");
  leg[996]->SetBorderSize(0);
  leg[996]->SetMargin(0.01);

  leg[998] = new TLegend(0.18, 0.70, 0.71, 0.88, tleg.c_str());
  leg[998]->SetBorderSize(0);
  leg[998]->SetMargin(0.1);

  //leg[997] = new TLegend(0.28, 0.64, 0.56, 0.72, "FTFP_BERT_EMM");
  leg[997] = new TLegend(0.6, 0.2, 0.88, 0.4, "FTFP_BERT_EMM");
  leg[997]->SetBorderSize(0);
  leg[997]->SetMargin(0.1);

  leg[ip] = new TLegend(0.18, 0.74, 0.72, 0.88);
  leg[ip]->SetBorderSize(0);
  leg[ip]->SetMargin(0.15);

  gr[ip + 800] = new TGraphErrors(np, mom, sig, ww, err);
  gr[ip + 800]->SetMarkerColor(1);
  gr[ip + 800]->SetMarkerStyle(20);
  gr[ip + 800]->SetMarkerSize(marsize);
  gr[ip + 800]->Draw("P SAME 9");
  leg[ip]->AddEntry(gr[ip + 800], tleg.c_str(), "p");

  double fact = 1.0;

  for (int i = 0; i < nn; ++i) {
    int idx = i * 100;
    double xi2 = 0.;
    double del = 0.;
    double del2 = 0.;
    for (int j = 0; j < np; ++j) {
      std::string fname = rdir[i] + partsF[ip] + "_RR_" + ttt[j] + "gev.root";
      ff[idx + j] = new TFile(fname.c_str());
      if (!ff[idx + j])
        continue;
      TH1F* h1 = (TH1F*)ff[idx + j]->Get("testbeam/edepN");
      if (!h1) {
        cout << "Error open testbeam/edepN for i=" << i << " j= " << j << endl;
        continue;
      }
      vx[j] = mom[j] * fact;
      vmean[j] = h1->GetMean();
      vrms[j] = h1->GetRMS();
      verr[j] = vrms[j] / sn;
      rmean[j] = vmean[j] / sig[j];
      double w1 = verr[j] / vmean[j];
      double w2 = err[j] / sig[j];
      double w3 = rmean[j] * std::sqrt(w1 * w1 + w2 * w2);
      rerr[j] = rmean[j] * w1;
      double d1 = abs(1.0 - rmean[j]);
      del2 = max(del2, d1);
      d1 /= w3;
      del = max(del, d1);
      xi2 += d1 * d1;
      if (ip <= 1) {
        vrms[j] /= vmean[j];
        verr[j] /= vmean[j];
        rrms[j] = vrms[j] / x1[j];
        w1 = verr[j] / vrms[j];
        //w2 = err[j]/x1[j];
        //rerrr[j] = rrms[j]*std::sqrt(2*w1*w1 + w2*w2);
        rerrr[j] = rrms[j] * w1 * std::sqrt(2);

        if (nnmin <= i) {
          TH1F* h2 = (TH1F*)ff[idx + j]->Get("testbeam/emhcN");
          if (!h2) {
            cout << "Error open testbeam/emhcN for i= " << i << " j= " << j << endl;
            continue;
          }
          hmean[j] = h2->GetMean();
          hrms[j] = h2->GetRMS();
          double stat = sqrt(1 / h2->GetEntries());
          herr[j] = hrms[j] * stat;
          hrmean[j] = hmean[j] / x2[j];
          w1 = herr[j] / hmean[j];
          //w2 = err[j]/x2[j];
          herr[j] *= sq2;
          //hverr[j] = hrmean[j]*std::sqrt(w1*w1 + w2*w2);
          hverr[j] = hrmean[j] * w1;

          w1 = herr[j] / hrms[j];
          //w2 = err[j]/x4[j];
          hrms[j] /= hmean[j];
          hvrms[j] = hrms[j] / x4[j];
          hrrms[j] = hrms[j] * w1 * sq2;
          //hrerr[j] = hvrms[j]*std::sqrt(2*w1*w1 + w2*w2);
          hrerr[j] = hvrms[j] * w1 * std::sqrt(2);
        }

        TH1F* h3 = (TH1F*)ff[idx + j]->Get("testbeam/edecS");
        if (!h3) {
          cout << "Error open testbeam/edecS for i= " << i << " j= " << j << endl;
          continue;
        }
        double tot = h3->GetEntries();
        double ecal = 0.0;
        for (int k = 1; k <= 28; ++k) {
          ecal += h3->GetBinContent(k);
          //cout << k << ".  " << ecal << endl;
        }
        //cout << "=== Sum: ecal= " << ecal << "  tot= " << tot << "  " << ttt[j] << " GeV/c" << endl;
        mip[j] = ecal / tot;
        mipr[j] = sqrt(ecal) / tot;
        rmip[j] = mip[j] / x3[j];
        w1 = mipr[j] / mip[j];
        //w2 = err[j]/x3[j];
        //rmipr[j] = rmip[j]*std::sqrt(w1*w1 + w2*w2);
        rmipr[j] = rmip[j] * w1;
      }
    }
    xi2 /= (np - 1);
    cout << "Stat: Xi2= " << xi2 << " abs(del/RMS)max= " << del << " abs(del)max= " << del2 << "  " << res[i] << " "
         << partsN[ip] << endl;
    if (verb == 2) {
      cout << "====== ECAL+HCAL =====" << endl;
      for (int kk = 0; kk < np; ++kk) {
        cout << "p= " << vx[kk] << " sim= " << vmean[kk] << " tb= " << sig[kk] << " sim/tb= " << rmean[kk]
             << " rms= " << rerr[kk] << " experr= " << q1[kk + 1] << endl;
      }
      if (ip <= 1) {
        cout << "====== HCAL if MIP in ECAL =====" << endl;
        for (int kk = 0; kk < np; ++kk) {
          cout << "p= " << vx[kk] << " sim= " << hmean[kk] << " tb= " << x2[kk] << " sim/tb= " << hrmean[kk]
               << " rms= " << hverr[kk] << endl;
        }
        cout << "====== RMS =====" << endl;
        for (int kk = 0; kk < np; ++kk) {
          cout << "p= " << vx[kk] << " sim= " << vrms[kk] << " tb= " << x1[kk] << " sim/tb= " << rrms[kk]
               << " rms= " << rerrr[kk] << endl;
        }
        cout << "====== MIP =====" << endl;
        for (int kk = 0; kk < np; ++kk) {
          cout << "p= " << vx[kk] << " sim= " << mip[kk] << " tb= " << x3[kk] << " sim/tb= " << rmip[kk]
               << " rms= " << rmipr[kk] << endl;
        }
      }
    }
    gr[idx] = new TGraphErrors(np, vx, vmean, ww, verr);
    gr[idx + 1] = new TGraphErrors(np, vx, rmean, ww, rerr);
    if (ip <= 1) {
      gr[idx + 2] = new TGraphErrors(np, vx, vrms, ww, verr);
      gr[idx + 3] = new TGraphErrors(np, vx, rrms, ww, rerrr);
      gr[idx + 2]->SetMarkerColor(col[i]);
      gr[idx + 2]->SetLineColor(col[i]);
      gr[idx + 2]->SetMarkerStyle(mar[i]);
      gr[idx + 2]->SetMarkerSize(marsize);

      gr[idx + 3]->SetMarkerColor(col[i]);
      gr[idx + 3]->SetLineColor(col[i]);
      gr[idx + 3]->SetMarkerStyle(mar[i]);
      gr[idx + 3]->SetMarkerSize(marsize);

      gr[idx + 6] = new TGraphErrors(np, vx, mip, ww, mipr);
      gr[idx + 7] = new TGraphErrors(np, vx, rmip, ww, rmipr);
      gr[idx + 6]->SetMarkerColor(col[i]);
      gr[idx + 6]->SetMarkerStyle(mar[i]);
      gr[idx + 6]->SetMarkerSize(marsize);
      gr[idx + 7]->SetMarkerColor(col[i]);
      gr[idx + 7]->SetMarkerStyle(mar[i]);
      gr[idx + 7]->SetMarkerSize(marsize);

      if (nnmin <= i) {
        gr[idx + 14] = new TGraphErrors(np, vx, hmean, ww, herr);
        gr[idx + 15] = new TGraphErrors(np, vx, hrmean, ww, hverr);
        gr[idx + 16] = new TGraphErrors(np, vx, hrms, ww, hrrms);
        gr[idx + 17] = new TGraphErrors(np, vx, hvrms, ww, hrerr);

        gr[idx + 14]->SetMarkerColor(col[i]);
        gr[idx + 14]->SetLineColor(col[i]);
        gr[idx + 14]->SetMarkerStyle(mar[i]);
        gr[idx + 14]->SetMarkerSize(marsize);

        gr[idx + 15]->SetMarkerColor(col[i]);
        gr[idx + 15]->SetLineColor(col[i]);
        gr[idx + 15]->SetMarkerStyle(mar[i]);
        gr[idx + 15]->SetMarkerSize(marsize);

        gr[idx + 16]->SetMarkerColor(col[i]);
        gr[idx + 16]->SetLineColor(col[i]);
        gr[idx + 16]->SetMarkerStyle(mar[i]);
        gr[idx + 16]->SetMarkerSize(marsize);

        gr[idx + 17]->SetMarkerColor(col[i]);
        gr[idx + 17]->SetLineColor(col[i]);
        gr[idx + 17]->SetMarkerStyle(mar[i]);
        gr[idx + 17]->SetMarkerSize(marsize);
      }
    }
    gr[idx]->SetMarkerColor(col[i]);
    gr[idx]->SetMarkerStyle(mar[i]);
    gr[idx]->SetMarkerSize(marsize);
    gr[idx]->Draw("P SAME 9");
    leg[ip]->AddEntry(gr[idx], res[i].c_str(), "p");
    gr[idx + 1]->SetMarkerColor(col[i]);
    gr[idx + 1]->SetMarkerStyle(mar[i]);
    gr[idx + 1]->SetMarkerSize(marsize);
    fact *= 1.02;
  }
  leg[ip]->Draw("SAME 9");
  leg[999]->Draw("SAME 9");
  //leg[997]->Draw("SAME 9");
  pad1->Update();
  pad1->Modified();

  pad->cd();
  pad->Update();
  pad->Modified();

  gStyle->SetPadBottomMargin(0.3);
  gStyle->SetPadTopMargin(0.10);
  gStyle->SetLabelSize(0.07, "x");
  gStyle->SetLabelSize(0.07, "y");
  gStyle->SetTitleOffset(1.2, "x");
  gStyle->SetTitleOffset(0.7, "y");
  gStyle->SetTitleSize(0.07, "x");
  gStyle->SetTitleSize(0.07, "y");

  TPad* pad2 = new TPad(partsF[ip].c_str(), "pad2", 0, 0, 1, 0.3);
  pad2->SetBottomMargin(0.2);
  pad2->SetTopMargin(0.1);
  pad2->Draw();
  pad2->cd();

  gPad->SetLogx();
  hh[ip + 10] = pad2->DrawFrame(xmin[ip], rmin[ip], xmax[ip], rmax[ip], title.c_str());
  hh[ip + 10]->GetXaxis()->SetTitle("p_{Beam} (GeV/c)");
  hh[ip + 10]->GetYaxis()->SetTitle("MC/Data");

  PlotRMSArea(np, mom, sig, err, xmin[ip], xmax[ip]);

  for (int i = 0; i < nn; ++i) {
    int idx = i * 100;
    gr[idx + 1]->Draw("P SAME 9");
  }
  //leg[998]->Draw("SAME 9");
  leg[999]->Draw("SAME 9");
  leg[996]->Draw("SAME 9");
  pad2->Update();
  pad2->Modified();
  pad->cd();
  pad->Update();
  pad->Modified();
  std::string pl = "Amean_" + partsF[ip] + pltype;
  pad->Print(pl.c_str());
  delete pad;
}

void PlotRmsTotal(int ip, int np, double* mom, double* sig, double* err) {
  TPad* pad = new TCanvas("c2", "c2", 700, 900);

  gStyle->SetPadBottomMargin(0.15);
  gStyle->SetPadTopMargin(0.10);
  gStyle->SetLabelSize(0.035, "x");
  gStyle->SetLabelSize(0.035, "y");
  gStyle->SetTitleOffset(1.4, "x");
  gStyle->SetTitleOffset(1.6, "y");
  gStyle->SetTitleSize(0.03, "x");
  gStyle->SetTitleSize(0.03, "y");

  TPad* pad1 = new TPad(partsF[ip].c_str(), "pad1", 0, 0.3, 1, 1);
  pad1->SetBottomMargin(0.15);
  pad1->SetTopMargin(0.1);
  pad1->Update();
  pad1->Modified();
  pad1->Draw();
  pad1->cd();

  std::string tleg = "2006 Test Beam Data (" + partsN[ip] + ")";
  std::string title = "";
  gPad->SetLogx();
  hh[ip + 20] = pad1->DrawFrame(xmin[ip], 0.0, xmax[ip], 1.5, title.c_str());
  hh[ip + 20]->GetXaxis()->SetTitle("p_{Beam} (GeV/c)");
  hh[ip + 20]->GetYaxis()->SetTitle("Resolution RMS/E_{Mean}");

  gr[ip + 810] = new TGraphErrors(np, mom, sig, ww, err);
  gr[ip + 810]->SetMarkerColor(1);
  gr[ip + 810]->SetMarkerStyle(20);
  gr[ip + 810]->SetMarkerSize(marsize);
  gr[ip + 810]->Draw("P SAME 9");

  for (int i = 0; i < nn; ++i) {
    int idx = i * 100;
    gr[idx + 2]->Draw("P SAME 9");
  }

  leg[ip]->Draw("SAME 9");
  leg[999]->Draw("SAME 9");
  pad1->Update();
  pad1->Modified();

  pad->cd();
  pad->Update();
  pad->Modified();

  gStyle->SetPadBottomMargin(0.3);
  gStyle->SetPadTopMargin(0.10);
  gStyle->SetLabelSize(0.07, "x");
  gStyle->SetLabelSize(0.07, "y");
  gStyle->SetTitleOffset(1.2, "x");
  gStyle->SetTitleOffset(0.7, "y");
  gStyle->SetTitleSize(0.07, "x");
  gStyle->SetTitleSize(0.07, "y");

  TPad* pad2 = new TPad(partsF[ip].c_str(), "pad2", 0, 0, 1, 0.3);
  pad2->SetBottomMargin(0.2);
  pad2->SetTopMargin(0.1);
  pad2->Draw();
  pad2->cd();

  gPad->SetLogx();
  hh[ip + 30] = pad2->DrawFrame(xmin[ip], 0.7, xmax[ip], 1.3, title.c_str());
  hh[ip + 30]->GetXaxis()->SetTitle("p_{Beam} (GeV/c)");
  hh[ip + 30]->GetYaxis()->SetTitle("MC/Data");

  PlotRMSArea(np, mom, sig, err, xmin[ip], xmax[ip]);
  for (int i = 0; i < nn; ++i) {
    int idx = i * 100;
    gr[idx + 3]->Draw("P SAME 9");
  }

  //leg[998]->Draw("SAME 9");
  leg[999]->Draw("SAME 9");
  leg[996]->Draw("SAME 9");
  pad2->Update();
  pad2->Modified();
  pad->cd();
  pad->Update();
  pad->Modified();
  std::string pl = "Arms_" + partsF[ip] + pltype;
  pad->Print(pl.c_str());
  delete pad;
}

void PlotMeanHcal(int ip, int np, double* mom, double* sig, double* err) {
  TPad* pad = new TCanvas("c3", "c3", 700, 900);

  gStyle->SetPadBottomMargin(0.15);
  gStyle->SetPadTopMargin(0.10);
  gStyle->SetLabelSize(0.035, "x");
  gStyle->SetLabelSize(0.035, "y");
  gStyle->SetTitleOffset(1.4, "x");
  gStyle->SetTitleOffset(1.6, "y");
  gStyle->SetTitleSize(0.03, "x");
  gStyle->SetTitleSize(0.03, "y");

  TPad* pad1 = new TPad(partsF[ip].c_str(), "pad1", 0, 0.3, 1, 1);
  pad1->SetBottomMargin(0.15);
  pad1->SetTopMargin(0.1);
  pad1->Update();
  pad1->Modified();
  pad1->Draw();
  pad1->cd();

  std::string tleg = "2006 Test Beam Data (" + partsN[ip] + ")";
  std::string title = "";
  gPad->SetLogx();
  hh[ip + 40] = pad1->DrawFrame(xmin[ip], 0.0, xmax[ip], 1.2, title.c_str());
  hh[ip + 40]->GetXaxis()->SetTitle("p_{Beam} (GeV/c)");
  hh[ip + 40]->GetYaxis()->SetTitle("Mean of E_{Measured}/p_{Beam} (MIP Ecal)");

  gr[ip + 820] = new TGraphErrors(np, mom, sig, ww, err);
  gr[ip + 820]->SetMarkerColor(1);
  gr[ip + 820]->SetMarkerStyle(20);
  gr[ip + 820]->SetMarkerSize(marsize);
  gr[ip + 820]->Draw("P SAME");

  for (int i = 0; i < nn; ++i) {
    if (i < nnmin)
      continue;
    int idx = i * 100 + 14;
    gr[idx]->Draw("P SAME 9");
  }

  leg[ip]->Draw("SAME 9");
  leg[999]->Draw("SAME 9");
  pad1->Update();
  pad1->Modified();

  pad->cd();
  pad->Update();
  pad->Modified();

  gStyle->SetPadBottomMargin(0.3);
  gStyle->SetPadTopMargin(0.10);
  gStyle->SetLabelSize(0.07, "x");
  gStyle->SetLabelSize(0.07, "y");
  gStyle->SetTitleOffset(1.2, "x");
  gStyle->SetTitleOffset(0.7, "y");
  gStyle->SetTitleSize(0.07, "x");
  gStyle->SetTitleSize(0.07, "y");

  TPad* pad2 = new TPad(partsF[ip].c_str(), "pad2", 0, 0, 1, 0.3);
  pad2->SetBottomMargin(0.2);
  pad2->SetTopMargin(0.1);
  pad2->Draw();
  pad2->cd();

  gPad->SetLogx();
  hh[ip + 50] = pad2->DrawFrame(xmin[ip], 0.7, xmax[ip], 1.4, title.c_str());
  hh[ip + 50]->GetXaxis()->SetTitle("p_{Beam} (GeV/c)");
  hh[ip + 50]->GetYaxis()->SetTitle("MC/Data");

  PlotRMSArea(np, mom, sig, err, xmin[ip], xmax[ip]);

  for (int i = 0; i < nn; ++i) {
    if (i < nnmin)
      continue;
    int idx = i * 100 + 15;
    gr[idx]->Draw("P SAME 9");
  }
  //leg[998]->Draw("SAME 9");
  leg[999]->Draw("SAME 9");
  leg[996]->Draw("SAME 9");
  pad2->Update();
  pad2->Modified();
  pad->cd();
  pad->Update();
  pad->Modified();
  std::string pl = "Ahcal_" + partsF[ip] + pltype;
  pad->Print(pl.c_str());
  delete pad;
}

void PlotRmsHcal(int ip, int np, double* mom, double* sig, double* err) {
  TPad* pad = new TCanvas("c5", "c5", 700, 900);

  gStyle->SetPadBottomMargin(0.15);
  gStyle->SetPadTopMargin(0.10);
  gStyle->SetLabelSize(0.035, "x");
  gStyle->SetLabelSize(0.035, "y");
  gStyle->SetTitleOffset(1.4, "x");
  gStyle->SetTitleOffset(1.6, "y");
  gStyle->SetTitleSize(0.03, "x");
  gStyle->SetTitleSize(0.03, "y");

  TPad* pad1 = new TPad(partsF[ip].c_str(), "pad1", 0, 0.3, 1, 1);
  pad1->SetBottomMargin(0.15);
  pad1->SetTopMargin(0.1);
  pad1->Update();
  pad1->Modified();
  pad1->Draw();
  pad1->cd();

  std::string tleg = "2006 Test Beam Data (" + partsN[ip] + ")";
  std::string title = "";
  gPad->SetLogx();
  hh[ip + 40] = pad1->DrawFrame(xmin[ip], 0.0, xmax[ip], 2.0, title.c_str());
  hh[ip + 40]->GetXaxis()->SetTitle("p_{Beam} (GeV/c)");
  hh[ip + 40]->GetYaxis()->SetTitle("Resolution RMS/E_{Mean} (MIP Ecal)");

  gr[ip + 830] = new TGraphErrors(np, mom, sig, ww, err);
  gr[ip + 830]->SetMarkerColor(1);
  gr[ip + 830]->SetMarkerStyle(20);
  gr[ip + 830]->SetMarkerSize(marsize);
  gr[ip + 830]->Draw("P SAME 9");

  for (int i = 0; i < nn; ++i) {
    if (i < nnmin)
      continue;
    int idx = i * 100 + 16;
    gr[idx]->Draw("P SAME 9");
  }

  leg[ip]->Draw("SAME 9");
  leg[999]->Draw("SAME 9");
  pad1->Update();
  pad1->Modified();

  pad->cd();
  pad->Update();
  pad->Modified();

  gStyle->SetPadBottomMargin(0.3);
  gStyle->SetPadTopMargin(0.10);
  gStyle->SetLabelSize(0.07, "x");
  gStyle->SetLabelSize(0.07, "y");
  gStyle->SetTitleOffset(1.2, "x");
  gStyle->SetTitleOffset(0.7, "y");
  gStyle->SetTitleSize(0.07, "x");
  gStyle->SetTitleSize(0.07, "y");

  TPad* pad2 = new TPad(partsF[ip].c_str(), "pad2", 0, 0, 1, 0.3);
  pad2->SetBottomMargin(0.2);
  pad2->SetTopMargin(0.1);
  pad2->Draw();
  pad2->cd();

  gPad->SetLogx();
  hh[ip + 50] = pad2->DrawFrame(xmin[ip], 0.7, xmax[ip], 1.4, title.c_str());
  hh[ip + 50]->GetXaxis()->SetTitle("p_{Beam} (GeV/c)");
  hh[ip + 50]->GetYaxis()->SetTitle("MC/Data");

  PlotRMSArea(np, mom, sig, err, xmin[ip], xmax[ip]);

  for (int i = 0; i < nn; ++i) {
    if (i < nnmin)
      continue;
    int idx = i * 100 + 17;
    gr[idx]->Draw("P SAME 9");
  }
  //leg[998]->Draw("SAME 9");
  leg[999]->Draw("SAME 9");
  leg[996]->Draw("SAME 9");
  pad2->Update();
  pad2->Modified();
  pad->cd();
  pad->Update();
  pad->Modified();
  std::string pl = "Ahrms_" + partsF[ip] + pltype;
  pad->Print(pl.c_str());
  delete pad;
}

void PlotMIP(int ip, int np, double* mom, double* sig, double* err) {
  TPad* pad = new TCanvas("c4", "c4", 700, 900);

  gStyle->SetPadBottomMargin(0.15);
  gStyle->SetPadTopMargin(0.10);
  gStyle->SetLabelSize(0.035, "x");
  gStyle->SetLabelSize(0.035, "y");
  gStyle->SetTitleOffset(1.4, "x");
  gStyle->SetTitleOffset(1.6, "y");
  gStyle->SetTitleSize(0.03, "x");
  gStyle->SetTitleSize(0.03, "y");

  TPad* pad1 = new TPad(partsF[ip].c_str(), "pad1", 0, 0.3, 1, 1);
  pad1->SetBottomMargin(0.15);
  pad1->SetTopMargin(0.1);
  pad1->Update();
  pad1->Modified();
  pad1->Draw();
  pad1->cd();

  std::string tleg = "2006 Test Beam Data (" + partsN[ip] + ")";
  std::string title = "";
  gPad->SetLogx();
  hh[ip + 60] = pad1->DrawFrame(xmin[ip], 0.0, xmax[ip], 1.0, title.c_str());
  hh[ip + 60]->GetXaxis()->SetTitle("p_{Beam} (GeV/c)");
  hh[ip + 60]->GetYaxis()->SetTitle("MIP Fraction");

  gr[ip + 840] = new TGraphErrors(np, mom, sig, ww, err);
  gr[ip + 840]->SetMarkerColor(1);
  gr[ip + 840]->SetMarkerStyle(20);
  gr[ip + 840]->SetMarkerSize(marsize);
  gr[ip + 840]->Draw("P SAME 9");

  for (int i = 0; i < nn; ++i) {
    int idx = i * 100;
    gr[idx + 6]->Draw("P SAME 9");
  }

  leg[ip]->Draw("SAME 9");
  leg[999]->Draw("SAME 9");
  pad1->Update();
  pad1->Modified();

  pad->cd();
  pad->Update();
  pad->Modified();

  gStyle->SetPadBottomMargin(0.3);
  gStyle->SetPadTopMargin(0.10);
  gStyle->SetLabelSize(0.07, "x");
  gStyle->SetLabelSize(0.07, "y");
  gStyle->SetTitleOffset(1.2, "x");
  gStyle->SetTitleOffset(0.7, "y");
  gStyle->SetTitleSize(0.07, "x");
  gStyle->SetTitleSize(0.07, "y");

  TPad* pad2 = new TPad(partsF[ip].c_str(), "pad2", 0, 0, 1, 0.3);
  pad2->SetBottomMargin(0.2);
  pad2->SetTopMargin(0.1);
  pad2->Draw();
  pad2->cd();

  gPad->SetLogx();
  hh[ip + 70] = pad2->DrawFrame(xmin[ip], 0.7, xmax[ip], 1.4, title.c_str());
  hh[ip + 70]->GetXaxis()->SetTitle("p_{Beam} (GeV/c)");
  hh[ip + 70]->GetYaxis()->SetTitle("MC/Data");

  PlotRMSArea(np, mom, sig, err, xmin[ip], xmax[ip]);

  for (int i = 0; i < nn; ++i) {
    int idx = i * 100;
    gr[idx + 7]->Draw("P SAME 9");
  }
  //  leg[998]->Draw("SAME 9");
  leg[999]->Draw("SAME 9");
  leg[996]->Draw("SAME 9");
  pad2->Update();
  pad2->Modified();
  pad->cd();
  pad->Update();
  pad->Modified();
  std::string pl = "Amip_" + partsF[ip] + pltype;
  pad->Print(pl.c_str());
  delete pad;
}

void TB06Plots2() {
  gROOT->SetStyle("Plain");
  gStyle->SetPadBottomMargin(0.15);
  gStyle->SetPadTopMargin(0.10);
  gStyle->SetPadLeftMargin(0.15);
  gStyle->SetPadRightMargin(0.05);
  gStyle->SetPadBorderMode(0);

  PlotMeanTotal(0, 15, ppin, spin, wss, tpin, rpin, hpin, mpin, whpin);
  PlotRmsTotal(0, 15, ppin, rpin, wss);
  PlotMeanHcal(0, 15, ppin, hpin, wss);
  PlotRmsHcal(0, 15, ppin, whpin, wss);
  PlotMIP(0, 15, ppin, mpin, wss);

  PlotMeanTotal(1, 11, pp, sp, wss, tp, rp, hp, mp, whp);
  PlotRmsTotal(1, 11, pp, rp, wss);
  PlotMeanHcal(1, 11, pp, hp, wss);
  PlotRmsHcal(1, 11, pp, whp, wss);
  PlotMIP(1, 11, pp, mp, wss);

  PlotMeanTotal(2, 9, ppip, spip, wpip, tpip, ww, ww, ww, ww);

  PlotMeanTotal(3, 8, pkp, skp, wkp, tkp, ww, ww, ww, ww);
  PlotMeanTotal(4, 8, pkn, skn, wkn, tkn, ww, ww, ww, ww);
  PlotMeanTotal(5, 9, ppbar, spbar, wpbar, tpbar, ww, ww, ww, ww);
}
