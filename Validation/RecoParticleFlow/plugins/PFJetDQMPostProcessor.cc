// -*- C++ -*-
//
// Package:    Validation/RecoParticleFlow
// Class:      PFJetDQMPostProcessor.cc
//
// Main Developer:   "Juska Pekkanen"
// Original Author:  "Kenichi Hatakeyama"
//

#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "TCanvas.h"

//
// class declaration
//

class PFJetDQMPostProcessor : public DQMEDHarvester {
public:
  explicit PFJetDQMPostProcessor(const edm::ParameterSet&);
  ~PFJetDQMPostProcessor() override;

private:
  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;
  void fitResponse(TH1F* hreso,
                   TH1F* h_genjet_pt,
                   int ptbinlow,
                   int ietahigh,
                   double recoptcut,
                   double& resp,
                   double& resp_err,
                   double& reso,
                   double& reso_err);
  double getRespUnc(double width, double width_err, double mean, double mean_err);

  std::vector<std::string> jetResponseDir;
  std::string genjetDir;
  std::vector<double> ptBins;
  std::vector<double> etaBins;

  double recoptcut;

  bool debug = false;

  std::string seta(double eta) {
    std::string seta = std::to_string(int(eta * 10.));
    if (seta.length() < 2)
      seta = "0" + seta;
    return seta;
  }

  std::string spt(double ptmin, double ptmax) {
    std::string spt = std::to_string(int(ptmin)) + "_" + std::to_string(int(ptmax));
    return spt;
  }
};

// Some switches
//
// constuctors and destructor
//
PFJetDQMPostProcessor::PFJetDQMPostProcessor(const edm::ParameterSet& iConfig) {
  jetResponseDir = iConfig.getParameter<std::vector<std::string>>("jetResponseDir");
  genjetDir = iConfig.getParameter<std::string>("genjetDir");
  ptBins = iConfig.getParameter<std::vector<double>>("ptBins");
  etaBins = iConfig.getParameter<std::vector<double>>("etaBins");
  recoptcut = iConfig.getParameter<double>("recoPtCut");
}

PFJetDQMPostProcessor::~PFJetDQMPostProcessor() {}

// ------------ method called right after a run ends ------------
void PFJetDQMPostProcessor::dqmEndJob(DQMStore::IBooker& ibook_, DQMStore::IGetter& iget_) {
  for (unsigned int idir = 0; idir < jetResponseDir.size(); idir++) {
    iget_.setCurrentFolder(genjetDir);
    std::vector<std::string> sME_genjets = iget_.getMEs();
    std::for_each(sME_genjets.begin(), sME_genjets.end(), [&](auto& s) { s.insert(0, genjetDir); });
    //for (unsigned int i=0; i<sME_genjets.size(); i++) std::cout << sME_genjets[i] << std::endl;

    iget_.setCurrentFolder(jetResponseDir[idir]);
    std::vector<std::string> sME_response = iget_.getMEs();
    std::for_each(sME_response.begin(), sME_response.end(), [&](auto& s) { s.insert(0, jetResponseDir[idir]); });
    //for (unsigned int i=0; i<sME_response.size(); i++) std::cout << sME_response[i] << std::endl;

    iget_.setCurrentFolder(jetResponseDir[idir]);

    double ptBinsArray[ptBins.size()];
    unsigned int nPtBins = ptBins.size() - 1;
    std::copy(ptBins.begin(), ptBins.end(), ptBinsArray);
    //for(unsigned int ipt = 0; ipt < ptBins.size(); ++ipt) std::cout << ptBins[ipt] << std::endl;

    std::string stitle;
    char ctitle[50];
    std::vector<MonitorElement*> vME_presponse;
    std::vector<MonitorElement*> vME_preso;
    std::vector<MonitorElement*> vME_preso_rms;

    MonitorElement* me;
    TH1F* h_resp;
    TH1F* h_genjet_pt;

    //
    // Response distributions
    //
    for (unsigned int ieta = 1; ieta < etaBins.size(); ++ieta) {
      stitle = genjetDir + "genjet_pt" + "_eta" + seta(etaBins[ieta]);
      //std::cout << ieta << " " << stitle << std::endl;

      std::vector<std::string>::const_iterator it = std::find(sME_genjets.begin(), sME_genjets.end(), stitle);
      if (it == sME_genjets.end())
        continue;
      me = iget_.get(stitle);
      h_genjet_pt = (TH1F*)me->getTH1F();

      stitle = "presponse_eta" + seta(etaBins[ieta]);
      // adding "Raw" to the title of raw jet response histograms
      if (jetResponseDir[idir].find("noJEC") != std::string::npos) {
        sprintf(ctitle, "Raw Jet pT response, %4.1f<|#eta|<%4.1f", etaBins[ieta - 1], etaBins[ieta]);
      } else {
        sprintf(ctitle, "Jet pT response, %4.1f<|#eta|<%4.1f", etaBins[ieta - 1], etaBins[ieta]);
      }
      TH1F* h_presponse = new TH1F(stitle.c_str(), ctitle, nPtBins, ptBinsArray);

      stitle = "preso_eta" + seta(etaBins[ieta]);
      if (jetResponseDir[idir].find("noJEC") != std::string::npos) {
        sprintf(ctitle, "Raw Jet pT resolution, %4.1f<|#eta|<%4.1f", etaBins[ieta - 1], etaBins[ieta]);
      } else {
        sprintf(ctitle, "Jet pT resolution, %4.1f<|#eta|<%4.1f", etaBins[ieta - 1], etaBins[ieta]);
      }
      TH1F* h_preso = new TH1F(stitle.c_str(), ctitle, nPtBins, ptBinsArray);

      stitle = "preso_eta" + seta(etaBins[ieta]) + "_rms";
      if (jetResponseDir[idir].find("noJEC") != std::string::npos) {
        sprintf(ctitle, "Raw Jet pT resolution using RMS, %4.1f<|#eta|<%4.1f", etaBins[ieta - 1], etaBins[ieta]);
      } else {
        sprintf(ctitle, "Jet pT resolution using RMS, %4.1f<|#eta|<%4.1f", etaBins[ieta - 1], etaBins[ieta]);
      }
      TH1F* h_preso_rms = new TH1F(stitle.c_str(), ctitle, nPtBins, ptBinsArray);

      for (unsigned int ipt = 0; ipt < ptBins.size() - 1; ++ipt) {
        stitle = jetResponseDir[idir] + "reso_dist_" + spt(ptBins[ipt], ptBins[ipt + 1]) + "_eta" + seta(etaBins[ieta]);
        std::vector<std::string>::const_iterator it = std::find(sME_response.begin(), sME_response.end(), stitle);
        if (it == sME_response.end())
          continue;
        me = iget_.get(stitle);
        h_resp = (TH1F*)me->getTH1F();

        // Fit-based
        double resp = 1.0, resp_err = 0.0, reso = 0.0, reso_err = 0.0;
        fitResponse(h_resp, h_genjet_pt, ipt, ieta, recoptcut, resp, resp_err, reso, reso_err);

        h_presponse->SetBinContent(ipt + 1, resp);
        h_presponse->SetBinError(ipt + 1, resp_err);
        h_preso->SetBinContent(ipt + 1, reso);
        h_preso->SetBinError(ipt + 1, reso_err);

        // RMS-based
        double std = h_resp->GetStdDev();
        double std_error = h_resp->GetStdDevError();

        // Scale each bin with mean response
        double mean = 1.0;
        double mean_error = 0.0;
        double err = 0.0;
        if (h_resp->GetMean() > 0) {
          mean = h_resp->GetMean();
          mean_error = h_resp->GetMeanError();

          // Scale resolution by response.
          std /= mean;
          std_error /= mean;

          err = getRespUnc(std, std_error, mean, mean_error);
        }

        h_preso_rms->SetBinContent(ipt + 1, std);
        h_preso_rms->SetBinError(ipt + 1, err);

      }  // ipt

      stitle = "presponse_eta" + seta(etaBins[ieta]);
      me = ibook_.book1D(stitle.c_str(), h_presponse);
      vME_presponse.push_back(me);

      stitle = "preso_eta" + seta(etaBins[ieta]);
      me = ibook_.book1D(stitle.c_str(), h_preso);
      vME_preso.push_back(me);

      stitle = "preso_eta" + seta(etaBins[ieta]) + "_rms";
      me = ibook_.book1D(stitle.c_str(), h_preso_rms);
      vME_preso_rms.push_back(me);

    }  // ieta

    //
    // Checks
    //
    if (debug) {
      for (std::vector<MonitorElement*>::const_iterator i = vME_presponse.begin(); i != vME_presponse.end(); ++i)
        (*i)->getTH1F()->Print();
      for (std::vector<MonitorElement*>::const_iterator i = vME_preso.begin(); i != vME_preso.end(); ++i)
        (*i)->getTH1F()->Print();
      for (std::vector<MonitorElement*>::const_iterator i = vME_preso_rms.begin(); i != vME_preso_rms.end(); ++i)
        (*i)->getTH1F()->Print();
    }
  }
}

void PFJetDQMPostProcessor::fitResponse(TH1F* hreso,
                                        TH1F* h_genjet_pt,
                                        int ptbinlow,
                                        int ietahigh,
                                        double recoptcut,
                                        double& resp,
                                        double& resp_err,
                                        double& reso,
                                        double& reso_err) {
  // This 'smartfitter' is converted from the original Python smart_fit() -function
  // implemented in test/helperFunctions.py. See that file for more commentary.
  // Juska 23 May 2019

  // Only do plots if needed for debugging
  // NOTE a directory called 'debug' must exist in the working directory
  //
  bool doPlots = false;

  double ptlow = ptBins[ptbinlow];
  double pthigh = ptBins[ptbinlow + 1];

  // Take range by Mikko's advice: -1.5 and + 1.5 * RMS width

  double rmswidth = hreso->GetStdDev();
  double rmsmean = hreso->GetMean();
  double fitlow = rmsmean - 1.5 * rmswidth;
  fitlow = TMath::Max(recoptcut / ptlow, fitlow);
  double fithigh = rmsmean + 1.5 * rmswidth;

  TF1* fg = new TF1("mygaus", "gaus", fitlow, fithigh);
  TF1* fg2 = new TF1("fg2", "TMath::Gaus(x,[0],[1],true)*[2]", fitlow, fithigh);

  hreso->Fit("mygaus", "RQN");

  fg2->SetParameter(0, fg->GetParameter(1));
  fg2->SetParameter(1, fg->GetParameter(2));

  // Extract ngenjet in the current pT bin from the genjet histo
  float ngenjet = h_genjet_pt->GetBinContent(ptbinlow + 1);

  // Here the fit is forced to take the area of ngenjets.
  // The area is further normalized for the response histogram x-axis length
  // (3) and number of bins (100)
  fg2->FixParameter(2, ngenjet * 3. / 100.);

  hreso->Fit("fg2", "RQN");

  fitlow = fg2->GetParameter(0) - 1.5 * fg2->GetParameter(1);
  fitlow = TMath::Max(15. / ptlow, fitlow);
  fithigh = fg2->GetParameter(0) + 1.5 * fg2->GetParameter(1);

  fg2->SetRange(fitlow, fithigh);

  hreso->Fit("fg2", "RQ");

  fg->SetRange(0, 3);
  fg2->SetRange(0, 3);
  fg->SetLineWidth(2);
  fg2->SetLineColor(kGreen + 2);

  hreso->GetXaxis()->SetRangeUser(0, 2);

  // Save plots to a subdirectory if asked (debug-directory must exist!)
  if (doPlots & (hreso->GetEntries() > 0)) {
    TCanvas* cfit = new TCanvas(Form("respofit_%i", int(ptlow)), "respofit", 600, 600);
    hreso->Draw("ehist");
    fg->Draw("same");
    fg2->Draw("same");
    cfit->SaveAs(
        Form("debug/respo_smartfit_%04d_%i_eta%s.pdf", (int)ptlow, (int)pthigh, seta(etaBins[ietahigh]).c_str()));
  }

  resp = fg2->GetParameter(0);
  resp_err = fg2->GetParError(0);

  // Scale resolution by response. Avoid division by zero.
  if (0 == resp)
    reso = 0;
  else
    reso = fg2->GetParameter(1) / resp;
  reso_err = fg2->GetParError(1) / resp;

  reso_err = getRespUnc(reso, reso_err, resp, resp_err);
}

// Calculate resolution uncertainty
double PFJetDQMPostProcessor::getRespUnc(double width, double width_err, double mean, double mean_err) {
  if (0 == width || 0 == mean)
    return 0;
  return TMath::Sqrt(pow(width_err, 2) / pow(width, 2) + pow(mean_err, 2) / pow(mean, 2)) * width;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFJetDQMPostProcessor);
