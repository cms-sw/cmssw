#ifndef Validation_RecoVertex_SVResolutionBundle_h
#define Validation_RecoVertex_SVResolutionBundle_h

// Package:    Validation/RecoVertex
// Class:      SVResolutionBundle
//
/**\class SVResolutionBundle Validation/RecoVertex/interface/SVResolutionBundle.h

 Description: Bundle of monitoring elements for the SecondaryVertexAnalyzer,
              grouping residual and pull histograms for a single vertex
              quantity at one place (e.g. x-position residuals and pulls vs
              decay length, eta, and nTracks).

 Original Author: Jan Schulz
*/

#include "DQMServices/Core/interface/DQMBookingHelpers.h"

// =============================================================================
// SVResolutionBundle
//
// Generic resolution bundle: residuals and pulls of any vertex quantity
// vs decay length, transverse radius, pseudorapidity, pt, and nTracks.
// =============================================================================

class SVResolutionBundle {
public:
  using IBooker = dqm::reco::DQMStore::IBooker;

  struct BinConfig {
    struct Bins {
      int nBins;
      double min, max;
    };

    Bins decayLength;    // decay length axis
    Bins decayLength2D;  // transverse decay length (2D) axis
    Bins eta;            // eta axis
    Bins pt;             // pt axis
    Bins nTracks;        // track multiplicity axis
  };

  SVResolutionBundle() = default;
  virtual ~SVResolutionBundle() = default;

  // Fill all six residual/pull histograms for one matched reco-sim pair.
  void fill(const double decayLength,
            const double decayLength2D,
            const double eta,
            const double pt,
            const int nTracks,
            const double residual,
            const double pull) {
    if (h_res)
      h_res->Fill(residual);
    if (h_res_vs_decayLength)
      h_res_vs_decayLength->Fill(decayLength, residual);
    if (h_res_vs_decayLength2D)
      h_res_vs_decayLength2D->Fill(decayLength2D, residual);
    if (h_res_vs_eta)
      h_res_vs_eta->Fill(eta, residual);
    if (h_res_vs_pt)
      h_res_vs_pt->Fill(pt, residual);
    if (h_res_vs_nTracks)
      h_res_vs_nTracks->Fill(nTracks, residual);
    if (h_pull)
      h_pull->Fill(pull);
    if (h_pull_vs_decayLength)
      h_pull_vs_decayLength->Fill(decayLength, pull);
    if (h_pull_vs_decayLength2D)
      h_pull_vs_decayLength2D->Fill(decayLength2D, pull);
    if (h_pull_vs_eta)
      h_pull_vs_eta->Fill(eta, pull);
    if (h_pull_vs_pt)
      h_pull_vs_pt->Fill(pt, pull);
    if (h_pull_vs_nTracks)
      h_pull_vs_nTracks->Fill(nTracks, pull);
  }

  // Book all histograms for this resolution bundle.
  void bookResolutions(IBooker &ibooker,
                       // Common bins settings
                       const BinConfig &bins,
                       // residual axis
                       const std::string &name,
                       const int resNBins,
                       const double resMin,
                       const double resMax) {
    h_res = dqm::booking::book1DIfLogX(
        ibooker, false, (name + "_res").c_str(), (name + " residuals;Residuals").c_str(), resNBins, resMin, resMax);
    h_res_vs_decayLength =
        dqm::booking::book2DIfLogX(ibooker,
                                   false,
                                   (name + "_res_vs_decayLength").c_str(),
                                   (name + " residuals vs decay length;Decay length L_{3D} [cm];Residuals").c_str(),
                                   bins.decayLength.nBins,
                                   bins.decayLength.min,
                                   bins.decayLength.max,
                                   resNBins,
                                   resMin,
                                   resMax);
    h_res_vs_decayLength2D = dqm::booking::book2DIfLogX(
        ibooker,
        false,
        (name + "_res_vs_decayLength2D").c_str(),
        (name + " residuals vs transverse decay length (2D);Transverse decay length L_{2D} [cm];Residuals").c_str(),
        bins.decayLength2D.nBins,
        bins.decayLength2D.min,
        bins.decayLength2D.max,
        resNBins,
        resMin,
        resMax);
    h_res_vs_eta = dqm::booking::book2DIfLogX(ibooker,
                                              false,
                                              (name + "_res_vs_eta").c_str(),
                                              (name + " residuals vs SV eta;SimVertex #eta;Residuals").c_str(),
                                              bins.eta.nBins,
                                              bins.eta.min,
                                              bins.eta.max,
                                              resNBins,
                                              resMin,
                                              resMax);
    h_res_vs_pt = dqm::booking::book2DIfLogX(ibooker,
                                             true,
                                             (name + "_res_vs_pt").c_str(),
                                             (name + " residuals vs SV pt;SimVertex p_T;Residuals").c_str(),
                                             bins.pt.nBins,
                                             bins.pt.min,
                                             bins.pt.max,
                                             resNBins,
                                             resMin,
                                             resMax);
    h_res_vs_nTracks =
        dqm::booking::book2DIfLogX(ibooker,
                                   false,
                                   (name + "_res_vs_nTracks").c_str(),
                                   (name + " residuals vs track multiplicity;N tracks;Residuals").c_str(),
                                   bins.nTracks.nBins,
                                   bins.nTracks.min,
                                   bins.nTracks.max,
                                   resNBins,
                                   resMin,
                                   resMax);
    h_pull = dqm::booking::book1DIfLogX(
        ibooker, false, (name + "_pull").c_str(), (name + " pulls;Pulls").c_str(), 100, -10., 10.);
    h_pull_vs_decayLength =
        dqm::booking::book2DIfLogX(ibooker,
                                   false,
                                   (name + "_pull_vs_decayLength").c_str(),
                                   (name + " pulls vs decay length;Decay length L_{3D} [cm];Pulls").c_str(),
                                   bins.decayLength.nBins,
                                   bins.decayLength.min,
                                   bins.decayLength.max,
                                   100,
                                   -10.,
                                   10.);
    h_pull_vs_decayLength2D = dqm::booking::book2DIfLogX(
        ibooker,
        false,
        (name + "_pull_vs_decayLength2D").c_str(),
        (name + " pulls vs transverse decay length (2D);Transverse decay length L_{2D} [cm];Pulls").c_str(),
        bins.decayLength2D.nBins,
        bins.decayLength2D.min,
        bins.decayLength2D.max,
        100,
        -10.,
        10.);
    h_pull_vs_eta = dqm::booking::book2DIfLogX(ibooker,
                                               false,
                                               (name + "_pull_vs_eta").c_str(),
                                               (name + " pulls vs vertex eta;SimVertex #eta;Pulls").c_str(),
                                               bins.eta.nBins,
                                               bins.eta.min,
                                               bins.eta.max,
                                               100,
                                               -10.,
                                               10.);
    h_pull_vs_pt = dqm::booking::book2DIfLogX(ibooker,
                                              true,
                                              (name + "_pull_vs_pt").c_str(),
                                              (name + " pulls vs vertex pt;SimVertex p_T;Pulls").c_str(),
                                              bins.pt.nBins,
                                              bins.pt.min,
                                              bins.pt.max,
                                              100,
                                              -10.,
                                              10.);
    h_pull_vs_nTracks = dqm::booking::book2DIfLogX(ibooker,
                                                   false,
                                                   (name + "_pull_vs_nTracks").c_str(),
                                                   (name + " pulls vs track multiplicity;N tracks;Pulls").c_str(),
                                                   bins.nTracks.nBins,
                                                   bins.nTracks.min,
                                                   bins.nTracks.max,
                                                   100,
                                                   -10.,
                                                   10.);
  }

  template <typename ModificationFunc, typename... Args>
  void modifyHistograms(ModificationFunc modify, Args &&...args) {
    for (auto h : {h_res_vs_decayLength,
                   h_res_vs_decayLength2D,
                   h_res_vs_eta,
                   h_res_vs_pt,
                   h_res_vs_nTracks,
                   h_pull_vs_decayLength,
                   h_pull_vs_decayLength2D,
                   h_pull_vs_eta,
                   h_pull_vs_pt,
                   h_pull_vs_nTracks}) {
      if (h)
        modify(h, std::forward<Args>(args)...);
    }
  }

private:
  dqm::reco::MonitorElement *h_res = nullptr;
  dqm::reco::MonitorElement *h_res_vs_decayLength = nullptr;
  dqm::reco::MonitorElement *h_res_vs_decayLength2D = nullptr;
  dqm::reco::MonitorElement *h_res_vs_eta = nullptr;
  dqm::reco::MonitorElement *h_res_vs_pt = nullptr;
  dqm::reco::MonitorElement *h_res_vs_nTracks = nullptr;
  dqm::reco::MonitorElement *h_pull = nullptr;
  dqm::reco::MonitorElement *h_pull_vs_decayLength = nullptr;
  dqm::reco::MonitorElement *h_pull_vs_decayLength2D = nullptr;
  dqm::reco::MonitorElement *h_pull_vs_eta = nullptr;
  dqm::reco::MonitorElement *h_pull_vs_pt = nullptr;
  dqm::reco::MonitorElement *h_pull_vs_nTracks = nullptr;
};

#endif  // Validation_RecoVertex_SVResolutionBundle_h
