#ifndef Validation_RecoVertex_SVTrackQualityBundle_h
#define Validation_RecoVertex_SVTrackQualityBundle_h

// Package:    Validation/RecoVertex
// Class:      SVTrackQualityBundle
//
/**\class SVTrackQualityBundle Validation/RecoVertex/interface/SVTrackQualityBundle.h

 Description: Bundle of monitoring elements for the SecondaryVertexAnalyzer,
              grouping all histograms for the track content quality at one
              place (e.g. track purity or the number of shared tracks).

 Original Author: Jan Schulz
*/

#include "DQMServices/Core/interface/DQMBookingHelpers.h"
// SVTrackQualityBundle.h
class SVTrackQualityBundle {
  using IBooker = dqm::reco::DQMStore::IBooker;

public:
  // fill function for pure 1D histograms
  void fill(double purity, double efficiency, double nShared) {
    h_purity->Fill(purity);
    h_efficiency->Fill(efficiency);
    h_nShared->Fill(nShared);
  }

  // fill function for 2D histograms
  void fill(double variable, double purity, double efficiency, double nShared) {
    h_purity->Fill(variable, purity);
    h_efficiency->Fill(variable, efficiency);
    h_nShared->Fill(variable, nShared);
    p_purity->Fill(variable, purity);
    p_efficiency->Fill(variable, efficiency);
  }

  // booker for pure 1D histograms
  void bookHistograms(IBooker &ibooker) {
    h_purity = ibooker.book1D(
        "trackPurity",
        "Track purity per matched RecoSV;Purity = nSharedTracks(RecoSV, SimSV) / nTracks(RecoSV);Sim-matched RecoSVs",
        50,
        0.,
        1.);
    h_efficiency = ibooker.book1D("trackEfficiency",
                                  "Track efficiency per matched SimSV;Efficiency = nSharedTracks(RecoSV, SimSV) / "
                                  "nMatchedRecoTracks(SimSV);Reco-matched SimSVs",
                                  50,
                                  0.,
                                  1.);
    h_nShared = ibooker.book1D("nSharedTracks", "N shared tracks;N shared tracks;Entries", 20, -0.5, 19.5);
  }

  // booker for 2D histograms variable vs. efficiency/purity/nShared
  void bookHistograms(IBooker &ibooker,
                      const bool logScale,
                      const std::string &name,
                      const std::string &xlabel,
                      const int nBins,
                      const double valMin,
                      const double valMax) {
    h_purity = dqm::booking::book2DIfLogX(
        ibooker,
        logScale,
        ("trackPurity_vs_" + name).c_str(),
        ("Track purity per matched RecoSV;" + xlabel + ";Track purity = nSharedTracks(RecoSV, SimSV)").c_str(),
        nBins,
        valMin,
        valMax,
        50,
        0.,
        1.0001);
    h_efficiency =
        dqm::booking::book2DIfLogX(ibooker,
                                   logScale,
                                   ("trackEfficiency_vs_" + name).c_str(),
                                   ("Track efficiency per matched SimSV;" + xlabel +
                                    ";Track efficiency = nSharedTracks(RecoSV, SimSV) / nMatchedRecoTracks(SimSV)")
                                       .c_str(),
                                   nBins,
                                   valMin,
                                   valMax,
                                   50,
                                   0.,
                                   1.0001);
    h_nShared = dqm::booking::book2DIfLogX(
        ibooker,
        logScale,
        ("nSharedTracks_vs_" + name).c_str(),
        ("N(shared tracks) of matched Sim-Reco pairs;" + xlabel + ";N shared tracks").c_str(),
        nBins,
        valMin,
        valMax,
        20,
        -0.5,
        19.5);
    p_purity = dqm::booking::bookProfileIfLogX(
        ibooker,
        logScale,
        ("trackPurityProfile_vs_" + name).c_str(),
        ("Average track purity of matched RecoSVs;" + xlabel + ";Average track purity = nSharedTracks(RecoSV, SimSV)")
            .c_str(),
        nBins,
        valMin,
        valMax,
        0,
        1,
        " ");
    p_efficiency = dqm::booking::bookProfileIfLogX(
        ibooker,
        logScale,
        ("trackEfficiencyProfile_vs_" + name).c_str(),
        ("Average track efficiency of matched SimSVs;" + xlabel +
         ";Average track efficiency = nSharedTracks(RecoSV, SimSV) / nMatchedRecoTracks(SimSV)")
            .c_str(),
        nBins,
        valMin,
        valMax,
        0,
        1,
        " ");
  }

private:
  // plain histograms
  dqm::reco::MonitorElement *h_purity = nullptr;
  dqm::reco::MonitorElement *h_efficiency = nullptr;
  dqm::reco::MonitorElement *h_nShared = nullptr;
  // profiles
  dqm::reco::MonitorElement *p_purity = nullptr;
  dqm::reco::MonitorElement *p_efficiency = nullptr;
};

#endif  // Validation_RecoVertex_SVTrackQualityBundle_h
