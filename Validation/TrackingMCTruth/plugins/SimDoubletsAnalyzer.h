#ifndef Validation_TrackingMCTruth_SimDoubletsAnalyzer_h
#define Validation_TrackingMCTruth_SimDoubletsAnalyzer_h

// Package:    Validation/TrackingMCTruth
// Class:      SimDoubletsAnalyzer
//
/**\class SimDoubletsAnalyzer SimDoubletsAnalyzer.cc Validation/TrackingMCTruth/plugins/SimDoubletsAnalyzer.cc

 Description: DQM analyzer for true RecHit doublets (SimDoublets) of the inner tracker

 Implementation:
    This analyzer takes as input collection SimDoublets produced by the SimDoubletsProducer. Like the producer,
    you have two versions, one for Phase 1 and one for Phase 2.
    The analyzer's main purpose is to study the goodness of the pixel RecHit doublet creation in the patatrack
    pixel track reconstruction. This depends on a bunch of cuts, meaning a doublet of two RecHits is only formed
    if it passes all of those cuts. The SimDoublets represent the true and therefore ideal doublets of 
    simulated TrackingParticles (Note that in the SimDoublet production you may apply selections on the TPs).
    
    For this study, the analyzer produces histograms for the variables that are cut on during the real 
    reconstruction. Each distribution is produced twice:
     1. for all SimDoublets (name = `{variablename}`)
     2. for those SimDoublets which actually pass all selection cuts (name = `pass_{variablename}`)
    The cut values can be passed in the configuration of the analyzer and should default to the ones used in 
    reconstruction. The distributions of the cut variables can be found in the folder:
        SimDoublets/cutParameters/
    In there, you have one subfolder global/ where you find the cut parameters which are set globally for the 
    entire pixel detector. Additionally, you have for each configured layer pair a subfolder with all layer-pair-
    dependent cut parameters. Those are labeled `lp_{innerLayerId}_{outerLayerId}`.

    Besides this, also general distributions can be found in 
        SimDoublets/general/
    These are e.g. histograms for the number of SimDoublets per layer pair or SimDoublet efficiencies depending 
    on the cut values set in the configuration.

*/
//
// Original Author:  Luca Ferragina, Elena Vernazza, Jan Schulz
//         Created:  Thu, 16 Jan 2025 13:46:21 GMT
//
//

// includes
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "SimDataFormats/TrackingAnalysis/interface/SimDoublets.h"

#include <vector>
#include <string>
#include <map>

namespace simdoublets {
  struct CellCutVariables;
  struct ClusterSizeCutManager;
  void BinLogX(TH1*);
}  // namespace simdoublets

// -------------------------------------------------------------------------------------------------------------
// class declaration
// -------------------------------------------------------------------------------------------------------------

template <typename TrackerTraits>
class SimDoubletsAnalyzer : public DQMEDAnalyzer {
public:
  explicit SimDoubletsAnalyzer(const edm::ParameterSet&);
  ~SimDoubletsAnalyzer() override;

  void dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  // this is simply a little helper to allow us to book histograms easier
  // it automatically books both a pass and tot histogram and allows us to fill them easily
  class CoupledMonitorElement {
  public:
    CoupledMonitorElement() {}
    ~CoupledMonitorElement() {}

    template <typename... Args>
    void fill(const bool pass, Args... args) {
      if (pass)
        h_pass_->Fill(args...);
      h_total_->Fill(args...);
    }

    template <typename... Args>
    void book1D(DQMStore::IBooker& ibooker,
                const std::string& name,
                const std::string& title,
                const std::string& xlabel,
                const std::string& ylabel,
                Args... args) {
      const std::string& xylabels = "; " + xlabel + "; " + ylabel;
      h_pass_ = ibooker.book1D("pass_" + name, title + " (pass)" + xylabels, args...);
      h_total_ = ibooker.book1D(name, title + " (all)" + xylabels, args...);
    }

    template <typename... Args>
    void book2D(DQMStore::IBooker& ibooker,
                const std::string& name,
                const std::string& title,
                const std::string& xlabel,
                const std::string& ylabel,
                Args... args) {
      const std::string& xylabels = "; " + xlabel + "; " + ylabel;
      h_pass_ = ibooker.book2D("pass_" + name, title + " (pass)" + xylabels, args...);
      h_total_ = ibooker.book2D(name, title + " (all)" + xylabels, args...);
    }

    template <typename... Args>
    void book1DLogX(DQMStore::IBooker& ibooker,
                    const std::string& name,
                    const std::string& title,
                    const std::string& xlabel,
                    const std::string& ylabel,
                    Args&&... args) {
      const std::string& xylabels = "; " + xlabel + "; " + ylabel;
      auto hp = std::make_unique<TH1F>(
          ("pass_" + name).c_str(), (title + " (pass)" + xylabels).c_str(), std::forward<Args>(args)...);
      auto ht =
          std::make_unique<TH1F>(name.c_str(), (title + " (all)" + xylabels).c_str(), std::forward<Args>(args)...);
      simdoublets::BinLogX(hp.get());
      simdoublets::BinLogX(ht.get());
      h_pass_ = ibooker.book1D("pass_" + name, hp.release());
      h_total_ = ibooker.book1D(name, ht.release());
    }

  private:
    MonitorElement* h_pass_ = nullptr;
    MonitorElement* h_total_ = nullptr;
  };

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  // function to apply cuts and set doublet to alive if it passes and to killed otherwise
  void applyCuts(SimDoublets::Doublet&,
                 bool const,
                 int const,
                 simdoublets::CellCutVariables const&,
                 simdoublets::ClusterSizeCutManager const&) const;

  //  function that fills all histograms for cut variables (in folder cutParameters)
  void fillCutHistograms(SimDoublets::Doublet const&,
                         bool const,
                         int const,
                         simdoublets::CellCutVariables const&,
                         simdoublets::ClusterSizeCutManager const&);

  // ------------ member data ------------

  const TrackerTopology* trackerTopology_ = nullptr;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topology_getToken_;
  const edm::EDGetTokenT<SimDoubletsCollection> simDoublets_getToken_;

  // map that takes the layerPairId as defined in the SimDoublets
  // and gives the position of the histogram in the histogram vector
  std::map<int, int> layerPairId2Index_;

  // cutting parameters
  std::vector<double> cellMinz_;
  std::vector<double> cellMaxz_;
  std::vector<int> cellPhiCuts_;
  std::vector<double> cellMaxr_;
  int cellMinYSizeB1_;
  int cellMinYSizeB2_;
  int cellMaxDYSize12_;
  int cellMaxDYSize_;
  int cellMaxDYPred_;
  double cellZ0Cut_;
  double cellPtCut_;
  double CAThetaCutBarrel_over_ptmin_;
  double CAThetaCutForward_over_ptmin_;
  double dcaCutInnerTriplet_;
  double dcaCutOuterTriplet_;
  double hardCurvCut_;

  std::string folder_;  // main folder in the DQM file

  // monitor elements
  // profiles to be filled
  MonitorElement* h_effSimDoubletsPerTPVsPt_;
  MonitorElement* h_effSimDoubletsPerTPVsEta_;
  // histograms of TrackingParticles
  CoupledMonitorElement h_numTPVsPt_;
  CoupledMonitorElement h_numTPVsEta_;
  // histograms of SimDoublets
  CoupledMonitorElement h_layerPairs_;
  CoupledMonitorElement h_numSkippedLayers_;
  CoupledMonitorElement h_numSimDoubletsPerTrackingParticle_;
  CoupledMonitorElement h_numLayersPerTrackingParticle_;
  CoupledMonitorElement h_numVsPt_;
  CoupledMonitorElement h_numVsEta_;
  CoupledMonitorElement h_z0_;
  CoupledMonitorElement h_curvatureR_;
  CoupledMonitorElement h_pTFromR_;
  CoupledMonitorElement h_YsizeB1_;
  CoupledMonitorElement h_YsizeB2_;
  CoupledMonitorElement h_DYsize12_;
  CoupledMonitorElement h_DYsize_;
  CoupledMonitorElement h_DYPred_;
  // vectors of histograms (one hist per layer pair)
  std::vector<CoupledMonitorElement> hVector_dr_;
  std::vector<CoupledMonitorElement> hVector_dphi_;
  std::vector<CoupledMonitorElement> hVector_idphi_;
  std::vector<CoupledMonitorElement> hVector_innerZ_;
  std::vector<CoupledMonitorElement> hVector_Ysize_;
  std::vector<CoupledMonitorElement> hVector_DYsize_;
  std::vector<CoupledMonitorElement> hVector_DYPred_;
  // histograms of doublet connections
  CoupledMonitorElement h_CAThetaCutBarrel_;
  CoupledMonitorElement h_CAThetaCutForward_;
  CoupledMonitorElement h_dcaCutInnerTriplet_;
  CoupledMonitorElement h_dcaCutOuterTriplet_;
  CoupledMonitorElement h_hardCurvCut_;
};

#endif