#ifndef Validation_TrackingMCTruth_SimPixelTrackAnalyzer_h
#define Validation_TrackingMCTruth_SimPixelTrackAnalyzer_h

// Package:    Validation/TrackingMCTruth
// Class:      SimPixelTrackAnalyzer
//
/**\class SimPixelTrackAnalyzer SimPixelTrackAnalyzer.cc Validation/TrackingMCTruth/plugins/SimPixelTrackAnalyzer.cc

 Description: DQM analyzer for true RecHit doublets (SimDoublets) of the inner tracker

 Implementation:
    This analyzer takes as input collection SimPixelTracks produced by the SimPixelTrackProducer. Like the producer,
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

#include "SimDataFormats/TrackingAnalysis/interface/SimPixelTrack.h"

#include <vector>
#include <string>
#include <map>

namespace simdoublets {
  struct CellCutVariables;
  template <typename TrackerTraits>
  struct ClusterSizeCutManager;
  void BinLogX(TH1*);
  void BinLogY(TH1*);

  // struct keeping all the true parameters of the TrackingParticle / RecoTrack
  struct TrackTruth {
    double eta;
    double pt;
    double phi;
    double dz;
    double dxy;
    double vertpos;
    int pdgId{0};
  };
}  // namespace simdoublets

// -------------------------------------------------------------------------------------------------------------
// class declaration
// -------------------------------------------------------------------------------------------------------------

template <typename TrackerTraits>
class SimPixelTrackAnalyzer : public DQMEDAnalyzer {
public:
  explicit SimPixelTrackAnalyzer(const edm::ParameterSet&);
  ~SimPixelTrackAnalyzer() override;

  void dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  // small struct keeping all cut parameters
  struct CAGeometryParams {
    //Constructor from ParameterSet
    CAGeometryParams(edm::ParameterSet const& iConfig, double const ptmin)
        : caDCACuts_(iConfig.getParameter<std::vector<double>>("caDCACuts")),
          phiCuts_(iConfig.getParameter<std::vector<int>>("phiCuts")),
          ptCuts_(iConfig.getParameter<std::vector<double>>("ptCuts")),
          minInner_(iConfig.getParameter<std::vector<double>>("minInner")),
          maxInner_(iConfig.getParameter<std::vector<double>>("maxInner")),
          minOuter_(iConfig.getParameter<std::vector<double>>("minOuter")),
          maxOuter_(iConfig.getParameter<std::vector<double>>("maxOuter")),
          maxDZ_(iConfig.getParameter<std::vector<double>>("maxDZ")),
          minDZ_(iConfig.getParameter<std::vector<double>>("minDZ")),
          maxDR_(iConfig.getParameter<std::vector<double>>("maxDR")) {
      for (double const caThetaCut : iConfig.getParameter<std::vector<double>>("caThetaCuts")) {
        caThetaCuts_over_ptmin_.push_back(caThetaCut / ptmin);
      }
      for (double const isBar : iConfig.getParameter<std::vector<int>>("isBarrel")) {
        isBarrel_.push_back((bool)isBar);
      }
    }

    // Layers params
    std::vector<bool> isBarrel_;
    std::vector<double> caThetaCuts_over_ptmin_;
    const std::vector<double> caDCACuts_;

    // Cells params
    const std::vector<int> phiCuts_;
    const std::vector<double> ptCuts_;
    const std::vector<double> minInner_;
    const std::vector<double> maxInner_;
    const std::vector<double> minOuter_;
    const std::vector<double> maxOuter_;
    const std::vector<double> maxDZ_;
    const std::vector<double> minDZ_;
    const std::vector<double> maxDR_;
  };

  // this is simply a little helper to allow us to book histograms easier
  // it automatically books both a pass and tot histogram and allows us to fill them easily
  class CoupledMonitorElement {
  public:
    CoupledMonitorElement() {}
    virtual ~CoupledMonitorElement() = default;

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
                const int nBins,
                const double valMin,
                const double valMax,
                Args... args) {
      const std::string& xylabels = "; " + xlabel + "; " + ylabel;
      h_pass_ = ibooker.book1D("pass_" + name, title + " (pass)" + xylabels, nBins, valMin, valMax, args...);
      h_total_ = ibooker.book1D(name, title + " (all)" + xylabels, nBins, valMin, valMax, args...);
      bookExtraCutHistos(ibooker, name, xlabel, nBins, valMin, valMax);
    }

    template <typename... Args>
    void book2D(DQMStore::IBooker& ibooker,
                const std::string& name,
                const std::string& title,
                const std::string& xlabel,
                const std::string& ylabel,
                const int nBins,
                const double valMin,
                const double valMax,
                Args... args) {
      const std::string& xylabels = "; " + xlabel + "; " + ylabel;
      h_pass_ = ibooker.book2D("pass_" + name, title + " (pass)" + xylabels, nBins, valMin, valMax, args...);
      h_total_ = ibooker.book2D(name, title + " (all)" + xylabels, nBins, valMin, valMax, args...);
      bookExtraCutHistos(ibooker, name, xlabel, nBins, valMin, valMax);
    }

    template <typename... Args>
    void book1DLogX(DQMStore::IBooker& ibooker,
                    const std::string& name,
                    const std::string& title,
                    const std::string& xlabel,
                    const std::string& ylabel,
                    const int nBins,
                    const double valMin,
                    const double valMax,
                    Args&&... args) {
      const std::string& xylabels = "; " + xlabel + "; " + ylabel;
      auto hp = std::make_unique<TH1F>(("pass_" + name).c_str(),
                                       (title + " (pass)" + xylabels).c_str(),
                                       nBins,
                                       valMin,
                                       valMax,
                                       std::forward<Args>(args)...);
      auto ht = std::make_unique<TH1F>(
          name.c_str(), (title + " (all)" + xylabels).c_str(), nBins, valMin, valMax, std::forward<Args>(args)...);
      simdoublets::BinLogX(hp.get());
      simdoublets::BinLogX(ht.get());
      h_pass_ = ibooker.book1D("pass_" + name, hp.release());
      h_total_ = ibooker.book1D(name, ht.release());
      bookExtraCutHistos(ibooker, name, xlabel, nBins, valMin, valMax);
    }

    template <typename... Args>
    void book2DLogX(DQMStore::IBooker& ibooker,
                    const std::string& name,
                    const std::string& title,
                    const std::string& xlabel,
                    const std::string& ylabel,
                    const int nBins,
                    const double valMin,
                    const double valMax,
                    Args&&... args) {
      const std::string& xylabels = "; " + xlabel + "; " + ylabel;
      auto hp = std::make_unique<TH2F>(("pass_" + name).c_str(),
                                       (title + " (pass)" + xylabels).c_str(),
                                       nBins,
                                       valMin,
                                       valMax,
                                       std::forward<Args>(args)...);
      auto ht = std::make_unique<TH2F>(
          name.c_str(), (title + " (all)" + xylabels).c_str(), nBins, valMin, valMax, std::forward<Args>(args)...);
      simdoublets::BinLogX(hp.get());
      simdoublets::BinLogX(ht.get());
      h_pass_ = ibooker.book2D("pass_" + name, hp.release());
      h_total_ = ibooker.book2D(name, ht.release());
      bookExtraCutHistos(ibooker, name, xlabel, nBins, valMin, valMax);
    }

    template <typename... Args>
    void book2DLogY(DQMStore::IBooker& ibooker,
                    const std::string& name,
                    const std::string& title,
                    const std::string& xlabel,
                    const std::string& ylabel,
                    const int nBins,
                    const double valMin,
                    const double valMax,
                    Args&&... args) {
      const std::string& xylabels = "; " + xlabel + "; " + ylabel;
      auto hp = std::make_unique<TH2F>(("pass_" + name).c_str(),
                                       (title + " (pass)" + xylabels).c_str(),
                                       nBins,
                                       valMin,
                                       valMax,
                                       std::forward<Args>(args)...);
      auto ht = std::make_unique<TH2F>(
          name.c_str(), (title + " (all)" + xylabels).c_str(), nBins, valMin, valMax, std::forward<Args>(args)...);
      simdoublets::BinLogY(hp.get());
      simdoublets::BinLogY(ht.get());
      h_pass_ = ibooker.book2D("pass_" + name, hp.release());
      h_total_ = ibooker.book2D(name, ht.release());
      bookExtraCutHistos(ibooker, name, xlabel, nBins, valMin, valMax);
    }

  protected:
    virtual void bookExtraCutHistos(DQMStore::IBooker& ibooker,
                                    const std::string& name,
                                    const std::string& title,
                                    const int cutNBins,
                                    const double cutMin,
                                    const double cutMax) {}
    MonitorElement* h_pass_ = nullptr;
    MonitorElement* h_total_ = nullptr;
  };

  // special class version for histograms of cut variables
  class CoupledCutMonitorElement : public CoupledMonitorElement {
  public:
    CoupledCutMonitorElement() {}
    ~CoupledCutMonitorElement() override = default;

    template <typename... Args>
    void fillCut(const bool pass, simdoublets::TrackTruth const& trackTruth, const double inner, Args... args) {
      if (pass)
        this->h_pass_->Fill(args...);
      this->h_total_->Fill(args...);
      h_z_eta_->Fill(trackTruth.dz, trackTruth.eta, args...);
      h_z_->Fill(trackTruth.dz, args...);
      h_eta_->Fill(trackTruth.eta, args...);
      h_inner_->Fill(inner, args...);
    }

    void fillPassThisCut(const bool passThisCut) { h_passThisCut_->Fill(0.5, passThisCut); }

  private:
    void bookExtraCutHistos(DQMStore::IBooker& ibooker,
                            const std::string& name,
                            const std::string& title,
                            const int cutNBins,
                            const double cutMin,
                            const double cutMax) override {
      int etaNBins = 90;
      double etaMin = -4.5;
      double etaMax = 4.5;
      int zNBins = 80;
      double zMin = -20.;
      double zMax = 20.;
      int innerNBins = 300;
      double innerMin = -150;
      double innerMax = 150;
      h_z_eta_ = ibooker.bookProfile2D(name + "_zEta",
                                       title + "; Vertex z coordinate [cm]; Pseudorapidity #eta",
                                       innerNBins,
                                       zMin,
                                       zMax,
                                       etaNBins,
                                       etaMin,
                                       etaMax,
                                       -1.0e5,
                                       1.0e5,
                                       " ");
      h_inner_ = ibooker.book2D(name + "_inner",
                                title + "; z or r coordinate of inner RecHit [cm]; " + title,
                                innerNBins,
                                innerMin,
                                innerMax,
                                cutNBins,
                                cutMin,
                                cutMax);
      h_z_ = ibooker.book2D(
          name + "_z", title + "; Vertex z coordinate [cm]; " + title, zNBins, zMin, zMax, cutNBins, cutMin, cutMax);
      h_eta_ = ibooker.book2D(
          name + "_eta", title + "; Pseudorapidity #eta; " + title, etaNBins, etaMin, etaMax, cutNBins, cutMin, cutMax);
      h_passThisCut_ = ibooker.bookProfile(
          name + "_passThisCut", title + ";;Fraction of Doublets passing this cut", 1, 0.0, 1, 0.0, 1.0, " ");
    }

    // additional histograms for cuts
    MonitorElement* h_z_eta_ = nullptr;
    MonitorElement* h_z_ = nullptr;
    MonitorElement* h_eta_ = nullptr;
    MonitorElement* h_inner_ = nullptr;
    MonitorElement* h_passThisCut_ = nullptr;
  };

  // this is simply a little helper to allow us to book SimNtuplet histograms easier
  // it automatically books all histograms and allows us to fill them easily
  struct SimNtupletMonitorElement {
  public:
    struct histogramBlock {
      void book(DQMStore::IBooker& ibook,
                const std::string& simNtupletName,
                const std::string& statusTag,
                const std::string& statusLabel) {
        int pTNBins = 200;
        double pTmin = log10(0.01);
        double pTmax = log10(1000);
        const auto name = ("num_pt_" + statusTag);
        const auto title =
            (simNtupletName + statusLabel + ";True transverse momentum p_{T} [GeV];Number of TrackingParticles");
        auto h = std::make_unique<TH1F>(name.c_str(), title.c_str(), pTNBins, pTmin, pTmax);
        simdoublets::BinLogX(h.get());
        h_pt = ibook.book1D(name.c_str(), h.release());

        int etaNBins = 90;
        double etamin = -4.5;
        double etamax = 4.5;
        const auto name2 = ("num_eta_" + statusTag);
        const auto title2 = (simNtupletName + statusLabel + ";True pseudorapidity #eta;Number of TrackingParticles");
        h_eta = ibook.book1D(name2.c_str(), title2.c_str(), etaNBins, etamin, etamax);

        int vertPosNBins = 40;
        double vertPosmin = log10(0.01);
        double vertPosmax = log10(100);
        const auto name3 = ("num_vertpos_" + statusTag);
        const auto title3 = (simNtupletName + statusLabel +
                             ";True radial vertex position r_{vertex} [cm];Number of "
                             "TrackingParticles");
        auto h3 = std::make_unique<TH1F>(name3.c_str(), title3.c_str(), vertPosNBins, vertPosmin, vertPosmax);
        simdoublets::BinLogX(h3.get());
        h_vertpos = ibook.book1D(name3.c_str(), h3.release());
      }
      void fill(simdoublets::TrackTruth const& trackTruth) {
        h_pt->Fill(trackTruth.pt);
        h_eta->Fill(trackTruth.eta);
        h_vertpos->Fill(trackTruth.vertpos);
      }
      MonitorElement* h_pt{nullptr};
      MonitorElement* h_eta{nullptr};
      MonitorElement* h_vertpos{nullptr};
    };
    void bookHistograms(DQMStore::IBooker& ibook, const std::string& simNtupletName) {
      alive_.book(ibook, simNtupletName, "Alive", " (alive)");
      undefDoubletCuts_.book(ibook, simNtupletName, "UndefDoubletCuts", " (with undef doublet cuts)");
      undefConnectionCuts_.book(ibook, simNtupletName, "UndefConnectionCuts", " (with undef connection cuts)");
      missingLayerPair_.book(ibook, simNtupletName, "MissingLayerPair", " (with missing layer pair)");
      killedDoublets_.book(ibook, simNtupletName, "KilledDoublets", " (killed by doublet cuts)");
      killedDoubletConnections_.book(
          ibook, simNtupletName, "KilledConnections", " (killed by doublet connection cuts)");
      killedTripletConnections_.book(
          ibook, simNtupletName, "KilledTripletConnections", " (killed by triplet connection cuts)");
      tooShort_.book(ibook, simNtupletName, "TooShort", " (3 RecHits but still shorter than the threshold)");
      notStartingPair_.book(ibook,
                            simNtupletName,
                            "NotStartingPair",
                            " (has first doublet in layer pair not considered for starting Ntuplets)");
    }

    histogramBlock alive_;
    histogramBlock undefDoubletCuts_;
    histogramBlock undefConnectionCuts_;
    histogramBlock missingLayerPair_;
    histogramBlock killedDoublets_;
    histogramBlock killedDoubletConnections_;
    histogramBlock killedTripletConnections_;
    histogramBlock tooShort_;
    histogramBlock notStartingPair_;
  };

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  // function to apply cuts and set doublet to alive if it passes and to killed otherwise
  void applyCuts(SimPixelTrack::Doublet&,
                 SimPixelTrack const&,
                 bool const,
                 bool const,
                 int const,
                 simdoublets::CellCutVariables const&,
                 simdoublets::ClusterSizeCutManager<TrackerTraits> const&);

  // function that fills all histograms for cut variables (in folder CAParameters)
  void fillCutHistograms(SimPixelTrack::Doublet const&,
                         bool const,
                         bool const,
                         int const,
                         simdoublets::CellCutVariables const&,
                         simdoublets::ClusterSizeCutManager<TrackerTraits> const&,
                         simdoublets::TrackTruth const&);

  // function that fills all histograms of SimDoublets (in folder SimDoublets)
  void fillSimDoubletHistograms(SimPixelTrack::Doublet const&, simdoublets::TrackTruth const&);

  // function that fills all histograms of SimNtuplets (in folder SimNtuplets)
  void fillSimNtupletHistograms(SimPixelTrack const&, simdoublets::TrackTruth const&);

  // function that fills all general histograms (in folder general)
  void fillGeneralHistograms(SimPixelTrack const&, simdoublets::TrackTruth const&, int const, int const, int const);

  // function that trys to find a valid Ntuplet for the given SimPixelTrack using the given geometry configuration
  // (layer pairs, starting pairs, minimum number of hits) ignoring all cuts on doublets/connections and returns if it was able to find one
  bool configAllowsForValidNtuplet(SimPixelTrack const&) const;

  // ------------ member data ------------

  const TrackerTopology* trackerTopology_ = nullptr;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topology_getToken_;
  const edm::EDGetTokenT<SimPixelTrackCollection> simPixelTracks_getToken_;

  // number of layers in total
  int numLayers_;

  // map that takes the layerPairId as defined in the SimPixelTrack
  // and gives the position of the histogram in the histogram vector
  std::map<int, int> layerPairId2Index_;

  // set that contains all the layerPairId as defined in the SimPixelTrack
  // that are considered as a starting points for Ntuplets
  std::set<int> startingPairs_;

  // cutting parameters
  CAGeometryParams cellCuts_;
  const int minYsizeB1_;
  const int minYsizeB2_;
  const int maxDYsize12_;
  const int maxDYsize_;
  const int maxDYPred_;
  const double cellZ0Cut_;
  const double cellPtCut_;
  const double hardCurvCut_;
  const int minNumDoubletsPerNtuplet_;

  std::string folder_;  // main folder in the DQM file
  // inputIsRecoTracks_: - set to false if SimPixelTracks were produced based on TrackingParticles (truth information)
  //                     - set to true if they were produced based on reconstructed tracks
  bool inputIsRecoTracks_;

  // monitor elements
  // TO = TrackingParticle or Track
  // profiles to be filled
  MonitorElement* h_effSimDoubletsPerTOVsPt_;
  MonitorElement* h_effSimDoubletsPerTOVsEta_;
  MonitorElement* h_numLayersVsEtaPt_;
  MonitorElement* h_effConfigLimitVsEta_;
  MonitorElement* h_effConfigLimitVsPt_;
  // histograms of TrackingParticles
  CoupledMonitorElement h_numTOVsPt_;
  CoupledMonitorElement h_numTOVsEta_;
  CoupledMonitorElement h_numTOVsPhi_;
  CoupledMonitorElement h_numTOVsDxy_;
  CoupledMonitorElement h_numTOVsDz_;
  CoupledMonitorElement h_numTOVsVertpos_;
  CoupledMonitorElement h_numTOVsChi2_;
  CoupledMonitorElement h_numRecHitsVsPt_;
  CoupledMonitorElement h_numRecHitsVsEta_;
  CoupledMonitorElement h_numRecHitsVsDxy_;
  CoupledMonitorElement h_numRecHitsVsDz_;
  CoupledMonitorElement h_numRecHitsVsChi2_;
  CoupledMonitorElement h_numTOVsEtaPt_;
  CoupledMonitorElement h_numTOVsEtaPhi_;
  CoupledMonitorElement h_numTOVsPhiPt_;
  CoupledMonitorElement h_numSimDoubletsPerTrackingObject_;
  CoupledMonitorElement h_numSkippedLayersPerTrackingObject_;
  CoupledMonitorElement h_numRecHitsPerTrackingObject_;
  CoupledMonitorElement h_numLayersPerTrackingObject_;
  CoupledMonitorElement h_numSkippedLayersVsEta_;
  CoupledMonitorElement h_numLayersVsEta_;
  CoupledMonitorElement h_numSkippedLayersVsPt_;
  CoupledMonitorElement h_numSkippedLayersVsNumLayers_;
  CoupledMonitorElement h_numSkippedLayersVsNumRecHits_;
  CoupledMonitorElement h_numLayersVsPt_;
  CoupledMonitorElement h_numTOVsPdgId_;
  CoupledMonitorElement h_numRecHitsPerLayer_;
  // histograms of SimDoublets
  CoupledMonitorElement h_layerPairs_;
  CoupledMonitorElement h_numSkippedLayers_;
  CoupledMonitorElement h_num_vs_pt_;
  CoupledMonitorElement h_num_vs_eta_;
  CoupledMonitorElement h_num_vs_vertpos_;
  CoupledCutMonitorElement h_z0_;
  CoupledCutMonitorElement h_curvatureR_;
  CoupledCutMonitorElement h_pTFromR_;
  CoupledCutMonitorElement h_YsizeB1_;
  CoupledCutMonitorElement h_YsizeB2_;
  CoupledCutMonitorElement h_DYsize12_;
  CoupledCutMonitorElement h_DYsize_;
  CoupledCutMonitorElement h_DYPred_;
  // vectors of histograms (one hist per layer pair)
  std::vector<CoupledCutMonitorElement> hVector_z0_;
  std::vector<CoupledCutMonitorElement> hVector_curvatureR_;
  std::vector<CoupledCutMonitorElement> hVector_pTFromR_;
  std::vector<CoupledCutMonitorElement> hVector_dz_;
  std::vector<CoupledCutMonitorElement> hVector_dr_;
  std::vector<CoupledCutMonitorElement> hVector_dphi_;
  std::vector<CoupledCutMonitorElement> hVector_idphi_;
  std::vector<CoupledCutMonitorElement> hVector_inner_;
  std::vector<CoupledCutMonitorElement> hVector_outer_;
  std::vector<CoupledCutMonitorElement> hVector_Ysize_;
  std::vector<CoupledCutMonitorElement> hVector_DYsize_;
  std::vector<CoupledCutMonitorElement> hVector_DYPred_;
  // histograms of doublet connections
  CoupledCutMonitorElement h_hardCurvCut_;
  CoupledCutMonitorElement h_dCurvCut_;
  CoupledCutMonitorElement h_curvRatioCut_;
  // vectors of historgrams (one per layer)
  std::vector<CoupledCutMonitorElement> hVector_caThetaCut_;
  std::vector<CoupledCutMonitorElement> hVector_caDCACut_;
  std::vector<CoupledCutMonitorElement> hVector_firstHitR_;
  // histograms of the most alive Ntuplet per TP
  CoupledMonitorElement h_bestNtuplet_numRecHits_;
  CoupledMonitorElement h_bestNtuplet_firstLayerId_;
  CoupledMonitorElement h_bestNtuplet_lastLayerId_;
  CoupledMonitorElement h_bestNtuplet_layerSpan_;
  CoupledMonitorElement h_bestNtuplet_firstVsSecondLayer_;
  CoupledMonitorElement h_bestNtuplet_firstLayerVsEta_;
  CoupledMonitorElement h_bestNtuplet_lastLayerVsEta_;
  CoupledMonitorElement h_bestNtuplet_numSkippedLayersVsNumLayers_;
  MonitorElement* h_aliveNtuplet_fracNumRecHits_eta_;
  MonitorElement* h_aliveNtuplet_fracNumRecHits_pt_;
  // histograms of the longest Ntuplet per TP
  CoupledMonitorElement h_longNtuplet_numRecHits_;
  CoupledMonitorElement h_longNtuplet_firstLayerId_;
  CoupledMonitorElement h_longNtuplet_lastLayerId_;
  CoupledMonitorElement h_longNtuplet_layerSpan_;
  CoupledMonitorElement h_longNtuplet_firstVsSecondLayer_;
  CoupledMonitorElement h_longNtuplet_firstLayerVsEta_;
  CoupledMonitorElement h_longNtuplet_lastLayerVsEta_;
  CoupledMonitorElement h_longNtuplet_numSkippedLayersVsNumLayers_;
  // status of the most alive and longest SimNtuplet per TP
  SimNtupletMonitorElement h_bestNtuplet_;
  SimNtupletMonitorElement h_longNtuplet_;
};

#endif