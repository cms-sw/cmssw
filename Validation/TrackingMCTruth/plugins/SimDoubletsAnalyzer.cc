// -*- C++ -*-
//
// Package:    Validation/TrackingMCTruth
// Class:      SimDoubletsAnalyzer
//

// user include files
#include "Validation/TrackingMCTruth/plugins/SimDoubletsAnalyzer.h"

#include "DataFormats/Histograms/interface/MonitorElementCollection.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/approx_atan2.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDoubletsAnalyzer.h"

#include <cstddef>

namespace simdoublets {
  // class that calculate and stores all cut variables for a given doublet
  struct CellCutVariables {
    void calculateCutVariables(SimDoublets::Doublet const& doublet) {
      // inner RecHit properties
      inner_z_ = doublet.innerGlobalPos().z();
      double inner_r = doublet.innerGlobalPos().perp();
      double inner_phi = doublet.innerGlobalPos().barePhi();  // returns float, whereas .phi() returns phi object
      int inner_iphi = unsafe_atan2s<7>(doublet.innerGlobalPos().y(), doublet.innerGlobalPos().x());
      // outer RecHit properties
      double outer_z = doublet.outerGlobalPos().z();
      double outer_r = doublet.outerGlobalPos().perp();
      double outer_phi = doublet.outerGlobalPos().barePhi();
      int outer_iphi = unsafe_atan2s<7>(doublet.outerGlobalPos().y(), doublet.outerGlobalPos().x());

      // relative properties
      dz_ = outer_z - inner_z_;
      dr_ = outer_r - inner_r;
      dphi_ = reco::deltaPhi(inner_phi, outer_phi);
      idphi_ = std::min(std::abs(int16_t(outer_iphi - inner_iphi)), std::abs(int16_t(inner_iphi - outer_iphi)));

      // longitudinal impact parameter with respect to the beamspot
      z0_ = std::abs(inner_r * outer_z - inner_z_ * outer_r) / dr_;

      // radius of the circle defined by the two RecHits and the beamspot
      curvature_ = 1.f / 2.f * std::sqrt((dr_ / dphi_) * (dr_ / dphi_) + (inner_r * outer_r));

      // pT that this curvature radius corresponds to
      pT_ = curvature_ / 87.78f;

      // cluster size variables
      Ysize_ = doublet.innerRecHit()->cluster()->sizeY();
      DYsize_ = std::abs(Ysize_ - doublet.outerRecHit()->cluster()->sizeY());
      DYPred_ = std::abs(Ysize_ - int(std::abs(dz_ / dr_) * pixelTopology::Phase2::dzdrFact + 0.5f));
    }

    // methods to get the cut variables
    double inner_z() const { return inner_z_; }
    double dz() const { return dz_; }
    double dr() const { return dr_; }
    double dphi() const { return dphi_; }
    double z0() const { return z0_; }
    double curvature() const { return curvature_; }
    double pT() const { return pT_; }
    int idphi() const { return idphi_; }
    int Ysize() const { return Ysize_; }
    int DYsize() const { return DYsize_; }
    int DYPred() const { return DYPred_; }

  private:
    double inner_z_, dz_, dr_, dphi_, z0_, curvature_, pT_;  // double-valued variables
    int idphi_, Ysize_, DYsize_, DYPred_;                    // integer-valued variables
  };

  // class to help keep track of which cluster size cuts are applied
  struct ClusterSizeCutManager {
    // flags indicating to which cluster size cuts the doublet is subject to
    enum class CutStatusBit : uint8_t {
      subjectToYsizeB1 = 1,
      subjectToYsizeB2 = 1 << 1,
      subjectToDYsize = 1 << 2,
      subjectToDYsize12 = 1 << 3,
      subjectToDYPred = 1 << 4
    };

    // reset to "not subject to any cut"
    void reset() { status_ = 0; }

    // set is subject to cuts...
    void setSubjectToYsizeB1() { status_ |= uint8_t(CutStatusBit::subjectToYsizeB1); }
    void setSubjectToYsizeB2() { status_ |= uint8_t(CutStatusBit::subjectToYsizeB2); }
    void setSubjectToDYsize() { status_ |= uint8_t(CutStatusBit::subjectToDYsize); }
    void setSubjectToDYsize12() { status_ |= uint8_t(CutStatusBit::subjectToDYsize12); }
    void setSubjectToDYPred() { status_ |= uint8_t(CutStatusBit::subjectToDYPred); }

    // check if is subject to cuts...
    bool isSubjectToYsizeB1() const { return status_ & uint8_t(CutStatusBit::subjectToYsizeB1); }
    bool isSubjectToYsizeB2() const { return status_ & uint8_t(CutStatusBit::subjectToYsizeB2); }
    bool isSubjectToDYsize() const { return status_ & uint8_t(CutStatusBit::subjectToDYsize); }
    bool isSubjectToDYsize12() const { return status_ & uint8_t(CutStatusBit::subjectToDYsize12); }
    bool isSubjectToDYPred() const { return status_ & uint8_t(CutStatusBit::subjectToDYPred); }

    // function that determines for a given doublet which cuts should be applied
    void setSubjectsToCuts(SimDoublets::Doublet const& doublet) {
      // determine the moduleId
      const GeomDetUnit* geomDetUnit = doublet.innerRecHit()->det();
      const uint32_t moduleId = geomDetUnit->index();

      // define bools needed to decide on cutting parameters
      const bool innerInB1 = (doublet.innerLayerId() == 0);
      const bool innerInB2 = (doublet.innerLayerId() == 1);
      const bool isOuterLadder = (0 == (moduleId / 8) % 2);  // check if this even makes sense in Phase-2
      const bool innerInBarrel = (doublet.innerLayerId() < 4);
      const bool outerInBarrel = (doublet.outerLayerId() < 4);

      // YsizeB1 & YsizeB2 cuts
      if (!outerInBarrel) {
        if (innerInB1 && isOuterLadder) {
          setSubjectToYsizeB1();
        }
        if (innerInB2) {
          setSubjectToYsizeB2();
        }
      }

      // DYsize, DYsizeB12 & DYPred cuts
      if (innerInBarrel) {
        if (outerInBarrel) {  // onlyBarrel
          if (innerInB1 && isOuterLadder) {
            setSubjectToDYsize12();
          } else if (!innerInB1) {
            setSubjectToDYsize();
          }
        } else {  // not onlyBarrel
          setSubjectToDYPred();
        }
      }
    }

  private:
    uint8_t status_{0};
  };

  // helper function that takes the layerPairId and returns two strings with the
  // inner and outer layer id
  std::pair<std::string, std::string> getInnerOuterLayerNames(int const layerPairId) {
    // make a string from the Id (int)
    std::string index = std::to_string(layerPairId);
    // determine inner and outer layer name
    std::string innerLayerName;
    std::string outerLayerName;
    if (index.size() < 3) {
      innerLayerName = "0";
      outerLayerName = index;
    } else if (index.size() == 3) {
      innerLayerName = index.substr(0, 1);
      outerLayerName = index.substr(1, 3);
    } else {
      innerLayerName = index.substr(0, 2);
      outerLayerName = index.substr(2, 4);
    }
    if (outerLayerName[0] == '0') {
      outerLayerName = outerLayerName.substr(1, 2);
    }

    return {innerLayerName, outerLayerName};
  }

  // make bins logarithmic
  void BinLogX(TH1* h) {
    TAxis* axis = h->GetXaxis();
    int bins = axis->GetNbins();

    float from = axis->GetXmin();
    float to = axis->GetXmax();
    float width = (to - from) / bins;
    std::vector<float> new_bins(bins + 1, 0);

    for (int i = 0; i <= bins; i++) {
      new_bins[i] = TMath::Power(10, from + i * width);
    }
    axis->Set(bins, new_bins.data());
  }

  // function to produce histogram with log scale on x (taken from MultiTrackValidator)
  template <typename... Args>
  dqm::reco::MonitorElement* make1DLogX(dqm::reco::DQMStore::IBooker& ibook, Args&&... args) {
    auto h = std::make_unique<TH1F>(std::forward<Args>(args)...);
    BinLogX(h.get());
    const auto& name = h->GetName();
    return ibook.book1D(name, h.release());
  }

  // function to produce profile with log scale on x (taken from MultiTrackValidator)
  template <typename... Args>
  dqm::reco::MonitorElement* makeProfileLogX(dqm::reco::DQMStore::IBooker& ibook, Args&&... args) {
    auto h = std::make_unique<TProfile>(std::forward<Args>(args)...);
    BinLogX(h.get());
    const auto& name = h->GetName();
    return ibook.bookProfile(name, h.release());
  }

  // function that checks if two vector share a common element
  template <typename T>
  bool haveCommonElement(std::vector<T> const& v1, std::vector<T> const& v2) {
    return std::find_first_of(v1.begin(), v1.end(), v2.begin(), v2.end()) != v1.end();
  }

  // fillDescriptionsCommon: description that is identical for Phase 1 and 2
  template <typename TrackerTraits>
  void fillDescriptionsCommon(edm::ParameterSetDescription& desc) {
    desc.add<std::string>("folder", "Tracking/TrackingMCTruth/SimDoublets");

    // cut parameters with scalar values
    desc.add<int>("cellMaxDYSize12", TrackerTraits::maxDYsize12)
        ->setComment("Maximum difference in cluster size for B1/B2");
    desc.add<int>("cellMaxDYSize", TrackerTraits::maxDYsize)->setComment("Maximum difference in cluster size");
    desc.add<int>("cellMaxDYPred", TrackerTraits::maxDYPred)
        ->setComment("Maximum difference between actual and expected cluster size of inner RecHit");
  }
}  // namespace simdoublets

// -------------------------------------------------------------------------------------------------------------
// constructors and destructor
// -------------------------------------------------------------------------------------------------------------

template <typename TrackerTraits>
SimDoubletsAnalyzer<TrackerTraits>::SimDoubletsAnalyzer(const edm::ParameterSet& iConfig)
    : topology_getToken_(esConsumes<TrackerTopology, TrackerTopologyRcd>()),
      simDoublets_getToken_(consumes(iConfig.getParameter<edm::InputTag>("simDoubletsSrc"))),
      cellMinz_(iConfig.getParameter<std::vector<double>>("cellMinz")),
      cellMaxz_(iConfig.getParameter<std::vector<double>>("cellMaxz")),
      cellPhiCuts_(iConfig.getParameter<std::vector<int>>("cellPhiCuts")),
      cellMaxr_(iConfig.getParameter<std::vector<double>>("cellMaxr")),
      cellMinYSizeB1_(iConfig.getParameter<int>("cellMinYSizeB1")),
      cellMinYSizeB2_(iConfig.getParameter<int>("cellMinYSizeB2")),
      cellMaxDYSize12_(iConfig.getParameter<int>("cellMaxDYSize12")),
      cellMaxDYSize_(iConfig.getParameter<int>("cellMaxDYSize")),
      cellMaxDYPred_(iConfig.getParameter<int>("cellMaxDYPred")),
      cellZ0Cut_(iConfig.getParameter<double>("cellZ0Cut")),
      cellPtCut_(iConfig.getParameter<double>("cellPtCut")),
      folder_(iConfig.getParameter<std::string>("folder")) {
  // get layer pairs from configuration
  std::vector<int> layerPairs{iConfig.getParameter<std::vector<int>>("layerPairs")};

  // number of configured layer pairs
  size_t numLayerPairs = layerPairs.size() / 2;

  // fill the map of layer pairs
  for (size_t i{0}; i < numLayerPairs; i++) {
    int layerPairId = 100 * layerPairs[2 * i] + layerPairs[2 * i + 1];
    layerPairId2Index_.insert({layerPairId, i});
  }

  // resize all histogram vectors, so that we can fill them according to the
  // layerPairIndex saved in the map that we just created
  hVector_dr_.resize(numLayerPairs);
  hVector_dphi_.resize(numLayerPairs);
  hVector_idphi_.resize(numLayerPairs);
  hVector_innerZ_.resize(numLayerPairs);
  hVector_Ysize_.resize(numLayerPairs);
  hVector_DYsize_.resize(numLayerPairs);
  hVector_DYPred_.resize(numLayerPairs);
  hVector_pass_dr_.resize(numLayerPairs);
  hVector_pass_idphi_.resize(numLayerPairs);
  hVector_pass_innerZ_.resize(numLayerPairs);
}

template <typename TrackerTraits>
SimDoubletsAnalyzer<TrackerTraits>::~SimDoubletsAnalyzer() {}

// -------------------------------------------------------------------------------------------------------------
// member functions
// -------------------------------------------------------------------------------------------------------------

template <typename TrackerTraits>
void SimDoubletsAnalyzer<TrackerTraits>::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {}

// function to apply cuts and set doublet to alive if it passes and to killed otherwise
template <typename TrackerTraits>
void SimDoubletsAnalyzer<TrackerTraits>::applyCuts(SimDoublets::Doublet& doublet,
                                                   int const layerPairIdIndex,
                                                   simdoublets::CellCutVariables const& cellCutVariables,
                                                   simdoublets::ClusterSizeCutManager const& clusterSizeCutManager) const {
  // apply all cuts, break after the first cut that kills the doublet
  // z window cut
  if (cellCutVariables.inner_z() < cellMinz_[layerPairIdIndex] ||
      cellCutVariables.inner_z() > cellMaxz_[layerPairIdIndex]) {
    doublet.setKilled();
    return;
  }
  // z0cutoff
  if (cellCutVariables.dr() > cellMaxr_[layerPairIdIndex] || cellCutVariables.dr() < 0 ||
      cellCutVariables.z0() > cellZ0Cut_) {
    doublet.setKilled();
    return;
  }
  // ptcut
  if (cellCutVariables.pT() < cellPtCut_) {
    doublet.setKilled();
    return;
  }
  // iphicut
  if (cellCutVariables.idphi() > cellPhiCuts_[layerPairIdIndex]) {
    doublet.setKilled();
    return;
  }
  // YsizeB1 cut
  if (clusterSizeCutManager.isSubjectToYsizeB1() && (cellCutVariables.Ysize() < cellMinYSizeB1_)) {
    doublet.setKilled();
    return;
  }
  // YsizeB2 cut
  if (clusterSizeCutManager.isSubjectToYsizeB2() && (cellCutVariables.Ysize() < cellMinYSizeB2_)) {
    doublet.setKilled();
    return;
  }
  // DYsize12 cut
  if (clusterSizeCutManager.isSubjectToDYsize12() && (cellCutVariables.DYsize() > cellMaxDYSize12_)) {
    doublet.setKilled();
    return;
  }
  // DYsize cut
  if (clusterSizeCutManager.isSubjectToDYsize() && (cellCutVariables.DYsize() > cellMaxDYSize_)) {
    doublet.setKilled();
    return;
  }
  // DYPred cut
  if (clusterSizeCutManager.isSubjectToDYPred() && (cellCutVariables.DYPred() > cellMaxDYPred_)) {
    doublet.setKilled();
    return;
  }

  // if the function is still going, the doublet survived
  doublet.setAlive();
}

template <typename TrackerTraits>
void SimDoubletsAnalyzer<TrackerTraits>::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  // get tracker topology
  trackerTopology_ = &iSetup.getData(topology_getToken_);

  // get simDoublets
  SimDoubletsCollection const& simDoubletsCollection = iEvent.get(simDoublets_getToken_);

  // create vectors for inner and outer RecHits of SimDoublets passing all cuts
  std::vector<SiPixelRecHitRef> innerRecHitsPassing;
  std::vector<SiPixelRecHitRef> outerRecHitsPassing;

  // initialize a bunch of variables that we will use in the coming for loops
  double true_pT, true_eta;
  int numSimDoublets, pass_numSimDoublets, layerPairId, layerPairIdIndex;

  // initialize the manager for keeping track of which cluster cuts are applied to the inidividual doublets
  simdoublets::ClusterSizeCutManager clusterSizeCutManager;

  // initialize the structure holding the cut variables for an individual doublet
  simdoublets::CellCutVariables cellCutVariables;

  // loop over SimDoublets (= loop over TrackingParticles)
  for (auto const& simDoublets : simDoubletsCollection) {
    // get true pT of the TrackingParticle
    true_pT = simDoublets.trackingParticle()->pt();
    true_eta = simDoublets.trackingParticle()->eta();

    // create the true RecHit doublets of the TrackingParticle
    auto doublets = simDoublets.getSimDoublets(trackerTopology_);

    // number of SimDoublets of the Tracking Particle
    numSimDoublets = doublets.size();
    // number of SimDoublets of the Tracking Particle passing all cuts
    pass_numSimDoublets = 0;

    // fill histograms for number of SimDoublets
    h_numSimDoubletsPerTrackingParticle_->Fill(numSimDoublets);
    h_numLayersPerTrackingParticle_->Fill(simDoublets.numLayers());

    // fill histograms for number of TrackingParticles
    h_numTPVsPt_->Fill(true_pT);
    h_numTPVsEta_->Fill(true_eta);

    // clear passing inner and outer RecHits
    innerRecHitsPassing.clear();
    outerRecHitsPassing.clear();

    // loop over those doublets
    for (auto& doublet : doublets) {
      // reset clusterSizeCutManager to "no cluster cuts applied"
      clusterSizeCutManager.reset();

      // calculate the cut variables for the given doublet
      cellCutVariables.calculateCutVariables(doublet);

      // ----------------------------------------------------------
      // general plots (general folder)
      // ----------------------------------------------------------

      // outer layer vs inner layer of SimDoublets
      h_layerPairs_->Fill(doublet.innerLayerId(), doublet.outerLayerId());

      // number of skipped layers by SimDoublets
      h_numSkippedLayers_->Fill(doublet.numSkippedLayers());

      // ----------------------------------------------------------
      // layer pair independent cuts (global folder)
      // ----------------------------------------------------------

      // longitudinal impact parameter with respect to the beamspot
      h_z0_->Fill(cellCutVariables.z0());

      // radius of the circle defined by the two RecHits and the beamspot
      h_curvatureR_->Fill(cellCutVariables.curvature());

      // pT that this curvature radius corresponds to
      h_pTFromR_->Fill(cellCutVariables.pT());

      // ----------------------------------------------------------
      // layer pair dependent cuts (sub-folders for layer pairs)
      // ----------------------------------------------------------

      // first, get layer pair Id and exclude layer pairs that are not considered
      layerPairId = doublet.layerPairId();
      if (layerPairId2Index_.find(layerPairId) == layerPairId2Index_.end()) {
        // if not considered set the doublet as killed
        doublet.setKilled();
        continue;
      }

      // get the position of the layer pair in the vectors of histograms
      layerPairIdIndex = layerPairId2Index_.at(layerPairId);

      // determine which cluster size cuts the doublet is subject to
      clusterSizeCutManager.setSubjectsToCuts(doublet);

      // apply the cuts for doublet building according to the set cut values
      applyCuts(doublet, layerPairIdIndex, cellCutVariables, clusterSizeCutManager);

      // dr = (outer_r - inner_r) histogram
      hVector_dr_[layerPairIdIndex]->Fill(cellCutVariables.dr());

      // dphi histogram
      hVector_dphi_[layerPairIdIndex]->Fill(cellCutVariables.dphi());
      hVector_idphi_[layerPairIdIndex]->Fill(cellCutVariables.idphi());

      // z of the inner RecHit histogram
      hVector_innerZ_[layerPairIdIndex]->Fill(cellCutVariables.inner_z());

      // ----------------------------------------------------------
      // cluster size cuts (global + sub-folders for layer pairs)
      // ----------------------------------------------------------

      // cluster size in local y histogram
      hVector_Ysize_[layerPairIdIndex]->Fill(cellCutVariables.Ysize());

      // histograms for clusterCut
      // YsizeB1 cut
      if (clusterSizeCutManager.isSubjectToYsizeB1()) {
        h_YsizeB1_->Fill(cellCutVariables.Ysize());
      }
      // YsizeB2 cut
      if (clusterSizeCutManager.isSubjectToYsizeB2()) {
        h_YsizeB2_->Fill(cellCutVariables.Ysize());
      }

      // histograms for zSizeCut
      // DYsize12 cut
      if (clusterSizeCutManager.isSubjectToDYsize12()) {
        hVector_DYsize_[layerPairIdIndex]->Fill(cellCutVariables.DYsize());
        h_DYsize12_->Fill(cellCutVariables.DYsize());
      }
      // DYsize cut
      if (clusterSizeCutManager.isSubjectToDYsize()) {
        hVector_DYsize_[layerPairIdIndex]->Fill(cellCutVariables.DYsize());
        h_DYsize_->Fill(cellCutVariables.DYsize());
      }
      // DYPred cut
      if (clusterSizeCutManager.isSubjectToDYPred()) {
        hVector_DYPred_[layerPairIdIndex]->Fill(cellCutVariables.DYPred());
        h_DYPred_->Fill(cellCutVariables.DYPred());
      }

      // ----------------------------------------------------------
      // all kinds of plots for doublets passing all cuts
      // ----------------------------------------------------------

      // fill the number histograms
      // histogram of all valid doublets
      h_numVsPt_->Fill(true_pT);
      h_numVsEta_->Fill(true_eta);

      // if the doublet passes all cuts
      if (doublet.isAlive()) {
        // increment number of SimDoublets passing all cuts
        pass_numSimDoublets++;

        // fill histogram of doublets that pass all cuts
        h_pass_layerPairs_->Fill(doublet.innerLayerId(), doublet.outerLayerId());
        h_pass_numVsPt_->Fill(true_pT);
        h_pass_numVsEta_->Fill(true_eta);

        // also put the inner/outer RecHit in the respective vector
        innerRecHitsPassing.push_back(doublet.innerRecHit());
        outerRecHitsPassing.push_back(doublet.outerRecHit());

        // fill pass_ histograms
        h_pass_z0_->Fill(cellCutVariables.z0());
        h_pass_pTFromR_->Fill(cellCutVariables.pT());
        hVector_pass_dr_[layerPairIdIndex]->Fill(cellCutVariables.dr());
        hVector_pass_idphi_[layerPairIdIndex]->Fill(cellCutVariables.idphi());
        hVector_pass_innerZ_[layerPairIdIndex]->Fill(cellCutVariables.inner_z());
        if (clusterSizeCutManager.isSubjectToDYPred()) {
          h_pass_DYPred_->Fill(cellCutVariables.DYPred());
        }
        if (clusterSizeCutManager.isSubjectToDYsize()) {
          h_pass_DYsize_->Fill(cellCutVariables.DYsize());
        }
        if (clusterSizeCutManager.isSubjectToDYsize12()) {
          h_pass_DYsize12_->Fill(cellCutVariables.DYsize());
        }
        if (clusterSizeCutManager.isSubjectToYsizeB1()) {
          h_pass_YsizeB1_->Fill(cellCutVariables.Ysize());
        }
        if (clusterSizeCutManager.isSubjectToYsizeB2()) {
          h_pass_YsizeB2_->Fill(cellCutVariables.Ysize());
        }
      }
    }  // end loop over those doublets

    // Now check if the TrackingParticle is reconstructable by at least two conencted SimDoublets surviving the cuts
    if (simdoublets::haveCommonElement<SiPixelRecHitRef>(innerRecHitsPassing, outerRecHitsPassing)) {
      h_pass_numTPVsPt_->Fill(true_pT);
      h_pass_numTPVsEta_->Fill(true_eta);
    }

    // Fill the efficiency profile per Tracking Particle only if the TP has at least one SimDoublet
    if (numSimDoublets > 0) {
      h_effSimDoubletsPerTPVsEta_->Fill(true_eta, pass_numSimDoublets / numSimDoublets);
      h_effSimDoubletsPerTPVsPt_->Fill(true_pT, pass_numSimDoublets / numSimDoublets);
    }
  }  // end loop over SimDoublets (= loop over TrackingParticles)
}

// booking the histograms
template <typename TrackerTraits>
void SimDoubletsAnalyzer<TrackerTraits>::bookHistograms(DQMStore::IBooker& ibook,
                                                        edm::Run const& run,
                                                        edm::EventSetup const& iSetup) {
  // set some common parameters
  int pTNBins = 50;
  double pTmin = log10(0.01);
  double pTmax = log10(1000);
  int etaNBins = 80;
  double etamin = -4.;
  double etamax = 4.;

  // ----------------------------------------------------------
  // booking general histograms (general folder)
  // ----------------------------------------------------------

  ibook.setCurrentFolder(folder_ + "/general");

  // overview histograms and profiles
  h_effSimDoubletsPerTPVsPt_ =
      simdoublets::makeProfileLogX(ibook,
                                   "efficiencyPerTP_vs_pT",
                                   "SimDoublets efficiency per TP vs p_{T}; TP transverse momentum p_{T} [GeV]; "
                                   "Average fraction of SimDoublets per TP passing all cuts",
                                   pTNBins,
                                   pTmin,
                                   pTmax,
                                   0,
                                   1,
                                   " ");
  h_effSimDoubletsPerTPVsEta_ = ibook.bookProfile("efficiencyPerTP_vs_eta",
                                                  "SimDoublets efficiency per TP vs #eta; TP transverse momentum #eta; "
                                                  "Average fraction of SimDoublets per TP passing all cuts",
                                                  etaNBins,
                                                  etamin,
                                                  etamax,
                                                  0,
                                                  1,
                                                  " ");
  h_layerPairs_ = ibook.book2D("layerPairs",
                               "Layer pairs in SimDoublets; Inner layer ID; Outer layer ID",
                               TrackerTraits::numberOfLayers,
                               -0.5,
                               -0.5 + TrackerTraits::numberOfLayers,
                               TrackerTraits::numberOfLayers,
                               -0.5,
                               -0.5 + TrackerTraits::numberOfLayers);
  h_pass_layerPairs_ = ibook.book2D("pass_layerPairs",
                                    "Layer pairs in SimDoublets passing all cuts; Inner layer ID; Outer layer ID",
                                    TrackerTraits::numberOfLayers,
                                    -0.5,
                                    -0.5 + TrackerTraits::numberOfLayers,
                                    TrackerTraits::numberOfLayers,
                                    -0.5,
                                    -0.5 + TrackerTraits::numberOfLayers);
  h_numSkippedLayers_ = ibook.book1D(
      "numSkippedLayers", "Number of skipped layers; Number of skipped layers; Number of SimDoublets", 16, -1.5, 14.5);
  h_numSimDoubletsPerTrackingParticle_ =
      ibook.book1D("numSimDoubletsPerTrackingParticle",
                   "Number of SimDoublets per Tracking Particle; Number of SimDoublets; Number of Tracking Particles",
                   31,
                   -0.5,
                   30.5);
  h_numLayersPerTrackingParticle_ =
      ibook.book1D("numLayersPerTrackingParticle",
                   "Number of layers hit by Tracking Particle; Number of layers; Number of Tracking Particles",
                   29,
                   -0.5,
                   28.5);
  h_numTPVsPt_ = simdoublets::make1DLogX(
      ibook,
      "numTPVsPt",
      "Total number of TrackingParticles; True transverse momentum p_{T} [GeV]; Total number of TrackingParticles",
      pTNBins,
      pTmin,
      pTmax);
  h_pass_numTPVsPt_ = simdoublets::make1DLogX(ibook,
                                              "pass_numTPVsPt",
                                              "Reconstructable TrackingParticles (two or more connected SimDoublets "
                                              "pass cuts); True transverse momentum p_{T} [GeV]; "
                                              "Number of reconstructable TrackingParticles",
                                              pTNBins,
                                              pTmin,
                                              pTmax);
  h_numTPVsEta_ =
      ibook.book1D("numTPVsEta",
                   "Total number of TrackingParticles; True pseudorapidity #eta; Total number of TrackingParticles",
                   etaNBins,
                   etamin,
                   etamax);
  h_pass_numTPVsEta_ = ibook.book1D("pass_numTPVsEta",
                                    "Reconstructable TrackingParticles (two or more connected SimDoublets "
                                    "pass cuts); True pseudorapidity #eta; Number of reconstructable TrackingParticles",
                                    etaNBins,
                                    etamin,
                                    etamax);
  h_numVsPt_ = simdoublets::make1DLogX(
      ibook,
      "numVsPt",
      "Total number of SimDoublets; True transverse momentum p_{T} [GeV]; Total number of SimDoublets",
      pTNBins,
      pTmin,
      pTmax);
  h_pass_numVsPt_ = simdoublets::make1DLogX(ibook,
                                            "pass_numVsPt",
                                            "Number of passing SimDoublets; True transverse momentum p_{T} [GeV]; "
                                            "Number of SimDoublets passing all cuts",
                                            pTNBins,
                                            pTmin,
                                            pTmax);
  h_numVsEta_ = ibook.book1D("numVsEta",
                             "Total number of SimDoublets; True pseudorapidity #eta; Total number of SimDoublets",
                             etaNBins,
                             etamin,
                             etamax);
  h_pass_numVsEta_ =
      ibook.book1D("pass_numVsEta",
                   "Number of SimDoublets; True pseudorapidity #eta; Number of SimDoublets passing all cuts",
                   etaNBins,
                   etamin,
                   etamax);

  // -------------------------------------------------------------
  // booking layer pair independent cut histograms (global folder)
  // -------------------------------------------------------------

  ibook.setCurrentFolder(folder_ + "/cutParameters/global");

  // histogram for z0cutoff  (z0Cut)
  h_z0_ = ibook.book1D("z0", "z_{0}; Longitudinal impact parameter z_{0} [cm]; Number of SimDoublets", 51, -1, 50);
  h_pass_z0_ = ibook.book1D(
      "pass_z0",
      "z_{0} of SimDoublets passing all cuts; Longitudinal impact parameter z_{0} [cm]; Number of SimDoublets",
      51,
      -1,
      50);

  // histograms for ptcut  (ptCut)
  h_curvatureR_ = ibook.book1D(
      "curvatureR", "Curvature from SimDoublet+beamspot; Curvature radius [cm] ; Number of SimDoublets", 100, 0, 1000);
  h_pTFromR_ = simdoublets::make1DLogX(
      ibook,
      "pTFromR",
      "Transverse momentum from curvature; Transverse momentum p_{T} [GeV]; Number of SimDoublets",
      pTNBins,
      pTmin,
      pTmax);
  h_pass_pTFromR_ = simdoublets::make1DLogX(ibook,
                                            "pass_pTFromR",
                                            "Transverse momentum from curvature of SimDoublets passing all cuts; "
                                            "Transverse momentum p_{T} [GeV]; Number of SimDoublets",
                                            pTNBins,
                                            pTmin,
                                            pTmax);

  // histograms for clusterCut  (minYsizeB1 and minYsizeB2)
  h_YsizeB1_ = ibook.book1D(
      "YsizeB1",
      "Cluster size along z (inner from B1); Size along z of inner cluster [num of pixels]; Number of SimDoublets",
      51,
      -1,
      50);
  h_YsizeB2_ = ibook.book1D(
      "YsizeB2",
      "Cluster size along z (inner not from B1); Size along z of inner cluster [num of pixels]; Number of SimDoublets",
      51,
      -1,
      50);
  h_pass_YsizeB1_ = ibook.book1D("pass_YsizeB1",
                                 "Cluster size along z of SimDoublets passing all cuts (inner from B1); Size along z "
                                 "of inner cluster [num of pixels]; Number of SimDoublets",
                                 51,
                                 -1,
                                 50);
  h_pass_YsizeB2_ = ibook.book1D("pass_YsizeB2",
                                 "Cluster size along z of SimDoublets passing all cuts (inner not from B1); Size along "
                                 "z of inner cluster [num of pixels]; Number of SimDoublets",
                                 51,
                                 -1,
                                 50);

  // histograms for zSizeCut  (maxDYsize12, maxDYsize and maxDYPred)
  h_DYsize12_ =
      ibook.book1D("DYsize12",
                   "Difference in cluster size along z (inner from B1); Absolute difference in cluster size along z of "
                   "the two RecHits [num of pixels]; Number of SimDoublets",
                   31,
                   -1,
                   30);
  h_DYsize_ = ibook.book1D("DYsize",
                           "Difference in cluster size along z; Absolute difference in cluster size along z of the two "
                           "RecHits [num of pixels]; Number of SimDoublets",
                           31,
                           -1,
                           30);
  h_DYPred_ = ibook.book1D("DYPred",
                           "Difference between actual and predicted cluster size along z of inner cluster; Absolute "
                           "difference [num of pixels]; Number of SimDoublets",
                           201,
                           -1,
                           200);
  h_pass_DYsize12_ = ibook.book1D("pass_DYsize12",
                                  "Difference in cluster size along z of SimDoublets passing all cuts (inner from B1); "
                                  "Absolute difference in cluster size along z of "
                                  "the two RecHits [num of pixels]; Number of SimDoublets",
                                  31,
                                  -1,
                                  30);
  h_pass_DYsize_ =
      ibook.book1D("pass_DYsize",
                   "Difference in cluster size along z of SimDoublets passing all cuts; Absolute difference in "
                   "cluster size along z of the two RecHits [num of pixels]; Number of SimDoublets",
                   31,
                   -1,
                   30);
  h_pass_DYPred_ =
      ibook.book1D("pass_DYPred",
                   "Difference between actual and predicted cluster size along z of inner cluster of SimDoublets "
                   "passing all cuts; Absolute difference [num of pixels]; Number of SimDoublets",
                   201,
                   -1,
                   200);

  // -----------------------------------------------------------------------
  // booking layer pair dependent histograms (sub-folders for layer pairs)
  // -----------------------------------------------------------------------

  // loop through valid layer pairs and add for each one booked hist per vector
  for (auto id = layerPairId2Index_.begin(); id != layerPairId2Index_.end(); ++id) {
    // get the position of the layer pair in the histogram vectors
    int layerPairIdIndex = id->second;

    // get layer names from the layer pair Id
    auto layerNames = simdoublets::getInnerOuterLayerNames(id->first);
    std::string innerLayerName = layerNames.first;
    std::string outerLayerName = layerNames.second;

    // name the sub-folder for the layer pair "lp_${innerLayerId}_${outerLayerId}"
    std::string subFolderName = "/cutParameters/lp_" + innerLayerName + "_" + outerLayerName;

    // layer mentioning in histogram titles
    std::string layerTitle = "(layers (" + innerLayerName + "," + outerLayerName + "))";

    // set folder to the sub-folder for the layer pair
    ibook.setCurrentFolder(folder_ + subFolderName);

    // histogram for z0cutoff  (maxr)
    hVector_dr_.at(layerPairIdIndex) = ibook.book1D(
        "dr",
        "dr of RecHit pair " + layerTitle + "; dr between outer and inner RecHit [cm]; Number of SimDoublets",
        31,
        -1,
        30);
    hVector_pass_dr_.at(layerPairIdIndex) = ibook.book1D(
        "pass_dr",
        "dr of RecHit pair " + layerTitle +
            " for SimDoublets passing all cuts; dr between outer and inner RecHit [cm]; Number of SimDoublets",
        31,
        -1,
        30);

    // histograms for iphicut  (phiCuts)
    hVector_dphi_.at(layerPairIdIndex) = ibook.book1D(
        "dphi",
        "dphi of RecHit pair " + layerTitle + "; d#phi between outer and inner RecHit [rad]; Number of SimDoublets",
        50,
        -M_PI,
        M_PI);
    hVector_idphi_.at(layerPairIdIndex) =
        ibook.book1D("idphi",
                     "idphi of RecHit pair " + layerTitle +
                         "; Absolute int d#phi between outer and inner RecHit; Number of SimDoublets",
                     50,
                     0,
                     1000);
    hVector_pass_idphi_.at(layerPairIdIndex) = ibook.book1D("pass_idphi",
                                                            "idphi of RecHit pair " + layerTitle +
                                                                " for SimDoublets passing all cuts; Absolute int d#phi "
                                                                "between outer and inner RecHit; Number of SimDoublets",
                                                            50,
                                                            0,
                                                            1000);

    // histogram for z window  (minz and maxz)
    hVector_innerZ_.at(layerPairIdIndex) =
        ibook.book1D("innerZ",
                     "z of the inner RecHit " + layerTitle + "; z of inner RecHit [cm]; Number of SimDoublets",
                     100,
                     -300,
                     300);
    hVector_pass_innerZ_.at(layerPairIdIndex) =
        ibook.book1D("pass_innerZ",
                     "z of the inner RecHit " + layerTitle +
                         " for SimDoublets passing all cuts; z of inner RecHit [cm]; Number of SimDoublets",
                     100,
                     -300,
                     300);

    // histograms for cluster size and size differences
    hVector_DYsize_.at(layerPairIdIndex) =
        ibook.book1D("DYsize",
                     "Difference in cluster size along z between outer and inner RecHit " + layerTitle +
                         "; Absolute difference in cluster size along z of the two "
                         "RecHits [num of pixels]; Number of SimDoublets",
                     51,
                     -1,
                     50);
    hVector_DYPred_.at(layerPairIdIndex) =
        ibook.book1D("DYPred",
                     "Difference between actual and predicted cluster size along z of inner cluster " + layerTitle +
                         "; Absolute difference [num of pixels]; Number of SimDoublets",
                     51,
                     -1,
                     50);
    hVector_Ysize_.at(layerPairIdIndex) = ibook.book1D(
        "Ysize",
        "Cluster size along z " + layerTitle + "; Size along z of inner cluster [num of pixels]; Number of SimDoublets",
        51,
        -1,
        50);
  }
}

// -------------------------------------------------------------------------------------------------------------
// fillDescriptions
// -------------------------------------------------------------------------------------------------------------

// dummy default fillDescription
template <typename TrackerTraits>
void SimDoubletsAnalyzer<TrackerTraits>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.addWithDefaultLabel(desc);
}

// fillDescription for Phase 1
template <>
void SimDoubletsAnalyzer<pixelTopology::Phase1>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  simdoublets::fillDescriptionsCommon<pixelTopology::Phase1>(desc);

  // input source for SimDoublets
  desc.add<edm::InputTag>("simDoubletsSrc", edm::InputTag("simDoubletsProducerPhase1"));

  // layer pairs in reconstruction
  desc.add<std::vector<int>>(
          "layerPairs",
          std::vector<int>(std::begin(phase1PixelTopology::layerPairs), std::end(phase1PixelTopology::layerPairs)))
      ->setComment(
          "Array of length 2*NumberOfPairs where the elements at 2i and 2i+1 are the inner and outer layers of layer "
          "pair i");

  // cutting parameters
  desc.add<int>("cellMinYSizeB1", 36)->setComment("Minimum cluster size for inner RecHit from B1");
  desc.add<int>("cellMinYSizeB2", 28)->setComment("Minimum cluster size for inner RecHit not from B1");
  desc.add<double>("cellZ0Cut", 12.0)->setComment("Maximum longitudinal impact parameter");
  desc.add<double>("cellPtCut", 0.5)->setComment("Minimum tranverse momentum");
  desc.add<std::vector<double>>(
          "cellMinz", std::vector<double>(std::begin(phase1PixelTopology::minz), std::end(phase1PixelTopology::minz)))
      ->setComment("Minimum z of inner RecHit for each layer pair");
  desc.add<std::vector<double>>(
          "cellMaxz", std::vector<double>(std::begin(phase1PixelTopology::maxz), std::end(phase1PixelTopology::maxz)))
      ->setComment("Maximum z of inner RecHit for each layer pair");
  desc.add<std::vector<int>>(
          "cellPhiCuts",
          std::vector<int>(std::begin(phase1PixelTopology::phicuts), std::end(phase1PixelTopology::phicuts)))
      ->setComment("Cuts in delta phi for cells");
  desc.add<std::vector<double>>(
          "cellMaxr", std::vector<double>(std::begin(phase1PixelTopology::maxr), std::end(phase1PixelTopology::maxr)))
      ->setComment("Cut for dr of cells");

  descriptions.addWithDefaultLabel(desc);
}

// fillDescription for Phase 2
template <>
void SimDoubletsAnalyzer<pixelTopology::Phase2>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  simdoublets::fillDescriptionsCommon<pixelTopology::Phase2>(desc);

  // input source for SimDoublets
  desc.add<edm::InputTag>("simDoubletsSrc", edm::InputTag("simDoubletsProducerPhase2"));

  // layer pairs in reconstruction
  desc.add<std::vector<int>>(
          "layerPairs",
          std::vector<int>(std::begin(phase2PixelTopology::layerPairs), std::end(phase2PixelTopology::layerPairs)))
      ->setComment(
          "Array of length 2*NumberOfPairs where the elements at 2i and 2i+1 are the inner and outer layers of layer "
          "pair i");

  // cutting parameters
  desc.add<int>("cellMinYSizeB1", 25)->setComment("Minimum cluster size for inner RecHit from B1");
  desc.add<int>("cellMinYSizeB2", 15)->setComment("Minimum cluster size for inner RecHit not from B1");
  desc.add<double>("cellZ0Cut", 7.5)->setComment("Maximum longitudinal impact parameter");
  desc.add<double>("cellPtCut", 0.85)->setComment("Minimum tranverse momentum");
  desc.add<std::vector<double>>(
          "cellMinz", std::vector<double>(std::begin(phase2PixelTopology::minz), std::end(phase2PixelTopology::minz)))
      ->setComment("Minimum z of inner RecHit for each layer pair");
  desc.add<std::vector<double>>(
          "cellMaxz", std::vector<double>(std::begin(phase2PixelTopology::maxz), std::end(phase2PixelTopology::maxz)))
      ->setComment("Maximum z of inner RecHit for each layer pair");
  desc.add<std::vector<int>>(
          "cellPhiCuts",
          std::vector<int>(std::begin(phase2PixelTopology::phicuts), std::end(phase2PixelTopology::phicuts)))
      ->setComment("Cuts in delta phi for cells");
  desc.add<std::vector<double>>(
          "cellMaxr", std::vector<double>(std::begin(phase2PixelTopology::maxr), std::end(phase2PixelTopology::maxr)))
      ->setComment("Cut for dr of cells");

  descriptions.addWithDefaultLabel(desc);
}

// define two plugins for Phase 1 and 2
using SimDoubletsAnalyzerPhase1 = SimDoubletsAnalyzer<pixelTopology::Phase1>;
using SimDoubletsAnalyzerPhase2 = SimDoubletsAnalyzer<pixelTopology::Phase2>;

// define this as a plug-in
DEFINE_FWK_MODULE(SimDoubletsAnalyzerPhase1);
DEFINE_FWK_MODULE(SimDoubletsAnalyzerPhase2);
