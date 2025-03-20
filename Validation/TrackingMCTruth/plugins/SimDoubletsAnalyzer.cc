// -*- C++ -*-
//
// Package:    Validation/TrackingMCTruth
// Class:      SimDoubletsAnalyzer
//

// user include files
#include "Validation/TrackingMCTruth/plugins/SimDoubletsAnalyzer.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/Histograms/interface/MonitorElementCollection.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/approx_atan2.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "RecoTracker/PixelSeeding/interface/CircleEq.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDoubletsAnalyzer.h"

#include <cstddef>

namespace simdoublets {
  // class that calculate and stores all cut variables for a given doublet
  struct CellCutVariables {
    void calculateCutVariables(SimDoublets::Doublet const& doublet, SimDoublets const& simDoublets) {
      // inner RecHit properties
      GlobalPoint inner_globalPosition = doublet.innerGlobalPos();
      inner_z_ = inner_globalPosition.z();
      double inner_r = inner_globalPosition.perp();
      double inner_phi = inner_globalPosition.barePhi();  // returns float, whereas .phi() returns phi object
      double inner_x = inner_globalPosition.x();
      double inner_y = inner_globalPosition.y();
      int inner_iphi = unsafe_atan2s<7>(inner_globalPosition.y(), inner_x);
      // outer RecHit properties
      GlobalPoint outer_globalPosition = doublet.outerGlobalPos();
      double outer_z = outer_globalPosition.z();
      double outer_r = outer_globalPosition.perp();
      double outer_x = outer_globalPosition.x();
      double outer_y = outer_globalPosition.y();
      double outer_phi = outer_globalPosition.barePhi();
      int outer_iphi = unsafe_atan2s<7>(outer_globalPosition.y(), outer_globalPosition.x());

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

      // cuts on doublet connections (loop over all inner neighboring doublets)
      for (auto const& neighbor : doublet.innerNeighborsView()) {
        // get the inner RecHit of the inner neighbor
        GlobalPoint neighbor_globalPosition = simDoublets.getSimDoublet(neighbor.index()).innerGlobalPos();
        double neighbor_z = neighbor_globalPosition.z();
        double neighbor_r = neighbor_globalPosition.perp();
        double neighbor_x = neighbor_globalPosition.x();
        double neighbor_y = neighbor_globalPosition.y();

        // alignement cut variable in R-Z assuming ptmin = 1 GeV
        double radius_diff = std::abs(neighbor_r - outer_r);
        double distance_13_squared = radius_diff * radius_diff + (neighbor_z - outer_z) * (neighbor_z - outer_z);
        double tan_12_13_half_mul_distance_13_squared = fabs(
            neighbor_z * (inner_r - outer_r) + inner_z_ * (outer_r - neighbor_r) + outer_z * (neighbor_r - inner_r));
        double denominator = std::sqrt(distance_13_squared) * radius_diff;
        CAThetaCut_.push_back(tan_12_13_half_mul_distance_13_squared / denominator);

        // alignement cut variables in x-y
        CircleEq<double> eq(neighbor_x, neighbor_y, inner_x, inner_y, outer_x, outer_y);
        hardCurvCut_.push_back(eq.curvature());
        dcaCut_.push_back(std::abs(eq.dca0() / std::abs(eq.curvature())));
      }
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
    std::vector<double> const& CAThetaCut() const { return CAThetaCut_; }
    std::vector<double> const& dcaCut() const { return dcaCut_; }
    std::vector<double> const& hardCurvCut() const { return hardCurvCut_; }
    double CAThetaCut(int i) const { return CAThetaCut_.at(i); }
    double dcaCut(int i) const { return dcaCut_.at(i); }
    double hardCurvCut(int i) const { return hardCurvCut_.at(i); }

  private:
    double inner_z_, dz_, dr_, dphi_, z0_, curvature_, pT_;  // double-valued variables
    int idphi_, Ysize_, DYsize_, DYPred_;                    // integer-valued variables
    std::vector<double> CAThetaCut_, dcaCut_, hardCurvCut_;  // doublet connection cut variables
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
      const bool onlyBarrel = innerInBarrel && outerInBarrel;

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
      if ((!(innerInB1) || isOuterLadder) && (innerInBarrel || onlyBarrel)) {
        if (onlyBarrel) {  // onlyBarrel
          if (innerInB1) {
            setSubjectToDYsize12();
          } else {
            setSubjectToDYsize();
          }
        } else {  // innerInBarrel but not onlyBarrel
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

    // cut for minimum number of RecHits required for an Ntuplet
    desc.add<int>("minHitsPerNtuplet", 4)->setComment("Cut on minimum number of RecHits required for an Ntuplet");

    // cut parameters for connecting doublets
    desc.add<double>("CAThetaCutBarrel", 0.002)->setComment("Cut on RZ alignement for Barrel in GeV");
    desc.add<double>("CAThetaCutForward", 0.003)->setComment("Cut on RZ alignement for Forward in GeV");
    desc.add<double>("ptmin", 0.9)
        ->setComment(
            "Minimum tranverse momentum considered for the multiple scattering expectation when checking alignement in "
            "R-z plane of two doublets in GeV");
    desc.add<double>("dcaCutInnerTriplet", 0.15)->setComment("Cut on origin radius when the inner hit is on BPix1");
    desc.add<double>("dcaCutOuterTriplet", 0.25)->setComment("Cut on origin radius when the inner hit is not on BPix1");
    desc.add<double>("hardCurvCut", 1. / (0.35 * 87.))
        ->setComment("Cut on minimum curvature, used in DCA ntuplet selection");

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
      CAThetaCutBarrel_over_ptmin_(iConfig.getParameter<double>("CAThetaCutBarrel") /
                                   iConfig.getParameter<double>("ptmin")),
      CAThetaCutForward_over_ptmin_(iConfig.getParameter<double>("CAThetaCutForward") /
                                    iConfig.getParameter<double>("ptmin")),
      dcaCutInnerTriplet_(iConfig.getParameter<double>("dcaCutInnerTriplet")),
      dcaCutOuterTriplet_(iConfig.getParameter<double>("dcaCutOuterTriplet")),
      hardCurvCut_(iConfig.getParameter<double>("hardCurvCut")),
      minNumDoubletsPerNtuplet_(iConfig.getParameter<int>("minHitsPerNtuplet") - 1),
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
void SimDoubletsAnalyzer<TrackerTraits>::applyCuts(
    SimDoublets::Doublet& doublet,
    bool const hasValidNeighbors,
    int const layerPairIdIndex,
    simdoublets::CellCutVariables const& cellCutVariables,
    simdoublets::ClusterSizeCutManager const& clusterSizeCutManager) const {
  // -------------------------------------------------------------------------
  //  apply cuts for doublet creation
  // -------------------------------------------------------------------------

  if (/* z window cut */
      (cellCutVariables.inner_z() < cellMinz_[layerPairIdIndex] ||
       cellCutVariables.inner_z() > cellMaxz_[layerPairIdIndex]) ||
      /* z0cutoff */
      (cellCutVariables.dr() > cellMaxr_[layerPairIdIndex] || cellCutVariables.dr() < 0 ||
       cellCutVariables.z0() > cellZ0Cut_) ||
      /* ptcut */
      (cellCutVariables.pT() < cellPtCut_) ||
      /* idphicut */
      (cellCutVariables.idphi() > cellPhiCuts_[layerPairIdIndex]) ||
      /* YsizeB1 cut */
      (clusterSizeCutManager.isSubjectToYsizeB1() && (cellCutVariables.Ysize() < cellMinYSizeB1_)) ||
      /* YsizeB2 cut */
      (clusterSizeCutManager.isSubjectToYsizeB2() && (cellCutVariables.Ysize() < cellMinYSizeB2_)) ||
      /* DYsize12 cut */
      (clusterSizeCutManager.isSubjectToDYsize12() && (cellCutVariables.DYsize() > cellMaxDYSize12_)) ||
      /* DYsize cut */
      (clusterSizeCutManager.isSubjectToDYsize() && (cellCutVariables.DYsize() > cellMaxDYSize_)) ||
      /* DYPred cut */
      (clusterSizeCutManager.isSubjectToDYPred() && (cellCutVariables.DYPred() > cellMaxDYPred_))) {
    // if any of the cuts apply kill the doublet
    doublet.setKilledByCuts();
  } else {
    // if the function arrives here, the doublet survived
    doublet.setAlive();
  }

  // -------------------------------------------------------------------------
  //  apply cuts for doublet connections
  // -------------------------------------------------------------------------
  if (hasValidNeighbors) {
    // loop over the inner neighboring doublets of the doublet
    for (int i{0}; auto& neighbor : doublet.innerNeighbors()) {
      if (
          /* apply CAThetaCut
          use Barrel if the centered common RecHit of the connection is in the barrel
          else Forward */
          (cellCutVariables.CAThetaCut(i) >
           ((doublet.innerLayerId() < 4) ? CAThetaCutBarrel_over_ptmin_ : CAThetaCutForward_over_ptmin_)) ||
          /* apply hardCurvCut */
          (cellCutVariables.hardCurvCut(i) > hardCurvCut_) ||
          /* apply dcaCut 
          use Inner if the inner RecHit is on BPix 1
          else Outer */
          (cellCutVariables.dcaCut(i) >
           ((doublet.innerNeighborsInnerLayerId() == 0) ? dcaCutInnerTriplet_ : dcaCutOuterTriplet_))) {
        neighbor.setKilled();
      } else {
        neighbor.setAlive();
      }

      i++;
    }
  }
}

// function that fills all histograms for cut variables
template <typename TrackerTraits>
void SimDoubletsAnalyzer<TrackerTraits>::fillCutHistograms(
    SimDoublets::Doublet const& doublet,
    bool hasValidNeighbors,
    int const layerPairIdIndex,
    simdoublets::CellCutVariables const& cellCutVariables,
    simdoublets::ClusterSizeCutManager const& clusterSizeCutManager) {
  // check if the doublet passed all cuts
  bool passed = doublet.isAlive();

  // -------------------------------------------------------------------------
  //  layer pair independent cuts (global folder)
  // -------------------------------------------------------------------------
  // radius of the circle defined by the two RecHits and the beamspot
  h_curvatureR_.fill(passed, cellCutVariables.curvature());
  // pT that this curvature radius corresponds to
  h_pTFromR_.fill(passed, cellCutVariables.pT());
  // longitudinal impact parameter with respect to the beamspot
  h_z0_.fill(passed, cellCutVariables.z0());

  // -------------------------------------------------------------------------
  //  layer pair dependent cuts (sub-folders for layer pairs)
  // -------------------------------------------------------------------------
  // dr = (outer_r - inner_r) histogram
  hVector_dr_[layerPairIdIndex].fill(passed, cellCutVariables.dr());
  // dphi histogram
  hVector_dphi_[layerPairIdIndex].fill(passed, cellCutVariables.dphi());
  hVector_idphi_[layerPairIdIndex].fill(passed, cellCutVariables.idphi());
  // z of the inner RecHit histogram
  hVector_innerZ_[layerPairIdIndex].fill(passed, cellCutVariables.inner_z());

  // -------------------------------------------------------------------------
  //  cluster size cuts (global + sub-folders for layer pairs)
  // -------------------------------------------------------------------------
  // cluster size in local y histogram
  hVector_Ysize_[layerPairIdIndex].fill(passed, cellCutVariables.Ysize());
  // histograms for clusterCut
  // YsizeB1 cut
  if (clusterSizeCutManager.isSubjectToYsizeB1()) {
    h_YsizeB1_.fill(passed, cellCutVariables.Ysize());
  }
  // YsizeB2 cut
  if (clusterSizeCutManager.isSubjectToYsizeB2()) {
    h_YsizeB2_.fill(passed, cellCutVariables.Ysize());
  }
  // histograms for zSizeCut
  // DYsize12 cut
  if (clusterSizeCutManager.isSubjectToDYsize12()) {
    hVector_DYsize_[layerPairIdIndex].fill(passed, cellCutVariables.DYsize());
    h_DYsize12_.fill(passed, cellCutVariables.DYsize());
  }
  // DYsize cut
  if (clusterSizeCutManager.isSubjectToDYsize()) {
    hVector_DYsize_[layerPairIdIndex].fill(passed, cellCutVariables.DYsize());
    h_DYsize_.fill(passed, cellCutVariables.DYsize());
  }
  // DYPred cut
  if (clusterSizeCutManager.isSubjectToDYPred()) {
    hVector_DYPred_[layerPairIdIndex].fill(passed, cellCutVariables.DYPred());
    h_DYPred_.fill(passed, cellCutVariables.DYPred());
  }

  // -------------------------------------------------------------------------
  //  connection cuts (connectionCuts folder)
  // -------------------------------------------------------------------------
  bool passedConnect;
  // check if connection cut histograms should be filled
  if (hasValidNeighbors) {
    // loop over the inner neighboring doublets of the doublet
    for (int i{0}; auto const& neighbor : doublet.innerNeighborsView()) {
      // get the status of the connection
      passedConnect = neighbor.isAlive();

      // fill the histograms
      // hard curvature cut
      h_hardCurvCut_.fill(passedConnect, cellCutVariables.hardCurvCut(i));
      // dca cut
      if (doublet.innerNeighborsInnerLayerId() == 0) {
        h_dcaCutInnerTriplet_.fill(passedConnect, cellCutVariables.dcaCut(i));
      } else {
        h_dcaCutOuterTriplet_.fill(passedConnect, cellCutVariables.dcaCut(i));
      }
      // CATheta cut
      if (doublet.innerLayerId() < 4) {
        h_CAThetaCutBarrel_.fill(passedConnect, cellCutVariables.CAThetaCut(i));
      } else {
        h_CAThetaCutForward_.fill(passedConnect, cellCutVariables.CAThetaCut(i));
      }

      i++;
    }
  }
}

// main function that fills the histograms
template <typename TrackerTraits>
void SimDoubletsAnalyzer<TrackerTraits>::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  // get tracker topology
  trackerTopology_ = &iSetup.getData(topology_getToken_);
  // get simDoublets
  SimDoubletsCollection const& simDoubletsCollection = iEvent.get(simDoublets_getToken_);

  // initialize a bunch of variables that we will use in the coming for loops
  double true_pT, true_eta, relativeLength;
  int numSimDoublets, pass_numSimDoublets, layerPairId, layerPairIdIndex;
  bool passed, passedTP, hasValidNeighbors, isAlive;

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
    auto& doublets = simDoublets.buildAndGetSimDoublets(trackerTopology_);
    // number of SimDoublets of the Tracking Particle
    numSimDoublets = doublets.size();
    // number of SimDoublets of the Tracking Particle passing all cuts
    pass_numSimDoublets = 0;

    // loop over those doublets
    for (auto& doublet : doublets) {
      // reset clusterSizeCutManager to "no cluster cuts applied"
      clusterSizeCutManager.reset();

      // calculate the cut variables for the given doublet
      cellCutVariables.calculateCutVariables(doublet, simDoublets);

      // first, get layer pair Id and exclude layer pairs that are not considered
      layerPairId = doublet.layerPairId();
      if (layerPairId2Index_.find(layerPairId) != layerPairId2Index_.end()) {
        // get the position of the layer pair in the vectors of histograms
        layerPairIdIndex = layerPairId2Index_.at(layerPairId);

        // check if the SimDoublet's inner neighbors also are from a considered layer pair
        hasValidNeighbors = (doublet.numInnerNeighbors() > 0 &&
                             !(simDoublets.getSimDoublet(doublet.innerNeighborIndex(0)).isKilledByMissingLayerPair()));

        // determine which cluster size cuts the doublet is subject to
        clusterSizeCutManager.setSubjectsToCuts(doublet);

        // apply the cuts for doublet building according to the set cut values
        applyCuts(doublet, hasValidNeighbors, layerPairIdIndex, cellCutVariables, clusterSizeCutManager);

        // -------------------------------------------------------------------------
        //  cut histograms for SimDoublets (cutParameters folder)
        // -------------------------------------------------------------------------
        fillCutHistograms(doublet, hasValidNeighbors, layerPairIdIndex, cellCutVariables, clusterSizeCutManager);

      } else {
        // if not considered set the doublet as killed
        doublet.setKilledByMissingLayerPair();
      }

      passed = doublet.isAlive();  // true if doublet passed

      // ---------------------------------------------------------------------------
      //  general plots related to SimDoublets (general folder)
      // ---------------------------------------------------------------------------
      // fill histograms for SimDoublet numbers
      h_numVsPt_.fill(passed, true_pT);
      h_numVsEta_.fill(passed, true_eta);
      // layer pair combinations
      h_layerPairs_.fill(passed, doublet.innerLayerId(), doublet.outerLayerId());
      // number of skipped layers by SimDoublets
      h_numSkippedLayers_.fill(passed, doublet.numSkippedLayers());

      // if the doublet passes all cuts
      if (passed) {
        // increment number of SimDoublets passing all cuts
        pass_numSimDoublets++;
      }
    }  // end loop over those doublets

    // build the SimNtuplets based on the SimDoublets
    simDoublets.buildSimNtuplets(minNumDoubletsPerNtuplet_);

    // -----------------------------------------------------------------------------
    //  plots related to SimNtuplets (simNtuplet folder)
    // -----------------------------------------------------------------------------
    // Now check if the TrackingParticle has a surviving SimNtuplet
    passedTP = simDoublets.hasAliveSimNtuplet();

    // get the longest SimNtuplet of the TrackingParticle (if it exists)
    if (simDoublets.hasSimNtuplet()) {
      auto const& longNtuplet = simDoublets.longestSimNtuplet();

      // check if it is alive
      isAlive = longNtuplet.isAlive();

      // fill general longest SimNtuplet histogram
      h_longNtuplet_numRecHits_.fill(isAlive, longNtuplet.numRecHits());
      h_longNtuplet_firstLayerId_.fill(isAlive, longNtuplet.firstLayerId());
      h_longNtuplet_lastLayerId_.fill(isAlive, longNtuplet.lastLayerId());
      h_longNtuplet_layerSpan_.fill(isAlive, longNtuplet.firstLayerId(), longNtuplet.lastLayerId());

      // fill the respective histogram
      // 1. check if alive
      if (isAlive) {
        h_longNtuplet_alive_eta_->Fill(true_eta);
        h_longNtuplet_alive_pt_->Fill(true_pT);
      }
      // 2. if not alive, find out why (go in order of cut application)
      // A) the layer pair is missing, therefore no doublet is formed
      else if (longNtuplet.hasMissingLayerPair()) {
        h_longNtuplet_missingLayerPair_eta_->Fill(true_eta);
        h_longNtuplet_missingLayerPair_pt_->Fill(true_pT);
      }
      // B) the doublet got killed by doublet building cuts
      else if (longNtuplet.hasKilledDoublets()) {
        h_longNtuplet_killedDoublets_eta_->Fill(true_eta);
        h_longNtuplet_killedDoublets_pt_->Fill(true_pT);
      }
      // C) one of connections between the doublets got cut
      else if (longNtuplet.hasKilledConnections()) {
        h_longNtuplet_killedConnections_eta_->Fill(true_eta);
        h_longNtuplet_killedConnections_pt_->Fill(true_pT);
      }
      // D) the Ntuplet is build but does not meet the minimum length requirement
      else if (longNtuplet.isTooShort()) {
        h_longNtuplet_tooShort_eta_->Fill(true_eta);
        h_longNtuplet_tooShort_pt_->Fill(true_pT);
      }
      // E) if we arrive here something's wrong
      else if (longNtuplet.hasUndefDoubletCuts()) {
        h_longNtuplet_undefDoubletCuts_eta_->Fill(true_eta);
        h_longNtuplet_undefDoubletCuts_pt_->Fill(true_pT);
      }
      // F) or even wronger...
      else if (longNtuplet.hasUndefDoubletConnectionCuts()) {
        h_longNtuplet_undefConnectionCuts_eta_->Fill(true_eta);
        h_longNtuplet_undefConnectionCuts_pt_->Fill(true_pT);
      }

      // fill the longest alive histograms (if it exists)
      if (passedTP) {
        auto const& aliveNtuplet = simDoublets.longestAliveSimNtuplet();

        h_aliveNtuplet_numRecHits_->Fill(aliveNtuplet.numRecHits());
        h_aliveNtuplet_firstLayerId_->Fill(aliveNtuplet.firstLayerId());
        h_aliveNtuplet_lastLayerId_->Fill(aliveNtuplet.lastLayerId());
        h_aliveNtuplet_layerSpan_->Fill(aliveNtuplet.firstLayerId(), aliveNtuplet.lastLayerId());

        // relative length of alive SimNtuplet vs longest SimNtuplet
        relativeLength = aliveNtuplet.numRecHits() / longNtuplet.numRecHits();
        h_aliveNtuplet_fracNumRecHits_eta_->Fill(true_eta, relativeLength);
        h_aliveNtuplet_fracNumRecHits_pt_->Fill(true_pT, relativeLength);
      }
    }
    // fill all-histogram
    h_longNtuplet_pt_.fill(passedTP, true_pT);
    h_longNtuplet_eta_.fill(passedTP, true_eta);

    // -----------------------------------------------------------------------------
    //  general plots related to TrackingParticles (general folder)
    // -----------------------------------------------------------------------------
    // fill histograms for number of SimDoublets
    h_numSimDoubletsPerTrackingParticle_.fill(passedTP, numSimDoublets);
    h_numLayersPerTrackingParticle_.fill(passedTP, simDoublets.numLayers());
    // fill histograms for number of TrackingParticles
    h_numTPVsPt_.fill(passedTP, true_pT);
    h_numTPVsEta_.fill(passedTP, true_eta);
    // Fill the efficiency profile per Tracking Particle only if the TP has at least one SimDoublet
    if (numSimDoublets > 0) {
      h_effSimDoubletsPerTPVsEta_->Fill(true_eta, pass_numSimDoublets / numSimDoublets);
      h_effSimDoubletsPerTPVsPt_->Fill(true_pT, pass_numSimDoublets / numSimDoublets);
    }

    // clear SimDoublets and SimNtuplets of the TrackingParticle
    simDoublets.clearMutables();
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
  int etaNBins = 90;
  double etamin = -4.5;
  double etamax = 4.5;

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
  h_layerPairs_.book2D(ibook,
                       "layerPairs",
                       "Layer pairs in SimDoublets",
                       "Inner layer ID",
                       "Outer layer ID",
                       TrackerTraits::numberOfLayers,
                       -0.5,
                       -0.5 + TrackerTraits::numberOfLayers,
                       TrackerTraits::numberOfLayers,
                       -0.5,
                       -0.5 + TrackerTraits::numberOfLayers);
  h_numSkippedLayers_.book1D(ibook,
                             "numSkippedLayers",
                             "Number of skipped layers",
                             "Number of skipped layers",
                             "Number of SimDoublets",
                             16,
                             -1.5,
                             14.5);
  h_numSimDoubletsPerTrackingParticle_.book1D(ibook,
                                              "numSimDoubletsPerTrackingParticle",
                                              "Number of SimDoublets per Tracking Particle",
                                              "Number of SimDoublets",
                                              "Number of Tracking Particles",
                                              31,
                                              -0.5,
                                              30.5);
  h_numLayersPerTrackingParticle_.book1D(ibook,
                                         "numLayersPerTrackingParticle",
                                         "Number of layers hit by Tracking Particle",
                                         "Number of layers",
                                         "Number of Tracking Particles",
                                         29,
                                         -0.5,
                                         28.5);
  h_numTPVsPt_.book1DLogX(ibook,
                          "numTPVsPt",
                          "Number of TrackingParticles",
                          "True transverse momentum p_{T} [GeV]",
                          "Number of TrackingParticles",
                          pTNBins,
                          pTmin,
                          pTmax);
  h_numTPVsEta_.book1D(ibook,
                       "numTPVsEta",
                       "Number of TrackingParticles",
                       "True pseudorapidity #eta",
                       "Number of TrackingParticles",
                       etaNBins,
                       etamin,
                       etamax);
  h_numVsPt_.book1DLogX(ibook,
                        "numVsPt",
                        "Number of SimDoublets",
                        "True transverse momentum p_{T} [GeV]",
                        "Number of SimDoublets",
                        pTNBins,
                        pTmin,
                        pTmax);
  h_numVsEta_.book1D(ibook,
                     "numVsEta",
                     "Number of SimDoublets",
                     "True pseudorapidity #eta",
                     "Number of SimDoublets",
                     etaNBins,
                     etamin,
                     etamax);

  // --------------------------------------------------------------
  // booking layer pair independent cut histograms (global folder)
  // --------------------------------------------------------------

  ibook.setCurrentFolder(folder_ + "/cutParameters/global");

  // histogram for z0cutoff  (z0Cut)
  h_z0_.book1D(ibook,
               "z0",
               "z_{0} of SimDoublets",
               "Longitudinal impact parameter z_{0} [cm]",
               "Number of SimDoublets",
               51,
               -1,
               50);

  // histograms for ptcut  (ptCut)
  h_curvatureR_.book1DLogX(ibook,
                           "curvatureR",
                           "Curvature from 3 points of beamspot + RecHits of SimDoublets",
                           "Curvature radius [cm]",
                           "Number of SimDoublets",
                           100,
                           2,
                           4);
  h_pTFromR_.book1DLogX(ibook,
                        "pTFromR",
                        "Transverse momentum from curvature",
                        "Transverse momentum p_{T} [GeV]",
                        "Number of SimDoublets",
                        pTNBins,
                        pTmin,
                        pTmax);

  // histograms for clusterCut  (minYsizeB1 and minYsizeB2)
  h_YsizeB1_.book1D(ibook,
                    "YsizeB1",
                    "Cluster size along z of inner RecHit [from BPix1]",
                    "Size along z of inner cluster [num of pixels]",
                    "Number of SimDoublets",
                    51,
                    -1,
                    50);
  h_YsizeB2_.book1D(ibook,
                    "YsizeB2",
                    "Cluster size along z of inner RecHit [not from BPix1]",
                    "Size along z of inner cluster [num of pixels]",
                    "Number of SimDoublets",
                    51,
                    -1,
                    50);
  // histograms for zSizeCut  (maxDYsize12, maxDYsize and maxDYPred)
  h_DYsize12_.book1D(ibook,
                     "DYsize12",
                     "Difference in cluster size along z [inner from BPix1]",
                     "Absolute difference in cluster size along z of "
                     "the two RecHits [num of pixels]",
                     "Number of SimDoublets",
                     31,
                     -1,
                     30);
  h_DYsize_.book1D(ibook,
                   "DYsize",
                   "Difference in cluster size along z [inner not from BPix1]",
                   "Absolute difference in cluster size along z of the two RecHits [num of pixels]",
                   "Number of SimDoublets",
                   31,
                   -1,
                   30);
  h_DYPred_.book1D(ibook,
                   "DYPred",
                   "Difference between actual and predicted cluster size along z of inner cluster",
                   "Absolute difference [num of pixels]",
                   "Number of SimDoublets",
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
    hVector_dr_.at(layerPairIdIndex)
        .book1D(ibook,
                "dr",
                "dr of RecHit pair " + layerTitle,
                "dr between outer and inner RecHit [cm]",
                "Number of SimDoublets",
                31,
                -1,
                30);

    // histograms for iphicut  (phiCuts)
    hVector_dphi_.at(layerPairIdIndex)
        .book1D(ibook,
                "dphi",
                "dphi of RecHit pair " + layerTitle,
                "d#phi between outer and inner RecHit [rad]",
                "Number of SimDoublets",
                50,
                -M_PI,
                M_PI);
    hVector_idphi_.at(layerPairIdIndex)
        .book1D(ibook,
                "idphi",
                "idphi of RecHit pair " + layerTitle,
                "Absolute int d#phi between outer and inner RecHit",
                "Number of SimDoublets",
                50,
                0,
                1000);

    // histogram for z window  (minz and maxz)
    hVector_innerZ_.at(layerPairIdIndex)
        .book1D(ibook,
                "innerZ",
                "z of the inner RecHit " + layerTitle,
                "z of inner RecHit [cm]",
                "Number of SimDoublets",
                100,
                -300,
                300);

    // histograms for cluster size and size differences
    hVector_DYsize_.at(layerPairIdIndex)
        .book1D(ibook,
                "DYsize",
                "Difference in cluster size along z between outer and inner RecHit " + layerTitle,
                "Absolute difference in cluster size along z of the two RecHits [num of pixels]",
                "Number of SimDoublets",
                51,
                -1,
                50);
    hVector_DYPred_.at(layerPairIdIndex)
        .book1D(ibook,
                "DYPred",
                "Difference between actual and predicted cluster size along z of inner cluster " + layerTitle,
                "Absolute difference [num of pixels]",
                "Number of SimDoublets",
                51,
                -1,
                50);
    hVector_Ysize_.at(layerPairIdIndex)
        .book1D(ibook,
                "Ysize",
                "Cluster size along z " + layerTitle,
                "Size along z of inner cluster [num of pixels]",
                "Number of SimDoublets",
                51,
                -1,
                50);
  }

  // -----------------------------------------------------------------
  // booking connection cut histograms (connectionCuts folder)
  // -----------------------------------------------------------------

  ibook.setCurrentFolder(folder_ + "/cutParameters/connectionCuts");

  // histogram for areAlignedRZ
  h_CAThetaCutBarrel_.book1DLogX(ibook,
                                 "CAThetaBarrel_over_ptmin",
                                 "CATheta cut variable based on the area spaned by 3 RecHits of a pair of neighboring "
                                 "SimDoublets in R-z with the shared RecHit in the barrel",
                                 "CATheta cut variable",
                                 "Number of SimDoublet connections",
                                 51,
                                 -6,
                                 1);
  h_CAThetaCutForward_.book1DLogX(ibook,
                                  "CAThetaForward_over_ptmin",
                                  "CATheta cut variable based on the area spaned by 3 RecHits of a pair of neighboring "
                                  "SimDoublets in R-z with the shared RecHit in the endcap",
                                  "CATheta cut variable",
                                  "Number of SimDoublet connections",
                                  51,
                                  -6,
                                  1);

  // histogram for dcaCut (x-y alignement)
  h_hardCurvCut_.book1D(ibook,
                        "hardCurv",
                        "Curvature of a pair of neighboring SimDoublets",
                        "Curvature [1/cm]",
                        "Number of SimDoublet connections",
                        51,
                        -0.004,
                        0.004);
  h_dcaCutInnerTriplet_.book1DLogX(ibook,
                                   "dcaInner",
                                   "Closest transverse distance to beamspot based on the 3 RecHits of a pair of "
                                   "neighboring SimDoublets with the most inner RecHit on BPix1",
                                   "Transverse distance [cm]",
                                   "Number of SimDoublet connections",
                                   51,
                                   -6,
                                   1);
  h_dcaCutOuterTriplet_.book1DLogX(ibook,
                                   "dcaOuter",
                                   "Closest transverse distance to beamspot based on the 3 RecHits of a pair of "
                                   "neighboring SimDoublets with the most inner RecHit not on BPix1",
                                   "Transverse distance [cm]",
                                   "Number of SimDoublet connections",
                                   51,
                                   -6,
                                   1);

  // -----------------------------------------------------------------
  // booking SimNtuplet histograms (simNtuplets folder)
  // -----------------------------------------------------------------

  ibook.setCurrentFolder(folder_ + "/simNtuplets");

  // histograms of the TrackingParticles
  h_aliveNtuplet_numRecHits_ = ibook.book1D("alive_numRecHits",
                                            "Number of RecHits in longest alive SimNtuplet per TrackingParticle;Number "
                                            "of RecHits;Number of TrackingParticles",
                                            15,
                                            -0.5,
                                            14.5);
  h_aliveNtuplet_firstLayerId_ =
      ibook.book1D("alive_firstLayerId",
                   "First layer of longest alive SimNtuplet per TrackingParticle;Layer ID;Number of TrackingParticles",
                   TrackerTraits::numberOfLayers,
                   -0.5,
                   -0.5 + TrackerTraits::numberOfLayers);
  h_aliveNtuplet_lastLayerId_ =
      ibook.book1D("alive_lastLayerId",
                   "Last layer of longest alive SimNtuplet per TrackingParticle;Layer ID;Number of TrackingParticles",
                   TrackerTraits::numberOfLayers,
                   -0.5,
                   -0.5 + TrackerTraits::numberOfLayers);
  h_aliveNtuplet_layerSpan_ =
      ibook.book2D("alive_layerSpan",
                   "Layer span of  longest alive SimNtuplet per TrackingParticle;First layer ID;Last layer ID",
                   TrackerTraits::numberOfLayers,
                   -0.5,
                   -0.5 + TrackerTraits::numberOfLayers,
                   TrackerTraits::numberOfLayers,
                   -0.5,
                   -0.5 + TrackerTraits::numberOfLayers);
  h_aliveNtuplet_fracNumRecHits_pt_ =
      simdoublets::makeProfileLogX(ibook,
                                   "alive_fracNumRecHits_vs_pT",
                                   "Number of RecHits of longest alive SimNtuplet over number of RecHits of longest "
                                   "SimNtuplet per TrackingParticle; TP transverse momentum p_{T} [GeV]; "
                                   "Ratio",
                                   pTNBins,
                                   pTmin,
                                   pTmax,
                                   0,
                                   1,
                                   " ");
  h_aliveNtuplet_fracNumRecHits_eta_ =
      ibook.bookProfile("alive_fracNumRecHits_vs_eta",
                        "Number of RecHits of longest alive SimNtuplet over number of "
                        "RecHits of longest SimNtuplet per TrackingParticle; TP transverse momentum #eta; "
                        "Ratio",
                        etaNBins,
                        etamin,
                        etamax,
                        0,
                        1,
                        " ");
  h_longNtuplet_numRecHits_.book1D(ibook,
                                   "numRecHits",
                                   "Number of RecHits in longest SimNtuplet per TrackingParticle",
                                   "Number of RecHits",
                                   "Number of TrackingParticles",
                                   15,
                                   -0.5,
                                   14.5);
  h_longNtuplet_firstLayerId_.book1D(ibook,
                                     "firstLayerId",
                                     "First layer of longest SimNtuplet per TrackingParticle",
                                     "Layer ID",
                                     "Number of TrackingParticles",
                                     TrackerTraits::numberOfLayers,
                                     -0.5,
                                     -0.5 + TrackerTraits::numberOfLayers);
  h_longNtuplet_lastLayerId_.book1D(ibook,
                                    "lastLayerId",
                                    "Last layer of longest SimNtuplet per TrackingParticle",
                                    "Layer ID",
                                    "Number of TrackingParticles",
                                    TrackerTraits::numberOfLayers,
                                    -0.5,
                                    -0.5 + TrackerTraits::numberOfLayers);
  h_longNtuplet_layerSpan_.book2D(ibook,
                                  "layerSpan",
                                  "Layer span of longest SimNtuplet per TrackingParticle",
                                  "First layer ID",
                                  "Last layer ID",
                                  TrackerTraits::numberOfLayers,
                                  -0.5,
                                  -0.5 + TrackerTraits::numberOfLayers,
                                  TrackerTraits::numberOfLayers,
                                  -0.5,
                                  -0.5 + TrackerTraits::numberOfLayers);
  h_longNtuplet_pt_.book1DLogX(ibook,
                               "pt",
                               "Longest SimNtuplets per TrackingParticle (all, even no RecHits)",
                               "True transverse momentum p_{T} [GeV]",
                               "Number of TrackingParticles",
                               pTNBins,
                               pTmin,
                               pTmax);
  h_longNtuplet_eta_.book1D(ibook,
                            "eta",
                            "Longest SimNtuplets per TrackingParticle (all, even no RecHits)",
                            "True pseudorapidity #eta",
                            "Number of TrackingParticles",
                            etaNBins,
                            etamin,
                            etamax);

  // histograms of the longest SimNtuplets of the TrackingParticles
  h_longNtuplet_alive_pt_ = simdoublets::make1DLogX(ibook,
                                                    "pt_alive",
                                                    "Longest SimNtuplets per TrackingParticle (alive);True "
                                                    "transverse momentum p_{T} [GeV];Number of TrackingParticles",
                                                    pTNBins,
                                                    pTmin,
                                                    pTmax);

  h_longNtuplet_undefDoubletCuts_pt_ =
      simdoublets::make1DLogX(ibook,
                              "pt_undefDoubletCuts",
                              "Longest SimNtuplets per TrackingParticle (with undef doublet cuts);True "
                              "transverse momentum p_{T} [GeV];Number of TrackingParticles",
                              pTNBins,
                              pTmin,
                              pTmax);

  h_longNtuplet_undefConnectionCuts_pt_ =
      simdoublets::make1DLogX(ibook,
                              "pt_undefConnectionCuts",
                              "Longest SimNtuplets per TrackingParticle (with undef connection cuts);True "
                              "transverse momentum p_{T} [GeV];Number of TrackingParticles",
                              pTNBins,
                              pTmin,
                              pTmax);

  h_longNtuplet_missingLayerPair_pt_ =
      simdoublets::make1DLogX(ibook,
                              "pt_missingLayerPair",
                              "Longest SimNtuplets per TrackingParticle (with missing layer pair);True "
                              "transverse momentum p_{T} [GeV];Number of TrackingParticles",
                              pTNBins,
                              pTmin,
                              pTmax);

  h_longNtuplet_killedDoublets_pt_ =
      simdoublets::make1DLogX(ibook,
                              "pt_killedDoublets",
                              "Longest SimNtuplets per TrackingParticle (killed by doublet cuts);True "
                              "transverse momentum p_{T} [GeV];Number of TrackingParticles",
                              pTNBins,
                              pTmin,
                              pTmax);

  h_longNtuplet_killedConnections_pt_ =
      simdoublets::make1DLogX(ibook,
                              "pt_killedConnections",
                              "Longest SimNtuplets per TrackingParticle (killed by connection cuts);True "
                              "transverse momentum p_{T} [GeV];Number of TrackingParticles",
                              pTNBins,
                              pTmin,
                              pTmax);

  h_longNtuplet_tooShort_pt_ = simdoublets::make1DLogX(
      ibook,
      "pt_tooShort",
      "Longest SimNtuplets per TrackingParticle (3 RecHits but still shorter than the threshold);True "
      "transverse momentum p_{T} [GeV];Number of TrackingParticles",
      pTNBins,
      pTmin,
      pTmax);

  h_longNtuplet_alive_eta_ = ibook.book1D(
      "eta_alive",
      "Longest SimNtuplets per TrackingParticle (alive);True pseudorapidity #eta;Number of TrackingParticles",
      etaNBins,
      etamin,
      etamax);

  h_longNtuplet_undefDoubletCuts_eta_ = ibook.book1D("eta_undefDoubletCuts",
                                                     "Longest SimNtuplets per TrackingParticle (with undef doublet "
                                                     "cuts);True pseudorapidity #eta;Number of TrackingParticles",
                                                     etaNBins,
                                                     etamin,
                                                     etamax);

  h_longNtuplet_undefConnectionCuts_eta_ =
      ibook.book1D("eta_undefConnectionCuts",
                   "Longest SimNtuplets per TrackingParticle (with undef connection cuts);True pseudorapidity "
                   "#eta;Number of TrackingParticles",
                   etaNBins,
                   etamin,
                   etamax);

  h_longNtuplet_missingLayerPair_eta_ = ibook.book1D("eta_missingLayerPair",
                                                     "Longest SimNtuplets per TrackingParticle (with missing layer "
                                                     "pair);True pseudorapidity #eta;Number of TrackingParticles",
                                                     etaNBins,
                                                     etamin,
                                                     etamax);

  h_longNtuplet_killedDoublets_eta_ = ibook.book1D("eta_killedDoublets",
                                                   "Longest SimNtuplets per TrackingParticle (killed by doublet "
                                                   "cuts);True pseudorapidity #eta;Number of TrackingParticles",
                                                   etaNBins,
                                                   etamin,
                                                   etamax);

  h_longNtuplet_killedConnections_eta_ = ibook.book1D("eta_killedConnections",
                                                      "Longest SimNtuplets per TrackingParticle (killed by connection "
                                                      "cuts);True pseudorapidity #eta;Number of TrackingParticles",
                                                      etaNBins,
                                                      etamin,
                                                      etamax);

  h_longNtuplet_tooShort_eta_ =
      ibook.book1D("eta_tooShort",
                   "Longest SimNtuplets per TrackingParticle (3 RecHits but still shorter than the "
                   "threshold);True pseudorapidity #eta;Number of TrackingParticles",
                   etaNBins,
                   etamin,
                   etamax);
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

  // cutting parameters for doublets
  desc.add<int>("cellMinYSizeB1", 25)->setComment("Minimum cluster size for inner RecHit from B1");
  desc.add<int>("cellMinYSizeB2", 15)->setComment("Minimum cluster size for inner RecHit not from B1");
  desc.add<double>("cellZ0Cut", 7.5)->setComment("Maximum longitudinal impact parameter");
  desc.add<double>("cellPtCut", 0.85)->setComment("Minimum tranverse momentum in GeV");
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
