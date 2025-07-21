// -*- C++ -*-
//
// Package:    Validation/TrackingMCTruth
// Class:      SimDoubletsAnalyzer
//

// user include files
#include "Validation/TrackingMCTruth/plugins/SimDoubletsAnalyzer.h"
#include <sys/types.h>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/Histograms/interface/MonitorElementCollection.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/approx_atan2.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimTracker/TrackAssociation/interface/TrackingParticleIP.h"
#include "RecoTracker/PixelSeeding/interface/CircleEq.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <cstddef>

namespace simdoublets {
  // class that calculate and stores all cut variables for a given doublet
  struct CellCutVariables {
    void calculateCutVariables(SimDoublets::Doublet const& doublet, SimDoublets const& simDoublets) {
      // inner RecHit properties
      GlobalPoint inner_globalPosition = doublet.innerGlobalPos();
      inner_z_ = inner_globalPosition.z();
      inner_r_ = inner_globalPosition.perp();
      double inner_phi = inner_globalPosition.barePhi();  // returns float, whereas .phi() returns phi object
      double inner_x = inner_globalPosition.x();
      double inner_y = inner_globalPosition.y();
      int inner_iphi = unsafe_atan2s<7>(inner_globalPosition.y(), inner_x);
      // outer RecHit properties
      GlobalPoint outer_globalPosition = doublet.outerGlobalPos();
      outer_z_ = outer_globalPosition.z();
      outer_r_ = outer_globalPosition.perp();
      double outer_x = outer_globalPosition.x();
      double outer_y = outer_globalPosition.y();
      double outer_phi = outer_globalPosition.barePhi();
      int outer_iphi = unsafe_atan2s<7>(outer_globalPosition.y(), outer_globalPosition.x());

      // relative properties
      dz_ = outer_z_ - inner_z_;
      dr_ = outer_r_ - inner_r_;
      dphi_ = reco::deltaPhi(inner_phi, outer_phi);
      idphi_ = std::min(std::abs(int16_t(outer_iphi - inner_iphi)), std::abs(int16_t(inner_iphi - outer_iphi)));

      // longitudinal impact parameter with respect to the beamspot
      z0_ = std::abs(inner_r_ * outer_z_ - inner_z_ * outer_r_) / dr_;

      // radius of the circle defined by the two RecHits and the beamspot
      curvature_ = 1.f / 2.f * std::sqrt((dr_ / dphi_) * (dr_ / dphi_) + (inner_r_ * outer_r_));

      // pT that this curvature radius corresponds to
      pT_ = curvature_ / 87.78f;

      // cluster size variables
      Ysize_ = doublet.innerClusterYSize();
      DYsize_ = std::abs(Ysize_ - doublet.outerClusterYSize());
      DYPred_ = std::abs(Ysize_ - int(std::abs(dz_ / dr_) * pixelTopology::Phase2::dzdrFact + 0.5f));

      // cuts on doublet connections (loop over all inner neighboring doublets)
      // reset them first
      CAThetaCut_.clear();
      dcaCut_.clear();
      hardCurvCut_.clear();
      // then, refill
      for (auto const& neighbor : doublet.innerNeighborsView()) {
        // get the inner RecHit of the inner neighbor
        GlobalPoint neighbor_globalPosition = simDoublets.getSimDoublet(neighbor.index()).innerGlobalPos();
        double neighbor_z = neighbor_globalPosition.z();
        double neighbor_r = neighbor_globalPosition.perp();
        double neighbor_x = neighbor_globalPosition.x();
        double neighbor_y = neighbor_globalPosition.y();

        // alignement cut variable in R-Z assuming ptmin = 1 GeV
        double radius_diff = std::abs(neighbor_r - outer_r_);
        double distance_13_squared = radius_diff * radius_diff + (neighbor_z - outer_z_) * (neighbor_z - outer_z_);
        double tan_12_13_half_mul_distance_13_squared =
            fabs(neighbor_z * (inner_r_ - outer_r_) + inner_z_ * (outer_r_ - neighbor_r) +
                 outer_z_ * (neighbor_r - inner_r_));
        double denominator = std::sqrt(distance_13_squared) * radius_diff;
        CAThetaCut_.push_back(tan_12_13_half_mul_distance_13_squared / denominator);

        // alignement cut variables in x-y
        CircleEq<double> eq(neighbor_x, neighbor_y, inner_x, inner_y, outer_x, outer_y);
        hardCurvCut_.push_back(std::abs(eq.curvature()));
        dcaCut_.push_back(std::abs(eq.dca0() / std::abs(eq.curvature())));
      }
    }

    // methods to get the cut variables
    double inner_z() const { return inner_z_; }
    double inner_r() const { return inner_r_; }
    double outer_z() const { return outer_z_; }
    double outer_r() const { return outer_r_; }
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
    double inner_z_, inner_r_, outer_z_, outer_r_, dz_, dr_;
    double dphi_, z0_, curvature_, pT_;                      // double-valued variables
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
      // first check if the inner cluster size is even positive
      // because if not (cluster at module edge), no cut will be applied no matter what
      if (doublet.innerClusterYSize() < 0) {
        return;
      }

      // determine the moduleId
      const int moduleId = doublet.innerModuleId();

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
          // also check if the outer cluster size is positive
          if (doublet.outerClusterYSize() > 0) {
            if (innerInB1) {
              setSubjectToDYsize12();
            } else {
              setSubjectToDYsize();
            }
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
  void BinLogY(TH1* h) {
    TAxis* axis = h->GetYaxis();
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
  template <typename... Args>
  dqm::reco::MonitorElement* makeProfile2DLogY(dqm::reco::DQMStore::IBooker& ibook, Args&&... args) {
    auto h = std::make_unique<TProfile2D>(std::forward<Args>(args)...);
    BinLogY(h.get());
    const auto& name = h->GetName();
    return ibook.bookProfile2D(name, h.release());
  }

  // function that checks if two vector share a common element
  template <typename T>
  bool haveCommonElement(std::vector<T> const& v1, std::vector<T> const& v2) {
    return std::find_first_of(v1.begin(), v1.end(), v2.begin(), v2.end()) != v1.end();
  }

  // fillDescriptionsCommon: description that is identical for Phase 1 and 2
  template <typename TrackerTraits>
  void fillDescriptionsCommon(edm::ParameterSetDescription& desc) {
    desc.add<std::string>("folder", "Tracking/TrackingMCTruth/SimPixelTracks");
    desc.add<bool>("inputIsRecoTracks", false)
        ->setComment(
            "Set to true if SimPixelTracks are built from reconstructed tracks instead of TrackingParticles. "
            "This will disable most plots (those relying on truth information) but still produce CAParameters");

    // cut for minimum number of RecHits required for an Ntuplet
    desc.add<uint>("minHitsPerNtuplet", 4)->setComment("Cut on minimum number of RecHits required for an Ntuplet");

    // Extension settings
    desc.add<int>("numLayersOT", 0)->setComment("Number of additional layers from the OT extension.");

    // cut parameters with scalar values
    desc.add<double>("ptmin", 0.9)
        ->setComment(
            "Minimum tranverse momentum considered for the multiple scattering expectation when checking alignement in "
            "R-z plane of two doublets in GeV");
    desc.add<double>("hardCurvCut", 1. / (0.35 * 87.))
        ->setComment("Cut on minimum curvature, used in DCA ntuplet selection");
    desc.add<int>("maxDYsize12", TrackerTraits::maxDYsize12)
        ->setComment("Maximum difference in cluster size for B1/B2");
    desc.add<int>("maxDYsize", TrackerTraits::maxDYsize)->setComment("Maximum difference in cluster size");
    desc.add<int>("maxDYPred", TrackerTraits::maxDYPred)
        ->setComment("Maximum difference between actual and expected cluster size of inner RecHit");
  }

  // Function that, for a pair of two layers, gives a unique pair Id (innerLayerId * 100 + outerLayerId).
  int getLayerPairId(uint8_t const innerLayerId, uint8_t const outerLayerId) {
    // calculate the unique layer pair Id as (innerLayerId * 100 + outerLayerId)
    return (innerLayerId * 100 + outerLayerId);
  }
}  // namespace simdoublets

// -------------------------------------------------------------------------------------------------------------
// constructors and destructor
// -------------------------------------------------------------------------------------------------------------

template <typename TrackerTraits>
SimDoubletsAnalyzer<TrackerTraits>::SimDoubletsAnalyzer(const edm::ParameterSet& iConfig)
    : topology_getToken_(esConsumes<TrackerTopology, TrackerTopologyRcd>()),
      simDoublets_getToken_(consumes(iConfig.getParameter<edm::InputTag>("simDoubletsSrc"))),
      numLayersOT_(iConfig.getParameter<int>("numLayersOT")),
      cellCuts_(
          CAGeometryParams(iConfig.getParameter<edm::ParameterSet>("geometry"), iConfig.getParameter<double>("ptmin"))),
      minYsizeB1_(iConfig.getParameter<int>("minYsizeB1")),
      minYsizeB2_(iConfig.getParameter<int>("minYsizeB2")),
      maxDYsize12_(iConfig.getParameter<int>("maxDYsize12")),
      maxDYsize_(iConfig.getParameter<int>("maxDYsize")),
      maxDYPred_(iConfig.getParameter<int>("maxDYPred")),
      cellZ0Cut_(iConfig.getParameter<double>("cellZ0Cut")),
      cellPtCut_(iConfig.getParameter<double>("cellPtCut")),
      hardCurvCut_(iConfig.getParameter<double>("hardCurvCut")),
      minNumDoubletsPerNtuplet_(iConfig.getParameter<uint>("minHitsPerNtuplet") - 1),
      folder_(iConfig.getParameter<std::string>("folder")),
      inputIsRecoTracks_(iConfig.getParameter<bool>("inputIsRecoTracks")) {
  edm::ParameterSet geometryConfig{iConfig.getParameter<edm::ParameterSet>("geometry")};
  // get layer pairs from configuration
  std::vector<uint> layerPairs{geometryConfig.getParameter<std::vector<uint>>("pairGraph")};

  // get staring layer pairs from configuration
  std::vector<uint> startingPairs{geometryConfig.getParameter<std::vector<uint>>("startingPairs")};

  // number of configured layer pairs
  size_t numLayerPairs = layerPairs.size() / 2;

  // fill the map of layer pairs
  for (size_t i{0}; i < numLayerPairs; i++) {
    int layerPairId = simdoublets::getLayerPairId(layerPairs[2 * i], layerPairs[2 * i + 1]);
    layerPairId2Index_.insert({layerPairId, i});

    // check if the layer pair is considered as starting point for Ntuplets
    bool isStartingPair = (std::find(startingPairs.begin(), startingPairs.end(), i) != startingPairs.end());
    if (isStartingPair) {
      startingPairs_.insert(layerPairId);
    }
  }

  // resize all histogram vectors, so that we can fill them according to the
  // layerPairIndex saved in the map that we just created
  hVector_dz_.resize(numLayerPairs);
  hVector_dr_.resize(numLayerPairs);
  hVector_dphi_.resize(numLayerPairs);
  hVector_idphi_.resize(numLayerPairs);
  hVector_innerR_.resize(numLayerPairs);
  hVector_innerZ_.resize(numLayerPairs);
  hVector_outerR_.resize(numLayerPairs);
  hVector_outerZ_.resize(numLayerPairs);
  hVector_Ysize_.resize(numLayerPairs);
  hVector_DYsize_.resize(numLayerPairs);
  hVector_DYPred_.resize(numLayerPairs);

  // resize other vectors according to number of layers
  numLayers_ = TrackerTraits::numberOfLayers + numLayersOT_;
  hVector_caThetaCut_.resize(numLayers_);
  hVector_caDCACut_.resize(numLayers_);
  hVector_firstHitR_.resize(numLayers_);
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

  if (/* inner r window cut */
      (cellCutVariables.inner_r() < cellCuts_.minInnerR_[layerPairIdIndex] ||
       cellCutVariables.inner_r() > cellCuts_.maxInnerR_[layerPairIdIndex]) ||
      /* outer r window cut */
      (cellCutVariables.outer_r() < cellCuts_.minOuterR_[layerPairIdIndex] ||
       cellCutVariables.outer_r() > cellCuts_.maxOuterR_[layerPairIdIndex]) ||
      /* inner z window cut */
      (cellCutVariables.inner_z() < cellCuts_.minInnerZ_[layerPairIdIndex] ||
       cellCutVariables.inner_z() > cellCuts_.maxInnerZ_[layerPairIdIndex]) ||
      /* outer z window cut */
      (cellCutVariables.outer_z() < cellCuts_.minOuterZ_[layerPairIdIndex] ||
       cellCutVariables.outer_z() > cellCuts_.maxOuterZ_[layerPairIdIndex]) ||
      /* dz window */
      cellCutVariables.dz() > cellCuts_.maxDZ_[layerPairIdIndex] ||
      cellCutVariables.dz() < cellCuts_.minDZ_[layerPairIdIndex] ||
      /* z0cutoff */
      (cellCutVariables.dr() > cellCuts_.maxDR_[layerPairIdIndex] || cellCutVariables.dr() < 0 ||
       cellCutVariables.z0() > cellZ0Cut_) ||
      /* ptcut */
      (cellCutVariables.pT() < cellPtCut_) ||
      /* idphicut */
      (cellCutVariables.idphi() > cellCuts_.phiCuts_[layerPairIdIndex]) ||
      /* YsizeB1 cut */
      (clusterSizeCutManager.isSubjectToYsizeB1() && (cellCutVariables.Ysize() < minYsizeB1_)) ||
      /* YsizeB2 cut */
      (clusterSizeCutManager.isSubjectToYsizeB2() && (cellCutVariables.Ysize() < minYsizeB2_)) ||
      /* DYsize12 cut */
      (clusterSizeCutManager.isSubjectToDYsize12() && (cellCutVariables.DYsize() > maxDYsize12_)) ||
      /* DYsize cut */
      (clusterSizeCutManager.isSubjectToDYsize() && (cellCutVariables.DYsize() > maxDYsize_)) ||
      /* DYPred cut */
      (clusterSizeCutManager.isSubjectToDYPred() && (cellCutVariables.DYPred() > maxDYPred_))) {
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
          // apply CAThetaCut
          (cellCutVariables.CAThetaCut(i) > cellCuts_.caThetaCuts_over_ptmin_.at(doublet.innerLayerId())) ||
          // apply hardCurvCut
          (cellCutVariables.hardCurvCut(i) > hardCurvCut_) ||
          // apply dcaCut
          (cellCutVariables.dcaCut(i) > cellCuts_.caDCACuts_.at(doublet.innerNeighborsInnerLayerId()))) {
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
  // potential cuts for inner r and dz
  hVector_innerR_[layerPairIdIndex].fill(passed, cellCutVariables.inner_r());
  hVector_dz_[layerPairIdIndex].fill(passed, cellCutVariables.dz());
  hVector_outerZ_[layerPairIdIndex].fill(passed, cellCutVariables.outer_z());
  hVector_outerR_[layerPairIdIndex].fill(passed, cellCutVariables.outer_r());

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
  // check if connection cut histograms should be filled
  if (hasValidNeighbors) {
    bool passedConnect;
    // loop over the inner neighboring doublets of the doublet
    for (int i{0}; auto const& neighbor : doublet.innerNeighborsView()) {
      // get the status of the connection
      passedConnect = neighbor.isAlive();

      // fill the histograms
      // hard curvature cut
      h_hardCurvCut_.fill(passedConnect, cellCutVariables.hardCurvCut(i));
      // dca cut
      hVector_caDCACut_.at(doublet.innerNeighborsInnerLayerId()).fill(passedConnect, cellCutVariables.dcaCut(i));
      // CATheta cut
      hVector_caThetaCut_.at(doublet.innerLayerId()).fill(passedConnect, cellCutVariables.CAThetaCut(i));

      i++;
    }
  }
}

//  function that fills all histograms of SimDoublets (in folder SimDoublets)
template <typename TrackerTraits>
void SimDoubletsAnalyzer<TrackerTraits>::fillSimDoubletHistograms(SimDoublets::Doublet const& doublet,
                                                                  double const true_eta,
                                                                  double const true_pT) {
  // check if doublet passed all cuts
  bool passed = doublet.isAlive();

  // layer pair combinations
  h_layerPairs_.fill(passed, doublet.innerLayerId(), doublet.outerLayerId());
  // number of skipped layers by SimDoublets
  h_numSkippedLayers_.fill(passed, doublet.numSkippedLayers());

  // if input are RecoTracks break here and don't fill the other histograms
  if (inputIsRecoTracks_)
    return;

  // fill histograms for SimDoublet numbers
  h_num_vs_pt_.fill(passed, true_pT);
  h_num_vs_eta_.fill(passed, true_eta);
}

//  function that fills all histograms of SimNtuplets (in folder SimNtuplets)
template <typename TrackerTraits>
void SimDoubletsAnalyzer<TrackerTraits>::fillSimNtupletHistograms(SimDoublets const& simDoublets,
                                                                  double const true_eta,
                                                                  double const true_pT) {
  // get the longest SimNtuplet of the TrackingParticle (if it exists)
  auto const& longNtuplet = simDoublets.longestSimNtuplet();

  // check if it is alive
  bool isAlive = longNtuplet.isAlive();

  // fill general longest SimNtuplet histogram
  h_longNtuplet_numRecHits_.fill(isAlive, longNtuplet.numRecHits());
  h_longNtuplet_firstLayerId_.fill(isAlive, longNtuplet.firstLayerId());
  h_longNtuplet_lastLayerId_.fill(isAlive, longNtuplet.lastLayerId());
  h_longNtuplet_layerSpan_.fill(isAlive, longNtuplet.firstLayerId(), longNtuplet.lastLayerId());
  h_longNtuplet_firstVsSecondLayer_.fill(isAlive, longNtuplet.firstLayerId(), longNtuplet.secondLayerId());
  h_longNtuplet_numSkippedLayersVsNumLayers_.fill(isAlive, longNtuplet.numRecHits(), longNtuplet.numSkippedLayers());

  // fill first RecHit r histogram
  if (simDoublets.numRecHits() > 0) {
    auto firstHitR = simDoublets.globalPositions(0).perp();
    auto firstHitLayerId = simDoublets.layerIds(0);
    hVector_firstHitR_[firstHitLayerId].fill(simDoublets.hasAliveSimNtuplet(), firstHitR);
  }

  // if input are RecoTracks break here and don't fill the other histograms
  if (inputIsRecoTracks_)
    return;

  h_longNtuplet_firstLayerVsEta_.fill(isAlive, true_eta, longNtuplet.firstLayerId());
  h_longNtuplet_lastLayerVsEta_.fill(isAlive, true_eta, longNtuplet.lastLayerId());

  // fill the respective histogram
  // 1. check if alive
  if (isAlive) {
    h_longNtuplet_alive_eta_->Fill(true_eta);
    h_longNtuplet_alive_pt_->Fill(true_pT);
  }
  // 2. if not alive, find out why (go in order of cut application)
  // A) the Ntuplet does not meet the minimum length requirement and will never be build
  else if (longNtuplet.isTooShort()) {
    h_longNtuplet_tooShort_eta_->Fill(true_eta);
    h_longNtuplet_tooShort_pt_->Fill(true_pT);
  }
  // B) a layer pair is missing, therefore no doublet is formed
  else if (longNtuplet.hasMissingLayerPair()) {
    h_longNtuplet_missingLayerPair_eta_->Fill(true_eta);
    h_longNtuplet_missingLayerPair_pt_->Fill(true_pT);
  }
  // C) a doublet got killed by doublet building cuts
  else if (longNtuplet.hasKilledDoublets()) {
    h_longNtuplet_killedDoublets_eta_->Fill(true_eta);
    h_longNtuplet_killedDoublets_pt_->Fill(true_pT);
  }
  // D) one of connections between the doublets got cut
  else if (longNtuplet.hasKilledConnections()) {
    h_longNtuplet_killedConnections_eta_->Fill(true_eta);
    h_longNtuplet_killedConnections_pt_->Fill(true_pT);
  }
  // E) the Ntuplet starts with a layer pair not considered for starting
  else if (longNtuplet.firstDoubletNotInStartingLayerPairs()) {
    h_longNtuplet_notStartingPair_eta_->Fill(true_eta);
    h_longNtuplet_notStartingPair_pt_->Fill(true_pT);
  }
  // F) if we arrive here something's wrong
  else if (longNtuplet.hasUndefDoubletCuts()) {
    h_longNtuplet_undefDoubletCuts_eta_->Fill(true_eta);
    h_longNtuplet_undefDoubletCuts_pt_->Fill(true_pT);
  }
  // G) or even wronger...
  else if (longNtuplet.hasUndefDoubletConnectionCuts()) {
    h_longNtuplet_undefConnectionCuts_eta_->Fill(true_eta);
    h_longNtuplet_undefConnectionCuts_pt_->Fill(true_pT);
  }

  // -------------------------------------------------------------------------------------
  // fill the most alive (best) histograms
  auto const& bestNtuplet = simDoublets.bestSimNtuplet();
  // check if it is alive
  isAlive = bestNtuplet.isAlive();

  // fill general longest SimNtuplet histogram
  h_bestNtuplet_numRecHits_.fill(isAlive, bestNtuplet.numRecHits());
  h_bestNtuplet_firstLayerId_.fill(isAlive, bestNtuplet.firstLayerId());
  h_bestNtuplet_lastLayerId_.fill(isAlive, bestNtuplet.lastLayerId());
  h_bestNtuplet_layerSpan_.fill(isAlive, bestNtuplet.firstLayerId(), bestNtuplet.lastLayerId());
  h_bestNtuplet_firstVsSecondLayer_.fill(isAlive, bestNtuplet.firstLayerId(), bestNtuplet.secondLayerId());
  h_bestNtuplet_firstLayerVsEta_.fill(isAlive, true_eta, bestNtuplet.firstLayerId());
  h_bestNtuplet_lastLayerVsEta_.fill(isAlive, true_eta, bestNtuplet.lastLayerId());
  h_bestNtuplet_numSkippedLayersVsNumLayers_.fill(isAlive, bestNtuplet.numRecHits(), bestNtuplet.numSkippedLayers());

  // fill the respective histogram
  // 1. check if alive
  if (isAlive) {
    h_bestNtuplet_alive_eta_->Fill(true_eta);
    h_bestNtuplet_alive_pt_->Fill(true_pT);
  }
  // 2. if not alive, find out why (go in order of cut application)
  // A) the Ntuplet does not meet the minimum length requirement and will never be build
  else if (bestNtuplet.isTooShort()) {
    h_bestNtuplet_tooShort_eta_->Fill(true_eta);
    h_bestNtuplet_tooShort_pt_->Fill(true_pT);
  }
  // B) a layer pair is missing, therefore no doublet is formed
  else if (bestNtuplet.hasMissingLayerPair()) {
    h_bestNtuplet_missingLayerPair_eta_->Fill(true_eta);
    h_bestNtuplet_missingLayerPair_pt_->Fill(true_pT);
  }
  // C) a doublet got killed by doublet building cuts
  else if (bestNtuplet.hasKilledDoublets()) {
    h_bestNtuplet_killedDoublets_eta_->Fill(true_eta);
    h_bestNtuplet_killedDoublets_pt_->Fill(true_pT);
  }
  // D) one of connections between the doublets got cut
  else if (bestNtuplet.hasKilledConnections()) {
    h_bestNtuplet_killedConnections_eta_->Fill(true_eta);
    h_bestNtuplet_killedConnections_pt_->Fill(true_pT);
  }
  // E) the Ntuplet starts with a layer pair not considered for starting
  else if (bestNtuplet.firstDoubletNotInStartingLayerPairs()) {
    h_bestNtuplet_notStartingPair_eta_->Fill(true_eta);
    h_bestNtuplet_notStartingPair_pt_->Fill(true_pT);
  }
  // F) if we arrive here something's wrong
  else if (bestNtuplet.hasUndefDoubletCuts()) {
    h_bestNtuplet_undefDoubletCuts_eta_->Fill(true_eta);
    h_bestNtuplet_undefDoubletCuts_pt_->Fill(true_pT);
  }
  // G) or even wronger...
  else if (bestNtuplet.hasUndefDoubletConnectionCuts()) {
    h_bestNtuplet_undefConnectionCuts_eta_->Fill(true_eta);
    h_bestNtuplet_undefConnectionCuts_pt_->Fill(true_pT);
  }
  // -------------------------------------------------------------------------------------
  if (simDoublets.hasAliveSimNtuplet()) {
    auto const& aliveNtuplet = simDoublets.longestAliveSimNtuplet();
    // relative length of alive SimNtuplet vs longest SimNtuplet
    double relativeLength = (double)aliveNtuplet.numRecHits() / (double)longNtuplet.numRecHits();
    h_aliveNtuplet_fracNumRecHits_eta_->Fill(true_eta, relativeLength);
    h_aliveNtuplet_fracNumRecHits_pt_->Fill(true_pT, relativeLength);
  }
}

// function that fills all general histograms (in folder general)
template <typename TrackerTraits>
void SimDoubletsAnalyzer<TrackerTraits>::fillGeneralHistograms(SimDoublets const& simDoublets,
                                                               double const true_eta,
                                                               double const true_pT,
                                                               int const pass_numSimDoublets,
                                                               int const numSimDoublets,
                                                               int const numSkippedLayers) {
  // Now check if the TrackingParticle has a surviving SimNtuplet
  bool passed = simDoublets.hasAliveSimNtuplet();

  // count number of RecHits in layers
  std::vector<int> countsRecHitsPerLayer(numLayers_, 0);
  for (auto const layerId : simDoublets.layerIds())
    countsRecHitsPerLayer.at(layerId)++;
  for (int layerId{0}; auto countRecHits : countsRecHitsPerLayer) {
    h_numRecHitsPerLayer_.fill(passed, layerId, countRecHits);
    layerId++;
  }

  double true_phi, true_dxy, true_dz;
  if (inputIsRecoTracks_) {
    auto track = simDoublets.track();
    true_phi = track->phi();
    auto bs = simDoublets.beamSpotPosition();
    reco::TrackBase::Point beamSpotPoint(bs.x(), bs.y(), bs.z());
    true_dxy = track->dxy(beamSpotPoint);
    true_dz = track->dz(beamSpotPoint);

    h_numTOVsChi2_.fill(passed, track->normalizedChi2());
  } else {
    auto trackingParticle = simDoublets.trackingParticle();
    true_phi = trackingParticle->phi();
    auto true_pdgId = trackingParticle->pdgId();
    auto momentum = trackingParticle->momentum();
    auto vertex = trackingParticle->vertex();
    true_dxy = TrackingParticleIP::dxy(vertex, momentum, simDoublets.beamSpotPosition());
    true_dz = TrackingParticleIP::dz(vertex, momentum, simDoublets.beamSpotPosition());

    h_numTOVsPdgId_.fill(passed, true_pdgId);
    // Fill the efficiency profile per Tracking Particle only if the TP has at least one SimDoublet
    if (numSimDoublets > 0) {
      h_effSimDoubletsPerTOVsEta_->Fill(true_eta, (double)pass_numSimDoublets / (double)numSimDoublets);
      h_effSimDoubletsPerTOVsPt_->Fill(true_pT, (double)pass_numSimDoublets / (double)numSimDoublets);
    }
  }

  // fill histograms for number of SimDoublets
  h_numSimDoubletsPerTrackingObject_.fill(passed, numSimDoublets);
  h_numRecHitsPerTrackingObject_.fill(passed, simDoublets.numRecHits());
  h_numLayersPerTrackingObject_.fill(passed, simDoublets.numLayers());
  h_numSkippedLayersPerTrackingObject_.fill(passed, numSkippedLayers);
  h_numSkippedLayersVsNumLayers_.fill(passed, simDoublets.numLayers(), numSkippedLayers);
  h_numSkippedLayersVsNumRecHits_.fill(passed, simDoublets.numRecHits(), numSkippedLayers);
  h_numRecHitsVsEta_.fill(passed, true_eta, simDoublets.numRecHits());
  h_numLayersVsEta_.fill(passed, true_eta, simDoublets.numLayers());
  h_numSkippedLayersVsEta_.fill(passed, true_eta, numSkippedLayers);
  h_numRecHitsVsPt_.fill(passed, true_pT, simDoublets.numRecHits());
  h_numLayersVsPt_.fill(passed, true_pT, simDoublets.numLayers());
  h_numSkippedLayersVsPt_.fill(passed, true_pT, numSkippedLayers);
  h_numLayersVsEtaPt_->Fill(true_eta, true_pT, simDoublets.numLayers());
  // fill histograms for number of TrackingParticles
  h_numTOVsPt_.fill(passed, true_pT);
  h_numTOVsEta_.fill(passed, true_eta);
  h_numTOVsPhi_.fill(passed, true_phi);
  h_numTOVsDxy_.fill(passed, true_dxy);
  h_numTOVsDz_.fill(passed, true_dz);
  h_numTOVsEtaPhi_.fill(passed, true_eta, true_phi);
  h_numTOVsEtaPt_.fill(passed, true_eta, true_pT);
  h_numTOVsPhiPt_.fill(passed, true_phi, true_pT);
}

// function that trys to find a valid Ntuplet for the given SimDoublets object using the given geometry configuration
// (layer pairs, starting pairs, minimum number of hits) ignoring all cuts on doublets/connections and returns if it was able to find one
template <typename TrackerTraits>
bool SimDoubletsAnalyzer<TrackerTraits>::configAllowsForValidNtuplet(SimDoublets const& simDoublets) const {
  // if the number of layers is less than the minimum requirement, don't even bother building anything...
  if (simDoublets.numLayers() < minNumDoubletsPerNtuplet_ + 1)
    return false;

  // initialize counter for the number of layers in the built Ntuplet
  int numLayers{0};
  // initialize bool to know if the building has started
  // (need to start at a valid starting pair)
  bool building{false};

  // get the layerId of the first RecHit of the TrackingParticle
  auto currentLayer = simDoublets.layerIds(0);

  // loop over the RecHits in order and try building an Ntuplet starting from the first valid starting pair
  for (int layerPairId{0}; auto nextLayer : simDoublets.layerIds()) {
    // get the layerPairId for the (currentLayer, nextLayer) pair
    layerPairId = simdoublets::getLayerPairId(currentLayer, nextLayer);

    // if the building has already started
    if (building) {
      // check if the combination (currentLayer, nextLayer) is a valid pair
      if (layerPairId2Index_.find(layerPairId) != layerPairId2Index_.end()) {
        // if yes, make the nextLayer the current and increment the numLayers
        currentLayer = nextLayer;
        numLayers++;
      }
    } else {
      // if the building did not start, check if the current pair is a valid starting point
      if (startingPairs_.contains(layerPairId)) {
        // enable building, make the nextLayer the current and set numLayers to 2
        building = true;
        currentLayer = nextLayer;
        numLayers = 2;
      }
    }
  }  // end loop over RecHits

  // check if the built Ntuplet is long enough to be used in reconstruction
  return numLayers > minNumDoubletsPerNtuplet_;
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
  double true_pT{0.}, true_eta{0.};
  int numSimDoublets, pass_numSimDoublets, layerPairId, layerPairIdIndex, numSkippedLayers;
  bool hasValidNeighbors;

  // initialize the manager for keeping track of which cluster cuts are applied to the inidividual doublets
  simdoublets::ClusterSizeCutManager clusterSizeCutManager;
  // initialize the structure holding the cut variables for an individual doublet
  simdoublets::CellCutVariables cellCutVariables;

  // loop over SimDoublets (= loop over TrackingParticles/RecoTracks)
  for (auto const& simDoublets : simDoubletsCollection) {
    if (inputIsRecoTracks_) {
      // get true quatities from the track
      auto track = simDoublets.track();
      true_pT = track->pt();
      true_eta = track->eta();
    } else {
      // get true quantities of the TrackingParticle
      auto trackingParticle = simDoublets.trackingParticle();
      true_pT = trackingParticle->pt();
      true_eta = trackingParticle->eta();

      // check if a valid Ntuplet is possible for the given TP and geometry ignoring any cuts and fill hists
      bool reconstructable = configAllowsForValidNtuplet(simDoublets);
      h_effConfigLimitVsEta_->Fill(true_eta, reconstructable);
      h_effConfigLimitVsPt_->Fill(true_pT, reconstructable);
    }

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
        //  cut histograms for SimDoublets (CAParameters folder)
        // -------------------------------------------------------------------------
        fillCutHistograms(doublet, hasValidNeighbors, layerPairIdIndex, cellCutVariables, clusterSizeCutManager);

      } else {
        // if not considered set the doublet as killed
        doublet.setKilledByMissingLayerPair();
      }

      // ---------------------------------------------------------------------------
      //  general plots related to SimDoublets (SimDoublets folder)
      // ---------------------------------------------------------------------------
      fillSimDoubletHistograms(doublet, true_eta, true_pT);

      // if the doublet passes all cuts, increment number of SimDoublets passing all cuts
      if (doublet.isAlive())
        pass_numSimDoublets++;
    }  // end loop over those doublets

    // build the SimNtuplets based on the SimDoublets
    simDoublets.buildSimNtuplets(startingPairs_, minNumDoubletsPerNtuplet_);

    // -----------------------------------------------------------------------------
    //  plots related to SimNtuplets (SimNtuplets folder)
    // -----------------------------------------------------------------------------
    // set the number of skipped layers by default to -1
    numSkippedLayers = -1;

    if (simDoublets.hasSimNtuplet()) {
      // get number of skipped layers from longest SimNtuplet and fill histos
      numSkippedLayers = simDoublets.longestSimNtuplet().numSkippedLayers();
      fillSimNtupletHistograms(simDoublets, true_eta, true_pT);
    }

    // -----------------------------------------------------------------------------
    //  general plots related to TrackingParticles (general folder)
    // -----------------------------------------------------------------------------
    fillGeneralHistograms(simDoublets, true_eta, true_pT, pass_numSimDoublets, numSimDoublets, numSkippedLayers);

    // clear SimDoublets and SimNtuplets of the TrackingParticle
    simDoublets.clearMutables();
  }  // end loop over SimDoublets (= loop over TrackingParticles/RecoTracks)
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
  int phiNBins = 36;
  double phimin = -3.14159;
  double phimax = 3.14159;
  int pdgIdmax = 340;
  int pdgIdmin = -pdgIdmax;
  int pdgIdNBins = pdgIdmax - pdgIdmin;
  std::string trackingObject = inputIsRecoTracks_ ? "PixelTrack" : "Tracking Particle";
  std::string doublet = inputIsRecoTracks_ ? "Doublet" : "SimDoublet";
  std::string ntuplet = inputIsRecoTracks_ ? "Ntuplet" : "SimNtuplet";

  // ----------------------------------------------------------
  // booking general histograms (general folder)
  // ----------------------------------------------------------

  ibook.setCurrentFolder(folder_ + "/general");

  // overview histograms and profiles
  h_numSimDoubletsPerTrackingObject_.book1D(ibook,
                                            "numSimDoublets",
                                            "Number of " + doublet + "s per " + trackingObject,
                                            "Number of " + doublet + "s",
                                            "Number of " + trackingObject + "s",
                                            31,
                                            -0.5,
                                            30.5);
  h_numRecHitsPerTrackingObject_.book1D(ibook,
                                        "numRecHits",
                                        "Number of RecHits per " + trackingObject,
                                        "Number of RecHits",
                                        "Number of " + trackingObject + "s",
                                        25,
                                        -0.5,
                                        24.5);
  h_numLayersPerTrackingObject_.book1D(ibook,
                                       "numLayers",
                                       "Number of layers hit by " + trackingObject,
                                       "Number of layers",
                                       "Number of " + trackingObject + "s",
                                       15,
                                       -0.5,
                                       14.5);
  h_numSkippedLayersPerTrackingObject_.book1D(ibook,
                                              "numSkippedLayers",
                                              "Number of layers skipped by " + trackingObject,
                                              "Number of skipped layers",
                                              "Number of " + trackingObject + "s",
                                              16,
                                              -1.5,
                                              14.5);
  h_numRecHitsPerLayer_.book2D(ibook,
                               "numRecHits_vs_layer",
                               "Number of RecHits by " + trackingObject + " per layer",
                               "Layer Id",
                               "Number of RecHits",
                               numLayers_,
                               -0.5,
                               -0.5 + numLayers_,
                               11,
                               -0.5,
                               10.5);
  h_numSkippedLayersVsNumLayers_.book2D(ibook,
                                        "numSkippedLayers_vs_numLayers",
                                        "Number of layers skipped by Tracking Particle vs layers",
                                        "Number of layers hit by Tracking Particle",
                                        "Number of skipped layers",
                                        15,
                                        -0.5,
                                        14.5,
                                        16,
                                        -1.5,
                                        14.5);
  h_numSkippedLayersVsNumRecHits_.book2D(ibook,
                                         "numSkippedLayers_vs_numRecHits",
                                         "Number of layers skipped by Tracking Particle vs layers",
                                         "Number of RecHits by Tracking Particle",
                                         "Number of skipped layers",
                                         25,
                                         -0.5,
                                         24.5,
                                         16,
                                         -1.5,
                                         14.5);
  h_numLayersVsEtaPt_ = simdoublets::makeProfile2DLogY(
      ibook,
      "numLayers_vs_etaPt",
      "Number of layers hit by Tracking Particle; TP pseudorapidity #eta; TP transverse momentum p_{T} [GeV]",
      etaNBins,
      etamin,
      etamax,
      pTNBins,
      pTmin,
      pTmax,
      -1,
      15,
      " ");
  h_numRecHitsVsEta_.book2D(ibook,
                            "numRecHits_vs_eta",
                            "Number of RecHits by Tracking Particle vs #eta",
                            "True pseudorapidity #eta",
                            "Number of RecHits",
                            etaNBins,
                            etamin,
                            etamax,
                            26,
                            -1.5,
                            24.5);
  h_numLayersVsEta_.book2D(ibook,
                           "numLayers_vs_eta",
                           "Number of layers hit by Tracking Particle vs #eta",
                           "True pseudorapidity #eta",
                           "Number of layers",
                           etaNBins,
                           etamin,
                           etamax,
                           16,
                           -1.5,
                           14.5);
  h_numSkippedLayersVsEta_.book2D(ibook,
                                  "numSkippedLayers_vs_eta",
                                  "Number of layers skipped by Tracking Particle vs #eta",
                                  "True pseudorapidity #eta",
                                  "Number of skipped layers",
                                  etaNBins,
                                  etamin,
                                  etamax,
                                  16,
                                  -1.5,
                                  14.5);
  h_numRecHitsVsPt_.book2DLogX(ibook,
                               "numRecHits_vs_pt",
                               "Number of RecHits by Tracking Particle",
                               "True transverse momentum p_{T} [GeV]",
                               "Number of RecHits",
                               pTNBins,
                               pTmin,
                               pTmax,
                               26,
                               -1.5,
                               24.5);
  h_numLayersVsPt_.book2DLogX(ibook,
                              "numLayers_vs_pt",
                              "Number of layers hit by Tracking Particle",
                              "True transverse momentum p_{T} [GeV]",
                              "Number of layers",
                              pTNBins,
                              pTmin,
                              pTmax,
                              16,
                              -1.5,
                              14.5);
  h_numSkippedLayersVsPt_.book2DLogX(ibook,
                                     "numSkippedLayers_vs_pt",
                                     "Number of layers skipped by Tracking Particle",
                                     "True transverse momentum p_{T} [GeV]",
                                     "Number of skipped layers",
                                     pTNBins,
                                     pTmin,
                                     pTmax,
                                     16,
                                     -1.5,
                                     14.5);
  h_numTOVsPt_.book1DLogX(ibook,
                          "num_vs_pt",
                          "Number of TrackingParticles",
                          "True transverse momentum p_{T} [GeV]",
                          "Number of TrackingParticles",
                          pTNBins,
                          pTmin,
                          pTmax);
  h_numTOVsEta_.book1D(ibook,
                       "num_vs_eta",
                       "Number of TrackingParticles",
                       "True pseudorapidity #eta",
                       "Number of TrackingParticles",
                       etaNBins,
                       etamin,
                       etamax);
  h_numTOVsPhi_.book1D(ibook,
                       "num_vs_phi",
                       "Number of TrackingParticles",
                       "True azimuth angle #phi",
                       "Number of TrackingParticles",
                       phiNBins,
                       phimin,
                       phimax);
  h_numTOVsDxy_.book1D(
      ibook, "num_vs_dxy", "Number of TrackingParticles", "True dxy [cm]", "Number of TrackingParticles", 200, -5, 5);
  h_numTOVsDz_.book1D(
      ibook, "num_vs_dz", "Number of TrackingParticles", "True dz [cm]", "Number of TrackingParticles", 20, -20, 20);
  h_numTOVsEtaPhi_.book2D(ibook,
                          "num_vs_etaPhi",
                          "Number of TrackingParticles",
                          "True pseudorapidity #eta",
                          "True azimuth angle #phi",
                          etaNBins,
                          etamin,
                          etamax,
                          phiNBins,
                          phimin,
                          phimax);
  h_numTOVsEtaPt_.book2DLogY(ibook,
                             "num_vs_etaPt",
                             "Number of TrackingParticles",
                             "True pseudorapidity #eta",
                             "True transverse momentum p_{T} [GeV]",
                             etaNBins,
                             etamin,
                             etamax,
                             pTNBins,
                             pTmin,
                             pTmax);
  h_numTOVsPhiPt_.book2DLogY(ibook,
                             "num_vs_phiPt",
                             "Number of TrackingParticles",
                             "True azimuth angle #phi",
                             "True transverse momentum p_{T} [GeV]",
                             phiNBins,
                             phimin,
                             phimax,
                             pTNBins,
                             pTmin,
                             pTmax);
  if (!inputIsRecoTracks_) {
    h_effConfigLimitVsPt_ =
        simdoublets::makeProfileLogX(ibook,
                                     "effConfigLimit_vs_pt",
                                     "Theoretical limit of the efficiency for the given CA geometry config independent "
                                     "of cuts vs p_{T}; TP transverse momentum p_{T} [GeV]; "
                                     "Maximum efficiency",
                                     pTNBins,
                                     pTmin,
                                     pTmax,
                                     0,
                                     1,
                                     " ");
    h_effConfigLimitVsEta_ = ibook.bookProfile("effConfigLimit_vs_eta",
                                               "Theoretical limit of the efficiency for the given CA geometry config "
                                               "independent of cuts vs #eta; TP pseudorapidity #eta; "
                                               "Maximum efficiency",
                                               etaNBins,
                                               etamin,
                                               etamax,
                                               0,
                                               1,
                                               " ");
    h_effSimDoubletsPerTOVsPt_ =
        simdoublets::makeProfileLogX(ibook,
                                     "effSimDoubletsPerTP_vs_pt",
                                     "SimDoublets efficiency per TP vs p_{T}; TP transverse momentum p_{T} [GeV]; "
                                     "Average fraction of SimDoublets per TP passing all cuts",
                                     pTNBins,
                                     pTmin,
                                     pTmax,
                                     0,
                                     1,
                                     " ");
    h_effSimDoubletsPerTOVsEta_ = ibook.bookProfile("effSimDoubletsPerTP_vs_eta",
                                                    "SimDoublets efficiency per TP vs #eta; TP pseudorapidity #eta; "
                                                    "Average fraction of SimDoublets per TP passing all cuts",
                                                    etaNBins,
                                                    etamin,
                                                    etamax,
                                                    0,
                                                    1,
                                                    " ");
    h_numTOVsPdgId_.book1D(ibook,
                           "num_vs_pdgId",
                           "Number of TrackingParticles",
                           "PDG ID",
                           "Number of TrackingParticles",
                           pdgIdNBins,
                           pdgIdmin,
                           pdgIdmax);
  }

  if (inputIsRecoTracks_) {
    h_numTOVsChi2_.book1D(ibook, "num_vs_chi2", "Number of Tracks", "#chi^2 / ndof", "Number of Tracks", 70, 0, 7);
  }

  // ----------------------------------------------------------
  // booking SimDoublet histograms (SimDoublets folder)
  // ----------------------------------------------------------

  ibook.setCurrentFolder(folder_ + "/SimDoublets");

  h_layerPairs_.book2D(ibook,
                       "layerPairs",
                       "Layer pairs in " + doublet + "s",
                       "Inner layer ID",
                       "Outer layer ID",
                       numLayers_,
                       -0.5,
                       -0.5 + numLayers_,
                       numLayers_,
                       -0.5,
                       -0.5 + numLayers_);
  h_numSkippedLayers_.book1D(ibook,
                             "numSkippedLayers",
                             "Number of skipped layers",
                             "Number of skipped layers",
                             "Number of " + doublet + "s",
                             16,
                             -1.5,
                             14.5);

  if (!inputIsRecoTracks_) {
    h_num_vs_pt_.book1DLogX(ibook,
                            "num_vs_pt",
                            "Number of " + doublet + "s",
                            "True transverse momentum p_{T} [GeV]",
                            "Number of " + doublet + "s",
                            pTNBins,
                            pTmin,
                            pTmax);
    h_num_vs_eta_.book1D(ibook,
                         "num_vs_eta",
                         "Number of " + doublet + "s",
                         "True pseudorapidity #eta",
                         "Number of " + doublet + "s",
                         etaNBins,
                         etamin,
                         etamax);
  }

  // --------------------------------------------------------------
  // booking layer pair independent cut histograms (global folder)
  // --------------------------------------------------------------

  ibook.setCurrentFolder(folder_ + "/CAParameters/doubletCuts/global");

  // histogram for z0cutoff  (z0Cut)
  h_z0_.book1D(ibook,
               "z0",
               "z_{0} of " + doublet + "s",
               "Longitudinal impact parameter z_{0} [cm]",
               "Number of " + doublet + "s",
               51,
               -1,
               50);

  // histograms for ptcut  (ptCut)
  h_curvatureR_.book1DLogX(ibook,
                           "curvatureR",
                           "Curvature from 3 points of beamspot + RecHits of " + doublet + "s",
                           "Curvature radius [cm]",
                           "Number of " + doublet + "s",
                           100,
                           2,
                           4);
  h_pTFromR_.book1DLogX(ibook,
                        "pTFromR",
                        "Transverse momentum from curvature",
                        "Transverse momentum p_{T} [GeV]",
                        "Number of " + doublet + "s",
                        pTNBins,
                        pTmin,
                        pTmax);

  // histograms for clusterCut  (minYsizeB1 and minYsizeB2)
  h_YsizeB1_.book1D(ibook,
                    "YsizeB1",
                    "Cluster size along z of inner RecHit [from BPix1]",
                    "Size along z of inner cluster [num of pixels]",
                    "Number of " + doublet + "s",
                    51,
                    -1,
                    50);
  h_YsizeB2_.book1D(ibook,
                    "YsizeB2",
                    "Cluster size along z of inner RecHit [not from BPix1]",
                    "Size along z of inner cluster [num of pixels]",
                    "Number of " + doublet + "s",
                    51,
                    -1,
                    50);
  // histograms for zSizeCut  (maxDYsize12, maxDYsize and maxDYPred)
  h_DYsize12_.book1D(ibook,
                     "DYsize12",
                     "Difference in cluster size along z [inner from BPix1]",
                     "Absolute difference in cluster size along z of "
                     "the two RecHits [num of pixels]",
                     "Number of " + doublet + "s",
                     31,
                     -1,
                     30);
  h_DYsize_.book1D(ibook,
                   "DYsize",
                   "Difference in cluster size along z [inner not from BPix1]",
                   "Absolute difference in cluster size along z of the two RecHits [num of pixels]",
                   "Number of " + doublet + "s",
                   31,
                   -1,
                   30);
  h_DYPred_.book1D(ibook,
                   "DYPred",
                   "Difference between actual and predicted cluster size along z of inner cluster",
                   "Absolute difference [num of pixels]",
                   "Number of " + doublet + "s",
                   76,
                   -1,
                   75);

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
    std::string subFolderName = "/CAParameters/doubletCuts/lp_" + innerLayerName + "_" + outerLayerName;

    // layer mentioning in histogram titles
    std::string layerTitle = "(layers (" + innerLayerName + "," + outerLayerName + "))";

    // set folder to the sub-folder for the layer pair
    ibook.setCurrentFolder(folder_ + subFolderName);

    // histogram for potential dz cut
    hVector_dz_.at(layerPairIdIndex)
        .book1D(ibook,
                "dz",
                "dz of RecHit pair " + layerTitle,
                "dz between outer and inner RecHit [cm]",
                "Number of " + doublet + "s",
                300,
                -150,
                150);

    // histogram for z0cutoff  (maxr)
    hVector_dr_.at(layerPairIdIndex)
        .book1D(ibook,
                "dr",
                "dr of RecHit pair " + layerTitle,
                "dr between outer and inner RecHit [cm]",
                "Number of " + doublet + "s",
                93,
                -1,
                30);

    // histograms for iphicut  (phiCuts)
    hVector_dphi_.at(layerPairIdIndex)
        .book1D(ibook,
                "dphi",
                "dphi of RecHit pair " + layerTitle,
                "d#phi between outer and inner RecHit [rad]",
                "Number of " + doublet + "s",
                50,
                -M_PI,
                M_PI);
    hVector_idphi_.at(layerPairIdIndex)
        .book1D(ibook,
                "idphi",
                "idphi of RecHit pair " + layerTitle,
                "Absolute int d#phi between outer and inner RecHit",
                "Number of " + doublet + "s",
                100,
                0,
                2000);

    // histogram for potential innerR cut
    hVector_innerR_.at(layerPairIdIndex)
        .book1D(ibook,
                "innerR",
                "r of the inner RecHit " + layerTitle,
                "r of inner RecHit [cm]",
                "Number of " + doublet + "s",
                600,
                0,
                60);

    // histogram for z window  (minz and maxz)
    hVector_innerZ_.at(layerPairIdIndex)
        .book1D(ibook,
                "innerZ",
                "z of the inner RecHit " + layerTitle,
                "z of inner RecHit [cm]",
                "Number of " + doublet + "s",
                600,
                -300,
                300);

    // histogram for potential outerR cut
    hVector_outerR_.at(layerPairIdIndex)
        .book1D(ibook,
                "outerR",
                "r of the outer RecHit " + layerTitle,
                "r of outer RecHit [cm]",
                "Number of " + doublet + "s",
                600,
                0,
                60);

    // histogram for potential outerZ cut
    hVector_outerZ_.at(layerPairIdIndex)
        .book1D(ibook,
                "outerZ",
                "z of the outer RecHit " + layerTitle,
                "z of outer RecHit [cm]",
                "Number of " + doublet + "s",
                600,
                -300,
                300);

    // histograms for cluster size and size differences
    hVector_DYsize_.at(layerPairIdIndex)
        .book1D(ibook,
                "DYsize",
                "Difference in cluster size along z between outer and inner RecHit " + layerTitle,
                "Absolute difference in cluster size along z of the two RecHits [num of pixels]",
                "Number of " + doublet + "s",
                51,
                -1,
                50);
    hVector_DYPred_.at(layerPairIdIndex)
        .book1D(ibook,
                "DYPred",
                "Difference between actual and predicted cluster size along z of inner cluster " + layerTitle,
                "Absolute difference [num of pixels]",
                "Number of " + doublet + "s",
                51,
                -1,
                50);
    hVector_Ysize_.at(layerPairIdIndex)
        .book1D(ibook,
                "Ysize",
                "Cluster size along z " + layerTitle,
                "Size along z of inner cluster [num of pixels]",
                "Number of " + doublet + "s",
                51,
                -1,
                50);
  }

  // -----------------------------------------------------------------
  // booking connection cut histograms (connectionCuts folder)
  // -----------------------------------------------------------------

  ibook.setCurrentFolder(folder_ + "/CAParameters/connectionCuts");

  // histogram for dcaCut (x-y alignement)
  h_hardCurvCut_.book1D(ibook,
                        "hardCurvCut",
                        "Curvature of a pair of neighboring " + doublet + "s",
                        "Curvature [1/cm]",
                        "Number of " + doublet + " connections",
                        50,
                        0,
                        0.04);

  // loop through layer ids
  for (auto id{0}; id < numLayers_; ++id) {
    // layer as string
    std::string idStr = std::to_string(id);

    // set folder to the sub-folder for the layer pair
    ibook.setCurrentFolder(folder_ + "/CAParameters/connectionCuts/layer_" + idStr);

    // histogram for areAlignedRZ
    hVector_caThetaCut_.at(id).book1DLogX(
        ibook,
        "caThetaCut_over_ptmin",
        "CATheta cut variable based on the area spaned by 3 RecHits of a pair of neighboring "
        "" + doublet +
            "s in R-z with the shared RecHit in layer " + idStr,
        "CATheta cut variable",
        "Number of " + doublet + " connections",
        51,
        -6,
        1);
    // histogram for dcaCut (x-y alignement)
    hVector_caDCACut_.at(id).book1DLogX(ibook,
                                        "caDCACut",
                                        "Closest transverse distance to beamspot based on the 3 RecHits of a pair of "
                                        "neighboring " +
                                            doublet + "s with the most inner RecHit on layer " + idStr,
                                        "Transverse distance [cm]",
                                        "Number of " + doublet + " connections",
                                        51,
                                        -6,
                                        1);
  }

  // -----------------------------------------------------------------
  // booking connection cut histograms (startingCuts folder)
  // -----------------------------------------------------------------

  // loop through layer ids
  for (auto id{0}; id < numLayers_; ++id) {
    // layer as string
    std::string idStr = std::to_string(id);

    // set folder to the sub-folder for the layer pair
    ibook.setCurrentFolder(folder_ + "/CAParameters/startingCuts/layer_" + idStr);

    // histogram for r of first hit
    hVector_firstHitR_.at(id).book1D(ibook,
                                         "firstHitR",
                                         "r coordinate of first hit of "
                                         "" + trackingObject +
                                             "s starting in layer " + idStr,
                                         "r coordinate [cm]",
                                         "Number of " + trackingObject + "s",
                                         600,
                                         0,
                                         60);
  }

  // ------------------------------------------------------------------------
  // booking most alive SimNtuplet histograms (simNtuplets/mostAlive folder)
  // ------------------------------------------------------------------------

  if (!inputIsRecoTracks_) {
    ibook.setCurrentFolder(folder_ + "/SimNtuplets/mostAlive");

    // histograms of the TrackingParticles

    h_bestNtuplet_numRecHits_.book1D(ibook,
                                     "numRecHits",
                                     "Number of RecHits in most alive SimNtuplet per TrackingParticle",
                                     "Number of RecHits",
                                     "Number of TrackingParticles",
                                     15,
                                     -0.5,
                                     14.5);
    h_bestNtuplet_firstLayerId_.book1D(ibook,
                                       "firstLayerId",
                                       "First layer of most alive SimNtuplet per TrackingParticle",
                                       "Layer ID",
                                       "Number of TrackingParticles",
                                       numLayers_,
                                       -0.5,
                                       -0.5 + numLayers_);
    h_bestNtuplet_lastLayerId_.book1D(ibook,
                                      "lastLayerId",
                                      "Last layer of most alive SimNtuplet per TrackingParticle",
                                      "Layer ID",
                                      "Number of TrackingParticles",
                                      numLayers_,
                                      -0.5,
                                      -0.5 + numLayers_);
    h_bestNtuplet_layerSpan_.book2D(ibook,
                                    "layerSpan",
                                    "Layer span of most alive SimNtuplet per TrackingParticle",
                                    "First layer ID",
                                    "Last layer ID",
                                    numLayers_,
                                    -0.5,
                                    -0.5 + numLayers_,
                                    numLayers_,
                                    -0.5,
                                    -0.5 + numLayers_);
    h_bestNtuplet_firstVsSecondLayer_.book2D(ibook,
                                             "firstVsSecondLayer",
                                             "First two layers of most alive SimNtuplet per TrackingParticle",
                                             "First layer ID",
                                             "Second layer ID",
                                             numLayers_,
                                             -0.5,
                                             -0.5 + numLayers_,
                                             numLayers_,
                                             -0.5,
                                             -0.5 + numLayers_);
    h_bestNtuplet_numSkippedLayersVsNumLayers_.book2D(
        ibook,
        "numSkippedLayers_vs_numLayers",
        "Number of layers skipped of most alive SimNtuplet by Tracking Particle vs layers",
        "Number of layers hit by Tracking Particle",
        "Number of skipped layers",
        15,
        -0.5,
        14.5,
        16,
        -1.5,
        14.5);
    h_aliveNtuplet_fracNumRecHits_pt_ =
        simdoublets::makeProfileLogX(ibook,
                                     "fracNumRecHits_vs_pt",
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
        ibook.bookProfile("fracNumRecHits_vs_eta",
                          "Number of RecHits of longest alive SimNtuplet over number of "
                          "RecHits of longest SimNtuplet per TrackingParticle; TP transverse momentum #eta; "
                          "Ratio",
                          etaNBins,
                          etamin,
                          etamax,
                          0,
                          1,
                          " ");
    h_bestNtuplet_firstLayerVsEta_.book2D(ibook,
                                          "firstLayer_vs_eta",
                                          "First layer of most alive SimNtuplet per TrackingParticle",
                                          "True pseudorapidity #eta",
                                          "First layer ID",
                                          etaNBins,
                                          etamin,
                                          etamax,
                                          numLayers_,
                                          -0.5,
                                          -0.5 + numLayers_);
    h_bestNtuplet_lastLayerVsEta_.book2D(ibook,
                                         "lastLayer_vs_eta",
                                         "Last layer of most alive SimNtuplet per TrackingParticle",
                                         "True pseudorapidity #eta",
                                         "Last layer ID",
                                         etaNBins,
                                         etamin,
                                         etamax,
                                         numLayers_,
                                         -0.5,
                                         -0.5 + numLayers_);

    // status histograms of the most alive SimNtuplets of the TrackingParticles
    h_bestNtuplet_alive_pt_ = simdoublets::make1DLogX(ibook,
                                                      "num_pt_alive",
                                                      "Most alive SimNtuplet per TrackingParticle (alive);True "
                                                      "transverse momentum p_{T} [GeV];Number of TrackingParticles",
                                                      pTNBins,
                                                      pTmin,
                                                      pTmax);

    h_bestNtuplet_undefDoubletCuts_pt_ =
        simdoublets::make1DLogX(ibook,
                                "num_pt_undefDoubletCuts",
                                "Most alive SimNtuplet per TrackingParticle (with undef doublet cuts);True "
                                "transverse momentum p_{T} [GeV];Number of TrackingParticles",
                                pTNBins,
                                pTmin,
                                pTmax);

    h_bestNtuplet_undefConnectionCuts_pt_ =
        simdoublets::make1DLogX(ibook,
                                "num_pt_undefConnectionCuts",
                                "Most alive SimNtuplet per TrackingParticle (with undef connection cuts);True "
                                "transverse momentum p_{T} [GeV];Number of TrackingParticles",
                                pTNBins,
                                pTmin,
                                pTmax);

    h_bestNtuplet_missingLayerPair_pt_ =
        simdoublets::make1DLogX(ibook,
                                "num_pt_missingLayerPair",
                                "Most alive SimNtuplet per TrackingParticle (with missing layer pair);True "
                                "transverse momentum p_{T} [GeV];Number of TrackingParticles",
                                pTNBins,
                                pTmin,
                                pTmax);

    h_bestNtuplet_killedDoublets_pt_ =
        simdoublets::make1DLogX(ibook,
                                "num_pt_killedDoublets",
                                "Most alive SimNtuplet per TrackingParticle (killed by doublet cuts);True "
                                "transverse momentum p_{T} [GeV];Number of TrackingParticles",
                                pTNBins,
                                pTmin,
                                pTmax);

    h_bestNtuplet_killedConnections_pt_ =
        simdoublets::make1DLogX(ibook,
                                "num_pt_killedConnections",
                                "Most alive SimNtuplet per TrackingParticle (killed by connection cuts);True "
                                "transverse momentum p_{T} [GeV];Number of TrackingParticles",
                                pTNBins,
                                pTmin,
                                pTmax);

    h_bestNtuplet_tooShort_pt_ = simdoublets::make1DLogX(
        ibook,
        "num_pt_tooShort",
        "Most alive SimNtuplet per TrackingParticle (3 RecHits but still shorter than the threshold);True "
        "transverse momentum p_{T} [GeV];Number of TrackingParticles",
        pTNBins,
        pTmin,
        pTmax);

    h_bestNtuplet_notStartingPair_pt_ =
        simdoublets::make1DLogX(ibook,
                                "num_pt_notStartingPair",
                                "Most alive SimNtuplet per TrackingParticle (has first doublet in layer pair not "
                                "considered for starting Ntuplets);True "
                                "transverse momentum p_{T} [GeV];Number of TrackingParticles",
                                pTNBins,
                                pTmin,
                                pTmax);

    h_bestNtuplet_alive_eta_ = ibook.book1D(
        "num_eta_alive",
        "Most alive SimNtuplet per TrackingParticle (alive);True pseudorapidity #eta;Number of TrackingParticles",
        etaNBins,
        etamin,
        etamax);

    h_bestNtuplet_undefDoubletCuts_eta_ = ibook.book1D("num_eta_undefDoubletCuts",
                                                       "Most alive SimNtuplet per TrackingParticle (with undef doublet "
                                                       "cuts);True pseudorapidity #eta;Number of TrackingParticles",
                                                       etaNBins,
                                                       etamin,
                                                       etamax);

    h_bestNtuplet_undefConnectionCuts_eta_ =
        ibook.book1D("num_eta_undefConnectionCuts",
                     "Most alive SimNtuplet per TrackingParticle (with undef connection cuts);True pseudorapidity "
                     "#eta;Number of TrackingParticles",
                     etaNBins,
                     etamin,
                     etamax);

    h_bestNtuplet_missingLayerPair_eta_ = ibook.book1D("num_eta_missingLayerPair",
                                                       "Most alive SimNtuplet per TrackingParticle (with missing layer "
                                                       "pair);True pseudorapidity #eta;Number of TrackingParticles",
                                                       etaNBins,
                                                       etamin,
                                                       etamax);

    h_bestNtuplet_killedDoublets_eta_ = ibook.book1D("num_eta_killedDoublets",
                                                     "Most alive SimNtuplet per TrackingParticle (killed by doublet "
                                                     "cuts);True pseudorapidity #eta;Number of TrackingParticles",
                                                     etaNBins,
                                                     etamin,
                                                     etamax);

    h_bestNtuplet_killedConnections_eta_ =
        ibook.book1D("num_eta_killedConnections",
                     "Most alive SimNtuplet per TrackingParticle (killed by connection "
                     "cuts);True pseudorapidity #eta;Number of TrackingParticles",
                     etaNBins,
                     etamin,
                     etamax);

    h_bestNtuplet_tooShort_eta_ =
        ibook.book1D("num_eta_tooShort",
                     "Most alive SimNtuplet per TrackingParticle (3 RecHits but still shorter than the "
                     "threshold);True pseudorapidity #eta;Number of TrackingParticles",
                     etaNBins,
                     etamin,
                     etamax);

    h_bestNtuplet_notStartingPair_eta_ =
        ibook.book1D("num_eta_notStartingPair",
                     "Most alive SimNtuplet per TrackingParticle (has first doublet in layer pair not considered for "
                     "starting Ntuplets);True pseudorapidity #eta;Number of TrackingParticles",
                     etaNBins,
                     etamin,
                     etamax);
  }

  // ---------------------------------------------------------------------
  // booking longest SimNtuplet histograms (simNtuplets/longest folder)
  // ---------------------------------------------------------------------

  ibook.setCurrentFolder(folder_ + "/SimNtuplets/longest");

  h_longNtuplet_numRecHits_.book1D(ibook,
                                   "numRecHits",
                                   "Number of RecHits in longest " + ntuplet + " per " + trackingObject,
                                   "Number of RecHits",
                                   "Number of " + trackingObject + "s",
                                   15,
                                   -0.5,
                                   14.5);
  h_longNtuplet_firstLayerId_.book1D(ibook,
                                     "firstLayerId",
                                     "First layer of longest " + ntuplet + " per " + trackingObject,
                                     "Layer ID",
                                     "Number of " + trackingObject + "s",
                                     numLayers_,
                                     -0.5,
                                     -0.5 + numLayers_);
  h_longNtuplet_lastLayerId_.book1D(ibook,
                                    "lastLayerId",
                                    "Last layer of longest " + ntuplet + " per " + trackingObject,
                                    "Layer ID",
                                    "Number of " + trackingObject + "s",
                                    numLayers_,
                                    -0.5,
                                    -0.5 + numLayers_);
  h_longNtuplet_layerSpan_.book2D(ibook,
                                  "layerSpan",
                                  "Layer span of longest " + ntuplet + " per " + trackingObject,
                                  "First layer ID",
                                  "Last layer ID",
                                  numLayers_,
                                  -0.5,
                                  -0.5 + numLayers_,
                                  numLayers_,
                                  -0.5,
                                  -0.5 + numLayers_);
  h_longNtuplet_firstVsSecondLayer_.book2D(ibook,
                                           "firstVsSecondLayer",
                                           "First two layers of longest " + ntuplet + " per " + trackingObject,
                                           "First layer ID",
                                           "Second layer ID",
                                           numLayers_,
                                           -0.5,
                                           -0.5 + numLayers_,
                                           numLayers_,
                                           -0.5,
                                           -0.5 + numLayers_);
  h_longNtuplet_numSkippedLayersVsNumLayers_.book2D(
      ibook,
      "numSkippedLayers_vs_numLayers",
      "Number of layers skipped of longest " + ntuplet + " per " + trackingObject,
      "Number of layers hit by Tracking Particle",
      "Number of skipped layers",
      15,
      -0.5,
      14.5,
      16,
      -1.5,
      14.5);

  if (!inputIsRecoTracks_) {
    h_longNtuplet_firstLayerVsEta_.book2D(ibook,
                                          "firstLayer_vs_eta",
                                          "First layer of longest SimNtuplet per TrackingParticle",
                                          "True pseudorapidity #eta",
                                          "First layer ID",
                                          etaNBins,
                                          etamin,
                                          etamax,
                                          numLayers_,
                                          -0.5,
                                          -0.5 + numLayers_);
    h_longNtuplet_lastLayerVsEta_.book2D(ibook,
                                         "lastLayer_vs_eta",
                                         "Last layer of longest SimNtuplet per TrackingParticle",
                                         "True pseudorapidity #eta",
                                         "Last layer ID",
                                         etaNBins,
                                         etamin,
                                         etamax,
                                         numLayers_,
                                         -0.5,
                                         -0.5 + numLayers_);

    // status histograms of the longest SimNtuplets of the TrackingParticles
    h_longNtuplet_alive_pt_ = simdoublets::make1DLogX(ibook,
                                                      "num_pt_alive",
                                                      "Longest SimNtuplets per TrackingParticle (alive);True "
                                                      "transverse momentum p_{T} [GeV];Number of TrackingParticles",
                                                      pTNBins,
                                                      pTmin,
                                                      pTmax);

    h_longNtuplet_undefDoubletCuts_pt_ =
        simdoublets::make1DLogX(ibook,
                                "num_pt_undefDoubletCuts",
                                "Longest SimNtuplets per TrackingParticle (with undef doublet cuts);True "
                                "transverse momentum p_{T} [GeV];Number of TrackingParticles",
                                pTNBins,
                                pTmin,
                                pTmax);

    h_longNtuplet_undefConnectionCuts_pt_ =
        simdoublets::make1DLogX(ibook,
                                "num_pt_undefConnectionCuts",
                                "Longest SimNtuplets per TrackingParticle (with undef connection cuts);True "
                                "transverse momentum p_{T} [GeV];Number of TrackingParticles",
                                pTNBins,
                                pTmin,
                                pTmax);

    h_longNtuplet_missingLayerPair_pt_ =
        simdoublets::make1DLogX(ibook,
                                "num_pt_missingLayerPair",
                                "Longest SimNtuplets per TrackingParticle (with missing layer pair);True "
                                "transverse momentum p_{T} [GeV];Number of TrackingParticles",
                                pTNBins,
                                pTmin,
                                pTmax);

    h_longNtuplet_killedDoublets_pt_ =
        simdoublets::make1DLogX(ibook,
                                "num_pt_killedDoublets",
                                "Longest SimNtuplets per TrackingParticle (killed by doublet cuts);True "
                                "transverse momentum p_{T} [GeV];Number of TrackingParticles",
                                pTNBins,
                                pTmin,
                                pTmax);

    h_longNtuplet_killedConnections_pt_ =
        simdoublets::make1DLogX(ibook,
                                "num_pt_killedConnections",
                                "Longest SimNtuplets per TrackingParticle (killed by connection cuts);True "
                                "transverse momentum p_{T} [GeV];Number of TrackingParticles",
                                pTNBins,
                                pTmin,
                                pTmax);

    h_longNtuplet_tooShort_pt_ = simdoublets::make1DLogX(
        ibook,
        "num_pt_tooShort",
        "Longest SimNtuplets per TrackingParticle (3 RecHits but still shorter than the threshold);True "
        "transverse momentum p_{T} [GeV];Number of TrackingParticles",
        pTNBins,
        pTmin,
        pTmax);

    h_longNtuplet_notStartingPair_pt_ =
        simdoublets::make1DLogX(ibook,
                                "num_pt_notStartingPair",
                                "Longest SimNtuplets per TrackingParticle (has first doublet in layer pair not "
                                "considered for starting Ntuplets);True "
                                "transverse momentum p_{T} [GeV];Number of TrackingParticles",
                                pTNBins,
                                pTmin,
                                pTmax);

    h_longNtuplet_alive_eta_ = ibook.book1D(
        "num_eta_alive",
        "Longest SimNtuplets per TrackingParticle (alive);True pseudorapidity #eta;Number of TrackingParticles",
        etaNBins,
        etamin,
        etamax);

    h_longNtuplet_undefDoubletCuts_eta_ = ibook.book1D("num_eta_undefDoubletCuts",
                                                       "Longest SimNtuplets per TrackingParticle (with undef doublet "
                                                       "cuts);True pseudorapidity #eta;Number of TrackingParticles",
                                                       etaNBins,
                                                       etamin,
                                                       etamax);

    h_longNtuplet_undefConnectionCuts_eta_ =
        ibook.book1D("num_eta_undefConnectionCuts",
                     "Longest SimNtuplets per TrackingParticle (with undef connection cuts);True pseudorapidity "
                     "#eta;Number of TrackingParticles",
                     etaNBins,
                     etamin,
                     etamax);

    h_longNtuplet_missingLayerPair_eta_ = ibook.book1D("num_eta_missingLayerPair",
                                                       "Longest SimNtuplets per TrackingParticle (with missing layer "
                                                       "pair);True pseudorapidity #eta;Number of TrackingParticles",
                                                       etaNBins,
                                                       etamin,
                                                       etamax);

    h_longNtuplet_killedDoublets_eta_ = ibook.book1D("num_eta_killedDoublets",
                                                     "Longest SimNtuplets per TrackingParticle (killed by doublet "
                                                     "cuts);True pseudorapidity #eta;Number of TrackingParticles",
                                                     etaNBins,
                                                     etamin,
                                                     etamax);

    h_longNtuplet_killedConnections_eta_ =
        ibook.book1D("num_eta_killedConnections",
                     "Longest SimNtuplets per TrackingParticle (killed by connection "
                     "cuts);True pseudorapidity #eta;Number of TrackingParticles",
                     etaNBins,
                     etamin,
                     etamax);

    h_longNtuplet_tooShort_eta_ =
        ibook.book1D("num_eta_tooShort",
                     "Longest SimNtuplets per TrackingParticle (3 RecHits but still shorter than the "
                     "threshold);True pseudorapidity #eta;Number of TrackingParticles",
                     etaNBins,
                     etamin,
                     etamax);

    h_longNtuplet_notStartingPair_eta_ =
        ibook.book1D("num_eta_notStartingPair",
                     "Longest SimNtuplets per TrackingParticle (has first doublet in layer pair not considered for "
                     "starting Ntuplets);True pseudorapidity #eta;Number of TrackingParticles",
                     etaNBins,
                     etamin,
                     etamax);
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
  int nPairs = pixelTopology::Phase1::nPairsForQuadruplets;

  edm::ParameterSetDescription desc;
  simdoublets::fillDescriptionsCommon<pixelTopology::Phase1>(desc);

  // input source for SimDoublets
  desc.add<edm::InputTag>("simDoubletsSrc", edm::InputTag("simDoubletsProducerPhase1"));

  // cutting parameters
  desc.add<int>("minYsizeB1", 36)->setComment("Minimum cluster size for inner RecHit from B1");
  desc.add<int>("minYsizeB2", 28)->setComment("Minimum cluster size for inner RecHit not from B1");
  desc.add<double>("cellZ0Cut", 12.0)->setComment("Maximum longitudinal impact parameter");
  desc.add<double>("cellPtCut", 0.5)->setComment("Minimum tranverse momentum");

  // layer-dependent parameters + layer pairs
  edm::ParameterSetDescription geometryParams;
  // layers params
  geometryParams
      .add<std::vector<double>>(
          "caDCACuts",
          std::vector<double>(std::begin(phase1PixelTopology::dcaCuts), std::end(phase1PixelTopology::dcaCuts)))
      ->setComment("Cut on RZ alignement. One per layer, the layer being the middle one for a triplet.");
  geometryParams
      .add<std::vector<double>>(
          "caThetaCuts",
          std::vector<double>(std::begin(phase1PixelTopology::thetaCuts), std::end(phase1PixelTopology::thetaCuts)))
      ->setComment("Cut on origin radius. One per layer, the layer being the innermost one for a triplet.");
  geometryParams.add<std::vector<unsigned int>>("startingPairs", {0u, 1u, 2u})
      ->setComment(
          "Array of variable length with the indices of the starting pairs for Ntuplet building");  //TODO could be parsed via an expression
  // cells params
  geometryParams
      .add<std::vector<unsigned int>>(
          "pairGraph",
          std::vector<uint>(std::begin(phase1PixelTopology::layerPairs), std::end(phase1PixelTopology::layerPairs)))
      ->setComment(
          "Array of length 2*NumberOfPairs where the elements at 2i and 2i+1 are the inner and outer layers of layer "
          "pair i considered in the CA for doublet building");
  geometryParams
      .add<std::vector<int>>(
          "phiCuts", std::vector<int>(std::begin(phase1PixelTopology::phicuts), std::end(phase1PixelTopology::phicuts)))
      ->setComment("Cuts in phi for cells");
  geometryParams
      .add<std::vector<double>>(
          "minInnerZ", std::vector<double>(std::begin(phase1PixelTopology::minz), std::end(phase1PixelTopology::minz)))
      ->setComment("Cuts in min z (on inner hit) for cells");
  geometryParams
      .add<std::vector<double>>(
          "maxInnerZ", std::vector<double>(std::begin(phase1PixelTopology::maxz), std::end(phase1PixelTopology::maxz)))
      ->setComment("Cuts in max z (on inner hit) for cells");
  geometryParams
      .add<std::vector<double>>(
          "maxDR", std::vector<double>(std::begin(phase1PixelTopology::maxr), std::end(phase1PixelTopology::maxr)))
      ->setComment("Cuts in max r for cells");
  geometryParams.add<std::vector<double>>("minOuterZ", std::vector<double>(nPairs, -10000))
      ->setComment("Cuts in min z (of outer hit) for cells");
  geometryParams.add<std::vector<double>>("maxOuterZ", std::vector<double>(nPairs, 10000))
      ->setComment("Cuts in max z (of outer hit) for cells");
  geometryParams.add<std::vector<double>>("minInnerR", std::vector<double>(nPairs, 0))
      ->setComment("Cuts in min r (of inner hit) for cells");
  geometryParams.add<std::vector<double>>("maxInnerR", std::vector<double>(nPairs, 10000))
      ->setComment("Cuts in max r (of inner hit) for cells");
  geometryParams.add<std::vector<double>>("minOuterR", std::vector<double>(nPairs, 0))
      ->setComment("Cuts in min r (of outer hit) for cells");
  geometryParams.add<std::vector<double>>("maxOuterR", std::vector<double>(nPairs, 10000))
      ->setComment("Cuts in max r (of outer hit) for cells");
  geometryParams.add<std::vector<double>>("minDZ", std::vector<double>(nPairs, -10000))
      ->setComment("Cuts in max dz for cells");
  geometryParams.add<std::vector<double>>("maxDZ", std::vector<double>(nPairs, 10000))
      ->setComment("Cuts in min dz for cells");

  desc.add<edm::ParameterSetDescription>("geometry", geometryParams)
      ->setComment("Layer pair graph, layer-dependent cut values.");

  descriptions.addWithDefaultLabel(desc);
}

// fillDescription for Phase 2
template <>
void SimDoubletsAnalyzer<pixelTopology::Phase2>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  int nPairs = pixelTopology::Phase2::nPairs;

  edm::ParameterSetDescription desc;
  simdoublets::fillDescriptionsCommon<pixelTopology::Phase2>(desc);

  // input source for SimDoublets
  desc.add<edm::InputTag>("simDoubletsSrc", edm::InputTag("simDoubletsProducerPhase2"));

  // cutting parameters for doublets
  desc.add<int>("minYsizeB1", 25)->setComment("Minimum cluster size for inner RecHit from B1");
  desc.add<int>("minYsizeB2", 15)->setComment("Minimum cluster size for inner RecHit not from B1");
  desc.add<double>("cellZ0Cut", 7.5)->setComment("Maximum longitudinal impact parameter");
  desc.add<double>("cellPtCut", 0.85)->setComment("Minimum tranverse momentum in GeV");

  // layer-dependent parameters + layer pairs
  edm::ParameterSetDescription geometryParams;
  // layers params
  geometryParams
      .add<std::vector<double>>(
          "caDCACuts",
          std::vector<double>(std::begin(phase2PixelTopology::dcaCuts), std::end(phase2PixelTopology::dcaCuts)))
      ->setComment("Cut on RZ alignement. One per layer, the layer being the middle one for a triplet.");
  geometryParams
      .add<std::vector<double>>(
          "caThetaCuts",
          std::vector<double>(std::begin(phase2PixelTopology::thetaCuts), std::end(phase2PixelTopology::thetaCuts)))
      ->setComment("Cut on origin radius. One per layer, the layer being the innermost one for a triplet.");
  geometryParams.add<std::vector<unsigned int>>("startingPairs", {0u, 1u, 2u})
      ->setComment(
          "Array of variable length with the indices of the starting pairs for Ntuplet building");  //TODO could be parsed via an expression
  // cells params
  geometryParams
      .add<std::vector<unsigned int>>(
          "pairGraph",
          std::vector<uint>(std::begin(phase2PixelTopology::layerPairs), std::end(phase2PixelTopology::layerPairs)))
      ->setComment(
          "Array of length 2*NumberOfPairs where the elements at 2i and 2i+1 are the inner and outer layers of layer "
          "pair i considered in the CA for doublet building");
  geometryParams
      .add<std::vector<int>>(
          "phiCuts", std::vector<int>(std::begin(phase2PixelTopology::phicuts), std::end(phase2PixelTopology::phicuts)))
      ->setComment("Cuts in phi for cells");
  geometryParams
      .add<std::vector<double>>(
          "minInnerZ", std::vector<double>(std::begin(phase2PixelTopology::minz), std::end(phase2PixelTopology::minz)))
      ->setComment("Cuts in min z (on inner hit) for cells");
  geometryParams
      .add<std::vector<double>>(
          "maxInnerZ", std::vector<double>(std::begin(phase2PixelTopology::maxz), std::end(phase2PixelTopology::maxz)))
      ->setComment("Cuts in max z (on inner hit) for cells");
  geometryParams
      .add<std::vector<double>>(
          "maxDR", std::vector<double>(std::begin(phase2PixelTopology::maxr), std::end(phase2PixelTopology::maxr)))
      ->setComment("Cuts in max r for cells");
  geometryParams.add<std::vector<double>>("minOuterZ", std::vector<double>(nPairs, -10000))
      ->setComment("Cuts in min z (of outer hit) for cells");
  geometryParams.add<std::vector<double>>("maxOuterZ", std::vector<double>(nPairs, 10000))
      ->setComment("Cuts in max z (of outer hit) for cells");
  geometryParams.add<std::vector<double>>("minInnerR", std::vector<double>(nPairs, 0))
      ->setComment("Cuts in min r (of inner hit) for cells");
  geometryParams.add<std::vector<double>>("maxInnerR", std::vector<double>(nPairs, 10000))
      ->setComment("Cuts in max r (of inner hit) for cells");
  geometryParams.add<std::vector<double>>("minOuterR", std::vector<double>(nPairs, 0))
      ->setComment("Cuts in min r (of outer hit) for cells");
  geometryParams.add<std::vector<double>>("maxOuterR", std::vector<double>(nPairs, 10000))
      ->setComment("Cuts in max r (of outer hit) for cells");
  geometryParams.add<std::vector<double>>("minDZ", std::vector<double>(nPairs, -10000))
      ->setComment("Cuts in max dz for cells");
  geometryParams.add<std::vector<double>>("maxDZ", std::vector<double>(nPairs, 10000))
      ->setComment("Cuts in min dz for cells");

  desc.add<edm::ParameterSetDescription>("geometry", geometryParams)
      ->setComment("Layer pair graph, layer-dependent cut values.");

  descriptions.addWithDefaultLabel(desc);
}

// define two plugins for Phase 1 and 2
using SimDoubletsAnalyzerPhase1 = SimDoubletsAnalyzer<pixelTopology::Phase1>;
using SimDoubletsAnalyzerPhase2 = SimDoubletsAnalyzer<pixelTopology::Phase2>;

// define this as a plug-in
DEFINE_FWK_MODULE(SimDoubletsAnalyzerPhase1);
DEFINE_FWK_MODULE(SimDoubletsAnalyzerPhase2);
