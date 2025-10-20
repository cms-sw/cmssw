// -*- C++ -*-
//
// Package:    Validation/TrackingMCTruth
// Class:      SimPixelTrackAnalyzer
//

// #define DOUBLETCUTS_PRINTOUTS
// #define LOSTNTUPLETS_PRINTOUTS

// user include files
#include "Validation/TrackingMCTruth/plugins/SimPixelTrackAnalyzer.h"
#include <sys/types.h>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
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
    void calculateCutVariables(SimPixelTrack::Doublet& doublet, SimPixelTrack const& simPixelTrack) {
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
      dCurvCut_.clear();
      curvRatioCut_.clear();
      tripletConnectionPassed_.clear();
      // then, refill
      for (auto& neighbor : doublet.innerNeighbors()) {
        // get the inner RecHit of the inner neighbor
        GlobalPoint neighbor_globalPosition = simPixelTrack.getSimDoublet(neighbor.index()).innerGlobalPos();
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
        double tripletCurvature = eq.curvature();
        neighbor.setCurvature(tripletCurvature);
        hardCurvCut_.push_back(std::abs(tripletCurvature));
        dcaCut_.push_back(std::abs(eq.dca0() / std::abs(tripletCurvature)));
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
    std::vector<double>& dCurvCut() const { return dCurvCut_; }
    std::vector<double>& curvRatioCut() const { return curvRatioCut_; }
    std::vector<bool>& tripletConnectionPassed() const { return tripletConnectionPassed_; }
    double CAThetaCut(int i) const { return CAThetaCut_.at(i); }
    double dcaCut(int i) const { return dcaCut_.at(i); }
    double hardCurvCut(int i) const { return hardCurvCut_.at(i); }
    double dCurvCut(int i) const { return dCurvCut_.at(i); }
    double curvRatioCut(int i) const { return curvRatioCut_.at(i); }
    bool tripletConnectionPassed(int i) const { return tripletConnectionPassed_.at(i); }

  private:
    double inner_z_, inner_r_, outer_z_, outer_r_, dz_, dr_;
    double dphi_, z0_, curvature_, pT_;                      // double-valued variables
    int idphi_, Ysize_, DYsize_, DYPred_;                    // integer-valued variables
    std::vector<double> CAThetaCut_, dcaCut_, hardCurvCut_;  // doublet connection cut variables
    mutable std::vector<double> dCurvCut_, curvRatioCut_;    // triplet connection cut variables
    mutable std::vector<bool> tripletConnectionPassed_;
  };

  template <typename TrackerTraits>
  bool moduleIsOuterLadder(int const moduleId);

  template <>
  bool moduleIsOuterLadder<pixelTopology::Phase1>(int const moduleId) {
    return (0 == (moduleId / 8) % 2);
  }

  template <>
  bool moduleIsOuterLadder<pixelTopology::Phase2>(int const moduleId) {
    return (0 != (moduleId / 18) % 2);
  }

  // class to help keep track of which cluster size cuts are applied
  template <typename TrackerTraits>
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
    void setSubjectsToCuts(SimPixelTrack::Doublet const& doublet) {
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
      const bool isOuterLadder = moduleIsOuterLadder<TrackerTraits>(moduleId);
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
    desc.add<bool>("includeOTBarrel", false)->setComment("If true, add barrel layers from the OT extension.");
    desc.add<bool>("includeOTDisks", false)->setComment("If true, add disk layers from the OT extension.");

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
SimPixelTrackAnalyzer<TrackerTraits>::SimPixelTrackAnalyzer(const edm::ParameterSet& iConfig)
    : topology_getToken_(esConsumes<TrackerTopology, TrackerTopologyRcd>()),
      simPixelTracks_getToken_(consumes(iConfig.getParameter<edm::InputTag>("simPixelTracksSrc"))),
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
  hVector_z0_.resize(numLayerPairs);
  hVector_curvatureR_.resize(numLayerPairs);
  hVector_pTFromR_.resize(numLayerPairs);
  hVector_dz_.resize(numLayerPairs);
  hVector_dr_.resize(numLayerPairs);
  hVector_dphi_.resize(numLayerPairs);
  hVector_idphi_.resize(numLayerPairs);
  hVector_inner_.resize(numLayerPairs);
  hVector_outer_.resize(numLayerPairs);
  hVector_Ysize_.resize(numLayerPairs);
  hVector_DYsize_.resize(numLayerPairs);
  hVector_DYPred_.resize(numLayerPairs);

  // resize other vectors according to number of layers
  // set the number of layers of the extension
  int numLayersOTBarrel = (iConfig.getParameter<bool>("includeOTBarrel")) ? 3 : 0;
  int numLayersOTDisks = (iConfig.getParameter<int>("includeOTDisks")) ? 5 : 0;
  numLayers_ = TrackerTraits::numberOfLayers + 2 * numLayersOTDisks + numLayersOTBarrel;
  hVector_caThetaCut_.resize(numLayers_);
  hVector_caDCACut_.resize(numLayers_);
  hVector_firstHitR_.resize(numLayers_);
}

template <typename TrackerTraits>
SimPixelTrackAnalyzer<TrackerTraits>::~SimPixelTrackAnalyzer() {}

// -------------------------------------------------------------------------------------------------------------
// member functions
// -------------------------------------------------------------------------------------------------------------

template <typename TrackerTraits>
void SimPixelTrackAnalyzer<TrackerTraits>::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {}

// function to apply cuts and set doublet to alive if it passes and to killed otherwise
template <typename TrackerTraits>
void SimPixelTrackAnalyzer<TrackerTraits>::applyCuts(
    SimPixelTrack::Doublet& doublet,
    SimPixelTrack const& simPixelTrack,
    bool const hasValidNeighbors,
    bool const hasValidTripletNeighbors,
    int const layerPairIdIndex,
    simdoublets::CellCutVariables const& cellCutVariables,
    simdoublets::ClusterSizeCutManager<TrackerTraits> const& clusterSizeCutManager) {
  // -------------------------------------------------------------------------
  //  apply cuts for doublet creation
  // -------------------------------------------------------------------------

  double inner = cellCuts_.isBarrel_[doublet.innerLayerId()] ? cellCutVariables.inner_z() : cellCutVariables.inner_r();
  double outer = cellCuts_.isBarrel_[doublet.outerLayerId()] ? cellCutVariables.outer_z() : cellCutVariables.outer_r();

  bool passInner{true}, passYsize{true}, passOuter{true}, passDPhi{true}, passDR{true}, passDZ{true}, passDYsize{true},
      passPt{true}, passZ0{true};

  /* inner r/z window cut */
  if (inner < cellCuts_.minInner_[layerPairIdIndex] || inner > cellCuts_.maxInner_[layerPairIdIndex])
    passInner = false;
  /* outer r/z window cut */
  if (outer < cellCuts_.minOuter_[layerPairIdIndex] || outer > cellCuts_.maxOuter_[layerPairIdIndex])
    passOuter = false;
  /* dz window */
  if (cellCutVariables.dz() > cellCuts_.maxDZ_[layerPairIdIndex] ||
      cellCutVariables.dz() < cellCuts_.minDZ_[layerPairIdIndex])
    passDZ = false;
  /* z0cutoff */
  if (cellCutVariables.dr() > cellCuts_.maxDR_[layerPairIdIndex] || cellCutVariables.dr() < 0)
    passDR = false;
  if (cellCutVariables.z0() > cellZ0Cut_)
    passZ0 = false;
  /* ptcut */
  if (cellCutVariables.pT() < cellCuts_.ptCuts_[layerPairIdIndex])
    passPt = false;
  /* idphicut */
  if (cellCutVariables.idphi() > cellCuts_.phiCuts_[layerPairIdIndex])
    passDPhi = false;
  /* YsizeB1/2 cut */
  if ((clusterSizeCutManager.isSubjectToYsizeB1() && (cellCutVariables.Ysize() < minYsizeB1_)) ||
      (clusterSizeCutManager.isSubjectToYsizeB2() && (cellCutVariables.Ysize() < minYsizeB2_)))
    passYsize = false;
  if (
      /* DYsize12 cut */
      (clusterSizeCutManager.isSubjectToDYsize12() && (cellCutVariables.DYsize() > maxDYsize12_)) ||
      /* DYsize cut */
      (clusterSizeCutManager.isSubjectToDYsize() && (cellCutVariables.DYsize() > maxDYsize_)) ||
      /* DYPred cut */
      (clusterSizeCutManager.isSubjectToDYPred() && (cellCutVariables.DYPred() > maxDYPred_)))
    passDYsize = false;
  if (!(passInner && passYsize && passOuter && passDZ && passDPhi && passDYsize && passPt && passDR && passZ0)) {
    // if any of the cuts apply kill the doublet
    doublet.setKilledByCuts();
  } else {
    // if the function arrives here, the doublet survived
    doublet.setAlive();
  }

  // fill pass this cut histograms
  h_z0_.fillPassThisCut(passZ0);
  h_pTFromR_.fillPassThisCut(passPt);
  h_YsizeB1_.fillPassThisCut(passYsize);
  h_YsizeB2_.fillPassThisCut(passYsize);
  h_DYsize12_.fillPassThisCut(passDYsize);
  h_DYsize_.fillPassThisCut(passDYsize);
  h_DYPred_.fillPassThisCut(passDYsize);
  hVector_z0_[layerPairIdIndex].fillPassThisCut(passZ0);
  hVector_pTFromR_[layerPairIdIndex].fillPassThisCut(passPt);
  hVector_dz_[layerPairIdIndex].fillPassThisCut(passDZ);
  hVector_dr_[layerPairIdIndex].fillPassThisCut(passDR);
  hVector_idphi_[layerPairIdIndex].fillPassThisCut(passDPhi);
  hVector_inner_[layerPairIdIndex].fillPassThisCut(passInner);
  hVector_outer_[layerPairIdIndex].fillPassThisCut(passOuter);
  hVector_Ysize_[layerPairIdIndex].fillPassThisCut(passYsize);
  hVector_DYsize_[layerPairIdIndex].fillPassThisCut(passDYsize);
  hVector_DYPred_[layerPairIdIndex].fillPassThisCut(passDYsize);

#ifdef DOUBLETCUTS_PRINTOUTS
  printf("%d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",
         layerPairIdIndex,
         doublet.innerLayerId(),
         doublet.outerLayerId(),
         0,
         0,
         passInner,
         passYsize,
         passOuter,
         passDZ,
         passZ0,
         passDR,
         passDPhi,
         passDYsize,
         passPt);
#endif

  // -------------------------------------------------------------------------
  //  apply cuts for doublet and triplet connections
  // -------------------------------------------------------------------------
  if (hasValidNeighbors) {
    // loop over the inner neighboring doublets of the doublet
    for (int i{0}; auto& neighbor : doublet.innerNeighbors()) {
      bool passCATheta{true}, passHardCurv{true}, passDca{true};

      // apply CAThetaCut
      if (cellCutVariables.CAThetaCut(i) > cellCuts_.caThetaCuts_over_ptmin_.at(doublet.innerLayerId()))
        passCATheta = false;
      // apply hardCurvCut
      if (cellCutVariables.hardCurvCut(i) > hardCurvCut_)
        passHardCurv = false;
      // apply dcaCut
      if (cellCutVariables.dcaCut(i) > cellCuts_.caDCACuts_.at(doublet.innerNeighborsInnerLayerId()))
        passDca = false;

      h_hardCurvCut_.fillPassThisCut(passHardCurv);
      hVector_caThetaCut_[doublet.innerLayerId()].fillPassThisCut(passCATheta);
      hVector_caDCACut_[doublet.innerNeighborsInnerLayerId()].fillPassThisCut(passDca);

      if (!(passCATheta && passHardCurv && passDca)) {
        neighbor.setKilled();
      } else {
        neighbor.setAlive();
      }

      // loop over the neighbors of the neighbors to apply cuts on triplet connections
      if (hasValidTripletNeighbors) {
        auto const& neighborDoublet = simPixelTrack.getSimDoublet(neighbor.index());
        for (int j{0}; auto const& tripletNeighbor : neighborDoublet.innerNeighborsView()) {
          /* DCurv cut*/
          double dCurv = std::abs(tripletNeighbor.curvature() - neighbor.curvature());
          cellCutVariables.dCurvCut().push_back(dCurv);
          /* curvRatio cut*/
          double curvRatio = tripletNeighbor.curvature() / neighbor.curvature();
          cellCutVariables.curvRatioCut().push_back(curvRatio);
          if (dCurv > 100000.) {
            neighbor.setKilledTripletConnection(j);
            cellCutVariables.tripletConnectionPassed().push_back(false);
          } else
            cellCutVariables.tripletConnectionPassed().push_back(true);

          j++;
        }
      }

      i++;
    }
  }
}

// function that fills all histograms for cut variables
template <typename TrackerTraits>
void SimPixelTrackAnalyzer<TrackerTraits>::fillCutHistograms(
    SimPixelTrack::Doublet const& doublet,
    bool hasValidNeighbors,
    bool hasValidTripletNeighbors,
    int const layerPairIdIndex,
    simdoublets::CellCutVariables const& cellCutVariables,
    simdoublets::ClusterSizeCutManager<TrackerTraits> const& clusterSizeCutManager,
    simdoublets::TrackTruth const& trackTruth) {
  // check if the doublet passed all cuts
  bool passed = doublet.isAlive();
  double inner = cellCuts_.isBarrel_[doublet.innerLayerId()] ? cellCutVariables.inner_z() : cellCutVariables.inner_r();
  double outer = cellCuts_.isBarrel_[doublet.outerLayerId()] ? cellCutVariables.outer_z() : cellCutVariables.outer_r();

  // -------------------------------------------------------------------------
  //  layer pair independent cuts (global folder)
  // -------------------------------------------------------------------------
  // radius of the circle defined by the two RecHits and the beamspot
  h_curvatureR_.fillCut(passed, trackTruth, inner, cellCutVariables.curvature());
  hVector_curvatureR_[layerPairIdIndex].fillCut(passed, trackTruth, inner, cellCutVariables.curvature());
  // pT that this curvature radius corresponds to
  h_pTFromR_.fillCut(passed, trackTruth, inner, cellCutVariables.pT());
  hVector_pTFromR_[layerPairIdIndex].fillCut(passed, trackTruth, inner, cellCutVariables.pT());
  // longitudinal impact parameter with respect to the beamspot
  h_z0_.fillCut(passed, trackTruth, inner, cellCutVariables.z0());
  hVector_z0_[layerPairIdIndex].fillCut(passed, trackTruth, inner, cellCutVariables.z0());

  // -------------------------------------------------------------------------
  //  layer pair dependent cuts (sub-folders for layer pairs)
  // -------------------------------------------------------------------------
  // dr = (outer_r - inner_r) histogram
  hVector_dr_[layerPairIdIndex].fillCut(passed, trackTruth, inner, cellCutVariables.dr());
  // dphi histogram
  hVector_dphi_[layerPairIdIndex].fillCut(passed, trackTruth, inner, cellCutVariables.dphi());
  hVector_idphi_[layerPairIdIndex].fillCut(passed, trackTruth, inner, cellCutVariables.idphi());
  // r/z of the inner and outer RecHit histograms
  hVector_inner_[layerPairIdIndex].fillCut(passed, trackTruth, inner, inner);
  hVector_outer_[layerPairIdIndex].fillCut(passed, trackTruth, inner, outer);
  // potential cuts for dz
  hVector_dz_[layerPairIdIndex].fillCut(passed, trackTruth, inner, cellCutVariables.dz());

  // -------------------------------------------------------------------------
  //  cluster size cuts (global + sub-folders for layer pairs)
  // -------------------------------------------------------------------------
  // cluster size in local y histogram
  hVector_Ysize_[layerPairIdIndex].fillCut(passed, trackTruth, inner, cellCutVariables.Ysize());
  // histograms for clusterCut
  // YsizeB1 cut
  if (clusterSizeCutManager.isSubjectToYsizeB1()) {
    h_YsizeB1_.fillCut(passed, trackTruth, inner, cellCutVariables.Ysize());
  }
  // YsizeB2 cut
  if (clusterSizeCutManager.isSubjectToYsizeB2()) {
    h_YsizeB2_.fillCut(passed, trackTruth, inner, cellCutVariables.Ysize());
  }
  // histograms for zSizeCut
  // DYsize12 cut
  if (clusterSizeCutManager.isSubjectToDYsize12()) {
    hVector_DYsize_[layerPairIdIndex].fillCut(passed, trackTruth, inner, cellCutVariables.DYsize());
    h_DYsize12_.fillCut(passed, trackTruth, inner, cellCutVariables.DYsize());
  }
  // DYsize cut
  if (clusterSizeCutManager.isSubjectToDYsize()) {
    hVector_DYsize_[layerPairIdIndex].fillCut(passed, trackTruth, inner, cellCutVariables.DYsize());
    h_DYsize_.fillCut(passed, trackTruth, inner, cellCutVariables.DYsize());
  }
  // DYPred cut
  if (clusterSizeCutManager.isSubjectToDYPred()) {
    hVector_DYPred_[layerPairIdIndex].fillCut(passed, trackTruth, inner, cellCutVariables.DYPred());
    h_DYPred_.fillCut(passed, trackTruth, inner, cellCutVariables.DYPred());
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

      // loop over the neighbors of the neighbors to fill histograms on triplet connections
      if (hasValidTripletNeighbors) {
        for (size_t j{0}; bool const passedTripletConnect : cellCutVariables.tripletConnectionPassed()) {
          // DCurv cut
          h_dCurvCut_.fill(passedTripletConnect, cellCutVariables.dCurvCut(j));
          // curvRatioCut
          h_curvRatioCut_.fill(passedTripletConnect, cellCutVariables.curvRatioCut(j));

          j++;
        }
      }

      i++;
    }
  }
}

//  function that fills all histograms of SimDoublets (in folder SimDoublets)
template <typename TrackerTraits>
void SimPixelTrackAnalyzer<TrackerTraits>::fillSimDoubletHistograms(SimPixelTrack::Doublet const& doublet,
                                                                    simdoublets::TrackTruth const& trackTruth) {
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
  h_num_vs_pt_.fill(passed, trackTruth.pt);
  h_num_vs_eta_.fill(passed, trackTruth.eta);
  h_num_vs_vertpos_.fill(passed, trackTruth.vertpos);
}

//  function that fills all histograms of SimNtuplets (in folder SimNtuplets)
template <typename TrackerTraits>
void SimPixelTrackAnalyzer<TrackerTraits>::fillSimNtupletHistograms(SimPixelTrack const& simPixelTrack,
                                                                    simdoublets::TrackTruth const& trackTruth) {
  // get the longest SimNtuplet of the TrackingParticle (if it exists)
  auto const& longNtuplet = simPixelTrack.longestSimNtuplet();

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
  if (simPixelTrack.numRecHits() > 0) {
    auto firstHitR = simPixelTrack.globalPositions(0).perp();
    auto firstHitLayerId = simPixelTrack.layerIds(0);
    hVector_firstHitR_[firstHitLayerId].fill(simPixelTrack.hasAliveSimNtuplet(), firstHitR);
  }

  // if input are RecoTracks break here and don't fill the other histograms
  if (inputIsRecoTracks_)
    return;

  h_longNtuplet_firstLayerVsEta_.fill(isAlive, trackTruth.eta, longNtuplet.firstLayerId());
  h_longNtuplet_lastLayerVsEta_.fill(isAlive, trackTruth.eta, longNtuplet.lastLayerId());

  // fill the respective histogram
  // 1. check if alive
  if (isAlive) {
    h_longNtuplet_.alive_.fill(trackTruth);
  }
  // 2. if not alive, find out why (go in order of cut application)
  // A) the Ntuplet does not meet the minimum length requirement and will never be build
  else if (longNtuplet.isTooShort()) {
    h_longNtuplet_.tooShort_.fill(trackTruth);
  }
  // B) a layer pair is missing, therefore no doublet is formed
  else if (longNtuplet.hasMissingLayerPair()) {
    h_longNtuplet_.missingLayerPair_.fill(trackTruth);
  }
  // C) a doublet got killed by doublet building cuts
  else if (longNtuplet.hasKilledDoublets()) {
    h_longNtuplet_.killedDoublets_.fill(trackTruth);
  }
  // D) one of connections between the doublets got cut
  else if (longNtuplet.hasKilledDoubletConnections()) {
    h_longNtuplet_.killedDoubletConnections_.fill(trackTruth);
  }
  // E) one of connections between the triplets got cut
  else if (longNtuplet.hasKilledTripletConnections()) {
    h_longNtuplet_.killedTripletConnections_.fill(trackTruth);
  }
  // F) the Ntuplet starts with a layer pair not considered for starting
  else if (longNtuplet.firstDoubletNotInStartingLayerPairs()) {
    h_longNtuplet_.notStartingPair_.fill(trackTruth);
  }
  // G) if we arrive here something's wrong
  else if (longNtuplet.hasUndefDoubletCuts()) {
    h_longNtuplet_.undefDoubletCuts_.fill(trackTruth);
  }
  // H) or even wronger...
  else if (longNtuplet.hasUndefDoubletConnectionCuts()) {
    h_longNtuplet_.undefConnectionCuts_.fill(trackTruth);
  }

  // -------------------------------------------------------------------------------------
  // fill the most alive (best) histograms
  auto const& bestNtuplet = simPixelTrack.bestSimNtuplet();
  // check if it is alive
  isAlive = bestNtuplet.isAlive();

  // fill general longest SimNtuplet histogram
  h_bestNtuplet_numRecHits_.fill(isAlive, bestNtuplet.numRecHits());
  h_bestNtuplet_firstLayerId_.fill(isAlive, bestNtuplet.firstLayerId());
  h_bestNtuplet_lastLayerId_.fill(isAlive, bestNtuplet.lastLayerId());
  h_bestNtuplet_layerSpan_.fill(isAlive, bestNtuplet.firstLayerId(), bestNtuplet.lastLayerId());
  h_bestNtuplet_firstVsSecondLayer_.fill(isAlive, bestNtuplet.firstLayerId(), bestNtuplet.secondLayerId());
  h_bestNtuplet_firstLayerVsEta_.fill(isAlive, trackTruth.eta, bestNtuplet.firstLayerId());
  h_bestNtuplet_lastLayerVsEta_.fill(isAlive, trackTruth.eta, bestNtuplet.lastLayerId());
  h_bestNtuplet_numSkippedLayersVsNumLayers_.fill(isAlive, bestNtuplet.numRecHits(), bestNtuplet.numSkippedLayers());

  // fill the respective histogram
  // 1. check if alive
  if (isAlive) {
    h_bestNtuplet_.alive_.fill(trackTruth);
  }
  // 2. if not alive, find out why (go in order of cut application)
  // A) the Ntuplet does not meet the minimum length requirement and will never be build
  else if (bestNtuplet.isTooShort()) {
    h_bestNtuplet_.tooShort_.fill(trackTruth);
  }
  // B) a layer pair is missing, therefore no doublet is formed
  else if (bestNtuplet.hasMissingLayerPair()) {
    h_bestNtuplet_.missingLayerPair_.fill(trackTruth);
  }
  // C) a doublet got killed by doublet building cuts
  else if (bestNtuplet.hasKilledDoublets()) {
    h_bestNtuplet_.killedDoublets_.fill(trackTruth);
  }
  // D) one of connections between the doublets got cut
  else if (bestNtuplet.hasKilledDoubletConnections()) {
    h_bestNtuplet_.killedDoubletConnections_.fill(trackTruth);
  }
  // E) one of connections between the triplets got cut
  else if (bestNtuplet.hasKilledTripletConnections()) {
    h_bestNtuplet_.killedTripletConnections_.fill(trackTruth);
  }
  // F) the Ntuplet starts with a layer pair not considered for starting
  else if (bestNtuplet.firstDoubletNotInStartingLayerPairs()) {
    h_bestNtuplet_.notStartingPair_.fill(trackTruth);
  }
  // G) if we arrive here something's wrong
  else if (bestNtuplet.hasUndefDoubletCuts()) {
    h_bestNtuplet_.undefDoubletCuts_.fill(trackTruth);
  }
  // H) or even wronger...
  else if (bestNtuplet.hasUndefDoubletConnectionCuts()) {
    h_bestNtuplet_.undefConnectionCuts_.fill(trackTruth);
  }
  // -------------------------------------------------------------------------------------
  if (simPixelTrack.hasAliveSimNtuplet()) {
    auto const& aliveNtuplet = simPixelTrack.longestAliveSimNtuplet();
    // relative length of alive SimNtuplet vs longest SimNtuplet
    double relativeLength = (double)aliveNtuplet.numRecHits() / (double)longNtuplet.numRecHits();
    h_aliveNtuplet_fracNumRecHits_eta_->Fill(trackTruth.eta, relativeLength);
    h_aliveNtuplet_fracNumRecHits_pt_->Fill(trackTruth.pt, relativeLength);
  }
}

// function that fills all general histograms (in folder general)
template <typename TrackerTraits>
void SimPixelTrackAnalyzer<TrackerTraits>::fillGeneralHistograms(SimPixelTrack const& simPixelTrack,
                                                                 simdoublets::TrackTruth const& trackTruth,
                                                                 int const pass_numSimDoublets,
                                                                 int const numSimDoublets,
                                                                 int const numSkippedLayers) {
  // Now check if the TrackingParticle has a surviving SimNtuplet
  bool passed = simPixelTrack.hasAliveSimNtuplet();

  // count number of RecHits in layers
  std::vector<int> countsRecHitsPerLayer(numLayers_, 0);
  for (auto const layerId : simPixelTrack.layerIds())
    countsRecHitsPerLayer.at(layerId)++;
  for (int layerId{0}; auto countRecHits : countsRecHitsPerLayer) {
    h_numRecHitsPerLayer_.fill(passed, layerId, countRecHits);
    layerId++;
  }

  if (inputIsRecoTracks_) {
    auto nChi2 = simPixelTrack.track()->normalizedChi2();
    h_numTOVsChi2_.fill(passed, nChi2);
    h_numRecHitsVsChi2_.fill(passed, nChi2, simPixelTrack.numRecHits());
  } else {
    h_numTOVsPdgId_.fill(passed, trackTruth.pdgId);
    // Fill the efficiency profile per Tracking Particle only if the TP has at least one SimDoublet
    if (numSimDoublets > 0) {
      h_effSimDoubletsPerTOVsEta_->Fill(trackTruth.eta, (double)pass_numSimDoublets / (double)numSimDoublets);
      h_effSimDoubletsPerTOVsPt_->Fill(trackTruth.pt, (double)pass_numSimDoublets / (double)numSimDoublets);
    }
  }

  // fill histograms for number of SimDoublets
  h_numSimDoubletsPerTrackingObject_.fill(passed, numSimDoublets);
  h_numRecHitsPerTrackingObject_.fill(passed, simPixelTrack.numRecHits());
  h_numLayersPerTrackingObject_.fill(passed, simPixelTrack.numLayers());
  h_numSkippedLayersPerTrackingObject_.fill(passed, numSkippedLayers);
  h_numSkippedLayersVsNumLayers_.fill(passed, simPixelTrack.numLayers(), numSkippedLayers);
  h_numSkippedLayersVsNumRecHits_.fill(passed, simPixelTrack.numRecHits(), numSkippedLayers);
  h_numRecHitsVsEta_.fill(passed, trackTruth.eta, simPixelTrack.numRecHits());
  h_numLayersVsEta_.fill(passed, trackTruth.eta, simPixelTrack.numLayers());
  h_numSkippedLayersVsEta_.fill(passed, trackTruth.eta, numSkippedLayers);
  h_numRecHitsVsPt_.fill(passed, trackTruth.pt, simPixelTrack.numRecHits());
  h_numLayersVsPt_.fill(passed, trackTruth.pt, simPixelTrack.numLayers());
  h_numSkippedLayersVsPt_.fill(passed, trackTruth.pt, numSkippedLayers);
  h_numLayersVsEtaPt_->Fill(trackTruth.eta, trackTruth.pt, simPixelTrack.numLayers());
  // fill histograms for number of TrackingParticles
  h_numTOVsPt_.fill(passed, trackTruth.pt);
  h_numTOVsEta_.fill(passed, trackTruth.eta);
  h_numTOVsPhi_.fill(passed, trackTruth.phi);
  h_numTOVsDxy_.fill(passed, trackTruth.dxy);
  h_numTOVsDz_.fill(passed, trackTruth.dz);
  h_numTOVsVertpos_.fill(passed, trackTruth.vertpos);
  h_numRecHitsVsDxy_.fill(passed, trackTruth.dxy, simPixelTrack.numRecHits());
  h_numRecHitsVsDz_.fill(passed, trackTruth.dz, simPixelTrack.numRecHits());
  h_numTOVsEtaPhi_.fill(passed, trackTruth.eta, trackTruth.phi);
  h_numTOVsEtaPt_.fill(passed, trackTruth.eta, trackTruth.pt);
  h_numTOVsPhiPt_.fill(passed, trackTruth.phi, trackTruth.pt);
}

// function that trys to find a valid Ntuplet for the given SimPixelTrack using the given geometry configuration
// (layer pairs, starting pairs, minimum number of hits) ignoring all cuts on doublets/connections and returns if it was able to find one
template <typename TrackerTraits>
bool SimPixelTrackAnalyzer<TrackerTraits>::configAllowsForValidNtuplet(SimPixelTrack const& simPixelTrack) const {
  // if the number of layers is less than the minimum requirement, don't even bother building anything...
  if (simPixelTrack.numLayers() < minNumDoubletsPerNtuplet_ + 1)
    return false;

  // initialize counter for the number of layers in the built Ntuplet
  int numLayers{0};
  // initialize bool to know if the building has started
  // (need to start at a valid starting pair)
  bool building{false};

  // get the layerId of the first RecHit of the TrackingParticle
  auto currentLayer = simPixelTrack.layerIds(0);

  // loop over the RecHits in order and try building an Ntuplet starting from the first valid starting pair
  for (int layerPairId{0}; auto nextLayer : simPixelTrack.layerIds()) {
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
void SimPixelTrackAnalyzer<TrackerTraits>::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  // get tracker topology
  trackerTopology_ = &iSetup.getData(topology_getToken_);
  // get SimPixelTracks
  SimPixelTrackCollection const& simPixelTrackCollection = iEvent.get(simPixelTracks_getToken_);

  // initialize a bunch of variables that we will use in the coming for loops
  int numSimDoublets, pass_numSimDoublets, layerPairId, layerPairIdIndex, numSkippedLayers;
  bool hasValidNeighbors, hasValidTripletNeighbors;

  // initialize the manager for keeping track of which cluster cuts are applied to the inidividual doublets
  simdoublets::ClusterSizeCutManager<TrackerTraits> clusterSizeCutManager;
  // initialize the structure holding the cut variables for an individual doublet
  simdoublets::CellCutVariables cellCutVariables;
  // initialize the structure holding the true TrackingParticle/RecoTrack parameters
  simdoublets::TrackTruth trackTruth{};

  // loop over SimDoublets (= loop over TrackingParticles/RecoTracks)
  for (auto const& simPixelTrack : simPixelTrackCollection) {
    if (inputIsRecoTracks_) {
      auto track = simPixelTrack.track();
      auto bs = simPixelTrack.beamSpotPosition();
      reco::TrackBase::Point beamSpotPoint(bs.x(), bs.y(), bs.z());
      trackTruth.dxy = track->dxy(beamSpotPoint);
      trackTruth.dz = track->dz(beamSpotPoint);
      trackTruth.phi = track->phi();
      trackTruth.pt = track->pt();
      trackTruth.eta = track->eta();
    } else {
      auto trackingParticle = simPixelTrack.trackingParticle();
      auto momentum = trackingParticle->momentum();
      auto vertex = trackingParticle->vertex();
      auto bs = simPixelTrack.beamSpotPosition();
      const math::XYZPoint beamSpotPoint(bs.x(), bs.y(), bs.z());
      const auto vertexTPwrtBS = vertex - beamSpotPoint;
      trackTruth.dxy = TrackingParticleIP::dxy(vertex, momentum, bs);
      trackTruth.dz = TrackingParticleIP::dz(vertex, momentum, bs);
      trackTruth.vertpos = std::sqrt(vertexTPwrtBS.perp2());
      trackTruth.phi = trackingParticle->phi();
      trackTruth.pt = trackingParticle->pt();
      trackTruth.eta = trackingParticle->eta();
      trackTruth.pdgId = trackingParticle->pdgId();

      // check if a valid Ntuplet is possible for the given TP and geometry ignoring any cuts and fill hists
      bool reconstructable = configAllowsForValidNtuplet(simPixelTrack);
      h_effConfigLimitVsEta_->Fill(trackTruth.eta, reconstructable);
      h_effConfigLimitVsPt_->Fill(trackTruth.pt, reconstructable);
    }

    // create the true RecHit doublets of the TrackingParticle
    auto& doublets = simPixelTrack.buildAndGetSimDoublets(trackerTopology_);
    // number of SimDoublets of the Tracking Particle
    numSimDoublets = doublets.size();
    // number of SimDoublets of the Tracking Particle passing all cuts
    pass_numSimDoublets = 0;

    // loop over those doublets
    for (auto& doublet : doublets) {
      // reset clusterSizeCutManager to "no cluster cuts applied"
      clusterSizeCutManager.reset();

      // calculate the cut variables for the given doublet
      cellCutVariables.calculateCutVariables(doublet, simPixelTrack);

      // first, get layer pair Id and exclude layer pairs that are not considered
      layerPairId = doublet.layerPairId();
      if (layerPairId2Index_.find(layerPairId) != layerPairId2Index_.end()) {
        // get the position of the layer pair in the vectors of histograms
        layerPairIdIndex = layerPairId2Index_.at(layerPairId);

        // function to check if a doublet has inner neighbors from a considered layer pair
        auto checkValidNeighbors = [&](SimPixelTrack::Doublet const& d) {
          return (d.numInnerNeighbors() > 0 &&
                  !(simPixelTrack.getSimDoublet(d.innerNeighborIndex(0)).isKilledByMissingLayerPair()));
        };

        // check if the SimDoublet's inner neighbors also are from a considered layer pair
        hasValidNeighbors = checkValidNeighbors(doublet);

        // check if the inner neighbors' neighbors also are from a considered layer pair
        hasValidTripletNeighbors =
            hasValidNeighbors && checkValidNeighbors(simPixelTrack.getSimDoublet(doublet.innerNeighborIndex(0)));

        // determine which cluster size cuts the doublet is subject to
        clusterSizeCutManager.setSubjectsToCuts(doublet);

        // apply the cuts for doublet building according to the set cut values
        applyCuts(doublet,
                  simPixelTrack,
                  hasValidNeighbors,
                  hasValidTripletNeighbors,
                  layerPairIdIndex,
                  cellCutVariables,
                  clusterSizeCutManager);

        // -------------------------------------------------------------------------
        //  cut histograms for SimDoublets (CAParameters folder)
        // -------------------------------------------------------------------------
        fillCutHistograms(doublet,
                          hasValidNeighbors,
                          hasValidTripletNeighbors,
                          layerPairIdIndex,
                          cellCutVariables,
                          clusterSizeCutManager,
                          trackTruth);

      } else {
        // if not considered set the doublet as killed
        doublet.setKilledByMissingLayerPair();
      }

      // ---------------------------------------------------------------------------
      //  general plots related to SimDoublets (SimDoublets folder)
      // ---------------------------------------------------------------------------
      fillSimDoubletHistograms(doublet, trackTruth);

      // if the doublet passes all cuts, increment number of SimDoublets passing all cuts
      if (doublet.isAlive())
        pass_numSimDoublets++;
    }  // end loop over those doublets

    // build the SimNtuplets based on the SimDoublets
    simPixelTrack.buildSimNtuplets(startingPairs_, minNumDoubletsPerNtuplet_);

    // -----------------------------------------------------------------------------
    //  plots related to SimNtuplets (SimNtuplets folder)
    // -----------------------------------------------------------------------------
    // set the number of skipped layers by default to -1
    numSkippedLayers = -1;

    if (simPixelTrack.hasSimNtuplet()) {
      // get number of skipped layers from longest SimNtuplet and fill histos
      numSkippedLayers = simPixelTrack.longestSimNtuplet().numSkippedLayers();
      fillSimNtupletHistograms(simPixelTrack, trackTruth);
    }

#ifdef LOSTNTUPLETS_PRINTOUTS
    if (!simPixelTrack.hasAliveSimNtuplet()) {
      printf(
          "\n------------------------------------------\n Lost Particle: pdgId %d, eta %f, pT %f, dz %f, dxy "
          "%f\n------------------------------------------\n",
          trackTruth.pdgId,
          trackTruth.eta,
          trackTruth.pt,
          trackTruth.dz,
          trackTruth.dxy);
      for (auto const& doublet : doublets) {
        printf(" doublet (%d, %d) is %s\n",
               doublet.innerLayerId(),
               doublet.outerLayerId(),
               doublet.isAlive() ? "alive" : "killed");
        for (auto const& neighbor : doublet.innerNeighborsView()) {
          printf("   - connection to %ld is %s \n", neighbor.index(), neighbor.isAlive() ? "alive" : "killed");
        }
      }
    }
#endif

    // -----------------------------------------------------------------------------
    //  general plots related to TrackingParticles (general folder)
    // -----------------------------------------------------------------------------
    fillGeneralHistograms(simPixelTrack, trackTruth, pass_numSimDoublets, numSimDoublets, numSkippedLayers);

    // clear SimDoublets and SimNtuplets of the TrackingParticle
    simPixelTrack.clearMutables();
  }  // end loop over SimPixelTrack (= loop over TrackingParticles/RecoTracks)
}

// booking the histograms
template <typename TrackerTraits>
void SimPixelTrackAnalyzer<TrackerTraits>::bookHistograms(DQMStore::IBooker& ibook,
                                                          edm::Run const& run,
                                                          edm::EventSetup const& iSetup) {
  // set some common parameters
  int pTNBins = 200;
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
  int vertPosNBins = 40;
  double vertPosmin = log10(0.01);
  double vertPosmax = log10(100);
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

  h_numRecHitsVsDxy_.book2D(
      ibook, "numRecHits_vs_dxy", "Number of Tracks", "dxy [cm]", "Number of RecHits", 200, -5, 5, 26, -1.5, 24.5);

  h_numRecHitsVsDz_.book2D(
      ibook, "numRecHits_vs_dz", "Number of Tracks", "dz [cm]", "Number of RecHits", 20, -20, 20, 26, -1.5, 24.5);

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
  h_numTOVsVertpos_.book1DLogX(ibook,
                               "num_vs_vertpos",
                               "Number of TrackingParticles",
                               "True radial vertex position r_{vertex} [cm]",
                               "Number of TrackingParticles",
                               vertPosNBins,
                               vertPosmin,
                               vertPosmax);
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
      ibook, "num_vs_dxy", "Number of TrackingParticles", "dxy [cm]", "Number of TrackingParticles", 200, -5, 5);
  h_numTOVsDz_.book1D(
      ibook, "num_vs_dz", "Number of TrackingParticles", "dz [cm]", "Number of TrackingParticles", 20, -20, 20);
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
    h_numRecHitsVsChi2_.book2D(
        ibook, "numRecHits_vs_chi2", "Number of Tracks", "#chi^2 / ndof", "Number of RecHits", 70, 0, 7, 26, -1.5, 24.5);
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
    h_num_vs_vertpos_.book1DLogX(ibook,
                                 "num_vs_vertpos",
                                 "Number of " + doublet + "s",
                                 "True radial vertex position r_{vertex} [cm]",
                                 "Number of " + doublet + "s",
                                 vertPosNBins,
                                 vertPosmin,
                                 vertPosmax);
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

    // histogram for z0cutoff  (z0)
    hVector_z0_.at(layerPairIdIndex)
        .book1D(ibook,
                "z0",
                "z_{0} of " + doublet + "s " + layerTitle,
                "Longitudinal impact parameter z_{0} [cm]",
                "Number of " + doublet + "s",
                51,
                -1,
                50);

    // histograms for ptcut  (ptCut)
    hVector_curvatureR_.at(layerPairIdIndex)
        .book1DLogX(ibook,
                    "curvatureR",
                    "Curvature from 3 points of beamspot + RecHits of " + doublet + "s " + layerTitle,
                    "Curvature radius [cm]",
                    "Number of " + doublet + "s",
                    100,
                    2,
                    4);
    hVector_pTFromR_.at(layerPairIdIndex)
        .book1DLogX(ibook,
                    "pTFromR",
                    "Transverse momentum from curvature " + layerTitle,
                    "Transverse momentum p_{T} [GeV]",
                    "Number of " + doublet + "s",
                    pTNBins,
                    pTmin,
                    pTmax);

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
                200,
                -M_PI / 16,
                M_PI / 16);
    hVector_idphi_.at(layerPairIdIndex)
        .book1D(ibook,
                "idphi",
                "idphi of RecHit pair " + layerTitle,
                "Absolute int d#phi between outer and inner RecHit",
                "Number of " + doublet + "s",
                100,
                0,
                2000);

    // histogram for inner (r/z) cuts
    if (cellCuts_.isBarrel_[std::stoi(innerLayerName)]) {
      hVector_inner_.at(layerPairIdIndex)
          .book1D(ibook,
                  "inner",
                  "z of the inner RecHit " + layerTitle,
                  "z of inner RecHit [cm]",
                  "Number of " + doublet + "s",
                  600,
                  -300,
                  300);
    } else {
      hVector_inner_.at(layerPairIdIndex)
          .book1D(ibook,
                  "inner",
                  "r of the inner RecHit " + layerTitle,
                  "r of inner RecHit [cm]",
                  "Number of " + doublet + "s",
                  600,
                  0,
                  60);
    }

    // histogram for outer (r/z) cut
    if (cellCuts_.isBarrel_[std::stoi(outerLayerName)]) {
      hVector_outer_.at(layerPairIdIndex)
          .book1D(ibook,
                  "outer",
                  "z of the outer RecHit " + layerTitle,
                  "z of outer RecHit [cm]",
                  "Number of " + doublet + "s",
                  600,
                  -300,
                  300);
    } else {
      hVector_outer_.at(layerPairIdIndex)
          .book1D(ibook,
                  "outer",
                  "r of the outer RecHit " + layerTitle,
                  "r of outer RecHit [cm]",
                  "Number of " + doublet + "s",
                  600,
                  0,
                  60);
    }

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

  // histogram for dCurvCut (x-y alignement of triplet connections)
  h_dCurvCut_.book1D(ibook,
                     "dCurvCut",
                     "Curvature difference of a pair of neighboring triplets",
                     "Absolute curvature difference [1/cm]",
                     "Number of triplet connections",
                     50,
                     0,
                     0.02);

  // histogram for curvRatioCut (x-y alignement of triplet connections)
  h_curvRatioCut_.book1D(ibook,
                         "curvRatioCut",
                         "Curvature ratio of a pair of neighboring triplets",
                         "Ratio of curvatures",
                         "Number of triplet connections",
                         200,
                         -3,
                         3);

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
    h_bestNtuplet_.bookHistograms(ibook, "Most alive SimNtuplet per TrackingParticle");
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
    h_longNtuplet_.bookHistograms(ibook, "Longest SimNtuplets per TrackingParticle");
  }
}

// -------------------------------------------------------------------------------------------------------------
// fillDescriptions
// -------------------------------------------------------------------------------------------------------------

// dummy default fillDescription
template <typename TrackerTraits>
void SimPixelTrackAnalyzer<TrackerTraits>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.addWithDefaultLabel(desc);
}

// fillDescription for Phase 1
template <>
void SimPixelTrackAnalyzer<pixelTopology::Phase1>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  int nPairs = pixelTopology::Phase1::nPairsForQuadruplets;

  edm::ParameterSetDescription desc;
  simdoublets::fillDescriptionsCommon<pixelTopology::Phase1>(desc);

  // input source for SimPixelTrack
  desc.add<edm::InputTag>("simPixelTrackSrc", edm::InputTag("simPixelTrackProducerPhase1"));

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
  geometryParams.add<std::vector<int>>("isBarrel", std::vector<int>(phase1PixelTopology::numberOfLayers, 1))
      ->setComment(
          "Bool vector with one element per layer that defines if the min/max cut for doublet building is applied in "
          "z (isBarrel->true) or r (isBarrel->false).");
  // cells params
  geometryParams
      .add<std::vector<unsigned int>>(
          "pairGraph",
          std::vector<unsigned int>(std::begin(phase1PixelTopology::layerPairs),
                                    std::begin(phase1PixelTopology::layerPairs) + (nPairs * 2)))
      ->setComment("CA graph");
  geometryParams
      .add<std::vector<int>>(
          "phiCuts",
          std::vector<int>(std::begin(phase1PixelTopology::phicuts), std::begin(phase1PixelTopology::phicuts) + nPairs))
      ->setComment("Cuts in phi for cells");
  geometryParams.add<std::vector<double>>("ptCuts", std::vector<double>(nPairs, 0.5))
      ->setComment("Minimum tranverse momentum");
  geometryParams
      .add<std::vector<double>>(
          "minInner",
          std::vector<double>(std::begin(phase1PixelTopology::minz), std::begin(phase1PixelTopology::minz) + nPairs))
      ->setComment("Cuts on inner hit's z (for barrel) or r (for endcap) for cells (min value)");
  geometryParams
      .add<std::vector<double>>(
          "maxInner",
          std::vector<double>(std::begin(phase1PixelTopology::maxz), std::begin(phase1PixelTopology::maxz) + nPairs))
      ->setComment("Cuts on inner hit's z (for barrel) or r (for endcap) for cells (max value)");
  geometryParams.add<std::vector<double>>("minOuter", std::vector<double>(nPairs, -10000))
      ->setComment("Cuts on outer hit's z (for barrel) or r (for endcap) for cells (min value)");
  geometryParams.add<std::vector<double>>("maxOuter", std::vector<double>(nPairs, 10000))
      ->setComment("Cuts on outer hit's z (for barrel) or r (for endcap) for cells (max value)");
  geometryParams
      .add<std::vector<double>>(
          "maxDR",
          std::vector<double>(std::begin(phase1PixelTopology::maxr), std::begin(phase1PixelTopology::maxr) + nPairs))
      ->setComment("Cuts in max dr for cells");
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
void SimPixelTrackAnalyzer<pixelTopology::Phase2>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  int nPairs = pixelTopology::Phase2::nPairs;

  edm::ParameterSetDescription desc;
  simdoublets::fillDescriptionsCommon<pixelTopology::Phase2>(desc);

  // input source for SimPixelTrack
  desc.add<edm::InputTag>("simPixelTrackSrc", edm::InputTag("simPixelTrackProducerPhase2"));

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
  geometryParams.add<std::vector<int>>("isBarrel", std::vector<int>(phase2PixelTopology::numberOfLayers, 1))
      ->setComment(
          "Bool vector with one element per layer that defines if the min/max cut for doublet building is applied in "
          "z (isBarrel->true) or r (isBarrel->false).");
  geometryParams
      .add<std::vector<unsigned int>>("startingPairs",
                                      {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32})
      ->setComment(
          "The list of the ids of pairs from which the CA ntuplets building may start.");  //TODO could be parsed via an expression
  // cells params
  geometryParams
      .add<std::vector<unsigned int>>(
          "pairGraph",
          std::vector<unsigned int>(std::begin(phase2PixelTopology::layerPairs),
                                    std::begin(phase2PixelTopology::layerPairs) + (nPairs * 2)))
      ->setComment("CA graph");
  geometryParams
      .add<std::vector<int>>(
          "phiCuts",
          std::vector<int>(std::begin(phase2PixelTopology::phicuts), std::begin(phase2PixelTopology::phicuts) + nPairs))
      ->setComment("Cuts in phi for cells");
  geometryParams.add<std::vector<double>>("ptCuts", std::vector<double>(nPairs, 0.85))
      ->setComment("Minimum tranverse momentum");
  geometryParams
      .add<std::vector<double>>(
          "minInner",
          std::vector<double>(std::begin(phase2PixelTopology::minz), std::begin(phase2PixelTopology::minz) + nPairs))
      ->setComment("Cuts on inner hit's z (for barrel) or r (for endcap) for cells (min value)");
  geometryParams
      .add<std::vector<double>>(
          "maxInner",
          std::vector<double>(std::begin(phase2PixelTopology::maxz), std::begin(phase2PixelTopology::maxz) + nPairs))
      ->setComment("Cuts on inner hit's z (for barrel) or r (for endcap) for cells (max value)");
  geometryParams.add<std::vector<double>>("minOuter", std::vector<double>(nPairs, -10000))
      ->setComment("Cuts on outer hit's z (for barrel) or r (for endcap) for cells (min value)");
  geometryParams.add<std::vector<double>>("maxOuter", std::vector<double>(nPairs, 10000))
      ->setComment("Cuts on outer hit's z (for barrel) or r (for endcap) for cells (max value)");
  geometryParams
      .add<std::vector<double>>(
          "maxDR",
          std::vector<double>(std::begin(phase2PixelTopology::maxr), std::begin(phase2PixelTopology::maxr) + nPairs))
      ->setComment("Cuts in max dr for cells");
  geometryParams.add<std::vector<double>>("minDZ", std::vector<double>(nPairs, -10000))
      ->setComment("Cuts in max dz for cells");
  geometryParams.add<std::vector<double>>("maxDZ", std::vector<double>(nPairs, 10000))
      ->setComment("Cuts in min dz for cells");

  desc.add<edm::ParameterSetDescription>("geometry", geometryParams)
      ->setComment("Layer pair graph, layer-dependent cut values.");

  descriptions.addWithDefaultLabel(desc);
}

// define two plugins for Phase 1 and 2
using SimPixelTrackAnalyzerPhase1 = SimPixelTrackAnalyzer<pixelTopology::Phase1>;
using SimPixelTrackAnalyzerPhase2 = SimPixelTrackAnalyzer<pixelTopology::Phase2>;

// define this as a plug-in
DEFINE_FWK_MODULE(SimPixelTrackAnalyzerPhase1);
DEFINE_FWK_MODULE(SimPixelTrackAnalyzerPhase2);
