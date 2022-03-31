#ifndef RecoTracker_TkTrackingRegions_L1MuonSeededTrackingRegionsProducer_h
#define RecoTracker_TkTrackingRegions_L1MuonSeededTrackingRegionsProducer_h

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/Record/interface/TrackerMultipleScatteringRecord.h"
#include "RecoTracker/TkMSParametrization/interface/MultipleScatteringParametrisationMaker.h"
#include "CLHEP/Vector/ThreeVector.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

/** class L1MuonSeededTrackingRegionsProducer
 *
 * eta-phi TrackingRegions producer in directions defined by L1 muon objects of interest
 * from a collection defined by the "input" parameter.
 *
 * Four operational modes are supported ("mode" parameter):
 *
 *   BeamSpotFixed:
 *     origin is defined by the beam spot
 *     z-half-length is defined by a fixed zErrorBeamSpot parameter
 *   BeamSpotSigma:
 *     origin is defined by the beam spot
 *     z-half-length is defined by nSigmaZBeamSpot * beamSpot.sigmaZ
 *   VerticesFixed:
 *     origins are defined by vertices from VertexCollection (use maximum MaxNVertices of them)
 *     z-half-length is defined by a fixed zErrorVetex parameter
 *   VerticesSigma:
 *     origins are defined by vertices from VertexCollection (use maximum MaxNVertices of them)
 *     z-half-length is defined by nSigmaZVertex * vetex.zError
 *
 *   If, while using one of the "Vertices" modes, there's no vertices in an event, we fall back into
 *   either BeamSpotSigma or BeamSpotFixed mode, depending on the positiveness of nSigmaZBeamSpot.
 *
 *   \author M. Oh.
 *       based on RecoTracker/TkTrackingRegions/plugins/CandidateSeededTrackingRegionsProducer.h
 */
class L1MuonSeededTrackingRegionsProducer : public TrackingRegionProducer {
public:
  typedef enum { BEAM_SPOT_FIXED, BEAM_SPOT_SIGMA, VERTICES_FIXED, VERTICES_SIGMA } Mode;

  explicit L1MuonSeededTrackingRegionsProducer(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iC)
      : token_field(iC.esConsumes()),
        l1MinPt_(iConfig.getParameter<double>("L1MinPt")),
        l1MaxEta_(iConfig.getParameter<double>("L1MaxEta")),
        l1MinQuality_(iConfig.getParameter<unsigned int>("L1MinQuality")),
        minPtBarrel_(iConfig.getParameter<double>("SetMinPtBarrelTo")),
        minPtEndcap_(iConfig.getParameter<double>("SetMinPtEndcapTo")),
        centralBxOnly_(iConfig.getParameter<bool>("CentralBxOnly")),
        propagatorName_(iConfig.getParameter<std::string>("Propagator")) {
    edm::ParameterSet regPSet = iConfig.getParameter<edm::ParameterSet>("RegionPSet");

    // operation mode
    std::string modeString = regPSet.getParameter<std::string>("mode");
    if (modeString == "BeamSpotFixed")
      m_mode = BEAM_SPOT_FIXED;
    else if (modeString == "BeamSpotSigma")
      m_mode = BEAM_SPOT_SIGMA;
    else if (modeString == "VerticesFixed")
      m_mode = VERTICES_FIXED;
    else if (modeString == "VerticesSigma")
      m_mode = VERTICES_SIGMA;
    else
      edm::LogError("L1MuonSeededTrackingRegionsProducer") << "Unknown mode string: " << modeString;

    // basic inputs
    token_input = iC.consumes<l1t::MuonBxCollection>(regPSet.getParameter<edm::InputTag>("input"));
    m_maxNRegions = regPSet.getParameter<int>("maxNRegions");
    token_beamSpot = iC.consumes<reco::BeamSpot>(regPSet.getParameter<edm::InputTag>("beamSpot"));
    m_maxNVertices = 1;
    if (m_mode == VERTICES_FIXED || m_mode == VERTICES_SIGMA) {
      token_vertex = iC.consumes<reco::VertexCollection>(regPSet.getParameter<edm::InputTag>("vertexCollection"));
      m_maxNVertices = regPSet.getParameter<int>("maxNVertices");
    }

    // RectangularEtaPhiTrackingRegion parameters:
    m_ptMin = regPSet.getParameter<double>("ptMin");
    m_originRadius = regPSet.getParameter<double>("originRadius");
    m_zErrorBeamSpot = regPSet.getParameter<double>("zErrorBeamSpot");
    m_ptRanges = regPSet.getParameter<std::vector<double>>("ptRanges");
    if (m_ptRanges.size() < 2) {
      edm::LogError("L1MuonSeededTrackingRegionsProducer") << "Size of ptRanges does not be less than 2" << std::endl;
    }
    m_deltaEtas = regPSet.getParameter<std::vector<double>>("deltaEtas");
    if (m_deltaEtas.size() != m_ptRanges.size() - 1) {
      edm::LogError("L1MuonSeededTrackingRegionsProducer")
          << "Size of deltaEtas does not match number of pt bins." << std::endl;
    }
    m_deltaPhis = regPSet.getParameter<std::vector<double>>("deltaPhis");
    if (m_deltaPhis.size() != m_ptRanges.size() - 1) {
      edm::LogError("L1MuonSeededTrackingRegionsProducer")
          << "Size of deltaPhis does not match number of pt bins." << std::endl;
    }

    m_precise = regPSet.getParameter<bool>("precise");
    m_whereToUseMeasurementTracker = RectangularEtaPhiTrackingRegion::stringToUseMeasurementTracker(
        regPSet.getParameter<std::string>("whereToUseMeasurementTracker"));
    if (m_whereToUseMeasurementTracker != RectangularEtaPhiTrackingRegion::UseMeasurementTracker::kNever) {
      token_measurementTracker =
          iC.consumes<MeasurementTrackerEvent>(regPSet.getParameter<edm::InputTag>("measurementTrackerName"));
    }

    m_searchOpt = regPSet.getParameter<bool>("searchOpt");

    // mode-dependent z-halflength of tracking regions
    if (m_mode == VERTICES_SIGMA)
      m_nSigmaZVertex = regPSet.getParameter<double>("nSigmaZVertex");
    if (m_mode == VERTICES_FIXED)
      m_zErrorVetex = regPSet.getParameter<double>("zErrorVetex");
    m_nSigmaZBeamSpot = -1.;
    if (m_mode == BEAM_SPOT_SIGMA) {
      m_nSigmaZBeamSpot = regPSet.getParameter<double>("nSigmaZBeamSpot");
      if (m_nSigmaZBeamSpot < 0.)
        edm::LogError("L1MuonSeededTrackingRegionsProducer")
            << "nSigmaZBeamSpot must be positive for BeamSpotSigma mode!";
    }
    if (m_precise) {
      token_msmaker = iC.esConsumes();
    }

    // MuonServiceProxy
    edm::ParameterSet serviceParameters = iConfig.getParameter<edm::ParameterSet>("ServiceParameters");
    service_ = std::make_unique<MuonServiceProxy>(serviceParameters, std::move(iC));
  }

  ~L1MuonSeededTrackingRegionsProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    // L1 muon selection parameters
    desc.add<std::string>("Propagator", "");
    desc.add<double>("L1MinPt", -1.);
    desc.add<double>("L1MaxEta", 5.0);
    desc.add<unsigned int>("L1MinQuality", 0);
    desc.add<double>("SetMinPtBarrelTo", 3.5);
    desc.add<double>("SetMinPtEndcapTo", 1.0);
    desc.add<bool>("CentralBxOnly", true);

    // Tracking region parameters
    edm::ParameterSetDescription descRegion;
    descRegion.add<std::string>("mode", "BeamSpotSigma");
    descRegion.add<edm::InputTag>("input", edm::InputTag(""));
    descRegion.add<int>("maxNRegions", 10);
    descRegion.add<edm::InputTag>("beamSpot", edm::InputTag("hltOnlineBeamSpot"));
    descRegion.add<edm::InputTag>("vertexCollection", edm::InputTag("notUsed"));
    descRegion.add<int>("maxNVertices", 1);
    descRegion.add<double>("ptMin", 0.0);
    descRegion.add<double>("originRadius", 0.2);
    descRegion.add<double>("zErrorBeamSpot", 24.2);
    descRegion.add<std::vector<double>>("ptRanges", {0., 1.e9});
    descRegion.add<std::vector<double>>("deltaEtas", {0.35});
    descRegion.add<std::vector<double>>("deltaPhis", {0.2});
    descRegion.add<bool>("precise", true);
    descRegion.add<double>("nSigmaZVertex", 3.);
    descRegion.add<double>("zErrorVetex", 0.2);
    descRegion.add<double>("nSigmaZBeamSpot", 4.);
    descRegion.add<std::string>("whereToUseMeasurementTracker", "Never");
    descRegion.add<edm::InputTag>("measurementTrackerName", edm::InputTag(""));
    descRegion.add<bool>("searchOpt", false);
    desc.add<edm::ParameterSetDescription>("RegionPSet", descRegion);

    // MuonServiceProxy for the propagation
    edm::ParameterSetDescription psd0;
    psd0.addUntracked<std::vector<std::string>>("Propagators", {"SteppingHelixPropagatorAny"});
    psd0.add<bool>("RPCLayers", false);
    psd0.addUntracked<bool>("UseMuonNavigation", false);
    desc.add<edm::ParameterSetDescription>("ServiceParameters", psd0);

    descriptions.add("hltIterL3MuonPixelTracksTrackingRegions", desc);
  }

  std::vector<std::unique_ptr<TrackingRegion>> regions(const edm::Event& iEvent,
                                                       const edm::EventSetup& iSetup) const override {
    std::vector<std::unique_ptr<TrackingRegion>> result;

    // pick up the candidate objects of interest
    edm::Handle<l1t::MuonBxCollection> muColl;
    iEvent.getByToken(token_input, muColl);
    if (muColl->size() == 0)
      return result;

    // always need the beam spot (as a fall back strategy for vertex modes)
    edm::Handle<reco::BeamSpot> bs;
    iEvent.getByToken(token_beamSpot, bs);
    if (!bs.isValid())
      return result;

    // this is a default origin for all modes
    GlobalPoint default_origin(bs->x0(), bs->y0(), bs->z0());

    // vector of origin & halfLength pairs:
    std::vector<std::pair<GlobalPoint, float>> origins;

    // fill the origins and halfLengths depending on the mode
    if (m_mode == BEAM_SPOT_FIXED || m_mode == BEAM_SPOT_SIGMA) {
      origins.push_back(std::make_pair(
          default_origin, (m_mode == BEAM_SPOT_FIXED) ? m_zErrorBeamSpot : m_nSigmaZBeamSpot * bs->sigmaZ()));
    } else if (m_mode == VERTICES_FIXED || m_mode == VERTICES_SIGMA) {
      edm::Handle<reco::VertexCollection> vertices;
      iEvent.getByToken(token_vertex, vertices);
      int n_vert = 0;
      for (reco::VertexCollection::const_iterator v = vertices->begin();
           v != vertices->end() && n_vert < m_maxNVertices;
           ++v) {
        if (v->isFake() || !v->isValid())
          continue;

        origins.push_back(std::make_pair(GlobalPoint(v->x(), v->y(), v->z()),
                                         (m_mode == VERTICES_FIXED) ? m_zErrorVetex : m_nSigmaZVertex * v->zError()));
        ++n_vert;
      }
      // no-vertex fall-back case:
      if (origins.empty()) {
        origins.push_back(std::make_pair(
            default_origin, (m_nSigmaZBeamSpot > 0.) ? m_nSigmaZBeamSpot * bs->z0Error() : m_zErrorBeamSpot));
      }
    }

    const MeasurementTrackerEvent* measurementTracker = nullptr;
    if (!token_measurementTracker.isUninitialized()) {
      measurementTracker = &iEvent.get(token_measurementTracker);
    }

    const auto& field = iSetup.getData(token_field);
    const MultipleScatteringParametrisationMaker* msmaker = nullptr;
    if (m_precise) {
      msmaker = &iSetup.getData(token_msmaker);
    }

    // create tracking regions (maximum MaxNRegions of them) in directions of the
    // objects of interest (we expect that the collection was sorted in decreasing pt order)
    int n_regions = 0;
    for (int ibx = muColl->getFirstBX(); ibx <= muColl->getLastBX() && n_regions < m_maxNRegions; ++ibx) {
      if (centralBxOnly_ && (ibx != 0))
        continue;

      for (auto it = muColl->begin(ibx); it != muColl->end(ibx) && n_regions < m_maxNRegions; it++) {
        unsigned int quality = it->hwQual();
        if (quality <= l1MinQuality_)
          continue;

        float pt = it->pt();
        float eta = it->eta();
        if (pt < l1MinPt_ || std::abs(eta) > l1MaxEta_)
          continue;

        float theta = 2 * atan(exp(-eta));
        float phi = it->phi();

        int valid_charge = it->hwChargeValid();
        int charge = it->charge();
        if (!valid_charge)
          charge = 0;

        int link = l1MuonTF_link_EMTFP_i_ + (int)(it->tfMuonIndex() / 3.);
        bool barrel = true;
        if ((link >= l1MuonTF_link_EMTFP_i_ && link <= l1MuonTF_link_EMTFP_f_) ||
            (link >= l1MuonTF_link_EMTFN_i_ && link <= l1MuonTF_link_EMTFN_f_))
          barrel = false;

        // propagate L1 FTS to BS
        service_->update(iSetup);
        const DetLayer* detLayer = nullptr;
        float radius = 0.;

        CLHEP::Hep3Vector vec(0., 1., 0.);
        vec.setTheta(theta);
        vec.setPhi(phi);

        DetId theid;
        // Get the det layer on which the state should be put
        if (barrel) {
          // MB2
          theid = DTChamberId(0, 2, 0);
          detLayer = service_->detLayerGeometry()->idToLayer(theid);

          const BoundSurface* sur = &(detLayer->surface());
          const BoundCylinder* bc = dynamic_cast<const BoundCylinder*>(sur);

          radius = std::abs(bc->radius() / sin(theta));

          if (pt < minPtBarrel_)
            pt = minPtBarrel_;
        } else {
          // ME2
          theid = theta < M_PI / 2. ? CSCDetId(1, 2, 0, 0, 0) : CSCDetId(2, 2, 0, 0, 0);

          detLayer = service_->detLayerGeometry()->idToLayer(theid);

          radius = std::abs(detLayer->position().z() / cos(theta));

          if (pt < minPtEndcap_)
            pt = minPtEndcap_;
        }
        vec.setMag(radius);
        GlobalPoint pos(vec.x(), vec.y(), vec.z());
        GlobalVector mom(pt * cos(phi), pt * sin(phi), pt * cos(theta) / sin(theta));
        GlobalTrajectoryParameters param(pos, mom, charge, &*service_->magneticField());

        AlgebraicSymMatrix55 mat;
        mat[0][0] = (sigma_qbpt_barrel_ / pt) * (sigma_qbpt_barrel_ / pt);  // sigma^2(charge/abs_momentum)
        if (!barrel)
          mat[0][0] = (sigma_qbpt_endcap_ / pt) * (sigma_qbpt_endcap_ / pt);

        //Assign q/pt = 0 +- 1/pt if charge has been declared invalid
        if (!valid_charge)
          mat[0][0] = (sigma_qbpt_invalid_charge_ / pt) * (sigma_qbpt_invalid_charge_ / pt);

        mat[1][1] = sigma_lambda_ * sigma_lambda_;  // sigma^2(lambda)
        mat[2][2] = sigma_phi_ * sigma_phi_;        // sigma^2(phi)
        mat[3][3] = sigma_x_ * sigma_x_;            // sigma^2(x_transverse))
        mat[4][4] = sigma_y_ * sigma_y_;            // sigma^2(y_transverse))

        CurvilinearTrajectoryError error(mat);

        const FreeTrajectoryState state(param, error);

        FreeTrajectoryState state_bs = service_->propagator(propagatorName_)->propagate(state, *bs.product());

        GlobalVector direction(state_bs.momentum().x(), state_bs.momentum().y(), state_bs.momentum().z());

        // set deltaEta and deltaPhi from L1 muon pt
        auto deltaEta = m_deltaEtas.at(0);
        auto deltaPhi = m_deltaPhis.at(0);
        if (it->pt() < m_ptRanges.back()) {
          auto lowEdge = std::upper_bound(m_ptRanges.begin(), m_ptRanges.end(), it->pt());
          deltaEta = m_deltaEtas.at(lowEdge - m_ptRanges.begin() - 1);
          deltaPhi = m_deltaPhis.at(lowEdge - m_ptRanges.begin() - 1);
        }

        for (size_t j = 0; j < origins.size() && n_regions < m_maxNRegions; ++j) {
          result.push_back(std::make_unique<RectangularEtaPhiTrackingRegion>(direction,
                                                                             origins[j].first,
                                                                             m_ptMin,
                                                                             m_originRadius,
                                                                             origins[j].second,
                                                                             deltaEta,
                                                                             deltaPhi,
                                                                             field,
                                                                             msmaker,
                                                                             m_precise,
                                                                             m_whereToUseMeasurementTracker,
                                                                             measurementTracker,
                                                                             m_searchOpt));
          ++n_regions;
        }
      }
    }
    edm::LogInfo("L1MuonSeededTrackingRegionsProducer") << "produced " << n_regions << " regions";

    return result;
  }

private:
  Mode m_mode;

  int m_maxNRegions;
  edm::EDGetTokenT<reco::VertexCollection> token_vertex;
  edm::EDGetTokenT<reco::BeamSpot> token_beamSpot;
  edm::EDGetTokenT<l1t::MuonBxCollection> token_input;
  int m_maxNVertices;

  float m_ptMin;
  float m_originRadius;
  float m_zErrorBeamSpot;
  std::vector<double> m_ptRanges;
  std::vector<double> m_deltaEtas;
  std::vector<double> m_deltaPhis;
  bool m_precise;
  edm::EDGetTokenT<MeasurementTrackerEvent> token_measurementTracker;
  RectangularEtaPhiTrackingRegion::UseMeasurementTracker m_whereToUseMeasurementTracker;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> token_field;
  edm::ESGetToken<MultipleScatteringParametrisationMaker, TrackerMultipleScatteringRecord> token_msmaker;
  bool m_searchOpt;

  float m_nSigmaZVertex;
  float m_zErrorVetex;
  float m_nSigmaZBeamSpot;

  const double l1MinPt_;
  const double l1MaxEta_;
  const unsigned l1MinQuality_;
  const double minPtBarrel_;
  const double minPtEndcap_;
  const bool centralBxOnly_;

  const std::string propagatorName_;
  std::unique_ptr<MuonServiceProxy> service_;

  // link number indices of the optical fibres that connect the uGMT with the track finders
  // EMTF+ : 36-41, OMTF+ : 42-47, BMTF : 48-59, OMTF- : 60-65, EMTF- : 66-71
  static constexpr int l1MuonTF_link_EMTFP_i_{36};
  static constexpr int l1MuonTF_link_EMTFP_f_{41};
  static constexpr int l1MuonTF_link_EMTFN_i_{66};
  static constexpr int l1MuonTF_link_EMTFN_f_{71};

  // fixed error matrix parameters for L1 muon FTS
  static constexpr double sigma_qbpt_barrel_{0.25};
  static constexpr double sigma_qbpt_endcap_{0.4};
  static constexpr double sigma_qbpt_invalid_charge_{1.0};
  static constexpr double sigma_lambda_{0.05};
  static constexpr double sigma_phi_{0.2};
  static constexpr double sigma_x_{20.0};
  static constexpr double sigma_y_{20.0};
};

#endif
