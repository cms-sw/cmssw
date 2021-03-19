// -*- C++ -*-
//
// Package:    trackJet/JetCoreMCtruthSeedGenerator
// Class:      JetCoreMCtruthSeedGenerator
//
/**\class JetCoreMCtruthSeedGenerator JetCoreMCtruthSeedGenerator.cc trackJet/JetCoreMCtruthSeedGenerator/plugins/JetCoreMCtruthSeedGenerator.cc
 Description: [one line class summary]
 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Valerio Bertacchi
//         Created:  Mon, 18 Dec 2017 16:35:04 GMT
//
//

// system include files

#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"

#include "TrackingTools/GeomPropagators/interface/StraightLinePlaneCrossing.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"

#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// class declaration
//

class JetCoreMCtruthSeedGenerator : public edm::one::EDProducer<edm::one::SharedResources> {
public:
  explicit JetCoreMCtruthSeedGenerator(const edm::ParameterSet&);
  ~JetCoreMCtruthSeedGenerator() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  double jetPt_;
  double jetEta_;
  double pitchX_ = 0.01;              //100 um (pixel pitch in X)
  double pitchY_ = 0.015;             //150 um (pixel pitch in Y)
  static constexpr int jetDimX = 30;  //pixel dimension of NN window on layer2
  static constexpr int jetDimY = 30;  //pixel dimension of NN window on layer2
  bool inclusiveConeSeed_ =
      true;  //true= fill tracks in a cone of deltaR_, false=fill tracks which have SimHit on globDet

private:
  void beginJob() override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------
  std::string propagatorName_;
  edm::ESHandle<MagneticField> magfield_;
  edm::ESHandle<GlobalTrackingGeometry> geometry_;
  edm::ESHandle<Propagator> propagator_;

  edm::EDGetTokenT<std::vector<reco::Vertex>> vertices_;
  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster>> pixelClusters_;
  edm::Handle<edmNew::DetSetVector<SiPixelCluster>> inputPixelClusters_;
  edm::EDGetTokenT<edm::View<reco::Candidate>> cores_;
  edm::EDGetTokenT<std::vector<SimTrack>> simtracksToken_;
  edm::EDGetTokenT<std::vector<SimVertex>> simvertexToken_;
  edm::EDGetTokenT<std::vector<PSimHit>> PSimHitToken_;
  edm::Handle<std::vector<PSimHit>> simhits_;

  double ptMin_;
  double deltaR_;
  double chargeFracMin_;
  double centralMIPCharge_;
  std::string pixelCPE_;

  std::pair<bool, Basic3DVector<float>> findIntersection(const GlobalVector&,
                                                         const reco::Candidate::Point&,
                                                         const GeomDet*);

  const GeomDet* DetectorSelector(int,
                                  const reco::Candidate&,
                                  GlobalVector,
                                  const reco::Vertex&,
                                  const TrackerTopology* const,
                                  const edmNew::DetSetVector<SiPixelCluster>&);

  std::vector<GlobalVector> splittedClusterDirections(
      const reco::Candidate&,
      const TrackerTopology* const,
      const PixelClusterParameterEstimator*,
      const reco::Vertex&,
      int,
      const edmNew::DetSetVector<SiPixelCluster>&);  //if not working,: args=2 auto

  std::vector<PSimHit> coreHitsFilling(std::vector<PSimHit>,
                                       const GeomDet*,
                                       GlobalVector,
                                       const reco::Vertex&);  //if not working,: args=0 auto

  std::pair<std::vector<SimTrack>, std::vector<SimVertex>> coreTracksFilling(
      std::vector<PSimHit>,
      const std::vector<SimTrack>,
      const std::vector<SimVertex>);  //if not working,: args=1,2 auto

  std::vector<std::array<double, 5>> seedParFilling(std::pair<std::vector<SimTrack>, std::vector<SimVertex>>,
                                                    const GeomDet*,
                                                    const reco::Candidate&);

  std::pair<std::vector<SimTrack>, std::vector<SimVertex>> coreTracksFillingDeltaR(
      const std::vector<SimTrack>,
      const std::vector<SimVertex>,
      const GeomDet*,
      const reco::Candidate&,
      const reco::Vertex&);  //if not working,: args=0,1 auto
};

JetCoreMCtruthSeedGenerator::JetCoreMCtruthSeedGenerator(const edm::ParameterSet& iConfig)
    :

      vertices_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices"))),
      pixelClusters_(
          consumes<edmNew::DetSetVector<SiPixelCluster>>(iConfig.getParameter<edm::InputTag>("pixelClusters"))),
      cores_(consumes<edm::View<reco::Candidate>>(iConfig.getParameter<edm::InputTag>("cores"))),
      simtracksToken_(consumes<std::vector<SimTrack>>(iConfig.getParameter<edm::InputTag>("simTracks"))),
      simvertexToken_(consumes<std::vector<SimVertex>>(iConfig.getParameter<edm::InputTag>("simVertex"))),
      PSimHitToken_(consumes<std::vector<PSimHit>>(iConfig.getParameter<edm::InputTag>("simHit"))),
      ptMin_(iConfig.getParameter<double>("ptMin")),
      deltaR_(iConfig.getParameter<double>("deltaR")),
      chargeFracMin_(iConfig.getParameter<double>("chargeFractionMin")),
      centralMIPCharge_(iConfig.getParameter<double>("centralMIPCharge")),
      pixelCPE_(iConfig.getParameter<std::string>("pixelCPE"))

{
  produces<TrajectorySeedCollection>();
  produces<reco::TrackCollection>();
}

JetCoreMCtruthSeedGenerator::~JetCoreMCtruthSeedGenerator() {}

void JetCoreMCtruthSeedGenerator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto result = std::make_unique<TrajectorySeedCollection>();
  auto resultTracks = std::make_unique<reco::TrackCollection>();

  using namespace edm;
  using namespace reco;

  iSetup.get<IdealMagneticFieldRecord>().get(magfield_);
  iSetup.get<GlobalTrackingGeometryRecord>().get(geometry_);
  iSetup.get<TrackingComponentsRecord>().get("AnalyticalPropagator", propagator_);

  const auto& inputPixelClusters_ = iEvent.get(pixelClusters_);
  const auto& simtracksVector = iEvent.get(simtracksToken_);
  const auto& simvertexVector = iEvent.get(simvertexToken_);
  const auto& simhits_ = iEvent.get(PSimHitToken_);
  const auto& vertices = iEvent.get(vertices_);
  const auto& cores = iEvent.get(cores_);

  edm::ESHandle<PixelClusterParameterEstimator> pixelCPEhandle;
  const PixelClusterParameterEstimator* pixelCPE;
  iSetup.get<TkPixelCPERecord>().get(pixelCPE_, pixelCPEhandle);
  pixelCPE = pixelCPEhandle.product();

  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  auto output = std::make_unique<edmNew::DetSetVector<SiPixelCluster>>();

  for (const auto& jet : cores) {  //jet loop

    if (jet.pt() > ptMin_) {
      std::set<long long int> ids;
      const reco::Vertex& jetVertex = vertices[0];

      std::vector<GlobalVector> splitClustDirSet =
          splittedClusterDirections(jet, tTopo, pixelCPE, jetVertex, 1, inputPixelClusters_);
      if (splitClustDirSet.empty()) {  //if layer 1 is broken find direcitons on layer 2
        splitClustDirSet = splittedClusterDirections(jet, tTopo, pixelCPE, jetVertex, 2, inputPixelClusters_);
      }
      if (inclusiveConeSeed_)
        splitClustDirSet.clear();
      splitClustDirSet.emplace_back(GlobalVector(jet.px(), jet.py(), jet.pz()));

      for (int cc = 0; cc < (int)splitClustDirSet.size(); cc++) {
        GlobalVector bigClustDir = splitClustDirSet[cc];

        jetEta_ = jet.eta();
        jetPt_ = jet.pt();

        const auto& jetVert = jetVertex;  //trackInfo filling

        std::vector<PSimHit> goodSimHit;

        const GeomDet* globDet = DetectorSelector(
            2, jet, bigClustDir, jetVertex, tTopo, inputPixelClusters_);  //select detector mostly hitten by the jet

        if (globDet == nullptr)
          continue;

        std::pair<std::vector<SimTrack>, std::vector<SimVertex>> goodSimTkVx;

        if (inclusiveConeSeed_) {
          goodSimTkVx = JetCoreMCtruthSeedGenerator::coreTracksFillingDeltaR(
              simtracksVector, simvertexVector, globDet, jet, jetVert);
        } else {
          std::vector<PSimHit> goodSimHit =
              JetCoreMCtruthSeedGenerator::coreHitsFilling(simhits_, globDet, bigClustDir, jetVertex);
          goodSimTkVx = JetCoreMCtruthSeedGenerator::coreTracksFilling(goodSimHit, simtracksVector, simvertexVector);
        }
        edm::LogInfo("PerfectSeeder") << "seed number in deltaR cone =" << goodSimTkVx.first.size();

        std::vector<std::array<double, 5>> seedVector =
            JetCoreMCtruthSeedGenerator::seedParFilling(goodSimTkVx, globDet, jet);
        edm::LogInfo("PerfectSeeder") << "seedVector.size()=" << seedVector.size();

        for (uint tk = 0; tk < seedVector.size(); tk++) {
          for (int pp = 0; pp < 5; pp++) {
            edm::LogInfo("PerfectSeeder")
                << "seed " << tk << ", int par " << pp << "=" << seedVector[tk][pp] << std::endl;
          }
          LocalPoint localSeedPoint = LocalPoint(seedVector[tk][0], seedVector[tk][1], 0);
          double track_theta = 2 * std::atan(std::exp(-seedVector[tk][2]));
          double track_phi = seedVector[tk][3];
          double pt = 1. / seedVector[tk][4];

          double normdirR = pt / sin(track_theta);
          const GlobalVector globSeedDir(
              GlobalVector::Polar(Geom::Theta<double>(track_theta), Geom::Phi<double>(track_phi), normdirR));
          LocalVector localSeedDir = globDet->surface().toLocal(globSeedDir);

          int64_t seedid = (int64_t(localSeedPoint.x() * 200.) << 0) + (int64_t(localSeedPoint.y() * 200.) << 16) +
                           (int64_t(seedVector[tk][2] * 400.) << 32) + (int64_t(track_phi * 400.) << 48);
          if (ids.count(seedid) != 0) {
            edm::LogInfo("PerfectSeeder") << "seed not removed with DeepCore cleaner";
          }
          ids.insert(seedid);

          //Covariance matrix, currently the hadrcoded variances = NN residuals width (see documentation of DeepCore)
          //in general: if are not compared with DeepCore but another algo-->to state-of-the art errors
          //The "perfect seeds" has no intrinsic error, but the CTF needs errors to propagate...
          float em[15] = {
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};  // (see LocalTrajectoryError for details), order as follow:
          em[0] = 0.15 * 0.15;                               // q/pt
          em[2] = 0.5e-5;                                    // dxdz
          em[5] = 0.5e-5;                                    // dydz
          em[9] = 2e-5;                                      // x
          em[14] = 2e-5;                                     // y
          long int detId = globDet->geographicalId();
          LocalTrajectoryParameters localParam(localSeedPoint, localSeedDir, TrackCharge(1));
          result->emplace_back(TrajectorySeed(PTrajectoryStateOnDet(localParam, pt, em, detId, /*surfaceSide*/ 0),
                                              edm::OwnVector<TrackingRecHit>(),
                                              PropagationDirection::alongMomentum));

          GlobalPoint globalSeedPoint = globDet->surface().toGlobal(localSeedPoint);
          reco::Track::CovarianceMatrix mm;
          resultTracks->emplace_back(
              reco::Track(1,
                          1,
                          reco::Track::Point(globalSeedPoint.x(), globalSeedPoint.y(), globalSeedPoint.z()),
                          reco::Track::Vector(globSeedDir.x(), globSeedDir.y(), globSeedDir.z()),
                          1,
                          mm));
          edm::LogInfo("PerfectSeeder") << "seed " << tk << ", out,  pt=" << pt << ", eta=" << globSeedDir.eta()
                                        << ", phi=" << globSeedDir.phi() << std::endl;
        }

      }  //bigcluster
    }    //jet > pt
  }      //jet
  iEvent.put(std::move(result));
  iEvent.put(std::move(resultTracks));
}

std::pair<bool, Basic3DVector<float>> JetCoreMCtruthSeedGenerator::findIntersection(
    const GlobalVector& dir, const reco::Candidate::Point& vertex, const GeomDet* det) {
  StraightLinePlaneCrossing vertexPlane(Basic3DVector<float>(vertex.x(), vertex.y(), vertex.z()),
                                        Basic3DVector<float>(dir.x(), dir.y(), dir.z()));

  std::pair<bool, Basic3DVector<float>> pos = vertexPlane.position(det->specificSurface());

  return pos;
}

const GeomDet* JetCoreMCtruthSeedGenerator::DetectorSelector(int llay,
                                                             const reco::Candidate& jet,
                                                             GlobalVector jetDir,
                                                             const reco::Vertex& jetVertex,
                                                             const TrackerTopology* const tTopo,
                                                             const edmNew::DetSetVector<SiPixelCluster>& clusters) {
  struct trkNumCompare {
    bool operator()(std::pair<int, const GeomDet*> x, std::pair<int, const GeomDet*> y) const {
      return x.first > y.first;
    }
  };
  std::set<std::pair<int, const GeomDet*>, trkNumCompare> track4detSet;

  double minDist = 0.0;
  GeomDet* output = (GeomDet*)nullptr;
  for (const auto& detset : clusters) {
    auto aClusterID = detset.id();
    if (DetId(aClusterID).subdetId() != 1)
      continue;
    const GeomDet* det = geometry_->idToDet(aClusterID);
    int lay = tTopo->layer(det->geographicalId());
    if (lay != llay)
      continue;
    std::pair<bool, Basic3DVector<float>> interPair =
        findIntersection(jetDir, (reco::Candidate::Point)jetVertex.position(), det);
    if (interPair.first == false)
      continue;
    Basic3DVector<float> inter = interPair.second;
    auto localInter = det->specificSurface().toLocal((GlobalPoint)inter);
    if ((minDist == 0.0 || std::abs(localInter.x()) < minDist) && std::abs(localInter.y()) < 3.35) {
      minDist = std::abs(localInter.x());
      output = (GeomDet*)det;
    }
  }  //detset
  return output;
}

std::vector<GlobalVector> JetCoreMCtruthSeedGenerator::splittedClusterDirections(
    const reco::Candidate& jet,
    const TrackerTopology* const tTopo,
    const PixelClusterParameterEstimator* pixelCPE,
    const reco::Vertex& jetVertex,
    int layer,
    const edmNew::DetSetVector<SiPixelCluster>& clusters) {
  std::vector<GlobalVector> clustDirs;
  for (const auto& detset_int : clusters) {
    const GeomDet* det_int = geometry_->idToDet(detset_int.id());
    int lay = tTopo->layer(det_int->geographicalId());
    if (lay != layer)
      continue;  //NB: saved bigclusetr on all the layers!!
    auto detUnit = *geometry_->idToDetUnit(detset_int.id());
    for (const auto& aCluster : detset_int) {
      GlobalPoint clustPos = det_int->surface().toGlobal(pixelCPE->localParametersV(aCluster, detUnit)[0].first);
      GlobalPoint vertexPos(jetVertex.position().x(), jetVertex.position().y(), jetVertex.position().z());
      GlobalVector clusterDir = clustPos - vertexPos;
      GlobalVector jetDir(jet.px(), jet.py(), jet.pz());
      if (Geom::deltaR(jetDir, clusterDir) < deltaR_) {
        clustDirs.emplace_back(clusterDir);
      }
    }
  }
  return clustDirs;
}

std::vector<PSimHit> JetCoreMCtruthSeedGenerator::coreHitsFilling(std::vector<PSimHit> simhits,
                                                                  const GeomDet* globDet,
                                                                  GlobalVector bigClustDir,
                                                                  const reco::Vertex& jetVertex) {
  std::vector<PSimHit> goodSimHit;
  for (const auto& sh : simhits) {
    const GeomDet* det = geometry_->idToDet(sh.detUnitId());
    if (det != globDet)
      continue;
    std::pair<bool, Basic3DVector<float>> interPair =
        findIntersection(bigClustDir, (reco::Candidate::Point)jetVertex.position(), det);
    if (interPair.first == false)
      continue;
    Basic3DVector<float> inter = interPair.second;
    auto localInter = det->specificSurface().toLocal((GlobalPoint)inter);

    if (std::abs((sh.localPosition()).x() - localInter.x()) / pitchX_ <= jetDimX / 2 &&
        std::abs((sh.localPosition()).y() - localInter.y()) / pitchY_ <= jetDimY / 2) {
      goodSimHit.emplace_back(sh);
    }
  }
  return goodSimHit;
}

std::pair<std::vector<SimTrack>, std::vector<SimVertex>> JetCoreMCtruthSeedGenerator::coreTracksFilling(
    std::vector<PSimHit> goodSimHit,
    const std::vector<SimTrack> simtracksVector,
    const std::vector<SimVertex> simvertexVector) {
  std::vector<SimTrack> goodSimTrk;
  std::vector<SimVertex> goodSimVtx;

  for (uint j = 0; j < simtracksVector.size(); j++) {
    for (std::vector<PSimHit>::const_iterator it = goodSimHit.begin(); it != goodSimHit.end(); ++it) {
      SimTrack st = simtracksVector[j];
      if (st.trackId() == (*it).trackId()) {
        for (uint v = 0; v < simvertexVector.size(); v++) {
          SimVertex sv = simvertexVector[v];
          if ((int)sv.vertexId() == (int)st.vertIndex()) {
            goodSimTrk.emplace_back(st);
            goodSimVtx.emplace_back(sv);
          }
        }
      }
    }
  }
  std::pair<std::vector<SimTrack>, std::vector<SimVertex>> output(goodSimTrk, goodSimVtx);
  return output;
}

std::pair<std::vector<SimTrack>, std::vector<SimVertex>> JetCoreMCtruthSeedGenerator::coreTracksFillingDeltaR(
    const std::vector<SimTrack> simtracksVector,
    const std::vector<SimVertex> simvertexVector,
    const GeomDet* globDet,
    const reco::Candidate& jet,
    const reco::Vertex& jetVertex) {
  std::vector<SimTrack> goodSimTrk;
  std::vector<SimVertex> goodSimVtx;

  GlobalVector jetDir(jet.px(), jet.py(), jet.pz());

  for (uint j = 0; j < simtracksVector.size(); j++) {
    SimTrack st = simtracksVector[j];
    GlobalVector trkDir(st.momentum().Px(), st.momentum().Py(), st.momentum().Pz());
    if (st.charge() == 0)
      continue;
    if (Geom::deltaR(jetDir, trkDir) < deltaR_) {
      for (uint v = 0; v < simvertexVector.size(); v++) {
        SimVertex sv = simvertexVector[v];
        if ((int)sv.vertexId() == (int)st.vertIndex()) {
          goodSimTrk.emplace_back(st);
          goodSimVtx.emplace_back(sv);
        }
      }
    }
  }
  std::pair<std::vector<SimTrack>, std::vector<SimVertex>> output(goodSimTrk, goodSimVtx);
  return output;
}

std::vector<std::array<double, 5>> JetCoreMCtruthSeedGenerator::seedParFilling(
    std::pair<std::vector<SimTrack>, std::vector<SimVertex>> goodSimTkVx,
    const GeomDet* globDet,
    const reco::Candidate& jet) {
  std::vector<std::array<double, 5>> output;
  std::vector<SimTrack> goodSimTrk = goodSimTkVx.first;
  std::vector<SimVertex> goodSimVtx = goodSimTkVx.second;

  edm::LogInfo("PerfectSeeder") << "goodSimTrk size" << goodSimTrk.size();
  for (uint j = 0; j < goodSimTrk.size(); j++) {
    SimTrack st = goodSimTrk[j];
    SimVertex sv = goodSimVtx[j];
    GlobalVector trkMom(st.momentum().x(), st.momentum().y(), st.momentum().z());
    GlobalPoint trkPos(sv.position().x(), sv.position().y(), sv.position().z());
    edm::LogInfo("PerfectSeeder") << "seed " << j << ", very int pt" << st.momentum().Pt()
                                  << ", eta=" << st.momentum().Eta() << ", phi=" << st.momentum().Phi()
                                  << "------ internal point=" << trkMom.x() << "," << trkMom.y() << "," << trkMom.z()
                                  << "," << trkPos.x() << "," << trkPos.y() << "," << trkPos.z() << std::endl;

    std::pair<bool, Basic3DVector<float>> trkInterPair;
    trkInterPair = findIntersection(trkMom, (reco::Candidate::Point)trkPos, globDet);
    if (trkInterPair.first == false) {
      GlobalVector jetDir(jet.px(), jet.py(), jet.pz());
      continue;
    }
    Basic3DVector<float> trkInter = trkInterPair.second;

    auto localTrkInter = globDet->specificSurface().toLocal((GlobalPoint)trkInter);  //trkInter->trkPos if par at vertex
    std::array<double, 5> tkPar{
        {localTrkInter.x(), localTrkInter.y(), st.momentum().Eta(), st.momentum().Phi(), 1 / st.momentum().Pt()}};
    output.emplace_back(tkPar);
  }
  return output;
}

// ------------ method called once each job just before starting event loop  ------------
void JetCoreMCtruthSeedGenerator::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void JetCoreMCtruthSeedGenerator::endJob() {}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void JetCoreMCtruthSeedGenerator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("vertices", edm::InputTag("offlinePrimaryVertices"));
  desc.add<edm::InputTag>("pixelClusters", edm::InputTag("siPixelClustersPreSplitting"));
  desc.add<edm::InputTag>("cores", edm::InputTag("jetsForCoreTracking"));
  desc.add<double>("ptMin", 300);
  desc.add<double>("deltaR", 0.3);
  desc.add<double>("chargeFractionMin", 18000.0);
  desc.add<edm::InputTag>("simTracks", edm::InputTag("g4SimHits"));
  desc.add<edm::InputTag>("simVertex", edm::InputTag("g4SimHits"));
  desc.add<edm::InputTag>("simHit", edm::InputTag("g4SimHits", "TrackerHitsPixelBarrelLowTof"));
  desc.add<double>("centralMIPCharge", 2.);
  desc.add<std::string>("pixelCPE", "PixelCPEGeneric");
  descriptions.add("JetCoreMCtruthSeedGenerator", desc);
}

DEFINE_FWK_MODULE(JetCoreMCtruthSeedGenerator);
