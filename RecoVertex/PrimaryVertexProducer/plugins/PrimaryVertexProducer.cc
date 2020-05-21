#include "RecoVertex/PrimaryVertexProducer/interface/PrimaryVertexProducer.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "RecoVertex/VertexTools/interface/GeometricAnnealing.h"

PrimaryVertexProducer::PrimaryVertexProducer(const edm::ParameterSet& conf) : theConfig(conf) {
  fVerbose = conf.getUntrackedParameter<bool>("verbose", false);

  trkToken = consumes<reco::TrackCollection>(conf.getParameter<edm::InputTag>("TrackLabel"));
  bsToken = consumes<reco::BeamSpot>(conf.getParameter<edm::InputTag>("beamSpotLabel"));
  f4D = false;

  // select and configure the track selection
  std::string trackSelectionAlgorithm =
      conf.getParameter<edm::ParameterSet>("TkFilterParameters").getParameter<std::string>("algorithm");
  if (trackSelectionAlgorithm == "filter") {
    theTrackFilter = new TrackFilterForPVFinding(conf.getParameter<edm::ParameterSet>("TkFilterParameters"));
  } else if (trackSelectionAlgorithm == "filterWithThreshold") {
    theTrackFilter = new HITrackFilterForPVFinding(conf.getParameter<edm::ParameterSet>("TkFilterParameters"));
  } else {
    throw VertexException("PrimaryVertexProducerAlgorithm: unknown track selection algorithm: " +
                          trackSelectionAlgorithm);
  }

  // select and configure the track clusterizer
  std::string clusteringAlgorithm =
      conf.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<std::string>("algorithm");
  if (clusteringAlgorithm == "gap") {
    theTrackClusterizer = new GapClusterizerInZ(
        conf.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<edm::ParameterSet>("TkGapClusParameters"));
  } else if (clusteringAlgorithm == "DA") {
    theTrackClusterizer = new DAClusterizerInZ(
        conf.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<edm::ParameterSet>("TkDAClusParameters"));
  }
  // provide the vectorized version of the clusterizer, if supported by the build
  else if (clusteringAlgorithm == "DA_vect") {
    theTrackClusterizer = new DAClusterizerInZ_vect(
        conf.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<edm::ParameterSet>("TkDAClusParameters"));
  } else if (clusteringAlgorithm == "DA2D_vect") {
    theTrackClusterizer = new DAClusterizerInZT_vect(
        conf.getParameter<edm::ParameterSet>("TkClusParameters").getParameter<edm::ParameterSet>("TkDAClusParameters"));
    f4D = true;
  }

  else {
    throw VertexException("PrimaryVertexProducerAlgorithm: unknown clustering algorithm: " + clusteringAlgorithm);
  }

  if (f4D) {
    trkTimesToken = consumes<edm::ValueMap<float> >(conf.getParameter<edm::InputTag>("TrackTimesLabel"));
    trkTimeResosToken = consumes<edm::ValueMap<float> >(conf.getParameter<edm::InputTag>("TrackTimeResosLabel"));
  }

  // select and configure the vertex fitters
  if (conf.exists("vertexCollections")) {
    std::vector<edm::ParameterSet> vertexCollections =
        conf.getParameter<std::vector<edm::ParameterSet> >("vertexCollections");

    for (const auto& vertexCollection : vertexCollections) {
      algo algorithm;
      std::string fitterAlgorithm = vertexCollection.getParameter<std::string>("algorithm");
      if (fitterAlgorithm == "KalmanVertexFitter") {
        algorithm.fitter = new KalmanVertexFitter();
      } else if (fitterAlgorithm == "AdaptiveVertexFitter") {
        algorithm.fitter =
            new AdaptiveVertexFitter(GeometricAnnealing(vertexCollection.getParameter<double>("chi2cutoff")));
      } else {
        throw VertexException("PrimaryVertexProducerAlgorithm: unknown algorithm: " + fitterAlgorithm);
      }
      algorithm.label = vertexCollection.getParameter<std::string>("label");
      algorithm.minNdof = vertexCollection.getParameter<double>("minNdof");
      algorithm.useBeamConstraint = vertexCollection.getParameter<bool>("useBeamConstraint");
      algorithm.vertexSelector =
          new VertexCompatibleWithBeam(VertexDistanceXY(), vertexCollection.getParameter<double>("maxDistanceToBeam"));
      algorithms.push_back(algorithm);

      produces<reco::VertexCollection>(algorithm.label);
    }
  } else {
    edm::LogWarning("MisConfiguration")
        << "this module's configuration has changed, please update to have a vertexCollections=cms.VPSet parameter.";

    algo algorithm;
    std::string fitterAlgorithm = conf.getParameter<std::string>("algorithm");
    if (fitterAlgorithm == "KalmanVertexFitter") {
      algorithm.fitter = new KalmanVertexFitter();
    } else if (fitterAlgorithm == "AdaptiveVertexFitter") {
      algorithm.fitter = new AdaptiveVertexFitter();
    } else {
      throw VertexException("PrimaryVertexProducerAlgorithm: unknown algorithm: " + fitterAlgorithm);
    }
    algorithm.label = "";
    algorithm.minNdof = conf.getParameter<double>("minNdof");
    algorithm.useBeamConstraint = conf.getParameter<bool>("useBeamConstraint");

    algorithm.vertexSelector = new VertexCompatibleWithBeam(
        VertexDistanceXY(),
        conf.getParameter<edm::ParameterSet>("PVSelParameters").getParameter<double>("maxDistanceToBeam"));

    algorithms.push_back(algorithm);
    produces<reco::VertexCollection>(algorithm.label);
  }
}

PrimaryVertexProducer::~PrimaryVertexProducer() {
  if (theTrackFilter)
    delete theTrackFilter;
  if (theTrackClusterizer)
    delete theTrackClusterizer;
  for (const auto& algorithm : algorithms) {
    if (algorithm.fitter)
      delete algorithm.fitter;
    if (algorithm.vertexSelector)
      delete algorithm.vertexSelector;
  }
}

void PrimaryVertexProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // get the BeamSpot, it will alwys be needed, even when not used as a constraint
  reco::BeamSpot beamSpot;
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  iEvent.getByToken(bsToken, recoBeamSpotHandle);
  if (recoBeamSpotHandle.isValid()) {
    beamSpot = *recoBeamSpotHandle;
  } else {
    edm::LogError("UnusableBeamSpot") << "No beam spot available from EventSetup";
  }

  bool validBS = true;
  VertexState beamVertexState(beamSpot);
  if ((beamVertexState.error().cxx() <= 0.) || (beamVertexState.error().cyy() <= 0.) ||
      (beamVertexState.error().czz() <= 0.)) {
    validBS = false;
    edm::LogError("UnusableBeamSpot") << "Beamspot with invalid errors " << beamVertexState.error().matrix();
  }

  // get RECO tracks from the event
  // `tks` can be used as a ptr to a reco::TrackCollection
  edm::Handle<reco::TrackCollection> tks;
  iEvent.getByToken(trkToken, tks);

  // interface RECO tracks to vertex reconstruction
  edm::ESHandle<TransientTrackBuilder> theB;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder", theB);
  std::vector<reco::TransientTrack> t_tks;

  if (f4D) {
    edm::Handle<edm::ValueMap<float> > trackTimesH;
    edm::Handle<edm::ValueMap<float> > trackTimeResosH;
    iEvent.getByToken(trkTimesToken, trackTimesH);
    iEvent.getByToken(trkTimeResosToken, trackTimeResosH);
    t_tks = (*theB).build(tks, beamSpot, *(trackTimesH.product()), *(trackTimeResosH.product()));
  } else {
    t_tks = (*theB).build(tks, beamSpot);
  }
  if (fVerbose) {
    std::cout << "RecoVertex/PrimaryVertexProducer"
              << "Found: " << t_tks.size() << " reconstructed tracks"
              << "\n";
  }

  // select tracks
  std::vector<reco::TransientTrack>&& seltks = theTrackFilter->select(t_tks);

  // clusterize tracks in Z
  std::vector<std::vector<reco::TransientTrack> >&& clusters = theTrackClusterizer->clusterize(seltks);

  if (fVerbose) {
    std::cout << " clustering returned  " << clusters.size() << " clusters  from " << seltks.size()
              << " selected tracks" << std::endl;
  }

  // vertex fits
  for (const auto& algorithm : algorithms) {
    auto result = std::make_unique<reco::VertexCollection>();
    reco::VertexCollection& vColl = (*result);

    std::vector<TransientVertex> pvs;
    for (const auto& cluster : clusters) {
      double sumwt = 0.;
      double sumwt2 = 0.;
      double sumw = 0.;
      double meantime = 0.;
      double vartime = 0.;
      if (f4D) {
        for (const auto& tk : cluster) {
          const double time = tk.timeExt();
          const double err = tk.dtErrorExt();
          const double inverr = err > 0. ? 1.0 / err : 0.;
          const double w = inverr * inverr;
          sumwt += w * time;
          sumwt2 += w * time * time;
          sumw += w;
        }
        meantime = sumwt / sumw;
        double sumsq = sumwt2 - sumwt * sumwt / sumw;
        double chisq = cluster.size() > 1 ? sumsq / double(cluster.size() - 1) : sumsq / double(cluster.size());
        vartime = chisq / sumw;
      }

      TransientVertex v;
      if (algorithm.useBeamConstraint && validBS && (cluster.size() > 1)) {
        v = algorithm.fitter->vertex(cluster, beamSpot);

        if (f4D) {
          if (v.isValid()) {
            auto err = v.positionError().matrix4D();
            err(3, 3) = vartime;
            v = TransientVertex(v.position(), meantime, err, v.originalTracks(), v.totalChiSquared());
          }
        }

      } else if (!(algorithm.useBeamConstraint) && (cluster.size() > 1)) {
        v = algorithm.fitter->vertex(cluster);

        if (f4D) {
          if (v.isValid()) {
            auto err = v.positionError().matrix4D();
            err(3, 3) = vartime;
            v = TransientVertex(v.position(), meantime, err, v.originalTracks(), v.totalChiSquared());
          }
        }

      }  // else: no fit ==> v.isValid()=False

      if (fVerbose) {
        if (v.isValid()) {
          std::cout << "x,y,z";
          if (f4D)
            std::cout << ",t";
          std::cout << "=" << v.position().x() << " " << v.position().y() << " " << v.position().z();
          if (f4D)
            std::cout << " " << v.time();
          std::cout << " cluster size = " << cluster.size() << std::endl;
        } else {
          std::cout << "Invalid fitted vertex,  cluster size=" << cluster.size() << std::endl;
        }
      }

      if (v.isValid() && (v.degreesOfFreedom() >= algorithm.minNdof) &&
          (!validBS || (*(algorithm.vertexSelector))(v, beamVertexState)))
        pvs.push_back(v);
    }  // end of cluster loop

    if (fVerbose) {
      std::cout << "PrimaryVertexProducerAlgorithm::vertices  candidates =" << pvs.size() << std::endl;
    }

    if (clusters.size() > 2 && clusters.size() > 2 * pvs.size())
      edm::LogWarning("PrimaryVertexProducer")
          << "more than half of candidate vertices lost " << pvs.size() << ' ' << clusters.size();

    if (pvs.empty() && seltks.size() > 5)
      edm::LogWarning("PrimaryVertexProducer")
          << "no vertex found with " << seltks.size() << " tracks and " << clusters.size() << " vertex-candidates";

    // sort vertices by pt**2  vertex (aka signal vertex tagging)
    if (pvs.size() > 1) {
      sort(pvs.begin(), pvs.end(), VertexHigherPtSquared());
    }

    // convert transient vertices returned by the theAlgo to (reco) vertices
    for (const auto& pv : pvs) {
      reco::Vertex v = pv;
      vColl.push_back(v);
    }

    if (vColl.empty()) {
      GlobalError bse(beamSpot.rotatedCovariance3D());
      if ((bse.cxx() <= 0.) || (bse.cyy() <= 0.) || (bse.czz() <= 0.)) {
        AlgebraicSymMatrix33 we;
        we(0, 0) = 10000;
        we(1, 1) = 10000;
        we(2, 2) = 10000;
        vColl.push_back(reco::Vertex(beamSpot.position(), we, 0., 0., 0));
        if (fVerbose) {
          std::cout << "RecoVertex/PrimaryVertexProducer: "
                    << "Beamspot with invalid errors " << bse.matrix() << std::endl;
          std::cout << "Will put Vertex derived from dummy-fake BeamSpot into Event.\n";
        }
      } else {
        vColl.push_back(reco::Vertex(beamSpot.position(), beamSpot.rotatedCovariance3D(), 0., 0., 0));
        if (fVerbose) {
          std::cout << "RecoVertex/PrimaryVertexProducer: "
                    << " will put Vertex derived from BeamSpot into Event.\n";
        }
      }
    }

    if (fVerbose) {
      int ivtx = 0;
      for (const auto& v : vColl) {
        std::cout << "recvtx " << ivtx++ << "#trk " << std::setw(3) << v.tracksSize() << " chi2 " << std::setw(4)
                  << v.chi2() << " ndof " << std::setw(3) << v.ndof() << " x " << std::setw(6) << v.position().x()
                  << " dx " << std::setw(6) << v.xError() << " y " << std::setw(6) << v.position().y() << " dy "
                  << std::setw(6) << v.yError() << " z " << std::setw(6) << v.position().z() << " dz " << std::setw(6)
                  << v.zError();
        if (f4D) {
          std::cout << " t " << std::setw(6) << v.t() << " dt " << std::setw(6) << v.tError();
        }
        std::cout << std::endl;
      }
    }

    iEvent.put(std::move(result), algorithm.label);
  }
}

void PrimaryVertexProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // offlinePrimaryVertices
  edm::ParameterSetDescription desc;
  {
    edm::ParameterSetDescription vpsd1;
    vpsd1.add<double>("maxDistanceToBeam", 1.0);
    vpsd1.add<std::string>("algorithm", "AdaptiveVertexFitter");
    vpsd1.add<bool>("useBeamConstraint", false);
    vpsd1.add<std::string>("label", "");
    vpsd1.add<double>("chi2cutoff", 2.5);
    vpsd1.add<double>("minNdof", 0.0);
    std::vector<edm::ParameterSet> temp1;
    temp1.reserve(2);
    {
      edm::ParameterSet temp2;
      temp2.addParameter<double>("maxDistanceToBeam", 1.0);
      temp2.addParameter<std::string>("algorithm", "AdaptiveVertexFitter");
      temp2.addParameter<bool>("useBeamConstraint", false);
      temp2.addParameter<std::string>("label", "");
      temp2.addParameter<double>("chi2cutoff", 2.5);
      temp2.addParameter<double>("minNdof", 0.0);
      temp1.push_back(temp2);
    }
    {
      edm::ParameterSet temp2;
      temp2.addParameter<double>("maxDistanceToBeam", 1.0);
      temp2.addParameter<std::string>("algorithm", "AdaptiveVertexFitter");
      temp2.addParameter<bool>("useBeamConstraint", true);
      temp2.addParameter<std::string>("label", "WithBS");
      temp2.addParameter<double>("chi2cutoff", 2.5);
      temp2.addParameter<double>("minNdof", 2.0);
      temp1.push_back(temp2);
    }
    desc.addVPSet("vertexCollections", vpsd1, temp1);
  }
  desc.addUntracked<bool>("verbose", false);
  {
    edm::ParameterSetDescription psd0;
    psd0.add<double>("maxNormalizedChi2", 10.0);
    psd0.add<double>("minPt", 0.0);
    psd0.add<std::string>("algorithm", "filter");
    psd0.add<double>("maxEta", 2.4);
    psd0.add<double>("maxD0Significance", 4.0);
    psd0.add<double>("maxD0Error", 1.0);
    psd0.add<double>("maxDzError", 1.0);
    psd0.add<std::string>("trackQuality", "any");
    psd0.add<int>("minPixelLayersWithHits", 2);
    psd0.add<int>("minSiliconLayersWithHits", 5);
    psd0.add<int>("numTracksThreshold", 0);  // HI only
    desc.add<edm::ParameterSetDescription>("TkFilterParameters", psd0);
  }
  desc.add<edm::InputTag>("beamSpotLabel", edm::InputTag("offlineBeamSpot"));
  desc.add<edm::InputTag>("TrackLabel", edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("TrackTimeResosLabel", edm::InputTag("dummy_default"));  // 4D only
  desc.add<edm::InputTag>("TrackTimesLabel", edm::InputTag("dummy_default"));      // 4D only
  {
    edm::ParameterSetDescription psd0;
    {
      edm::ParameterSetDescription psd1;
      psd1.addUntracked<bool>("verbose", false);
      psd1.addUntracked<double>("zdumpcenter", 0.);
      psd1.addUntracked<double>("zdumpwidth", 20.);
      psd1.addUntracked<bool>("use_vdt", false);  // obsolete, appears in HLT configs
      psd1.add<double>("d0CutOff", 3.0);
      psd1.add<double>("Tmin", 2.0);
      psd1.add<double>("delta_lowT", 0.001);
      psd1.add<double>("zmerge", 0.01);
      psd1.add<double>("dzCutOff", 3.0);
      psd1.add<double>("Tpurge", 2.0);
      psd1.add<int>("convergence_mode", 0);
      psd1.add<double>("delta_highT", 0.01);
      psd1.add<double>("Tstop", 0.5);
      psd1.add<double>("coolingFactor", 0.6);
      psd1.add<double>("vertexSize", 0.006);
      psd1.add<double>("uniquetrkweight", 0.8);
      psd1.add<double>("zrange", 4.0);

      psd1.add<double>("tmerge", 0.01);           // 4D only
      psd1.add<double>("dtCutOff", 4.);           // 4D only
      psd1.add<double>("t0Max", 1.0);             // 4D only
      psd1.add<double>("vertexSizeTime", 0.008);  // 4D only

      psd0.add<edm::ParameterSetDescription>("TkDAClusParameters", psd1);

      edm::ParameterSetDescription psd2;
      psd2.add<double>("zSeparation", 1.0);
      psd0.add<edm::ParameterSetDescription>("TkGapClusParameters", psd2);
    }
    psd0.add<std::string>("algorithm", "DA_vect");
    desc.add<edm::ParameterSetDescription>("TkClusParameters", psd0);
  }

  descriptions.add("primaryVertexProducer", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(PrimaryVertexProducer);
