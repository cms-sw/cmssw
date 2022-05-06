///////////////   OBSOLETE ////////////////////
#include "RecoVertex/PrimaryVertexProducer/interface/PrimaryVertexProducerAlgorithm.h"

#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

PrimaryVertexProducerAlgorithm::PrimaryVertexProducerAlgorithm(const edm::ParameterSet& conf) : theConfig(conf) {
  fVerbose = conf.getUntrackedParameter<bool>("verbose", false);
  trackLabel = conf.getParameter<edm::InputTag>("TrackLabel");
  beamSpotLabel = conf.getParameter<edm::InputTag>("beamSpotLabel");

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
  }

  else {
    throw VertexException("PrimaryVertexProducerAlgorithm: unknown clustering algorithm: " + clusteringAlgorithm);
  }

  // select and configure the vertex fitters
  std::vector<edm::ParameterSet> vertexCollections =
      conf.getParameter<std::vector<edm::ParameterSet> >("vertexCollections");

  for (std::vector<edm::ParameterSet>::const_iterator algoconf = vertexCollections.begin();
       algoconf != vertexCollections.end();
       algoconf++) {
    algo algorithm;
    std::string fitterAlgorithm = algoconf->getParameter<std::string>("algorithm");
    if (fitterAlgorithm == "KalmanVertexFitter") {
      algorithm.fitter = new KalmanVertexFitter();
    } else if (fitterAlgorithm == "AdaptiveVertexFitter") {
      algorithm.fitter = new AdaptiveVertexFitter();
    } else {
      throw VertexException("PrimaryVertexProducerAlgorithm: unknown algorithm: " + fitterAlgorithm);
    }
    algorithm.label = algoconf->getParameter<std::string>("label");
    algorithm.minNdof = algoconf->getParameter<double>("minNdof");
    algorithm.useBeamConstraint = algoconf->getParameter<bool>("useBeamConstraint");
    algorithm.vertexSelector =
        new VertexCompatibleWithBeam(VertexDistanceXY(), algoconf->getParameter<double>("maxDistanceToBeam"));
    algorithms.push_back(algorithm);
  }
}

PrimaryVertexProducerAlgorithm::~PrimaryVertexProducerAlgorithm() {
  if (theTrackFilter)
    delete theTrackFilter;
  if (theTrackClusterizer)
    delete theTrackClusterizer;
  for (std::vector<algo>::const_iterator algorithm = algorithms.begin(); algorithm != algorithms.end(); algorithm++) {
    if (algorithm->fitter)
      delete algorithm->fitter;
    if (algorithm->vertexSelector)
      delete algorithm->vertexSelector;
  }
}

//
// member functions
//

// obsolete method, unfortunately required through inheritance from  VertexReconstructor
std::vector<TransientVertex> PrimaryVertexProducerAlgorithm::vertices(
    const std::vector<reco::TransientTrack>& tracks) const {
  throw VertexException("PrimaryVertexProducerAlgorithm: cannot make a Primary Vertex without a beam spot");

  return std::vector<TransientVertex>();
}

std::vector<TransientVertex> PrimaryVertexProducerAlgorithm::vertices(const std::vector<reco::TransientTrack>& t_tks,
                                                                      const reco::BeamSpot& beamSpot,
                                                                      const std::string& label) const {
  bool validBS = true;
  VertexState beamVertexState(beamSpot);
  if ((beamVertexState.error().cxx() <= 0.) || (beamVertexState.error().cyy() <= 0.) ||
      (beamVertexState.error().czz() <= 0.)) {
    validBS = false;
    edm::LogError("UnusableBeamSpot") << "Beamspot with invalid errors " << beamVertexState.error().matrix();
  }

  //   // get RECO tracks from the event
  //   // `tks` can be used as a ptr to a reco::TrackCollection
  //   edm::Handle<reco::TrackCollection> tks;
  //   iEvent.getByLabel(trackLabel, tks);

  // select tracks
  std::vector<reco::TransientTrack> seltks = theTrackFilter->select(t_tks);

  // clusterize tracks in Z
  std::vector<std::vector<reco::TransientTrack> > clusters = theTrackClusterizer->clusterize(seltks);
  if (fVerbose) {
    std::cout << " clustering returned  " << clusters.size() << " clusters  from " << seltks.size()
              << " selected tracks" << std::endl;
  }

  // vertex fits
  for (std::vector<algo>::const_iterator algorithm = algorithms.begin(); algorithm != algorithms.end(); algorithm++) {
    if (!(algorithm->label == label))
      continue;

    //std::auto_ptr<reco::VertexCollection> result(new reco::VertexCollection);
    // reco::VertexCollection vColl;

    std::vector<TransientVertex> pvs;
    for (std::vector<std::vector<reco::TransientTrack> >::const_iterator iclus = clusters.begin();
         iclus != clusters.end();
         iclus++) {
      TransientVertex v;
      if (algorithm->useBeamConstraint && validBS && ((*iclus).size() > 1)) {
        v = algorithm->fitter->vertex(*iclus, beamSpot);

      } else if (!(algorithm->useBeamConstraint) && ((*iclus).size() > 1)) {
        v = algorithm->fitter->vertex(*iclus);

      }  // else: no fit ==> v.isValid()=False

      if (fVerbose) {
        if (v.isValid())
          std::cout << "x,y,z=" << v.position().x() << " " << v.position().y() << " " << v.position().z() << std::endl;
        else
          std::cout << "Invalid fitted vertex\n";
      }

      if (v.isValid() && (v.degreesOfFreedom() >= algorithm->minNdof) &&
          (!validBS || (*(algorithm->vertexSelector))(v, beamVertexState)))
        pvs.push_back(v);
    }  // end of cluster loop

    if (fVerbose) {
      std::cout << "PrimaryVertexProducerAlgorithm::vertices  candidates =" << pvs.size() << std::endl;
    }

    // sort vertices by pt**2  vertex (aka signal vertex tagging)
    if (pvs.size() > 1) {
      sort(pvs.begin(), pvs.end(), VertexHigherPtSquared());
    }

    return pvs;
  }

  std::vector<TransientVertex> dummy;
  return dummy;  //avoid compiler warning, should never be here
}
