#include "RecoVertex/PrimaryVertexProducer/interface/PrimaryVertexProducerAlgorithm.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertError.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"
#include <algorithm>

using namespace reco;

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
PrimaryVertexProducerAlgorithm::PrimaryVertexProducerAlgorithm(const edm::ParameterSet& conf)
  //FIXME extract relevant parts of config for components
  : theConfig(conf), theTrackFilter(conf), theTrackClusterizer(conf), 
    theVertexSelector(VertexDistanceXY(), 
		      conf.getParameter<double>("MaxDistanceToBeam"))
{
  edm::LogInfo("RecoVertex/PrimaryVertexProducerAlgorithm") 
    << "Initializing PV producer algorithm" << "\n";

  // initialization of vertex finder algorithm
  // configurable parameters
  // FIXME theFinder should not perform track selection
  float ptMin = theConfig.getParameter<double>("TrackPtMin");
  theFinder.setPtCut(ptMin);
  float minTrackCompatibilityToMainVertex 
    = theConfig.getParameter<double>("MinTrackCompatibilityToMainVertex");
  theFinder.setTrackCompatibilityCut(minTrackCompatibilityToMainVertex);
  float minTrackCompatibilityToOtherVertex 
    = theConfig.getParameter<double>("MinTrackCompatibilityToOtherVertex");
  theFinder.setTrackCompatibilityToSV(minTrackCompatibilityToOtherVertex);
  int maxNbVertices = theConfig.getParameter<int>("MaxNbVertices");
  theFinder.setMaxNbOfVertices(maxNbVertices);

  // FIXME introduce track selection (pt + i.p. cuts), track clustering 
  // and vertex selection (beam compatibility check)

}


PrimaryVertexProducerAlgorithm::~PrimaryVertexProducerAlgorithm()
{
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

vector<TransientVertex> 
PrimaryVertexProducerAlgorithm::vertices(const vector<reco::TransientTrack> & tracks) const
{
  vector<TransientVertex> pvs;
  try {
    
    // select tracks
    vector<reco::TransientTrack> seltks;
    for (vector<reco::TransientTrack>::const_iterator itk = tracks.begin();
	 itk != tracks.end(); itk++) {
      if (theTrackFilter(*itk)) seltks.push_back(*itk);
    }

    // clusterize tracks in Z
    vector< vector<reco::TransientTrack> > clusters = 
      theTrackClusterizer.clusterize(seltks);

    // look for primary vertices in each cluster
    vector<TransientVertex> pvCand;
    for (vector< vector<reco::TransientTrack> >::const_iterator iclus
	   = clusters.begin(); iclus != clusters.end(); iclus++) {

      vector<TransientVertex> pvFromClus = theFinder.vertices(*iclus);
      pvCand.reserve(pvCand.size() + pvFromClus.size());
      std::copy(pvFromClus.begin(), pvFromClus.end(), pvCand.end());
    }

    // select vertices compatible with beam
    for (vector<TransientVertex>::const_iterator ipv = pvCand.begin();
	 ipv != pvCand.begin(); ipv++) {
      if (theVertexSelector(*ipv)) pvs.push_back(*ipv);
    }
      

    /*
    // test with vertex fitter
    if (tracks.size() > 1) {
      KalmanVertexFitter kvf;
      TransientVertex tv = kvf.vertex(tracks);
      pvs.push_back(tv);
    }
    */
  }

  catch (std::exception & err) {
    edm::LogInfo("RecoVertex/PrimaryVertexProducerAlgorithm") 
      << "Exception while reconstructing tracker PV: " 
      << "\n" << err.what() << "\n";
  }

  return pvs;
  
}
