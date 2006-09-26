#include "RecoVertex/PrimaryVertexProducer/interface/PrimaryVertexProducerAlgorithm.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertError.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"
#include "FWCore/Utilities/interface/Exception.h"
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
  // extract relevant parts of config for components
  : theConfig(conf), 
    theTrackFilter(conf.getParameter<edm::ParameterSet>("TkFilterParameters")), 
    theTrackClusterizer(conf.getParameter<edm::ParameterSet>("TkClusParameters")), 
    theVertexSelector(VertexDistanceXY(), 
		      conf.getParameter<edm::ParameterSet>("PVSelParameters").getParameter<double>("maxDistanceToBeam"))
{
  edm::LogInfo("RecoVertex/PrimaryVertexProducerAlgorithm") 
    << "Initializing PV producer algorithm" << "\n";
  float testMaxDistanceToBeam = conf.getParameter<edm::ParameterSet>("PVSelParameters").getParameter<double>("maxDistanceToBeam");
  edm::LogInfo("RecoVertex/PrimaryVertexProducerAlgorithm") 
    << "PVSelParameters::maxDistanceToBeam = " 
    << conf.getParameter<edm::ParameterSet>("PVSelParameters").getParameter<double>("maxDistanceToBeam") << "\n";

  // FIXME move vertex chi2 cut in theVertexSelector
  // theFinder should not perform the final vertex cleanup
  float minVertexFitProb 
    = conf.getParameter<edm::ParameterSet>("PVSelParameters").getParameter<double>("minVertexFitProb");
  edm::LogInfo("RecoVertex/PrimaryVertexProducerAlgorithm") 
    << "PVSelParameters::minVertexFitProb = " 
    << conf.getParameter<edm::ParameterSet>("PVSelParameters").getParameter<double>("minVertexFitProb") << endl;
  theFinder.setVertexFitProbabilityCut(minVertexFitProb);

  // initialization of vertex finder algorithm
  // theFinder should not perform any track selection
  // theTrackFilter does it
  theFinder.setPtCut(0.);
  float minTrackCompatibilityToMainVertex 
    = conf.getParameter<edm::ParameterSet>("VtxFinderParameters").getParameter<double>("minTrackCompatibilityToMainVertex");
  edm::LogInfo("RecoVertex/PrimaryVertexProducerAlgorithm") 
    << "VtxFinderParameters::minTrackCompatibilityToMainVertex = " 
    << conf.getParameter<edm::ParameterSet>("VtxFinderParameters").getParameter<double>("minTrackCompatibilityToMainVertex") << endl;
  theFinder.setTrackCompatibilityCut(minTrackCompatibilityToMainVertex);
  float minTrackCompatibilityToOtherVertex 
    = conf.getParameter<edm::ParameterSet>("VtxFinderParameters").getParameter<double>("minTrackCompatibilityToOtherVertex");
  edm::LogInfo("RecoVertex/PrimaryVertexProducerAlgorithm") 
    << "VtxFinderParameters::minTrackCompatibilityToOtherVertex = " 
    << conf.getParameter<edm::ParameterSet>("VtxFinderParameters").getParameter<double>("minTrackCompatibilityToOtherVertex") << endl;
  theFinder.setTrackCompatibilityToSV(minTrackCompatibilityToOtherVertex);
  int maxNbVertices 
    = conf.getParameter<edm::ParameterSet>("VtxFinderParameters").getParameter<int>("maxNbVertices");
  edm::LogInfo("RecoVertex/PrimaryVertexProducerAlgorithm") 
    << "VtxFinderParameters::maxNbVertices = " 
    << conf.getParameter<edm::ParameterSet>("VtxFinderParameters").getParameter<int>("maxNbVertices") << endl;
  theFinder.setMaxNbOfVertices(maxNbVertices);

  edm::LogInfo("RecoVertex/PrimaryVertexProducerAlgorithm") 
    << "PV producer algorithm initialization: done" << "\n";

}


PrimaryVertexProducerAlgorithm::~PrimaryVertexProducerAlgorithm() 
{}


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
    //    cout << "PrimaryVertexProducerAlgorithm::vertices  input tracks="<<tracks.size() << endl;
    for (vector<reco::TransientTrack>::const_iterator itk = tracks.begin();
	 itk != tracks.end(); itk++) {
      if (theTrackFilter(*itk)) seltks.push_back(*itk);
    }
    //cout << "PrimaryVertexProducerAlgorithm::vertices  selected tracks=" << seltks.size() << endl;

    // clusterize tracks in Z
    vector< vector<reco::TransientTrack> > clusters = 
      theTrackClusterizer.clusterize(seltks);

    //cout << "PrimaryVertexProducerAlgorithm::vertices  clusters =" << clusters.size() << endl;

    // look for primary vertices in each cluster
    vector<TransientVertex> pvCand;
    int nclu=0;
    for (vector< vector<reco::TransientTrack> >::const_iterator iclus
	   = clusters.begin(); iclus != clusters.end(); iclus++) {
      //cout << "PrimaryVertexProducerAlgorithm::vertices  cluster =" << nclu << "  tracks" << (*iclus).size() << endl;

      
      /*
      if((*iclus).size()>1){
	vector<TransientVertex> pvFromClus = theFinder.vertices(*iclus);
	//cout << "PrimaryVertexProducerAlgorithm::vertices  finder found "  
	//     <<pvFromClus.size()
	//     << endl;
	//pvCand.reserve(pvCand.size() + pvFromClus.size());
	//std::copy(pvFromClus.begin(), pvFromClus.end(), pvCand.end());

	for (vector<TransientVertex>::const_iterator ipv = pvFromClus.begin();
	     ipv != pvFromClus.end(); ipv++) {
	  pvCand.push_back(*ipv);
	}
      }
      */


      if((*iclus).size()>1){
	KalmanVertexFitter kvf;
	TransientVertex v = kvf.vertex(*iclus);  // CachingVertex, converted by operater ?
	pvCand.push_back(v);
      }

      nclu++;
    }

    //cout << "PrimaryVertexProducerAlgorithm::vertices  candidates =" << pvCand.size() << endl;

    // select vertices compatible with beam
    int npv=0;
    for (vector<TransientVertex>::const_iterator ipv = pvCand.begin();
	 ipv != pvCand.end(); ipv++) {
      //cout << "PrimaryVertexProducerAlgorithm::vertices cand " << npv << " sel=" <<
      // theVertexSelector(*ipv) << "   z="  << ipv->position().z() << endl;
      //if (theVertexSelector(*ipv)) pvs.push_back(*ipv);
      pvs.push_back(*ipv);
      npv++;
    }
      
    //cout << "PrimaryVertexProducerAlgorithm::vertices  vertices =" << pvs.size() << endl;

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
