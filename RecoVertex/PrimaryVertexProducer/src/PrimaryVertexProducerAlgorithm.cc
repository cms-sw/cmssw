#include "RecoVertex/PrimaryVertexProducer/interface/PrimaryVertexProducerAlgorithm.h"
#include "RecoVertex/PrimaryVertexProducer/interface/VertexHigherPtSquared.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertError.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "RecoVertex/AdaptiveVertexFit/interface/AdaptiveVertexFitter.h"
#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"
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
  //float testMaxDistanceToBeam = conf.getParameter<edm::ParameterSet>("PVSelParameters").getParameter<double>("maxDistanceToBeam");
  edm::LogInfo("RecoVertex/PrimaryVertexProducerAlgorithm") 
    << "PVSelParameters::maxDistanceToBeam = " 
    << conf.getParameter<edm::ParameterSet>("PVSelParameters").getParameter<double>("maxDistanceToBeam") << "\n";

  fUseBeamConstraint = conf.getParameter<bool>("useBeamConstraint");
  fVerbose           = conf.getUntrackedParameter<bool>("verbose", false);
  std::string algorithm = conf.getParameter<std::string>("algorithm");
  fapply_finder = false;
  if (algorithm == "TrimmedKalmanFinder") {
    fapply_finder = true;
  } else if (algorithm=="KalmanVertexFitter") {
    theFitter=new KalmanVertexFitter();
  } else if( algorithm=="AdaptiveVertexFitter") {
    theFitter=new AdaptiveVertexFitter();
  } else {
    throw VertexException("PrimaryVertexProducerAlgorithm: unknown algorithm: " + algorithm);  
  }

  edm::LogInfo("RecoVertex/PrimaryVertexProducerAlgorithm") 
    << "Using " << algorithm << "\n";
  edm::LogInfo("RecoVertex/PrimaryVertexProducerAlgorithm") 
    << "beam-constraint  " << fUseBeamConstraint << "\n"; 

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
  
  if ( fapply_finder) {
    return theFinder.vertices( tracks );
  }
  vector<TransientVertex> pvs;
  try {
    // select tracks
    vector<reco::TransientTrack> seltks;
    // if(fVerbose){
    //  cout << "PrimaryVertexProducerAlgorithm::vertices  input tracks="<<tracks.size() << endl;
    //}
    for (vector<reco::TransientTrack>::const_iterator itk = tracks.begin();
	 itk != tracks.end(); itk++) {
      if (theTrackFilter(*itk)) seltks.push_back(*itk);
    }

    if(fVerbose){
      cout << "PrimaryVertexProducerAlgorithm::vertices  selected tracks=" << seltks.size() << endl;
    }

    // clusterize tracks in Z
    vector< vector<reco::TransientTrack> > clusters = 
      theTrackClusterizer.clusterize(seltks);

    if(fVerbose){
      cout << "PrimaryVertexProducerAlgorithm::vertices  clusters       =" << clusters.size() << endl;
      int i=0;
      for (vector< vector<reco::TransientTrack> >::const_iterator iclus
	     = clusters.begin(); iclus != clusters.end(); iclus++) {
	cout << "PrimaryVertexProducerAlgorithm::vertices  cluster  " << i++ << ")  tracks =" << (*iclus).size() << endl;
      }
    }

    // look for primary vertices in each cluster
    vector<TransientVertex> pvCand;
    int nclu=0;
    for (vector< vector<reco::TransientTrack> >::const_iterator iclus
	   = clusters.begin(); iclus != clusters.end(); iclus++) {
      if(fVerbose){
	cout << "PrimaryVertexProducerAlgorithm::vertices  cluster =" << nclu << "  tracks" << (*iclus).size() << endl;
      }


      if( fUseBeamConstraint &&((*iclus).size()>0) ){
	if (fVerbose){cout <<  "constrained fit with "<< (*iclus).size() << " tracks"  << endl;}
	try {
          TransientVertex v = theFitter->vertex(*iclus, theBeamSpot.position(), theBeamSpot.error());

	  if (fVerbose){
	    cout << "beamspot   x="<< theBeamSpot.position().x() 
		 << " y=" << theBeamSpot.position().y()
		 << " z=" << theBeamSpot.position().z()
		 << " dx=" << sqrt(theBeamSpot.error().cxx())
		 << " dy=" << sqrt(theBeamSpot.error().cyy())
		 << " dz=" << sqrt(theBeamSpot.error().czz())
		 << std::endl;
	    if (v.isValid()) cout << "x,y,z=" << v.position().x() <<" " << v.position().y() << " " <<  v.position().z() << endl;
	      else cout <<"Invalid fitted vertex\n";
	  }
	  if (v.isValid()) pvCand.push_back(v);
	}  catch (std::exception & err) {
	  edm::LogInfo("RecoVertex/PrimaryVertexProducerAlgorithm") 
	    << "Exception while fitting vertex: " 
	    << "\n" << err.what() << "\n";
	}

      }else if((*iclus).size()>1){
	if (fVerbose){cout <<  "unconstrained fit with "<< (*iclus).size() << " tracks"  << endl;}
	try {
	  TransientVertex v = theFitter->vertex(*iclus); 
	  if (fVerbose){
	    if (v.isValid()) cout << "x,y,z=" << v.position().x() <<" " << v.position().y() << " " <<  v.position().z() << endl;
	      else cout <<"Invalid fitted vertex\n";
	  }
	  if (v.isValid()) pvCand.push_back(v);
	}  catch (std::exception & err) {
	  edm::LogInfo("RecoVertex/PrimaryVertexProducerAlgorithm") 
	    << "Exception while fitting vertex: " 
	    << "\n" << err.what() << "\n";
	}
      }else if (fVerbose){
	cout <<  "cluster dropped" << endl;
      }



      nclu++;
    }// end of cluster loop


    if(fVerbose){
      cout << "PrimaryVertexProducerAlgorithm::vertices  candidates =" << pvCand.size() << endl;
    }

    // select vertices compatible with beam
    int npv=0;
    for (vector<TransientVertex>::const_iterator ipv = pvCand.begin();
	 ipv != pvCand.end(); ipv++) {
      if(fVerbose){
	cout << "PrimaryVertexProducerAlgorithm::vertices cand " << npv++ << " sel=" <<
	  theVertexSelector(*ipv) << "   z="  << ipv->position().z() << endl;
      }
      if (theVertexSelector(*ipv)) pvs.push_back(*ipv);
    }

    // sort vertices by pt**2  vertex (aka signal vertex tagging)
    sort(pvs.begin(), pvs.end(), VertexHigherPtSquared());
  
  }  catch (std::exception & err) {
    edm::LogInfo("RecoVertex/PrimaryVertexProducerAlgorithm") 
      << "Exception while reconstructing tracker PV: " 
      << "\n" << err.what() << "\n";
  } 

  return pvs;
  
}
