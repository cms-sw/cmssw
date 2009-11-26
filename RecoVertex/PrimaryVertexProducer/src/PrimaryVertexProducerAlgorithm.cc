#include "RecoVertex/PrimaryVertexProducer/interface/PrimaryVertexProducerAlgorithm.h"
#include "RecoVertex/PrimaryVertexProducer/interface/VertexHigherPtSquared.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "RecoVertex/AdaptiveVertexFit/interface/AdaptiveVertexFitter.h"
#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"
#include <algorithm>

using namespace reco;
//#define PV_EXTRA

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
  edm::LogInfo("RecoVertex/PrimaryVertexProducerAlgorithm") 
    << "PVSelParameters::maxDistanceToBeam = " 
    << conf.getParameter<edm::ParameterSet>("PVSelParameters").getParameter<double>("maxDistanceToBeam") << "\n";

  fUseBeamConstraint = conf.getParameter<bool>("useBeamConstraint");
  fVerbose           = conf.getUntrackedParameter<bool>("verbose", false);
  std::string algorithm = conf.getParameter<std::string>("algorithm");

  fapply_finder = false;
  if (algorithm == "TrimmedKalmanFinder") {
    fapply_finder = true;
    theFinder.setParameters(conf.getParameter<edm::ParameterSet>("VtxFinderParameters"));
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

  edm::LogInfo("RecoVertex/PrimaryVertexProducerAlgorithm") 
    << "PV producer algorithm initialization: done" << "\n";

}


PrimaryVertexProducerAlgorithm::~PrimaryVertexProducerAlgorithm() 
{
  if (theFitter) delete theFitter;
}


//
// member functions
//
vector<TransientVertex> 
PrimaryVertexProducerAlgorithm::vertices(const vector<reco::TransientTrack> & tracks) const
{

   throw VertexException("PrimaryVertexProducerAlgorithm: cannot make a Primary Vertex without a beam spot constraint " );

  /*  std::cout<< "PrimaryVertexProducer::vertices> Obsolete function, using dummy beamspot " << std::endl;
    reco::BeamSpot dummyBeamSpot;
    dummyBeamSpot.dummy();
    return vertices(tracks,dummyBeamSpot); */
   return vector<TransientVertex>();
}


vector<TransientVertex> 
PrimaryVertexProducerAlgorithm::vertices(const vector<reco::TransientTrack> & tracks,
					 const reco::BeamSpot & beamSpot) const
{
  bool validBS = true;
  VertexState beamVertexState(beamSpot);
  if ( (beamVertexState.error().cxx() <= 0.) || 
  	(beamVertexState.error().cyy() <= 0.) ||
  	(beamVertexState.error().czz() <= 0.) ) {
    validBS = false;
    edm::LogError("UnusableBeamSpot") << "Beamspot with invalid errors "<<beamVertexState.error().matrix();
  }

  if ( fapply_finder) {
        return theFinder.vertices( tracks );
  }
  vector<TransientVertex> pvs;


  // select tracks
  vector<TransientTrack> seltks;

  if (validBS){
    for (vector<reco::TransientTrack>::const_iterator itk = tracks.begin();
	 itk != tracks.end(); itk++) {
      if (theTrackFilter(*itk)) seltks.push_back(*itk);
    }
  } else {
    seltks = tracks;
  }

  if(fVerbose){
    cout << "PrimaryVertexProducerAlgorithm::vertices  selected tracks=" << seltks.size() << endl;
  }

#ifdef PV_EXTRA
  vector<double> clusterz, selector, cputime;  double tfit=0;
#endif

  // clusterize tracks in Z
  vector< vector<reco::TransientTrack> > clusters = 
    theTrackClusterizer.clusterize(seltks);


  // look for primary vertices in each cluster
  vector<TransientVertex> pvCand;
  int nclu=0;
  for (vector< vector<reco::TransientTrack> >::const_iterator iclus
	 = clusters.begin(); iclus != clusters.end(); iclus++) {


    if(fVerbose){
      cout << "PrimaryVertexProducerAlgorithm::vertices  cluster=" 
	   << nclu << "  tracks=" << (*iclus).size() << endl;
    }

    TransientVertex v;
    if( fUseBeamConstraint && validBS &&((*iclus).size()>1) ){
      if (fVerbose){cout <<  "constrained fit with "<< (*iclus).size() << " tracks"  << endl;}
      v = theFitter->vertex(*iclus, beamSpot);
      if (v.isValid()) pvCand.push_back(v);

      if (fVerbose){
	cout << "beamspot   x="<< beamVertexState.position().x() 
	     << " y=" << beamVertexState.position().y()
	     << " z=" << beamVertexState.position().z()
	     << " dx=" << sqrt(beamVertexState.error().cxx())
	     << " dy=" << sqrt(beamVertexState.error().cyy())
	     << " dz=" << sqrt(beamVertexState.error().czz())
	     << std::endl;
	if (v.isValid()) cout << "x,y,z=" << v.position().x() <<" " << v.position().y() << " " <<  v.position().z() << endl;
	else cout <<"Invalid fitted vertex\n";
      }

    }else if((*iclus).size()>1){
      if (fVerbose){cout <<  "unconstrained fit with "<< (*iclus).size() << " tracks"  << endl;}

      v = theFitter->vertex(*iclus); 
      if (v.isValid()) pvCand.push_back(v);

      if (fVerbose){
	if (v.isValid()) cout << "x,y,z=" << v.position().x() <<" " << v.position().y() << " " <<  v.position().z() << endl;
	else cout <<"Invalid fitted vertex\n";
      }

    }else if (fVerbose){
      cout <<  "cluster dropped" << endl;
    }

#ifdef PV_EXTRA
    cputime.push_back(tfit);
    if(v.isValid()){
      clusterz.push_back(v.position().z());
      if(validBS){
	selector.push_back(theVertexSelector(v,beamVertexState)? 1. : .0);
      }else{
	selector.push_back(-1);
      }
    }else{
      clusterz.push_back(0);
      selector.push_back(-2);
    }
#endif

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
	(validBS && theVertexSelector(*ipv,beamVertexState)) << "   z="  << ipv->position().z() << endl;
    }
    if (!validBS || theVertexSelector(*ipv,beamVertexState)) pvs.push_back(*ipv);
  }

  // sort vertices by pt**2  vertex (aka signal vertex tagging)
  sort(pvs.begin(), pvs.end(), VertexHigherPtSquared());
  

#ifdef PV_EXTRA
  // attach clusters as if they were vertices for test purposes
  // first "vertex" has all selected tracks
  GlobalError dummyError; // default constructor makes a zero matrix
  GlobalPoint pos(beamVertexState.position().x(),beamVertexState.position().y(), beamVertexState.position().z()); 
  pvs.push_back(TransientVertex(pos,dummyError, seltks, 0.,-1));

  int iclu=0;
  for (vector< vector<reco::TransientTrack> >::const_iterator iclus
	 = clusters.begin(); iclus != clusters.end(); iclus++) {
    GlobalPoint pos(selector[iclu],cputime[iclu],clusterz[iclu]); 
    // selector: 1=accepted, 0=rejected, -1=no beamspot, -2=invalid (fit failed)
    iclu++;
    TransientVertex tv(pos, dummyError, *iclus, 0., -2);
    pvs.push_back(tv);
  }
#endif

  return pvs;
  
}
