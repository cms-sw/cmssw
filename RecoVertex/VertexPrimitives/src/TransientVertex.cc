#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"
// #include "CommonReco/PatternTools/interface/RefittedRecTrack.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertError.h"
#include "TrackingTools/TransientTrack/interface/TrackTransientTrack.h"
#include <algorithm>

using namespace std;
using namespace reco;

TransientVertex::TransientVertex() : theVertexState(), theOriginalTracks(),
  theChi2(0), theNDF(0), vertexValid(false), withPrior(false),
  theWeightMapIsAvailable(false), theCovMapAvailable(false), 
  withRefittedTracks(false)
{}


TransientVertex::TransientVertex(const GlobalPoint & pos, const GlobalError & posError,
		     const vector<TransientTrack> & tracks, float chi2) :
//     Vertex(Vertex::Point(pos), RecoVertex::convertError(posError), 
// 	    chi2, (2.*theOriginalTracks.size() - 3.), tracks.size() ),
    theVertexState(pos, posError), theOriginalTracks(tracks),
    theChi2(chi2), theNDF(0), vertexValid(true), withPrior(false),
  theWeightMapIsAvailable(false), theCovMapAvailable(false), 
  withRefittedTracks(false)
{
  theNDF = 2.*theOriginalTracks.size() - 3.;
  //  addTracks(tracks);
}


TransientVertex::TransientVertex(const GlobalPoint & pos, const GlobalError & posError,
		     const vector<TransientTrack> & tracks, float chi2, float ndf) :
//     Vertex( Vertex::Point(pos), RecoVertex::convertError(posError), chi2, ndf, tracks.size() ),
    theVertexState(pos, posError), theOriginalTracks(tracks),
    theChi2(chi2), theNDF(ndf), vertexValid(true), withPrior(false),
    theWeightMapIsAvailable(false), theCovMapAvailable(false), 
    withRefittedTracks(false)
{
  //  addTracks(tracks);
}


TransientVertex::TransientVertex(const GlobalPoint & priorPos, const GlobalError & priorErr,
		     const GlobalPoint & pos, const GlobalError & posError,
		     const vector<TransientTrack> & tracks, float chi2) :
//     Vertex( Vertex::Point(pos), RecoVertex::convertError(posError), 
// 	    chi2, (2.*theOriginalTracks.size() - 3.), tracks.size() ),
    thePriorVertexState(priorPos, priorErr), theVertexState(pos, posError),
    theOriginalTracks(tracks), theChi2(chi2), theNDF(0), vertexValid(true),
    withPrior(true), theWeightMapIsAvailable(false), theCovMapAvailable(false),
    withRefittedTracks(false)
{
  theNDF = 2.*theOriginalTracks.size();
  //  addTracks(tracks);
}


TransientVertex::TransientVertex(const GlobalPoint & priorPos, const GlobalError & priorErr,
		     const GlobalPoint & pos, const GlobalError & posError,
		     const vector<TransientTrack> & tracks, float chi2, float ndf) :
//     Vertex( Vertex::Point(pos), RecoVertex::convertError(posError), chi2, ndf, tracks.size() ),
    thePriorVertexState(priorPos, priorErr), theVertexState(pos, posError),
    theOriginalTracks(tracks), theChi2(chi2), theNDF(ndf), vertexValid(true),
    withPrior(true), theWeightMapIsAvailable(false), theCovMapAvailable(false),
    withRefittedTracks(false)
{
  //  addTracks(tracks);
}


TransientVertex::TransientVertex(const VertexState & state, 
		     const vector<TransientTrack> & tracks, float chi2) : 
//     Vertex( Vertex::Point(state.position()), RecoVertex::convertError(state.error()), 
// 	    chi2, (2.*theOriginalTracks.size() - 3.), tracks.size() ),
  theVertexState(state), theOriginalTracks(tracks),
  theChi2(chi2), theNDF(0), vertexValid(true), withPrior(false),
  theWeightMapIsAvailable(false), theCovMapAvailable(false), 
  withRefittedTracks(false)
{
  theNDF = 2.*theOriginalTracks.size() - 3.;
}


TransientVertex::TransientVertex(const VertexState & state, 
		     const vector<TransientTrack> & tracks, float chi2, float ndf) : 
//     Vertex( Vertex::Point(state.position()), RecoVertex::convertError(state.error()), chi2, ndf, tracks.size() ),
    theVertexState(state), theOriginalTracks(tracks),
    theChi2(chi2), theNDF(ndf), vertexValid(true), withPrior(false),
    theWeightMapIsAvailable(false), theCovMapAvailable(false), 
    withRefittedTracks(false) 
{
  //  addTracks(tracks);
}


TransientVertex::TransientVertex(const VertexState & prior, 
				     const VertexState & state, 
				     const vector<TransientTrack> & tracks, 
				     float chi2) :
//     Vertex( Vertex::Point(state.position()), RecoVertex::convertError(state.error()), 
// 	    chi2, (2.*theOriginalTracks.size() - 3.), tracks.size() ),
    thePriorVertexState(prior), theVertexState(state),
    theOriginalTracks(tracks), theChi2(chi2), theNDF(0), vertexValid(true),
    withPrior(true), theWeightMapIsAvailable(false), theCovMapAvailable(false),
    withRefittedTracks(false)
{
  theNDF = 2.*theOriginalTracks.size();
  //  addTracks(tracks);
}


TransientVertex::TransientVertex(const VertexState & prior, 
		     const VertexState & state, 
		     const vector<TransientTrack> & tracks, 
		     float chi2, float ndf) :
//     Vertex( Vertex::Point(state.position()), RecoVertex::convertError(state.error()), chi2, ndf, tracks.size() ),
    thePriorVertexState(prior), theVertexState(state),
    theOriginalTracks(tracks), theChi2(chi2), theNDF(ndf), vertexValid(true),
    withPrior(true), theWeightMapIsAvailable(false),
    theCovMapAvailable(false), withRefittedTracks(false)
{
  //  addTracks(tracks);
}

void TransientVertex::weightMap(const TransientTrackToFloatMap & theMap)
{
  theWeightMap = theMap;
  theWeightMapIsAvailable = true;
//   removeTracks(); // remove trackrefs from reco::Vertex
//   addTracks( theOriginalTracks );
}

void TransientVertex::tkToTkCovariance(const TTtoTTmap covMap)
{
  theCovMap = covMap;
  withPrior = true;
}

float TransientVertex::trackWeight(const TransientTrack & track) const {
  if (!theWeightMapIsAvailable) {
    vector<TransientTrack>::const_iterator foundTrack = find(theOriginalTracks.begin(), 
    		theOriginalTracks.end(), track);
    return ((foundTrack != theOriginalTracks.end()) ? 1. : 0.);
  }
  TransientTrackToFloatMap::const_iterator it = theWeightMap.find(track);
  if (it !=  theWeightMap.end()) {
    return(it->second);
  }
  return 0.;

}

AlgebraicMatrix
TransientVertex::tkToTkCovariance(const TransientTrack& t1, const TransientTrack& t2) const
{
  if (!theCovMapAvailable) {
   throw VertexException("TransientVertex::Track-to-track covariance matrices not available");
  }
  const TransientTrack* tr1;
  const TransientTrack* tr2;
  if (t1<t2) {
    tr1 = &t1;
    tr2 = &t2;
  } else {
    tr1 = &t2;
    tr2 = &t1;
  }
  TTtoTTmap::const_iterator it = theCovMap.find(*tr1);
  if (it !=  theCovMap.end()) {
    const TTmap & tm = it->second;
    TTmap::const_iterator nit = tm.find(*tr2);
    if (nit != tm.end()) {
      return( nit->second);
    }
    else {
      throw VertexException("TransientVertex::requested Track-to-track covariance matrix does not exist");
    }
  }
  else {
    throw VertexException("TransientVertex::requested Track-to-track covariance matrix does not exist");
  }
}

TransientVertex::operator reco::Vertex() const
{
  Vertex vertex(Vertex::Point(theVertexState.position()),
	RecoVertex::convertError(theVertexState.error()), 
	totalChiSquared(), degreesOfFreedom(), theOriginalTracks.size() );
  for (vector<TransientTrack>::const_iterator i = theOriginalTracks.begin();
       i != theOriginalTracks.end(); ++i) {
    const TrackTransientTrack* ttt = dynamic_cast<const TrackTransientTrack*>((*i).basicTransientTrack());
    if ((ttt!=0) && (ttt->persistentTrackRef().isNonnull()))
    {
      vertex.add(ttt->persistentTrackRef());// FIXME: add wieghts: , trackWeight ( *i ) );
    }
  }
  return vertex;
}
