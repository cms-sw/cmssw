#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"
// #include "CommonReco/PatternTools/interface/RefittedRecTrack.h"
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
    theVertexState(pos, posError), theOriginalTracks(tracks),
    theChi2(chi2), theNDF(0), vertexValid(true), withPrior(false),
  theWeightMapIsAvailable(false), theCovMapAvailable(false), 
  withRefittedTracks(false)
{
  theNDF = 2.*theOriginalTracks.size() - 3.;
}


TransientVertex::TransientVertex(const GlobalPoint & pos, const GlobalError & posError,
		     const vector<TransientTrack> & tracks, float chi2, float ndf) :
    theVertexState(pos, posError), theOriginalTracks(tracks),
    theChi2(chi2), theNDF(ndf), vertexValid(true), withPrior(false),
    theWeightMapIsAvailable(false), theCovMapAvailable(false), 
    withRefittedTracks(false) {}


TransientVertex::TransientVertex(const GlobalPoint & priorPos, const GlobalError & priorErr,
		     const GlobalPoint & pos, const GlobalError & posError,
		     const vector<TransientTrack> & tracks, float chi2) :
    thePriorVertexState(priorPos, priorErr), theVertexState(pos, posError),
    theOriginalTracks(tracks), theChi2(chi2), theNDF(0), vertexValid(true),
    withPrior(true), theWeightMapIsAvailable(false), theCovMapAvailable(false),
    withRefittedTracks(false)
{
  theNDF = 2.*theOriginalTracks.size();
}


TransientVertex::TransientVertex(const GlobalPoint & priorPos, const GlobalError & priorErr,
		     const GlobalPoint & pos, const GlobalError & posError,
		     const vector<TransientTrack> & tracks, float chi2, float ndf) :
    thePriorVertexState(priorPos, priorErr), theVertexState(pos, posError),
    theOriginalTracks(tracks), theChi2(chi2), theNDF(ndf), vertexValid(true),
    withPrior(true), theWeightMapIsAvailable(false), theCovMapAvailable(false),
    withRefittedTracks(false) {}


TransientVertex::TransientVertex(const VertexState & state, 
		     const vector<TransientTrack> & tracks, float chi2) : 
  theVertexState(state), theOriginalTracks(tracks),
  theChi2(chi2), theNDF(0), vertexValid(true), withPrior(false),
  theWeightMapIsAvailable(false), theCovMapAvailable(false), 
  withRefittedTracks(false)
{
  theNDF = 2.*theOriginalTracks.size() - 3.;
}


TransientVertex::TransientVertex(const VertexState & state, 
		     const vector<TransientTrack> & tracks, float chi2, float ndf) : 
    theVertexState(state), theOriginalTracks(tracks),
    theChi2(chi2), theNDF(ndf), vertexValid(true), withPrior(false),
    theWeightMapIsAvailable(false), theCovMapAvailable(false), 
    withRefittedTracks(false) 
{}


TransientVertex::TransientVertex(const VertexState & prior, 
				     const VertexState & state, 
				     const vector<TransientTrack> & tracks, 
				     float chi2) :
    thePriorVertexState(prior), theVertexState(state),
    theOriginalTracks(tracks), theChi2(chi2), theNDF(0), vertexValid(true),
    withPrior(true), theWeightMapIsAvailable(false), theCovMapAvailable(false),
    withRefittedTracks(false)
{
  theNDF = 2.*theOriginalTracks.size();
}


TransientVertex::TransientVertex(const VertexState & prior, 
		     const VertexState & state, 
		     const vector<TransientTrack> & tracks, 
		     float chi2, float ndf) :
    thePriorVertexState(prior), theVertexState(state),
    theOriginalTracks(tracks), theChi2(chi2), theNDF(ndf), vertexValid(true),
    withPrior(true), theWeightMapIsAvailable(false),
    theCovMapAvailable(false), withRefittedTracks(false)
{}

void TransientVertex::weightMap(const TransientTrackToFloatMap & theMap)
{
  theWeightMap = theMap;
  theWeightMapIsAvailable = true;
}

void TransientVertex::tkToTkCovariance(const TTtoTTmap covMap)
{
  theCovMap = covMap;
  withPrior = true;
}

// TransientVertex::TransientVertex(const VertexState & state,
// 		     const vector<TransientTrack> & tracks, float chi2, float ndf, 
// 		     const TransientTrackToFloatMap & weightMap) :
//     theVertexState(state), theOriginalTracks(tracks), theChi2(chi2),
//     theNDF(ndf), vertexValid(true), withPrior(false),
//     theWeightMapIsAvailable(true),  theCovMapAvailable(false),
//     withRefittedTracks(false), theWeightMap(weightMap) {}


// Vertex::TrackPtrContainer TransientVertex::tracks() const
// {
//   TrackPtrContainer tc;
// 
//   for (vector<RecTrack>::const_iterator i = theOriginalTracks.begin();
//        i != theOriginalTracks.end(); ++i) {
//     const Track * tk = &(*i);
//     tc.push_back(tk);
//   }
//   return tc;
// }

float TransientVertex::trackWeight(const TransientTrack track) const {
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
  TransientTrack* tr1;
  TransientTrack* tr2;
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
