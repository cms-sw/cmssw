#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"
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
		     const std::vector<TransientTrack> & tracks, float chi2) :
    theVertexState(pos, posError), theOriginalTracks(tracks),
    theChi2(chi2), theNDF(0), vertexValid(true), withPrior(false),
  theWeightMapIsAvailable(false), theCovMapAvailable(false), 
  withRefittedTracks(false)
{
  theNDF = 2.*theOriginalTracks.size() - 3.;
  //  addTracks(tracks);
}


TransientVertex::TransientVertex(const GlobalPoint & pos, const GlobalError & posError,
		     const std::vector<TransientTrack> & tracks, float chi2, float ndf) :
    theVertexState(pos, posError), theOriginalTracks(tracks),
    theChi2(chi2), theNDF(ndf), vertexValid(true), withPrior(false),
    theWeightMapIsAvailable(false), theCovMapAvailable(false), 
    withRefittedTracks(false)
{
  //  addTracks(tracks);
}


TransientVertex::TransientVertex(const GlobalPoint & priorPos, const GlobalError & priorErr,
		     const GlobalPoint & pos, const GlobalError & posError,
		     const std::vector<TransientTrack> & tracks, float chi2) :
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
		     const std::vector<TransientTrack> & tracks, float chi2, float ndf) :
    thePriorVertexState(priorPos, priorErr), theVertexState(pos, posError),
    theOriginalTracks(tracks), theChi2(chi2), theNDF(ndf), vertexValid(true),
    withPrior(true), theWeightMapIsAvailable(false), theCovMapAvailable(false),
    withRefittedTracks(false)
{
  //  addTracks(tracks);
}


TransientVertex::TransientVertex(const VertexState & state, 
		     const std::vector<TransientTrack> & tracks, float chi2) : 
  theVertexState(state), theOriginalTracks(tracks),
  theChi2(chi2), theNDF(0), vertexValid(true), withPrior(false),
  theWeightMapIsAvailable(false), theCovMapAvailable(false), 
  withRefittedTracks(false)
{
  theNDF = 2.*theOriginalTracks.size() - 3.;
}


TransientVertex::TransientVertex(const VertexState & state, 
		     const std::vector<TransientTrack> & tracks, float chi2, float ndf) : 
    theVertexState(state), theOriginalTracks(tracks),
    theChi2(chi2), theNDF(ndf), vertexValid(true), withPrior(false),
    theWeightMapIsAvailable(false), theCovMapAvailable(false), 
    withRefittedTracks(false) 
{
  //  addTracks(tracks);
}


TransientVertex::TransientVertex(const VertexState & prior, 
				     const VertexState & state, 
				     const std::vector<TransientTrack> & tracks, 
				     float chi2) :
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
		     const std::vector<TransientTrack> & tracks, 
		     float chi2, float ndf) :
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

void TransientVertex::refittedTracks(
	const std::vector<reco::TransientTrack> & refittedTracks)
{
  if (refittedTracks.empty())
    throw VertexException("TransientVertex::refittedTracks: No refitted tracks stored in input container");
  theRefittedTracks = refittedTracks;
  withRefittedTracks = true;
}


void TransientVertex::tkToTkCovariance(const TTtoTTmap &covMap)
{
  theCovMap = covMap;
  theCovMapAvailable = true;
}

float TransientVertex::trackWeight(const TransientTrack & track) const {
  if (!theWeightMapIsAvailable) {
    std::vector<TransientTrack>::const_iterator foundTrack = find(theOriginalTracks.begin(), 
    		theOriginalTracks.end(), track);
    return ((foundTrack != theOriginalTracks.end()) ? 1. : 0.);
  }
  TransientTrackToFloatMap::const_iterator it = theWeightMap.find(track);
  if (it !=  theWeightMap.end()) {
    return(it->second);
  }
  return 0.;

}

AlgebraicMatrix33
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


TransientTrack TransientVertex::originalTrack(const TransientTrack & refTrack) const
{
  if (theRefittedTracks.empty())
	throw VertexException("No refitted tracks stored in vertex");
  std::vector<TransientTrack>::const_iterator it =
	find(theRefittedTracks.begin(), theRefittedTracks.end(), refTrack);
  if (it==theRefittedTracks.end())
	throw VertexException("Refitted track not found in list");
  size_t pos = it - theRefittedTracks.begin();
  return theOriginalTracks[pos];
}

TransientTrack TransientVertex::refittedTrack(const TransientTrack & track) const
{
  if (theRefittedTracks.empty())
	throw VertexException("No refitted tracks stored in vertex");
  std::vector<TransientTrack>::const_iterator it =
	find(theOriginalTracks.begin(), theOriginalTracks.end(), track);
  if (it==theOriginalTracks.end())
	throw VertexException("Track not found in list");
  size_t pos = it - theOriginalTracks.begin();
  return theRefittedTracks[pos];
}

TransientVertex::operator reco::Vertex() const
{
   //If the vertex is invalid, return an invalid TV !
  if (!isValid()) return Vertex();

  Vertex vertex(Vertex::Point(theVertexState.position()),
// 	RecoVertex::convertError(theVertexState.error()), 
	theVertexState.error().matrix_new(), 
	totalChiSquared(), degreesOfFreedom(), theOriginalTracks.size() );
  for (std::vector<TransientTrack>::const_iterator i = theOriginalTracks.begin();
       i != theOriginalTracks.end(); ++i) {
//     const TrackTransientTrack* ttt = dynamic_cast<const TrackTransientTrack*>((*i).basicTransientTrack());
//     if ((ttt!=0) && (ttt->persistentTrackRef().isNonnull()))
//     {
//       TrackRef tr = ttt->persistentTrackRef();
//       TrackBaseRef tbr(tr);
      if (withRefittedTracks) {
        
	vertex.add((*i).trackBaseRef(), refittedTrack(*i).track(), trackWeight ( *i ) );
      } else { 
	vertex.add((*i).trackBaseRef(), trackWeight ( *i ) );
      }
    //}
  }
  return vertex;
}
