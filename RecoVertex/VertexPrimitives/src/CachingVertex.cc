#include "RecoVertex/VertexPrimitives/interface/CachingVertex.h"
#include "RecoVertex/VertexPrimitives/interface/LinearizedTrackState.h"
#include "RecoVertex/VertexPrimitives/interface/RefCountedLinearizedTrackState.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoVertex/VertexPrimitives/interface/TTtoTTmap.h"
#include <map>

//to be removed
CachingVertex::CachingVertex(const GlobalPoint & pos, 
			     const GlobalError & posErr, 
			     const vector<RefCountedVertexTrack> & tks, 
			     float totalChiSq) 
  : theVertexState(pos, posErr),
    theChiSquared(totalChiSq), theNDF(0), theNDFAvailable(false), 
    theTracks(tks), theCovMapAvailable(false), withPrior(false), 
    theValid(true)

{}


//to be removed
CachingVertex::CachingVertex(const GlobalPoint & pos, 
			     const GlobalWeight & posWeight, 
			     const vector<RefCountedVertexTrack> & tks, 
			     float totalChiSq) 
  : theVertexState(pos, posWeight),
    theChiSquared(totalChiSq), theNDF(0), theNDFAvailable(false), 
    theTracks(tks), theCovMapAvailable(false), withPrior(false), 
    theValid(true)
{}


//to be removed
CachingVertex::CachingVertex(const AlgebraicVector & weightTimesPosition, 
			     const GlobalWeight & posWeight, 
			     const vector<RefCountedVertexTrack> & tks, 
			     float totalChiSq)
  : theVertexState(weightTimesPosition, posWeight),
    theChiSquared(totalChiSq), theNDF(0), theNDFAvailable(false), 
    theTracks(tks), theCovMapAvailable(false), withPrior(false), 
    theValid(true)
{}

CachingVertex::CachingVertex(const VertexState & aVertexState, 
			     const vector<RefCountedVertexTrack> & tks, 
			     float totalChiSq)
  : theVertexState(aVertexState),
    theChiSquared(totalChiSq), theNDF(0), theNDFAvailable(false), 
    theTracks(tks), theCovMapAvailable(false), withPrior(false), 
    theValid(true)
{}


CachingVertex::CachingVertex(const VertexState & aVertexState,
			     const vector<RefCountedVertexTrack> & tks, 
			     float totalChiSq, 
			     const TrackToTrackMap & covMap)
  : theVertexState(aVertexState),
    theChiSquared(totalChiSq), theNDF(0), theNDFAvailable(false),
    theTracks(tks), theCovMap(covMap), theCovMapAvailable(true), 
    withPrior(false), theValid(true)
{
  if (theCovMap.empty()) theCovMapAvailable = false;
}

CachingVertex::CachingVertex(const VertexState & priorVertexState, 
			     const VertexState & aVertexState, 
			     const vector<RefCountedVertexTrack> & tks, 
			     float totalChiSq)
  : theVertexState(aVertexState), theChiSquared(totalChiSq),
    theNDF(0), theNDFAvailable(false), theTracks(tks),
    theCovMapAvailable(false), thePriorVertexState(priorVertexState),
    withPrior(true), theValid(true)
{}

//to be removed
CachingVertex::CachingVertex(const GlobalPoint & priorPos, 
			     const GlobalError & priorErr,
			     const GlobalPoint & pos, 
			     const GlobalError & posErr, 
			     const vector<RefCountedVertexTrack> & tks, 
			     float totalChiSq) 
  : theVertexState(pos, posErr),
    theChiSquared(totalChiSq), theNDF(0), theNDFAvailable(false), 
    theTracks(tks), theCovMapAvailable(false), 
    thePriorVertexState(priorPos, priorErr), withPrior(true), theValid(true)
{}


//to be removed
CachingVertex::CachingVertex(const GlobalPoint & priorPos,
			     const GlobalError & priorErr, 
			     const GlobalPoint & pos, 
			     const GlobalWeight & posWeight, 
			     const vector<RefCountedVertexTrack> & tks, 
			     float totalChiSq) 
  : theVertexState(pos, posWeight),
    theChiSquared(totalChiSq), theNDF(0), theNDFAvailable(false), 
    theTracks(tks), theCovMapAvailable(false), 
    thePriorVertexState(priorPos, priorErr), withPrior(true), theValid(true)
{}


//to be removed
CachingVertex::CachingVertex(const GlobalPoint & priorPos, 
			     const GlobalError & priorErr,
			     const AlgebraicVector & weightTimesPosition, 
			     const GlobalWeight & posWeight, 
			     const vector<RefCountedVertexTrack> & tks, 
			     float totalChiSq)
  : theVertexState(weightTimesPosition, posWeight),
    theChiSquared(totalChiSq), theNDF(0), theNDFAvailable(false), 
    theTracks(tks), theCovMapAvailable(false), 
    thePriorVertexState(priorPos, priorErr), withPrior(true), theValid(true)
{}


CachingVertex::CachingVertex(const VertexState & priorVertexState, 
  			     const VertexState & aVertexState,
			     const vector<RefCountedVertexTrack> & tks, 
			     float totalChiSq, 
			     const TrackToTrackMap & covMap)
  : theVertexState(aVertexState), theChiSquared(totalChiSq),
    theNDF(0), theNDFAvailable(false), theTracks(tks),
    theCovMap(covMap), theCovMapAvailable(true), 
    thePriorVertexState(priorVertexState), withPrior(true), theValid(true)
{
  if (theCovMap.empty()) theCovMapAvailable = false;
}

CachingVertex::CachingVertex() 
  : theChiSquared(-1), theNDF(0), theNDFAvailable(false), theTracks(),
    theCovMapAvailable(false), withPrior(false), 
    theValid(false)
{}


GlobalPoint CachingVertex::position() const 
{
  return theVertexState.position();
}


GlobalError CachingVertex::error() const 
{
  return theVertexState.error();
}


GlobalWeight CachingVertex::weight() const 
{
  return theVertexState.weight();
}


AlgebraicVector CachingVertex::weightTimesPosition() const 
{
  return theVertexState.weightTimesPosition();
}


float CachingVertex::degreesOfFreedom() const 
{
  if (!theNDFAvailable) computeNDF();
  return theNDF;
}


// RefCountedVertexSeed CachingVertex::seedWithoutTracks() const 
// {
//   return theVertexState.seedWithoutTracks();
// }


void CachingVertex::computeNDF() const 
{
  theNDF = 0;
  for (vector<RefCountedVertexTrack>::const_iterator itk = theTracks.begin(); 
       itk != theTracks.end(); ++itk) {
    theNDF += (**itk).weight(); // adds up weights
  }
  theNDF *= 2.; // times 2df for each track
  if (!withPrior) theNDF -= 3.; // 3 position coordinates fitted
  theNDFAvailable = true;
}


AlgebraicMatrix 
CachingVertex::tkToTkCovariance(const RefCountedVertexTrack t1, 
				const RefCountedVertexTrack t2) const
{
  if (!tkToTkCovarianceIsAvailable()) {
   throw VertexException("CachingVertex::TkTkCovariance requested before been calculated");
  } 
  else {
    RefCountedVertexTrack tr1;
    RefCountedVertexTrack tr2;
    if(t1 < t2) {
      tr1 = t1;    
      tr2 = t2;
    }
    else {
      tr1 = t2;    
      tr2 = t1;
    }
    TrackToTrackMap::const_iterator it = theCovMap.find(tr1);
    if (it !=  theCovMap.end()) {
      const TrackMap & tm = it->second;
      TrackMap::const_iterator nit = tm.find(tr2);
      if (nit != tm.end()) {
	return( nit->second);
      }
      else {
	throw VertexException("CachingRecVertex::requested TkTkCovariance does not exist");
      }       
    }
    else {
      throw VertexException("CachingRecVertex::requested TkTkCovariance does not exist");
    }     
  }
}


CachingVertex::operator TransientVertex() const
{
  //If the vertex is invalid, return an invalid TV !
  if (!isValid()) return TransientVertex();

  typedef map<reco::TransientTrack, float> TransientTrackToFloatMap;

// Construct Track vector
  vector<reco::TransientTrack> ttVect;
  ttVect.reserve(theTracks.size());
  vector<reco::TransientTrack> refTTVect;
  TransientTrackToFloatMap theWeightMap;
  TTtoTTmap ttCovMap;
  float theMinWeight = 0.5;

  for (vector<RefCountedVertexTrack>::const_iterator i = theTracks.begin();
       i != theTracks.end(); ++i) {
    // discard tracks with too low weight
    if ((**i).weight() < theMinWeight) continue;

    reco::TransientTrack t1((**i).linearizedTrack()->track());
    ttVect.push_back(t1);
    //Fill in the weight map
    theWeightMap[t1] = (**i).weight();

    //Fill in the tk-to-tk covariance map
    if (theCovMapAvailable) {
      for (vector<RefCountedVertexTrack>::const_iterator j = (i+1);
	   j != theTracks.end(); ++j) {
	reco::TransientTrack t2((**j).linearizedTrack()->track());
	if (t1 < t2) {
	  ttCovMap[t1][t2] = tkToTkCovariance(*i, *j);
	} else {
	  ttCovMap[t2][t1] = tkToTkCovariance(*i, *j);
	}
      }
    }
    if ((**i).refittedStateAvailable()) {
      refTTVect.push_back( (**i).refittedState()->transientTrack()) ;
    }
  }
  TransientVertex tv;
  if (withPrior) {
    tv =  TransientVertex(priorVertexState(), vertexState(), ttVect, totalChiSquared(), degreesOfFreedom());
  } else {
    tv = TransientVertex(vertexState(), ttVect, totalChiSquared(), degreesOfFreedom());
  }
  tv.weightMap(theWeightMap);
  if (theCovMapAvailable) tv.tkToTkCovariance(ttCovMap);
  if (!refTTVect.empty()) tv.refittedTracks(refTTVect);
  return tv;
}
