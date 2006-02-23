#include "RecoVertex/VertexPrimitives/interface/CachingVertex.h"
#include "RecoVertex/VertexPrimitives/interface/LinearizedTrackState.h"
#include "RecoVertex/VertexPrimitives/interface/RefCountedLinearizedTrackState.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"

//to be removed
CachingVertex::CachingVertex(const GlobalPoint & pos, 
			     const GlobalError & posErr, 
			     const vector<RefCountedVertexTrack> & tks, 
			     float totalChiSq) 
  : theVertexState(pos, posErr),
    theChiSquared(totalChiSq), theNDF(0), theNDFAvailable(false), 
    theTracks(tks), theCovMapAvailable(false), withPrior(false)
{}


//to be removed
CachingVertex::CachingVertex(const GlobalPoint & pos, 
			     const GlobalWeight & posWeight, 
			     const vector<RefCountedVertexTrack> & tks, 
			     float totalChiSq) 
  : theVertexState(pos, posWeight),
    theChiSquared(totalChiSq), theNDF(0), theNDFAvailable(false), 
    theTracks(tks), theCovMapAvailable(false), withPrior(false)
{}


//to be removed
CachingVertex::CachingVertex(const AlgebraicVector & weightTimesPosition, 
			     const GlobalWeight & posWeight, 
			     const vector<RefCountedVertexTrack> & tks, 
			     float totalChiSq)
  : theVertexState(weightTimesPosition, posWeight),
    theChiSquared(totalChiSq), theNDF(0), theNDFAvailable(false), 
    theTracks(tks), theCovMapAvailable(false), withPrior(false)
{}

CachingVertex::CachingVertex(const VertexState & aVertexState, 
			     const vector<RefCountedVertexTrack> & tks, 
			     float totalChiSq)
  : theVertexState(aVertexState),
    theChiSquared(totalChiSq), theNDF(0), theNDFAvailable(false), 
    theTracks(tks), theCovMapAvailable(false), withPrior(false)
{}


CachingVertex::CachingVertex(const VertexState & aVertexState,
			     const vector<RefCountedVertexTrack> & tks, 
			     float totalChiSq, 
			     const TrackToTrackMap & covMap)
  : theVertexState(aVertexState),
    theChiSquared(totalChiSq), theNDF(0), theNDFAvailable(false),
    theTracks(tks), theCovMap(covMap), theCovMapAvailable(true), 
    withPrior(false)
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
    withPrior(true)
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
    thePriorVertexState(priorPos, priorErr), withPrior(true)
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
    thePriorVertexState(priorPos, priorErr), withPrior(true)
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
    thePriorVertexState(priorPos, priorErr), withPrior(true)
{}


CachingVertex::CachingVertex(const VertexState & priorVertexState, 
  			     const VertexState & aVertexState,
			     const vector<RefCountedVertexTrack> & tks, 
			     float totalChiSq, 
			     const TrackToTrackMap & covMap)
  : theVertexState(aVertexState), theChiSquared(totalChiSq),
    theNDF(0), theNDFAvailable(false), theTracks(tks),
    theCovMap(covMap), theCovMapAvailable(true), 
    thePriorVertexState(priorVertexState), withPrior(true)
{
  if (theCovMap.empty()) theCovMapAvailable = false;
}


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
