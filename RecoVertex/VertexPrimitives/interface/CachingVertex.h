#ifndef CachingVertex_H
#define CachingVertex_H

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalWeight.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

#include "RecoVertex/VertexPrimitives/interface/RefCountedVertexTrack.h"
// #include "RecoVertex/VertexPrimitives/interface/RefCountedVertexSeed.h"

#include "RecoVertex/VertexPrimitives/interface/TrackMap.h"
#include "RecoVertex/VertexPrimitives/interface/TrackToTrackMap.h"
#include "RecoVertex/VertexPrimitives/interface/VertexState.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"

#include <vector>
#include <map>

using namespace std;

/** Class for vertices fitted with Kalman and linear fit algorithms. 
 *  Provides access to temporary data to speed up the vertex update. 
 */

class CachingVertex {

public:

  /** Constructors
   */
  CachingVertex(const GlobalPoint & pos, const GlobalError & posErr, 
		const vector<RefCountedVertexTrack> & tks, float totalChiSq);

  CachingVertex(const GlobalPoint & pos, const GlobalWeight & posWeight, 
		const vector<RefCountedVertexTrack> & tks, float totalChiSq);

  CachingVertex(const AlgebraicVector & weightTimesPosition, 
		const GlobalWeight & posWeight, 
		const vector<RefCountedVertexTrack> & tks, 
		float totalChiSq);

  CachingVertex(const GlobalPoint & priorPos, const GlobalError & priorErr,
  		const GlobalPoint & pos, const GlobalError & posErr, 
		const vector<RefCountedVertexTrack> & tks, float totalChiSq);

  CachingVertex(const GlobalPoint & priorPos, const GlobalError & priorErr,
  		const GlobalPoint & pos, const GlobalWeight & posWeight, 
		const vector<RefCountedVertexTrack> & tks, float totalChiSq);

  CachingVertex(const GlobalPoint & priorPos, const GlobalError & priorErr,
  		const AlgebraicVector & weightTimesPosition, 
		const GlobalWeight & posWeight, 
		const vector<RefCountedVertexTrack> & tks, 
		float totalChiSq);


  CachingVertex(const VertexState & aVertexState, 
		const vector<RefCountedVertexTrack> & tks, float totalChiSq);

  CachingVertex(const VertexState & priorVertexState, 
  		const VertexState & aVertexState,
		const vector<RefCountedVertexTrack> & tks, float totalChiSq);

  CachingVertex(const VertexState & aVertexState,
		const vector<RefCountedVertexTrack> & tks, 
		float totalChiSq, const TrackToTrackMap & covMap);

  CachingVertex(const VertexState & priorVertexState, 
  		const VertexState & aVertexState,
		const vector<RefCountedVertexTrack> & tks, 
		float totalChiSq, const TrackToTrackMap & covMap);

  /** Constructor for invalid CachingVertex
   */
  CachingVertex();

  /** Access methods
   */      
  VertexState vertexState() const {return theVertexState;}
  VertexState priorVertexState() const {return thePriorVertexState;}
  GlobalPoint position() const;
  GlobalError error() const;
  GlobalWeight weight() const;
  AlgebraicVector weightTimesPosition() const;
  vector<RefCountedVertexTrack> tracks() const { return theTracks; }
  GlobalPoint priorPosition() const {return priorVertexState().position();}
  GlobalError priorError() const {return priorVertexState().error();}
  bool hasPrior() const {return withPrior;}
  bool isValid() const {return theValid;}

  /** Chi2, degrees of freedom. The latter may not be integer. 
   */
  float totalChiSquared() const { return theChiSquared; }
  float degreesOfFreedom() const;

//   /** conversion to VertexSeed
//    */
//   RefCountedVertexSeed seedWithoutTracks() const;

  /** Track to track covariance
   */
  AlgebraicMatrix tkToTkCovariance(const RefCountedVertexTrack t1, 
				   const RefCountedVertexTrack t2) const;
  bool tkToTkCovarianceIsAvailable() const { return theCovMapAvailable; }

  operator TransientVertex() const;

private:

  void computeNDF() const;

  mutable VertexState theVertexState;
  float theChiSquared;
  mutable float theNDF;
  mutable bool theNDFAvailable;
  vector<RefCountedVertexTrack> theTracks;
  TrackToTrackMap theCovMap;
  bool theCovMapAvailable;
  mutable VertexState thePriorVertexState;
  bool withPrior;

  bool theValid;
};

#endif
