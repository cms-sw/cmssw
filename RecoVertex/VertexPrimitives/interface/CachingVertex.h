#ifndef CachingVertex_H
#define CachingVertex_H

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalWeight.h"

#include "RecoVertex/VertexPrimitives/interface/VertexState.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"

#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
#include "RecoVertex/VertexPrimitives/interface/VertexTrack.h"

#include <vector>
#include <map>



/** Class for vertices fitted with Kalman and linear fit algorithms. 
 *  Provides access to temporary data to speed up the vertex update. 
 */


template <unsigned int N>
class CachingVertex {

public:

  typedef ReferenceCountingPointer<VertexTrack<N> > RefCountedVertexTrack;
  typedef ROOT::Math::SMatrix<double,N,N,ROOT::Math::MatRepSym<double,N> > AlgebraicSymMatrixNN;
  typedef ROOT::Math::SMatrix<double,N-2,N-2,ROOT::Math::MatRepStd<double,N-2,N-2> > AlgebraicMatrixMM;
  typedef std::map<RefCountedVertexTrack, AlgebraicMatrixMM > TrackMap;
  typedef std::map<RefCountedVertexTrack, TrackMap > TrackToTrackMap;

  /** Constructors
   */
  CachingVertex(const GlobalPoint & pos, const GlobalError & posErr, 
		const std::vector<RefCountedVertexTrack> & tks, float totalChiSq);

  CachingVertex(const GlobalPoint & pos, const GlobalWeight & posWeight, 
		const std::vector<RefCountedVertexTrack> & tks, float totalChiSq);

  CachingVertex(const AlgebraicVector3 & weightTimesPosition, 
		const GlobalWeight & posWeight, 
		const std::vector<RefCountedVertexTrack> & tks, 
		float totalChiSq);

  CachingVertex(const GlobalPoint & priorPos, const GlobalError & priorErr,
  		const GlobalPoint & pos, const GlobalError & posErr, 
		const std::vector<RefCountedVertexTrack> & tks, float totalChiSq);

  CachingVertex(const GlobalPoint & priorPos, const GlobalError & priorErr,
  		const GlobalPoint & pos, const GlobalWeight & posWeight, 
		const std::vector<RefCountedVertexTrack> & tks, float totalChiSq);

  CachingVertex(const GlobalPoint & priorPos, const GlobalError & priorErr,
  		const AlgebraicVector3 & weightTimesPosition, 
		const GlobalWeight & posWeight, 
		const std::vector<RefCountedVertexTrack> & tks, 
		float totalChiSq);


  CachingVertex(const VertexState & aVertexState, 
		const std::vector<RefCountedVertexTrack> & tks, float totalChiSq);

  CachingVertex(const VertexState & priorVertexState, 
  		const VertexState & aVertexState,
		const std::vector<RefCountedVertexTrack> & tks, float totalChiSq);

  CachingVertex(const VertexState & aVertexState,
		const std::vector<RefCountedVertexTrack> & tks, 
		float totalChiSq, const TrackToTrackMap & covMap);

  CachingVertex(const VertexState & priorVertexState, 
  		const VertexState & aVertexState,
		const std::vector<RefCountedVertexTrack> & tks, 
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
  AlgebraicVector3 weightTimesPosition() const;
  std::vector<RefCountedVertexTrack> tracks() const { return theTracks; }
  const std::vector<RefCountedVertexTrack> &tracksRef() const { return theTracks; }
  GlobalPoint priorPosition() const {return priorVertexState().position();}
  GlobalError priorError() const {return priorVertexState().error();}
  bool hasPrior() const {return withPrior;}
  bool isValid() const {return theValid;}

  /** Chi2, degrees of freedom. The latter may not be integer. 
   */
  float totalChiSquared() const { return theChiSquared; }
  float degreesOfFreedom() const;

  /** Track to track covariance
   */
  AlgebraicMatrixMM tkToTkCovariance(const RefCountedVertexTrack t1, 
				   const RefCountedVertexTrack t2) const;
  bool tkToTkCovarianceIsAvailable() const { return theCovMapAvailable; }

  operator TransientVertex() const;

private:

  void computeNDF() const;

  mutable VertexState theVertexState;
  float theChiSquared;
  mutable float theNDF;
  mutable bool theNDFAvailable;
  std::vector<RefCountedVertexTrack> theTracks;
  TrackToTrackMap theCovMap;
  bool theCovMapAvailable;
  mutable VertexState thePriorVertexState;
  bool withPrior;

  bool theValid;
};


#endif
