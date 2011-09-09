#ifndef VertexTrack_H
#define VertexTrack_H

#include "RecoVertex/VertexPrimitives/interface/LinearizedTrackState.h"
#include "RecoVertex/VertexPrimitives/interface/VertexState.h"
#include "RecoVertex/VertexPrimitives/interface/RefittedTrackState.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"
#include "Math/SMatrix.h"
#include "DataFormats/CLHEP/interface/Migration.h"

/** Track information relative to a track-to-vertex association. 
 *  The track weight corresponds to the distance 
 *  of the track to the seed position. 
 */

template <unsigned int N>
class VertexTrack : public ReferenceCounted {

public:

  typedef ROOT::Math::SVector<double,N> AlgebraicVectorN;
  typedef ROOT::Math::SMatrix<double,N-2,N-2,ROOT::Math::MatRepStd<double,N-2,N-2> > AlgebraicMatrixMM;
  typedef ROOT::Math::SMatrix<double,3,N-2,ROOT::Math::MatRepStd<double,3,N-2> > AlgebraicMatrix3M;
  typedef ROOT::Math::SMatrix<double,N+1,N+1,ROOT::Math::MatRepSym<double,N+1> > AlgebraicSymMatrixOO;

  //typedef ReferenceCountingPointer<VertexTrack<N> > RefCountedVertexTrack;
  typedef ReferenceCountingPointer<LinearizedTrackState<N> > RefCountedLinearizedTrackState;
  typedef ReferenceCountingPointer<RefittedTrackState<N> > RefCountedRefittedTrackState;

  /** Constructor with the linearized track data, vertex seed and weight
   */     
  VertexTrack(const RefCountedLinearizedTrackState lt, 
	      const VertexState v, 
	      float weight);

  /** Constructor with the linearized track data, vertex seed and weight
   *  and state at vertex, constrained by vertex
   */     
  VertexTrack(const RefCountedLinearizedTrackState lt, 
	      const VertexState v, 
	      float weight, const RefCountedRefittedTrackState & refittedState,
	      float smoothedChi2);

  /** Constructor with the linearized track data, vertex seed and weight
   *  and state and covariance at vertex, constrained by vertex
   */     
  VertexTrack(const RefCountedLinearizedTrackState lt, 
	      const VertexState v, 
	      float weight, const RefCountedRefittedTrackState & refittedState,
	      float smoothedChi2, const AlgebraicSymMatrixOO & fullCov);

  /** Access methods
   */ 
  RefCountedLinearizedTrackState linearizedTrack() const { return theLinTrack; }
  VertexState vertexState() const { return theVertexState; }
  float weight() const { return theWeight; }
  bool refittedStateAvailable() const { return stAvailable; }
  bool tkToVertexCovarianceAvailable() const { return covAvailable; }
  bool fullCovarianceAvailable() const { return covAvailable; }

  /**
   * The smoother track-chi2 (can be used to test the track-vertex compatibility).
   * Its value has a meaning only if the smoother has been run after the vertex
   * fit (track-refit) . Otherwise, the value returned is -1.
   */

  float smoothedChi2() const { return smoothedChi2_; }


  /** Track state with vertex constraint
   */
  RefCountedRefittedTrackState refittedState() const { 
    if (!refittedStateAvailable()) { 
      throw VertexException("VertexTrack::refitted state not available"); 
    }
    return theRefittedState;
  }

//   /** Track to vertex covariance 
//    */   
//   AlgebraicMatrix3M tkToVtxCovariance() const;

  /** Track to vertex covariance 
   */   
  AlgebraicSymMatrixOO fullCovariance() const {
    if (!tkToVertexCovarianceAvailable()) {
      throw VertexException("VertexTrack::track to vertex covariance not available"); 
    }
    return fullCovariance_;
  }

  /** Equality for finding a VertexTrack in a container
   *  Compares the RecTrack addresses
   */
  bool operator==(const VertexTrack<N> & data) const
  {
    return ((*data.linearizedTrack()) == (*linearizedTrack()));
  }

  /** Method helping Kalman vertex fit
   */
  AlgebraicVectorN refittedParamFromEquation() const;
 

private:

  RefCountedLinearizedTrackState theLinTrack;
  VertexState theVertexState;
  float theWeight;
  bool stAvailable;
  bool covAvailable;
  RefCountedRefittedTrackState theRefittedState;
  AlgebraicSymMatrixOO  fullCovariance_;
  ROOT::Math::SMatrix<double,6,6,ROOT::Math::MatRepSym<double,6> > b6;
  ROOT::Math::SMatrix<double,7,7,ROOT::Math::MatRepSym<double,7> > b7;
  float smoothedChi2_;
};

template <unsigned int N>
class VertexTrackEqual {
  public:
    typedef ReferenceCountingPointer<VertexTrack<N> > RefCountedVertexTrack;
    VertexTrackEqual( const RefCountedVertexTrack & t) : track_( t ) { }
    bool operator()( const RefCountedVertexTrack & t ) const { return t->operator==(*track_);}
  private:
    const RefCountedVertexTrack & track_;
};

#endif
