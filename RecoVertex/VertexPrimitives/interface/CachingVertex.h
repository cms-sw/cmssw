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
  typedef ROOT::Math::SMatrix<double, N, N, ROOT::Math::MatRepSym<double, N> > AlgebraicSymMatrixNN;
  typedef ROOT::Math::SMatrix<double, N - 2, N - 2, ROOT::Math::MatRepStd<double, N - 2, N - 2> > AlgebraicMatrixMM;
  typedef std::map<RefCountedVertexTrack, AlgebraicMatrixMM> TrackMap;
  typedef std::map<RefCountedVertexTrack, TrackMap> TrackToTrackMap;

  /** Constructors
   */
  // no time
  CachingVertex(const GlobalPoint &pos,
                const GlobalError &posErr,
                const std::vector<RefCountedVertexTrack> &tks,
                float totalChiSq);

  CachingVertex(const GlobalPoint &pos,
                const GlobalWeight &posWeight,
                const std::vector<RefCountedVertexTrack> &tks,
                float totalChiSq);

  CachingVertex(const AlgebraicVector3 &weightTimesPosition,
                const GlobalWeight &posWeight,
                const std::vector<RefCountedVertexTrack> &tks,
                float totalChiSq);

  CachingVertex(const GlobalPoint &priorPos,
                const GlobalError &priorErr,
                const AlgebraicVector3 &weightTimesPosition,
                const GlobalWeight &posWeight,
                const std::vector<RefCountedVertexTrack> &tks,
                float totalChiSq);

  // with time (tracks must have time as well)
  CachingVertex(const GlobalPoint &pos,
                const double time,
                const GlobalError &posTimeErr,
                const std::vector<RefCountedVertexTrack> &tks,
                float totalChiSq);

  CachingVertex(const GlobalPoint &pos,
                const double time,
                const GlobalWeight &posTimeWeight,
                const std::vector<RefCountedVertexTrack> &tks,
                float totalChiSq);

  CachingVertex(const AlgebraicVector4 &weightTimesPosition,
                const GlobalWeight &posTimeWeight,
                const std::vector<RefCountedVertexTrack> &tks,
                float totalChiSq);

  CachingVertex(const GlobalPoint &priorPos,
                const GlobalError &priorErr,
                const AlgebraicVector4 &weightTimesPosition,
                const GlobalWeight &posWeight,
                const std::vector<RefCountedVertexTrack> &tks,
                float totalChiSq);

  // either time or no time (depends on if the tracks/vertex states have times)
  CachingVertex(const GlobalPoint &priorPos,
                const GlobalError &priorErr,
                const GlobalPoint &pos,
                const GlobalError &posErr,
                const std::vector<RefCountedVertexTrack> &tks,
                float totalChiSq);

  CachingVertex(const GlobalPoint &priorPos,
                const double priorTime,
                const GlobalError &priorErr,
                const GlobalPoint &pos,
                const double time,
                const GlobalError &posErr,
                const std::vector<RefCountedVertexTrack> &tks,
                float totalChiSq);

  CachingVertex(const GlobalPoint &priorPos,
                const GlobalError &priorErr,
                const GlobalPoint &pos,
                const GlobalWeight &posWeight,
                const std::vector<RefCountedVertexTrack> &tks,
                float totalChiSq);

  CachingVertex(const GlobalPoint &priorPos,
                const double priorTime,
                const GlobalError &priorErr,
                const GlobalPoint &pos,
                const double time,
                const GlobalWeight &posWeight,
                const std::vector<RefCountedVertexTrack> &tks,
                float totalChiSq);

  CachingVertex(const VertexState &aVertexState, const std::vector<RefCountedVertexTrack> &tks, float totalChiSq);

  CachingVertex(const VertexState &priorVertexState,
                const VertexState &aVertexState,
                const std::vector<RefCountedVertexTrack> &tks,
                float totalChiSq);

  CachingVertex(const VertexState &aVertexState,
                const std::vector<RefCountedVertexTrack> &tks,
                float totalChiSq,
                const TrackToTrackMap &covMap);

  CachingVertex(const VertexState &priorVertexState,
                const VertexState &aVertexState,
                const std::vector<RefCountedVertexTrack> &tks,
                float totalChiSq,
                const TrackToTrackMap &covMap);

  /** Constructor for invalid CachingVertex
   */
  CachingVertex();

  /** Access methods
   */
  VertexState const &vertexState() const { return theVertexState; }
  VertexState const &priorVertexState() const { return thePriorVertexState; }
  GlobalPoint position() const;
  double time() const;
  GlobalError error() const;
  GlobalError error4D() const;
  GlobalWeight weight() const;
  GlobalWeight weight4D() const;
  AlgebraicVector3 weightTimesPosition() const;
  AlgebraicVector4 weightTimesPosition4D() const;
  std::vector<RefCountedVertexTrack> tracks() const { return theTracks; }
  const std::vector<RefCountedVertexTrack> &tracksRef() const { return theTracks; }
  GlobalPoint priorPosition() const { return priorVertexState().position(); }
  double priorTime() const { return priorVertexState().time(); }
  GlobalError priorError() const { return priorVertexState().error(); }
  GlobalError priorError4D() const { return priorVertexState().error4D(); }
  bool hasPrior() const { return withPrior; }
  bool isValid() const { return theValid; }
  bool is4D() const { return vertexIs4D; }

  /** Chi2, degrees of freedom. The latter may not be integer. 
   */
  float totalChiSquared() const { return theChiSquared; }
  float degreesOfFreedom() const;

  /** Track to track covariance
   */
  AlgebraicMatrixMM tkToTkCovariance(const RefCountedVertexTrack t1, const RefCountedVertexTrack t2) const;
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
  bool vertexIs4D;
};

#endif
