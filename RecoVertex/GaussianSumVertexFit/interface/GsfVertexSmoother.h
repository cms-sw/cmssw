#ifndef _GsfVertexSmoother_H_
#define _GsfVertexSmoother_H_


#include "RecoVertex/VertexPrimitives/interface/VertexSmoother.h"
#include "RecoVertex/GaussianSumVertexFit/interface/GsfVertexUpdator.h"
#include "RecoVertex/GaussianSumVertexFit/interface/GsfVertexMerger.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexUpdator.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexTrackUpdator.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanSmoothedVertexChi2Estimator.h"
#include "RecoVertex/GaussianSumVertexFit/interface/GsfVertexWeightCalculator.h"
#include "RecoVertex/VertexTools/interface/VertexTrackFactory.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/DeepCopyPointerByClone.h"
#include "RecoVertex/KalmanVertexFit/interface/KVFHelper.h"


/**
 *  The class which handles the track-refit and smoothed chi**2 calculations
 *  for the Gaussian Sum vertex fit. 
 *  The track-to-track covariance matrix calculation is not yet done.
 */

class GsfVertexSmoother : public VertexSmoother<5> {

public:

  /**
   *  The constructor
   *  \param limit Specifies whether the number of components of the vertex
   *			 should be limited
   */

  GsfVertexSmoother(bool limit, const GsfVertexMerger * merger);

  ~GsfVertexSmoother() override {}

  /**
   *  Methode which will refit the tracks with the vertex constraint
   *  and calculate the smoothed vertex chi**2
   *  \param vertex is the final estimate of the vertex, as given by the 
   *	last update.
   *  \return the final vertex estimate, with all the supplementary information
   */
  CachingVertex<5> smooth(const CachingVertex<5> & vertex) const override;

  /**
   *  Access methods
   */

  const VertexUpdator<5> * vertexUpdator() const
  {return &theUpdator;}

  /**
   * Clone method 
   */
  VertexSmoother<5> * clone() const override 
  {
    return new GsfVertexSmoother(* this);
  }
  
private:

  typedef CachingVertex<5>::RefCountedVertexTrack RefCountedVertexTrack;
  typedef VertexTrack<5>::RefCountedLinearizedTrackState RefCountedLinearizedTrackState;
  typedef VertexTrack<5>::RefCountedRefittedTrackState RefCountedRefittedTrackState;
  typedef VertexTrack<5>::AlgebraicMatrix3M AlgebraicMatrix3M;
  typedef VertexTrack<5>::AlgebraicSymMatrixOO AlgebraicSymMatrixOO;

  typedef std::vector<VertexState> VSC;
  typedef std::vector<RefCountedLinearizedTrackState> LTC;
  typedef std::pair<double, double> VtxTrkChi2Pair;
  typedef std::pair<RefCountedRefittedTrackState, double> TrackWeightPair;
  typedef std::pair<RefCountedRefittedTrackState, VtxTrkChi2Pair> TrackChi2Pair;
  typedef std::pair<TrackWeightPair, VtxTrkChi2Pair> RefittedTrackComponent;

  VertexState meanVertex(const VertexState & vertexA,
			 const VertexState & vertexB) const;

  TrackChi2Pair vertexAndTrackUpdate(const VertexState & oldVertex,
	const RefCountedVertexTrack track, const GlobalPoint & referencePosition) const;

  RefittedTrackComponent createNewComponent(const VertexState & oldVertex,
	 const RefCountedLinearizedTrackState linTrack, float weight) const;

  TrackChi2Pair assembleTrackComponents(
	const std::vector<RefittedTrackComponent> & trackComponents,
	const GlobalPoint & referencePosition) const;

  /**
   *  Methode which calculates the chi**2 between the prior and the fitted 
   *  vertex.
   *  This method takes into account multiple states.
   *  \param priorVertex The prior vertex state
   *  \param fittedVertex The fitted vertex state
   */
  double priorVertexChi2(const VertexState priorVertex, 
	const VertexState fittedVertex) const;

  bool limitComponents;
  DeepCopyPointerByClone<GsfVertexMerger> theMerger;
  GsfVertexUpdator theUpdator;
  KalmanVertexUpdator<5> kalmanVertexUpdator;
  KalmanSmoothedVertexChi2Estimator<5> smoothedChi2Estimator;
  KalmanVertexTrackUpdator<5> theVertexTrackUpdator;       
  GsfVertexWeightCalculator theWeightCalculator;
  VertexTrackFactory<5> theVTFactory;
  KVFHelper<5> helper;
};

#endif
