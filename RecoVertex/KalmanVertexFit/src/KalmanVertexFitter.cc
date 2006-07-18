#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexUpdator.h"
#include "RecoVertex/VertexTools/interface/SequentialVertexSmoother.h"
#include "RecoVertex/VertexTools/interface/DummyVertexSmoother.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexTrackUpdator.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanSmoothedVertexChi2Estimator.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanTrackToTrackCovCalculator.h"
#include "RecoVertex/LinearizationPointFinders/interface/FsmwLinearizationPointFinder.h"


KalmanVertexFitter::KalmanVertexFitter(const edm::ParameterSet& pSet,  bool useSmoothing )
{
  if (useSmoothing) {
    KalmanVertexTrackUpdator vtu;
    KalmanSmoothedVertexChi2Estimator vse;
    KalmanTrackToTrackCovCalculator covCalc;
    SequentialVertexSmoother smoother(vtu, vse, covCalc);
    theSequentialFitter 
      = new SequentialVertexFitter(pSet, FsmwLinearizationPointFinder(20, -2., 0.4, 10.), 
				   KalmanVertexUpdator(), 
				   smoother);
  }
  else {
    DummyVertexSmoother smoother;
    theSequentialFitter 
      = new SequentialVertexFitter(pSet, FsmwLinearizationPointFinder(20, -2., 0.4, 10.), 
				   KalmanVertexUpdator(), 
				   smoother);
  }
}
