#include "RecoVertex/AdaptiveVertexFit/interface/KalmanVertexSmoother.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexTrackUpdator.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanSmoothedVertexChi2Estimator.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanTrackToTrackCovCalculator.h"

KalmanVertexSmoother::KalmanVertexSmoother() : SequentialVertexSmoother (
    KalmanVertexTrackUpdator(), KalmanSmoothedVertexChi2Estimator(),
    KalmanTrackToTrackCovCalculator () )
{ }
