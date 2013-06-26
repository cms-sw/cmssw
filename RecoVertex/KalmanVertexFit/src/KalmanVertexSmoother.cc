#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexSmoother.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexTrackUpdator.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanSmoothedVertexChi2Estimator.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanTrackToTrackCovCalculator.h"

KalmanVertexSmoother::KalmanVertexSmoother() : SequentialVertexSmoother<5> (
    KalmanVertexTrackUpdator<5>(), KalmanSmoothedVertexChi2Estimator<5>(),
    KalmanTrackToTrackCovCalculator<5> () )
{}
