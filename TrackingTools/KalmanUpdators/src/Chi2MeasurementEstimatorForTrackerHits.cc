#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorForTrackerHits.h"
#include "TrackingTools/PatternTools/interface/MeasurementExtractor.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/Math/interface/invertPosDefMatrix.h"

std::pair<bool,double> 
Chi2MeasurementEstimatorForTrackerHits::estimate(
        const TrajectoryStateOnSurface& tsos,
        const TransientTrackingRecHit& aRecHit) const {
        if (!cacheUpToDate_) {
                AlgebraicVector5 par5 = tsos.localParameters().vector();
                tsosMeasuredParameters_[0] = par5[3]; 
                tsosMeasuredParameters_[1] = par5[4]; 
                const AlgebraicSymMatrix55 &err5 = tsos.localError().matrix();
                tsosMeasuredError_ = err5.Sub<AlgebraicSymMatrix22>(3,3);
                cacheUpToDate_ = true;
        }
        AlgebraicVector2     r = asSVector<2>(aRecHit.parameters())      - tsosMeasuredParameters_;
        AlgebraicSymMatrix22 R = asSMatrix<2>(aRecHit.parametersError()) + tsosMeasuredError_;
	invertPosDefMatrix(R);
        return returnIt( ROOT::Math::Similarity(r, R) );
}

