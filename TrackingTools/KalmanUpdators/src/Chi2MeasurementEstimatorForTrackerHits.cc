#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimatorForTrackerHits.h"
#include "TrackingTools/PatternTools/interface/MeasurementExtractor.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/Math/interface/invertPosDefMatrix.h"

Chi2MeasurementEstimatorForTrackerHits::Chi2MeasurementEstimatorForTrackerHits(const Chi2MeasurementEstimatorForTrackerHits& src) : Chi2MeasurementEstimatorBase(src), aHelper(nullptr) {}

std::pair<bool,double>
Chi2MeasurementEstimatorForTrackerHits::estimate(
        const TrajectoryStateOnSurface& tsos,
        const TransientTrackingRecHit& aRecHit) const {
        if (!aHelper) {
                const AlgebraicVector5& par5 = tsos.localParameters().vector();
                const AlgebraicSymMatrix55& err5 = tsos.localError().matrix();
                auto ptr = new AlgebraicHelper(par5.Sub<AlgebraicVector2>(3),
                                               err5.Sub<AlgebraicSymMatrix22>(3,3));
                AlgebraicHelper* expect = nullptr;
                bool exchanged = aHelper.compare_exchange_strong(expect, ptr);
                if (!exchanged) delete ptr;
        }
        AlgebraicVector2     r = asSVector<2>(aRecHit.parameters())      - (*aHelper).params();
        AlgebraicSymMatrix22 R = asSMatrix<2>(aRecHit.parametersError()) + (*aHelper).errors();
	invertPosDefMatrix(R);
        return returnIt( ROOT::Math::Similarity(r, R) );
}
