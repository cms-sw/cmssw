#ifndef RecoBTag_PositiveSideGhostTrackFitter_h
#define RecoBTag_PositiveSideGhostTrackFitter_h

#include <memory>
#include <vector>

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "RecoVertex/GhostTrackFitter/interface/GhostTrackFitter.h"

namespace reco {

class GhostTrackPredictioon;
class GhostTrackState;

class PositiveSideGhostTrackFitter : public GhostTrackFitter::FitterImpl {
    public:
	PositiveSideGhostTrackFitter(
			const GlobalPoint &origin,
			const GhostTrackFitter::FitterImpl &actualFitter) :
		origin_(origin), actualFitter_(actualFitter.clone()) {}
	~PositiveSideGhostTrackFitter() override {}

	PositiveSideGhostTrackFitter(
				const PositiveSideGhostTrackFitter &orig) :
		origin_(orig.origin_),
		actualFitter_(orig.actualFitter_->clone()) {}

	GhostTrackPrediction fit(
			const GhostTrackFitter::PredictionUpdater &updater,
			const GhostTrackPrediction &prior,
			std::vector<GhostTrackState> &states,
			double &ndof, double &chi2) override;

    private:
	FitterImpl *clone() const override
	{ return new PositiveSideGhostTrackFitter(*this); }

	GlobalPoint					origin_;
	std::unique_ptr<GhostTrackFitter::FitterImpl>	actualFitter_;
};

}

#endif // RecoBTag_PositiveSideGhostTrackFitter_h
