#ifndef RecoBTag_GhostTrackFitter_h
#define RecoBTag_GhostTrackFitter_h

#include <memory>
#include <vector>

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

#include "RecoVertex/GhostTrackFitter/interface/GhostTrack.h"
#include "RecoVertex/GhostTrackFitter/interface/GhostTrackState.h"
#include "RecoVertex/GhostTrackFitter/interface/GhostTrackPrediction.h"

namespace reco {

class GhostTrackFitter {
    public:
	GhostTrackFitter();
	virtual ~GhostTrackFitter();

	GhostTrack fit(const GlobalPoint &priorPosition,
	               const GlobalError &priorError,
	               const GlobalVector &direction,
	               double coneRadius,
	               const std::vector<TransientTrack> &tracks) const;

	GhostTrack fit(const GlobalPoint &priorPosition,
	               const GlobalError &priorError,
	               const GlobalVector &direction,
	               const GlobalError &directionError,
	               const std::vector<TransientTrack> &tracks) const;

	GhostTrack fit(const GhostTrackPrediction &prior,
	               const GlobalPoint &origin,
	               const std::vector<TransientTrack> &tracks) const;

	GhostTrack fit(const GhostTrackPrediction &prior,
	               const std::vector<TransientTrack> &tracks) const;

	class PredictionUpdater {
	    public:
		virtual ~PredictionUpdater() {}

		virtual PredictionUpdater *clone() const = 0;

		virtual GhostTrackPrediction update(
				const GhostTrackPrediction &pred,
				const GhostTrackState &state,
				double &ndof, double &chi2) const = 0;

		virtual void contribution(
				const GhostTrackPrediction &pred,
				const GhostTrackState &state,
				double &ndof, double &chi2,
		                bool withPredError = false) const = 0;
	};

	class FitterImpl {
	    public:
		virtual ~FitterImpl() {}

		virtual FitterImpl *clone() const = 0;

		virtual GhostTrackPrediction fit(
				const PredictionUpdater &updater,
				const GhostTrackPrediction &pred,
				std::vector<GhostTrackState> &states,
				double &ndof, double &chi2) = 0;
	};

	void setFitterImpl(const FitterImpl &fitterImpl)
	{ fitter.reset(fitterImpl.clone()); }

    protected:
	GhostTrack fit(FitterImpl &fitterImpl,
	               const GhostTrackPrediction &prior,
	               const std::vector<GhostTrackState> &states) const;

    private:
	std::unique_ptr<FitterImpl>		fitter;
	std::unique_ptr<PredictionUpdater>	updater;
};

}
#endif // RecoBTag_GhostTrackFitter_h
