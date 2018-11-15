#ifndef RecoBTag_AnnealingGhostTrackFitter_h
#define RecoBTag_AnnealingGhostTrackFitter_h

#include <vector>
#include <memory>

#include "RecoVertex/VertexTools/interface/AnnealingSchedule.h"

#include "RecoVertex/GhostTrackFitter/interface/GhostTrackFitter.h"
#include "RecoVertex/GhostTrackFitter/interface/SequentialGhostTrackFitter.h"

namespace reco {

class GhostTrackPrediction;
class GhostTrackState;

class AnnealingGhostTrackFitter : public SequentialGhostTrackFitter {
    public:
	AnnealingGhostTrackFitter();
	AnnealingGhostTrackFitter(const AnnealingGhostTrackFitter &orig) :
		annealing(orig.annealing->clone()),
		firstStep(orig.firstStep) {}
	~AnnealingGhostTrackFitter() override {}

    private:
	FitterImpl *clone() const override
	{ return new AnnealingGhostTrackFitter(*this); }

	bool stable(const GhostTrackPrediction &before,
	                    const GhostTrackPrediction &after) const override
	{
		return SequentialGhostTrackFitter::stable(before, after) &&
		       annealing->isAnnealed();
	}

	void reset() override { annealing->resetAnnealing(); firstStep = true; }
	void postFit(
			const GhostTrackFitter::PredictionUpdater &updater,
			const GhostTrackPrediction &pred,
			std::vector<GhostTrackState> &states) override;

	std::unique_ptr<AnnealingSchedule>	annealing;
	bool					firstStep;
};

}

#endif // RecoBTag_AnnealingGhostTrackFitter_h
