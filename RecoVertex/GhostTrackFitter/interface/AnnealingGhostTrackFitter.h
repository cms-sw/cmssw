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
	~AnnealingGhostTrackFitter() {}

    private:
	virtual FitterImpl *clone() const
	{ return new AnnealingGhostTrackFitter(*this); }

	virtual bool stable(const GhostTrackPrediction &before,
	                    const GhostTrackPrediction &after) const
	{
		return SequentialGhostTrackFitter::stable(before, after) &&
		       annealing->isAnnealed();
	}

	virtual void reset() { annealing->resetAnnealing(); firstStep = true; }
	virtual void postFit(
			const GhostTrackFitter::PredictionUpdater &updater,
			const GhostTrackPrediction &pred,
			std::vector<GhostTrackState> &states);

	std::auto_ptr<AnnealingSchedule>	annealing;
	bool					firstStep;
};

}

#endif // RecoBTag_AnnealingGhostTrackFitter_h
