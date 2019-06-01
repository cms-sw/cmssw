#include "SimTracker/TrackAssociation/interface/ResolutionModel.h"

class PerfectResolutionModel : public ResolutionModel {
public:
  PerfectResolutionModel(const edm::ParameterSet &conf) : ResolutionModel(conf) {}

  float getTimeResolution(const reco::Track &) const override { return 1e-6; }
  float getTimeResolution(const reco::PFCluster &) const override { return 1e-6; }
};

DEFINE_EDM_PLUGIN(ResolutionModelFactory, PerfectResolutionModel, "PerfectResolutionModel");
