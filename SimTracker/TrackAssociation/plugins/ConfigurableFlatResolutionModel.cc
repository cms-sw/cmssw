#include "SimTracker/TrackAssociation/interface/ResolutionModel.h"

class ConfigurableFlatResolutionModel : public ResolutionModel {
public:
  ConfigurableFlatResolutionModel(const edm::ParameterSet &conf)
      : ResolutionModel(conf), reso_(conf.getParameter<double>("resolutionInNs")) {}

  float getTimeResolution(const reco::Track &) const override { return reso_; }
  float getTimeResolution(const reco::PFCluster &) const override { return reso_; }

private:
  const float reso_;
};

DEFINE_EDM_PLUGIN(ResolutionModelFactory, ConfigurableFlatResolutionModel, "ConfigurableFlatResolutionModel");
