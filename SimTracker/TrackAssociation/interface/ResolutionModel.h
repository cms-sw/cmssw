#ifndef __SimTracker_TrackAssociation_ResolutionModel_h__
#define __SimTracker_TrackAssociation_ResolutionModel_h__

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>
#include <string>

class ResolutionModel {
public:
  ResolutionModel(const edm::ParameterSet &conf) : _modelName(conf.getParameter<std::string>("modelName")) {}
  virtual ~ResolutionModel() {}
  // get rid of things we should never use...
  ResolutionModel(const ResolutionModel &) = delete;
  ResolutionModel &operator=(const ResolutionModel &) = delete;

  virtual float getTimeResolution(const reco::Track &) const { return -1.f; }
  virtual float getTimeResolution(const reco::PFCluster &) const { return -1.f; }

  const std::string &name() const { return _modelName; }

private:
  const std::string _modelName;
};

#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory<ResolutionModel *(const edm::ParameterSet &)> ResolutionModelFactory;

#endif
