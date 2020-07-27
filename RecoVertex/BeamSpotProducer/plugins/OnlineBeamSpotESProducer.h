#ifndef RecoVertex_BeamSpotProducer_OnlineBeamSpotESProducer_H
#define RecoVertex_BeamSpotProducer_OnlineBeamSpotESProducer_H

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "CondFormats/DataRecord/interface/BeamSpotObjectsRcd.h"
//#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotOnlineObjects.h"
#include "CondFormats/DataRecord/interface/BeamSpotOnlineLegacyObjectsRcd.h"
#include "CondFormats/DataRecord/interface/BeamSpotOnlineHLTObjectsRcd.h"
#include "CondFormats/DataRecord/interface/BeamSpotTransientObjectsRcd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "FWCore/Framework/interface/ESProductHost.h"
#include "FWCore/Utilities/interface/ReusableObjectHolder.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <memory>

namespace edm {
  class ParameterSetDescription;

}

class OnlineBeamSpotESProducer : public edm::ESProducer {
public:
  OnlineBeamSpotESProducer(const edm::ParameterSet& p);
  ~OnlineBeamSpotESProducer() override;
  std::shared_ptr<const BeamSpotObjects> produce(const BeamSpotTransientObjectsRcd&);
  static void fillDescription(edm::ParameterSetDescription& desc);

private:
  const BeamSpotOnlineObjects* compareBS(const BeamSpotOnlineObjects* bs1, const BeamSpotOnlineObjects* bs2);
  const BeamSpotOnlineObjects* theHLTBS_;
  const BeamSpotOnlineObjects* theLegacyBS_;
  const BeamSpotObjects* transientBS_;
  BeamSpotObjects* fakeBS_;
  bool newHLT_;
  bool newLegacy_;
  //  std::string label_HLT_;

  edm::ESGetToken<BeamSpotObjects, BeamSpotTransientObjectsRcd> const bsToken_;
  edm::ESGetToken<BeamSpotOnlineObjects, BeamSpotOnlineHLTObjectsRcd> bsHLTToken_;
  edm::ESGetToken<BeamSpotOnlineObjects, BeamSpotOnlineLegacyObjectsRcd> bsLegacyToken_;
  using HostType =
      edm::ESProductHost<BeamSpotOnlineObjects, BeamSpotOnlineHLTObjectsRcd, BeamSpotOnlineLegacyObjectsRcd>;

  edm::ReusableObjectHolder<HostType> holder_;
};

#endif
