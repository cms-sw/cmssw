#ifndef RecoVertex_BeamSpotProducer_OfflineToTransientBeamSpotESProducer_H
#define RecoVertex_BeamSpotProducer_OfflineToTransientBeamSpotESProducer_H

#include "CondFormats/DataRecord/interface/BeamSpotObjectsRcd.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"
#include "CondFormats/DataRecord/interface/BeamSpotTransientObjectsRcd.h"

#include "FWCore/Framework/interface/ESProductHost.h"
#include "FWCore/Utilities/interface/ReusableObjectHolder.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <memory>

class OfflineToTransientBeamSpotESProducer : public edm::ESProducer {
public:
  OfflineToTransientBeamSpotESProducer(const edm::ParameterSet &p);
  ~OfflineToTransientBeamSpotESProducer() override;
  std::shared_ptr<const BeamSpotObjects> produce(const BeamSpotTransientObjectsRcd &);

private:
  //const BeamSpotObjects* theOfflineBS_;
  const BeamSpotObjects *transientBS_;
  edm::ESGetToken<BeamSpotObjects, BeamSpotTransientObjectsRcd> const bsToken_;
  edm::ESGetToken<BeamSpotObjects, BeamSpotObjectsRcd> bsOfflineToken_;
  using HostType = edm::ESProductHost<BeamSpotObjects, BeamSpotObjectsRcd>;

  edm::ReusableObjectHolder<HostType> holder_;
};

#endif
