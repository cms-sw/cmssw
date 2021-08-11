#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Utilities/interface/do_nothing_deleter.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "CondFormats/DataRecord/interface/BeamSpotObjectsRcd.h"
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
#include <iostream>
#include <string>

using namespace edm;

class OnlineBeamSpotESProducer : public edm::ESProducer {
public:
  OnlineBeamSpotESProducer(const edm::ParameterSet& p);
  std::shared_ptr<const BeamSpotObjects> produce(const BeamSpotTransientObjectsRcd&);
  static void fillDescriptions(edm::ConfigurationDescriptions& desc);

private:
  const BeamSpotOnlineObjects* compareBS(const BeamSpotOnlineObjects* bs1, const BeamSpotOnlineObjects* bs2);
  BeamSpotObjects fakeBS_;

  edm::ESGetToken<BeamSpotObjects, BeamSpotTransientObjectsRcd> const bsToken_;
  edm::ESGetToken<BeamSpotOnlineObjects, BeamSpotOnlineHLTObjectsRcd> bsHLTToken_;
  edm::ESGetToken<BeamSpotOnlineObjects, BeamSpotOnlineLegacyObjectsRcd> bsLegacyToken_;
};
OnlineBeamSpotESProducer::OnlineBeamSpotESProducer(const edm::ParameterSet& p) {
  auto cc = setWhatProduced(this);

  fakeBS_.SetBeamWidthX(0.1);
  fakeBS_.SetBeamWidthY(0.1);
  fakeBS_.SetSigmaZ(15.);
  fakeBS_.SetPosition(0.0001, 0.0001, 0.0001);
  fakeBS_.SetType(-1);

  bsHLTToken_ = cc.consumesFrom<BeamSpotOnlineObjects, BeamSpotOnlineHLTObjectsRcd>();
  bsLegacyToken_ = cc.consumesFrom<BeamSpotOnlineObjects, BeamSpotOnlineLegacyObjectsRcd>();
}

void OnlineBeamSpotESProducer::fillDescriptions(edm::ConfigurationDescriptions& desc) {
  edm::ParameterSetDescription dsc;
  desc.addWithDefaultLabel(dsc);
}

const BeamSpotOnlineObjects* OnlineBeamSpotESProducer::compareBS(const BeamSpotOnlineObjects* bs1,
                                                                 const BeamSpotOnlineObjects* bs2) {
  //Random logic so far ...
  if (bs1->GetSigmaZ() - 0.0001 > bs2->GetSigmaZ()) {  //just temporary for debugging
    if (bs1->GetSigmaZ() > 2.5) {
      return bs1;
    } else {
      return nullptr;
    }

  } else {
    if (bs2->GetSigmaZ() > 2.5) {
      return bs2;
    } else {
      return nullptr;
    }
  }
}

std::shared_ptr<const BeamSpotObjects> OnlineBeamSpotESProducer::produce(const BeamSpotTransientObjectsRcd& iRecord) {
  auto legacyRec = iRecord.tryToGetRecord<BeamSpotOnlineLegacyObjectsRcd>();
  auto hltRec = iRecord.tryToGetRecord<BeamSpotOnlineHLTObjectsRcd>();
  if (not legacyRec and not hltRec) {
    return std::shared_ptr<const BeamSpotObjects>(&fakeBS_, edm::do_nothing_deleter());
  }

  const BeamSpotOnlineObjects* best;
  if (legacyRec and hltRec) {
    best = compareBS(&legacyRec->get(bsLegacyToken_), &hltRec->get(bsHLTToken_));
  } else if (legacyRec) {
    best = &legacyRec->get(bsLegacyToken_);
  } else {
    best = &hltRec->get(bsHLTToken_);
  }
  if (best) {
    return std::shared_ptr<const BeamSpotObjects>(best, edm::do_nothing_deleter());
  } else {
    return std::shared_ptr<const BeamSpotObjects>(&fakeBS_, edm::do_nothing_deleter());
  }
};

DEFINE_FWK_EVENTSETUP_MODULE(OnlineBeamSpotESProducer);
