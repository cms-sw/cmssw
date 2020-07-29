#include "RecoVertex/BeamSpotProducer/plugins/OnlineBeamSpotESProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Utilities/interface/do_nothing_deleter.h"

#include <iostream>
#include <memory>
#include <string>

using namespace edm;

OnlineBeamSpotESProducer::OnlineBeamSpotESProducer(const edm::ParameterSet& p) {
  auto cc = setWhatProduced(this);

  transientBS_ = new BeamSpotObjects;
  theHLTBS_ = new BeamSpotOnlineObjects;
  theLegacyBS_ = new BeamSpotOnlineObjects;
  fakeBS_ = new BeamSpotOnlineObjects;
  fakeBS_->SetBeamWidthX(0.1);
  fakeBS_->SetBeamWidthY(0.1);
  fakeBS_->SetSigmaZ(15.);
  fakeBS_->SetPosition(0., 0., 0.);
  fakeBS_->SetType(-1);

  bsHLTToken_ = cc.consumesFrom<BeamSpotOnlineObjects, BeamSpotOnlineHLTObjectsRcd>();
  bsLegacyToken_ = cc.consumesFrom<BeamSpotOnlineObjects, BeamSpotOnlineLegacyObjectsRcd>();
}

const BeamSpotOnlineObjects* OnlineBeamSpotESProducer::compareBS(const BeamSpotOnlineObjects* bs1,
                                                                 const BeamSpotOnlineObjects* bs2) {
  //Random logic so far ...
  if (bs1->GetSigmaZ() - 0.0001 < bs2->GetSigmaZ()) {  //just temporary for debugging
    if (bs1->GetSigmaZ() > 5.) {
      return bs1;
    } else {
      return bs2;
    }

  } else {
    if (bs2->GetSigmaZ() > 5.) {
      return bs2;
    } else {
      return bs1;
    }
  }
}

OnlineBeamSpotESProducer::~OnlineBeamSpotESProducer() {
  delete theHLTBS_;
  delete theLegacyBS_;
  delete transientBS_;
  delete fakeBS_;
}

std::shared_ptr<const BeamSpotObjects> OnlineBeamSpotESProducer::produce(const BeamSpotTransientObjectsRcd& iRecord) {
  if (!(iRecord.tryToGetRecord<BeamSpotOnlineLegacyObjectsRcd>()) &&
      !(iRecord.tryToGetRecord<BeamSpotOnlineHLTObjectsRcd>())) {
    std::cout << "Sending out the fakeBS_" << std::endl;
    transientBS_ = fakeBS_;
    return std::shared_ptr<const BeamSpotObjects>(&(*transientBS_), edm::do_nothing_deleter());
  }

  auto host = holder_.makeOrGet([]() { return new HostType; });

  newHLT_ = false;
  newLegacy_ = false;
  if (iRecord.tryToGetRecord<BeamSpotOnlineHLTObjectsRcd>()) {
    host->ifRecordChanges<BeamSpotOnlineHLTObjectsRcd>(iRecord, [this, h = host.get()](auto const& rec) {
      newHLT_ = true;
      theHLTBS_ = &rec.get(bsHLTToken_);
    });
  }
  if (iRecord.tryToGetRecord<BeamSpotOnlineLegacyObjectsRcd>()) {
    host->ifRecordChanges<BeamSpotOnlineLegacyObjectsRcd>(iRecord, [this, h = host.get()](auto const& rec) {
      newLegacy_ = true;
      theLegacyBS_ = &rec.get(bsLegacyToken_);
    });
  }

  //we need to compare the transientBS_ which is the last one put in the event with whatever new one arrives. We cannot exclude we
  //have both new records at the same LS
  if (newHLT_ && !newLegacy_) {
    std::cout << "pippo1" << std::endl;
    //compare newHLT with transientBS_
    //temporary test
    //transientBS_ =  theHLTBS_;
    transientBS_->SetPosition(0, 0, 0);
    transientBS_->SetBeamWidthY(0.1);
    transientBS_->SetBeamWidthX(0.1);
    transientBS_->SetSigmaZ(5.0);
    //newHLT_= false;
    //transientBS_ = compareBS(theHLTBS_, transientBS_);
  }

  if (newLegacy_ && !newHLT_) {
    std::cout << "pippo2" << std::endl;
    //compare newLegacy_ with transientBS_
    //test
    transientBS_->SetPosition(1, 1, 1);
    transientBS_->SetBeamWidthY(0.1);
    transientBS_->SetBeamWidthX(0.1);
    transientBS_->SetSigmaZ(5.0);
    //transientBS_ = theLegacyBS_;
    //newLegacy_ = false;
    //transientBS_ = compareBS(theLegacyBS_, transientBS_);
  }
  if (newHLT_ && newLegacy_) {
    //compare newHLT_ with transientBS_ and then with newLegacy_
    std::cout << "pippo3" << std::endl;
    //test
    //transientBS_ = theHLTBS_;
    transientBS_->SetPosition(2, 2, 2);
    transientBS_->SetBeamWidthY(0.1);
    transientBS_->SetBeamWidthX(0.1);
    transientBS_->SetSigmaZ(5.0);
    //newHLT_ = false;
    //newLegacy_ = false;
    //transientBS_ = compareBS(theHLTBS_, transientBS_);
    // transientBS_ = compareBS(theLegacyBS_, transientBS_);
  };

  std::cout << "Transient " << *transientBS_ << std::endl;
  std::cout << "Legacy " << *theLegacyBS_ << std::endl;
  std::cout << "HLT " << *theHLTBS_ << std::endl;

  return std::shared_ptr<const BeamSpotObjects>(&(*transientBS_), edm::do_nothing_deleter());
};

DEFINE_FWK_EVENTSETUP_MODULE(OnlineBeamSpotESProducer);
