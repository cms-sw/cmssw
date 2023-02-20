#include "CondFormats/BeamSpotObjects/interface/BeamSpotOnlineObjects.h"
#include "CondFormats/DataRecord/interface/BeamSpotObjectsRcd.h"
#include "CondFormats/DataRecord/interface/BeamSpotOnlineHLTObjectsRcd.h"
#include "CondFormats/DataRecord/interface/BeamSpotOnlineLegacyObjectsRcd.h"
#include "CondFormats/DataRecord/interface/BeamSpotTransientObjectsRcd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESProductHost.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ReusableObjectHolder.h"
#include "FWCore/Utilities/interface/do_nothing_deleter.h"
#include <memory>
#include <string>

using namespace edm;

class OnlineBeamSpotESProducer : public edm::ESProducer {
public:
  OnlineBeamSpotESProducer(const edm::ParameterSet& p);
  std::shared_ptr<const BeamSpotObjects> produce(const BeamSpotTransientObjectsRcd&);
  static void fillDescriptions(edm::ConfigurationDescriptions& desc);

private:
  const BeamSpotOnlineObjects* compareBS(const BeamSpotOnlineObjects* bs1, const BeamSpotOnlineObjects* bs2);
  const BeamSpotOnlineObjects* checkSingleBS(const BeamSpotOnlineObjects* bs1);
  bool isGoodBS(const BeamSpotOnlineObjects* bs1) const;

  edm::ESGetToken<BeamSpotObjects, BeamSpotTransientObjectsRcd> const bsToken_;
  edm::ESGetToken<BeamSpotOnlineObjects, BeamSpotOnlineHLTObjectsRcd> bsHLTToken_;
  edm::ESGetToken<BeamSpotOnlineObjects, BeamSpotOnlineLegacyObjectsRcd> bsLegacyToken_;

  BeamSpotObjects fakeBS_;
  const int timeThreshold_;
  const double sigmaZThreshold_;
  const double sigmaXYThreshold_;
};

OnlineBeamSpotESProducer::OnlineBeamSpotESProducer(const edm::ParameterSet& p)
    // get parameters
    : timeThreshold_(p.getParameter<int>("timeThreshold")),
      sigmaZThreshold_(p.getParameter<double>("sigmaZThreshold")),
      sigmaXYThreshold_(p.getParameter<double>("sigmaXYThreshold") * 1E-4) {
  auto cc = setWhatProduced(this);

  fakeBS_.setBeamWidthX(0.1);
  fakeBS_.setBeamWidthY(0.1);
  fakeBS_.setSigmaZ(15.);
  fakeBS_.setPosition(0.0001, 0.0001, 0.0001);
  fakeBS_.setType(-1);

  bsHLTToken_ = cc.consumesFrom<BeamSpotOnlineObjects, BeamSpotOnlineHLTObjectsRcd>();
  bsLegacyToken_ = cc.consumesFrom<BeamSpotOnlineObjects, BeamSpotOnlineLegacyObjectsRcd>();
}

void OnlineBeamSpotESProducer::fillDescriptions(edm::ConfigurationDescriptions& desc) {
  edm::ParameterSetDescription dsc;
  dsc.add<int>("timeThreshold", 48)->setComment("hours");
  dsc.add<double>("sigmaZThreshold", 2.)->setComment("cm");
  dsc.add<double>("sigmaXYThreshold", 4.)->setComment("um");
  desc.addWithDefaultLabel(dsc);
}

const BeamSpotOnlineObjects* OnlineBeamSpotESProducer::compareBS(const BeamSpotOnlineObjects* bs1,
                                                                 const BeamSpotOnlineObjects* bs2) {
  // Current time to be compared with the one read from the payload
  auto currentTime =
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());

  // Get two beamspot creation times and compute the time difference wrt currentTime
  auto bs1time = std::chrono::microseconds(bs1->creationTime());
  auto diffBStime1 = (currentTime - bs1time).count();
  auto bs2time = std::chrono::microseconds(bs2->creationTime());
  auto diffBStime2 = (currentTime - bs2time).count();

  // Convert timeThreshold_ from hours to microseconds for comparison
  auto limitTime = std::chrono::microseconds((std::chrono::hours)timeThreshold_).count();

  // Logic to choose between the two BeamSpots:
  // 1. If both BS are older than limitTime retun fake BS
  // 2. If only one BS is newer than limitTime return it only if
  //     it passes isGoodBS (checks on sigmaZ, sigmaXY and fit convergence)
  // 3. If both are newer than the limit threshold return the BS that
  //     passes isGoodBS and has larger sigmaZ
  if (diffBStime1 > limitTime && diffBStime2 > limitTime) {
    edm::LogInfo("OnlineBeamSpotESProducer") << "Defaulting to fake because both payloads are too old.";
    return nullptr;
  } else if (diffBStime2 > limitTime) {
    if (isGoodBS(bs1)) {
      return bs1;
    } else {
      edm::LogInfo("OnlineBeamSpotESProducer")
          << "Defaulting to fake because the legacy Beam Spot is not suitable and HLT one is too old.";
      return nullptr;
    }
  } else if (diffBStime1 > limitTime) {
    if (isGoodBS(bs2)) {
      return bs2;
    } else {
      edm::LogInfo("OnlineBeamSpotESProducer")
          << "Defaulting to fake because the HLT Beam Spot is not suitable and the legacy one too old.";
      return nullptr;
    }
  } else {
    if (bs1->sigmaZ() > bs2->sigmaZ() && isGoodBS(bs1)) {
      return bs1;
    } else if (bs2->sigmaZ() >= bs1->sigmaZ() && isGoodBS(bs2)) {
      return bs2;
    } else {
      edm::LogInfo("OnlineBeamSpotESProducer")
          << "Defaulting to fake because despite both payloads are young enough, none has passed the fit sanity checks";
      return nullptr;
    }
  }
}

const BeamSpotOnlineObjects* OnlineBeamSpotESProducer::checkSingleBS(const BeamSpotOnlineObjects* bs1) {
  // Current time to be compared with the one read from the payload
  auto currentTime =
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());

  // Get the beamspot creation time and compute the time difference wrt currentTime
  auto bs1time = std::chrono::microseconds(bs1->creationTime());
  auto diffBStime1 = (currentTime - bs1time).count();

  // Convert timeThreshold_ from hours to microseconds for comparison
  auto limitTime = std::chrono::microseconds((std::chrono::hours)timeThreshold_).count();

  // Check that the BS is within the timeThreshold, converges and passes the sigmaZthreshold
  if (diffBStime1 < limitTime && isGoodBS(bs1)) {
    return bs1;
  } else {
    return nullptr;
  }
}

// This method is used to check the quality of the beamspot fit
bool OnlineBeamSpotESProducer::isGoodBS(const BeamSpotOnlineObjects* bs1) const {
  if (bs1->sigmaZ() > sigmaZThreshold_ && bs1->beamType() == reco::BeamSpot::Tracker &&
      bs1->beamWidthX() > sigmaXYThreshold_ && bs1->beamWidthY() > sigmaXYThreshold_)
    return true;
  else
    return false;
}

std::shared_ptr<const BeamSpotObjects> OnlineBeamSpotESProducer::produce(const BeamSpotTransientObjectsRcd& iRecord) {
  auto legacyRec = iRecord.tryToGetRecord<BeamSpotOnlineLegacyObjectsRcd>();
  auto hltRec = iRecord.tryToGetRecord<BeamSpotOnlineHLTObjectsRcd>();
  if (not legacyRec and not hltRec) {
    edm::LogInfo("OnlineBeamSpotESProducer") << "None of the Beam Spots in ES are available! \n returning a fake one.";
    return std::shared_ptr<const BeamSpotObjects>(&fakeBS_, edm::do_nothing_deleter());
  }

  const BeamSpotOnlineObjects* best;
  if (legacyRec and hltRec) {
    best = compareBS(&legacyRec->get(bsLegacyToken_), &hltRec->get(bsHLTToken_));
  } else if (legacyRec) {
    best = checkSingleBS(&legacyRec->get(bsLegacyToken_));
  } else {
    best = checkSingleBS(&hltRec->get(bsHLTToken_));
  }
  if (best) {
    return std::shared_ptr<const BeamSpotObjects>(best, edm::do_nothing_deleter());
  } else {
    return std::shared_ptr<const BeamSpotObjects>(&fakeBS_, edm::do_nothing_deleter());
    edm::LogInfo("OnlineBeamSpotESProducer")
        << "None of the Online BeamSpots in the ES is suitable, \n returning a fake one. ";
  }
};

DEFINE_FWK_EVENTSETUP_MODULE(OnlineBeamSpotESProducer);
