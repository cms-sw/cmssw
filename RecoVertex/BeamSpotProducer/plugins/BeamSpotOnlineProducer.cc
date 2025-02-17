/**_________________________________________________________________
   class:   BeamSpotOnlineProducer.h
   package: RecoVertex/BeamSpotProducer



 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)
 modified by: Simone Gennai, INFN MIB


________________________________________________________________**/

#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"
#include "CondFormats/DataRecord/interface/BeamSpotObjectsRcd.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotOnlineObjects.h"
#include "CondFormats/DataRecord/interface/BeamSpotTransientObjectsRcd.h"
#include "CondFormats/DataRecord/interface/BeamSpotOnlineHLTObjectsRcd.h"
#include "CondFormats/DataRecord/interface/BeamSpotOnlineLegacyObjectsRcd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"
#include "DataFormats/Scalers/interface/BeamSpotOnline.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

class BeamSpotOnlineProducer : public edm::stream::EDProducer<> {
public:
  /// constructor
  explicit BeamSpotOnlineProducer(const edm::ParameterSet& iConf);

  /// produce a beam spot class
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  ///Fill descriptor
  static void fillDescriptions(edm::ConfigurationDescriptions& iDesc);

private:
  // helper methods
  bool shouldShout(const edm::Event& iEvent) const;
  bool processRecords(const edm::Event& iEvent, const edm::EventSetup& iSetup, reco::BeamSpot& result, bool shoutMODE);
  void createBeamSpotFromRecord(const BeamSpotObjects& spotDB, reco::BeamSpot& result) const;
  template <typename TokenType>
  const BeamSpotOnlineObjects getBeamSpotFromRecord(const TokenType& token,
                                                    const edm::Event& event,
                                                    const edm::EventSetup& setup);
  const BeamSpotOnlineObjects& chooseBS(const BeamSpotOnlineObjects& bs1, const BeamSpotOnlineObjects& bs2);
  bool processScalers(const edm::Event& iEvent, reco::BeamSpot& result, bool shoutMODE);
  void createBeamSpotFromScaler(const BeamSpotOnline& spotOnline, reco::BeamSpot& result) const;
  bool isInvalidScaler(const BeamSpotOnline& spotOnline, bool shoutMODE) const;
  void createBeamSpotFromDB(const edm::EventSetup& iSetup, reco::BeamSpot& result, bool shoutMODE) const;

  // data members
  const bool changeFrame_;
  const double theMaxZ, theSetSigmaZ;
  double theMaxR2;
  const int timeThreshold_;
  const double sigmaZThreshold_, sigmaXYThreshold_;
  const bool useBSOnlineRecords_;
  const edm::EDGetTokenT<BeamSpotOnlineCollection> scalerToken_;
  const edm::EDGetTokenT<L1GlobalTriggerEvmReadoutRecord> l1GtEvmReadoutRecordToken_;
  const edm::ESGetToken<BeamSpotObjects, BeamSpotObjectsRcd> beamToken_;
  const edm::ESGetToken<BeamSpotOnlineObjects, BeamSpotOnlineLegacyObjectsRcd> beamTokenLegacy_;
  const edm::ESGetToken<BeamSpotOnlineObjects, BeamSpotOnlineHLTObjectsRcd> beamTokenHLT_;

  // watch IOV transition to emit warnings
  edm::ESWatcher<BeamSpotOnlineLegacyObjectsRcd> beamLegacyRcdESWatcher_;
  edm::ESWatcher<BeamSpotOnlineHLTObjectsRcd> beamHLTRcdESWatcher_;

  const unsigned int theBeamShoutMode;
  BeamSpotOnlineObjects fakeBS_;
};

using namespace edm;

BeamSpotOnlineProducer::BeamSpotOnlineProducer(const ParameterSet& iconf)
    : changeFrame_(iconf.getParameter<bool>("changeToCMSCoordinates")),
      theMaxZ(iconf.getParameter<double>("maxZ")),
      theSetSigmaZ(iconf.getParameter<double>("setSigmaZ")),
      timeThreshold_(iconf.getParameter<int>("timeThreshold")),
      sigmaZThreshold_(iconf.getParameter<double>("sigmaZThreshold")),
      sigmaXYThreshold_(iconf.getParameter<double>("sigmaXYThreshold") * 1E-4),
      useBSOnlineRecords_(iconf.getParameter<bool>("useBSOnlineRecords")),
      scalerToken_(useBSOnlineRecords_ ? edm::EDGetTokenT<BeamSpotOnlineCollection>()
                                       : consumes<BeamSpotOnlineCollection>(iconf.getParameter<InputTag>("src"))),
      l1GtEvmReadoutRecordToken_(consumes<L1GlobalTriggerEvmReadoutRecord>(iconf.getParameter<InputTag>("gtEvmLabel"))),
      beamToken_(esConsumes<BeamSpotObjects, BeamSpotObjectsRcd>()),
      beamTokenLegacy_(esConsumes<BeamSpotOnlineObjects, BeamSpotOnlineLegacyObjectsRcd>()),
      beamTokenHLT_(esConsumes<BeamSpotOnlineObjects, BeamSpotOnlineHLTObjectsRcd>()),
      theBeamShoutMode(iconf.getUntrackedParameter<unsigned int>("beamMode", 11)) {
  fakeBS_.setBeamWidthX(0.1);
  fakeBS_.setBeamWidthY(0.1);
  fakeBS_.setSigmaZ(15.);
  fakeBS_.setPosition(0.0001, 0.0001, 0.0001);
  fakeBS_.setType(reco::BeamSpot::Fake);
  fakeBS_.setCovariance(0, 0, 5e-10);
  fakeBS_.setCovariance(1, 1, 5e-10);
  fakeBS_.setCovariance(2, 2, 0.002);
  fakeBS_.setCovariance(3, 3, 0.002);
  fakeBS_.setCovariance(4, 4, 5e-11);
  fakeBS_.setCovariance(5, 5, 5e-11);
  fakeBS_.setCovariance(6, 6, 1e-09);
  theMaxR2 = iconf.getParameter<double>("maxRadius");
  theMaxR2 *= theMaxR2;

  produces<reco::BeamSpot>();
}

void BeamSpotOnlineProducer::fillDescriptions(edm::ConfigurationDescriptions& iDesc) {
  edm::ParameterSetDescription ps;
  ps.add<bool>("changeToCMSCoordinates", false);
  ps.add<double>("maxZ", 40.);
  ps.add<double>("setSigmaZ", -1.);
  ps.addUntracked<unsigned int>("beamMode", 11);
  ps.addOptional<InputTag>("src", InputTag("hltScalersRawToDigi"))->setComment("SCAL decommissioned after Run 2");
  ps.add<InputTag>("gtEvmLabel", InputTag(""));
  ps.add<double>("maxRadius", 2.0);
  ps.add<bool>("useBSOnlineRecords", false);
  ps.add<int>("timeThreshold", 48)->setComment("hours");
  ps.add<double>("sigmaZThreshold", 2.)->setComment("cm");
  ps.add<double>("sigmaXYThreshold", 4.)->setComment("um");
  iDesc.addWithDefaultLabel(ps);
}

void BeamSpotOnlineProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto result = std::make_unique<reco::BeamSpot>();

  // Determine if we should "shout" based on the beam mode
  bool shoutMODE = shouldShout(iEvent);
  bool fallBackToDB{false};

  // Use transient record if enabled
  if (useBSOnlineRecords_) {
    fallBackToDB = processRecords(iEvent, iSetup, *result, shoutMODE);
  } else {
    // Process online beam spot scalers
    fallBackToDB = processScalers(iEvent, *result, shoutMODE);
  }

  // Fallback to DB if necessary
  if (fallBackToDB) {
    createBeamSpotFromDB(iSetup, *result, shoutMODE);
  }

  iEvent.put(std::move(result));
}

bool BeamSpotOnlineProducer::shouldShout(const edm::Event& iEvent) const {
  edm::Handle<L1GlobalTriggerEvmReadoutRecord> gtEvmReadoutRecord;
  if (iEvent.getByToken(l1GtEvmReadoutRecordToken_, gtEvmReadoutRecord)) {
    return gtEvmReadoutRecord->gtfeWord().beamMode() == theBeamShoutMode;
  }
  return true;  // Default to "shout" if the record is missing
}

bool BeamSpotOnlineProducer::processRecords(const edm::Event& iEvent,
                                            const edm::EventSetup& iSetup,
                                            reco::BeamSpot& result,
                                            bool shoutMODE) {
  auto const spotDBLegacy = getBeamSpotFromRecord(beamTokenLegacy_, iEvent, iSetup);
  auto const spotDBHLT = getBeamSpotFromRecord(beamTokenHLT_, iEvent, iSetup);
  auto const& spotDB = chooseBS(spotDBLegacy, spotDBHLT);

  if (spotDB.beamType() != reco::BeamSpot::Tracker) {
    if (shoutMODE && (beamLegacyRcdESWatcher_.check(iSetup) || beamHLTRcdESWatcher_.check(iSetup))) {
      edm::LogWarning("BeamSpotOnlineProducer")
          << "Online Beam Spot producer falls back to DB value due to fake beamspot in transient record.";
    }
    return true;  // Trigger fallback to DB
  }

  // Create BeamSpot from transient record
  createBeamSpotFromRecord(spotDB, result);
  return false;  // No fallback needed
}

template <typename TokenType>
const BeamSpotOnlineObjects BeamSpotOnlineProducer::getBeamSpotFromRecord(const TokenType& token,
                                                                          const Event& event,
                                                                          const EventSetup& setup) {
  const auto& bs = setup.getData(token);
  if (bs.sigmaZ() < sigmaZThreshold_ || bs.beamWidthX() < sigmaXYThreshold_ || bs.beamWidthY() < sigmaXYThreshold_ ||
      bs.beamType() != reco::BeamSpot::Tracker) {
    return fakeBS_;
  }
  auto evtime = std::chrono::seconds(event.time().value() >> 32);
  auto bstime = std::chrono::microseconds(bs.creationTime());
  auto threshold = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::hours(timeThreshold_)).count();
  if ((evtime - bstime).count() > threshold) {
    return fakeBS_;
  }
  return bs;
}

const BeamSpotOnlineObjects& BeamSpotOnlineProducer::chooseBS(const BeamSpotOnlineObjects& bs1,
                                                              const BeamSpotOnlineObjects& bs2) {
  if (bs1.beamType() == reco::BeamSpot::Tracker && bs2.beamType() == reco::BeamSpot::Tracker) {
    return bs1.sigmaZ() > bs2.sigmaZ() ? bs1 : bs2;
  } else if (bs1.beamType() == reco::BeamSpot::Tracker) {
    return bs1;
  } else if (bs2.beamType() == reco::BeamSpot::Tracker) {
    return bs2;
  } else {
    return fakeBS_;
  }
}

void BeamSpotOnlineProducer::createBeamSpotFromRecord(const BeamSpotObjects& spotDB, reco::BeamSpot& result) const {
  double f = changeFrame_ ? -1.0 : 1.0;
  reco::BeamSpot::Point apoint(f * spotDB.x(), f * spotDB.y(), f * spotDB.z());

  reco::BeamSpot::CovarianceMatrix matrix;
  for (int i = 0; i < reco::BeamSpot::dimension; ++i) {
    for (int j = 0; j < reco::BeamSpot::dimension; ++j) {
      matrix(i, j) = spotDB.covariance(i, j);
    }
  }

  double sigmaZ = (theSetSigmaZ > 0) ? theSetSigmaZ : spotDB.sigmaZ();
  result = reco::BeamSpot(apoint, sigmaZ, spotDB.dxdz(), spotDB.dydz(), spotDB.beamWidthX(), matrix);
  result.setBeamWidthY(spotDB.beamWidthY());
  result.setEmittanceX(spotDB.emittanceX());
  result.setEmittanceY(spotDB.emittanceY());
  result.setbetaStar(spotDB.betaStar());
  result.setType(reco::BeamSpot::Tracker);
}

bool BeamSpotOnlineProducer::processScalers(const edm::Event& iEvent, reco::BeamSpot& result, bool shoutMODE) {
  edm::Handle<BeamSpotOnlineCollection> handleScaler;
  iEvent.getByToken(scalerToken_, handleScaler);

  if (handleScaler->empty()) {
    return true;  // Fallback to DB if scaler collection is empty
  }

  // Extract data from scaler
  BeamSpotOnline spotOnline = *(handleScaler->begin());
  createBeamSpotFromScaler(spotOnline, result);

  // Validate the scaler data
  if (isInvalidScaler(spotOnline, shoutMODE)) {
    return true;  // Trigger fallback to DB
  }
  return false;  // No fallback needed
}

void BeamSpotOnlineProducer::createBeamSpotFromScaler(const BeamSpotOnline& spotOnline, reco::BeamSpot& result) const {
  double f = changeFrame_ ? -1.0 : 1.0;
  reco::BeamSpot::Point apoint(f * spotOnline.x(), spotOnline.y(), f * spotOnline.z());

  reco::BeamSpot::CovarianceMatrix matrix;
  matrix(0, 0) = spotOnline.err_x() * spotOnline.err_x();
  matrix(1, 1) = spotOnline.err_y() * spotOnline.err_y();
  matrix(2, 2) = spotOnline.err_z() * spotOnline.err_z();
  matrix(3, 3) = spotOnline.err_sigma_z() * spotOnline.err_sigma_z();

  double sigmaZ = (theSetSigmaZ > 0) ? theSetSigmaZ : spotOnline.sigma_z();
  result = reco::BeamSpot(apoint, sigmaZ, spotOnline.dxdz(), f * spotOnline.dydz(), spotOnline.width_x(), matrix);
  result.setBeamWidthY(spotOnline.width_y());
  result.setEmittanceX(0.0);
  result.setEmittanceY(0.0);
  result.setbetaStar(0.0);
  result.setType(reco::BeamSpot::LHC);
}

bool BeamSpotOnlineProducer::isInvalidScaler(const BeamSpotOnline& spotOnline, bool shoutMODE) const {
  if (spotOnline.x() == 0 && spotOnline.y() == 0 && spotOnline.z() == 0 && spotOnline.width_x() == 0 &&
      spotOnline.width_y() == 0) {
    if (shoutMODE) {
      edm::LogWarning("BeamSpotOnlineProducer")
          << "Online Beam Spot producer falls back to DB due to zero scaler values.";
    }
    return true;
  }

  double r2 = spotOnline.x() * spotOnline.x() + spotOnline.y() * spotOnline.y();
  if (std::abs(spotOnline.z()) >= theMaxZ || r2 >= theMaxR2) {
    if (shoutMODE) {
      edm::LogError("BeamSpotOnlineProducer")
          << "Online Beam Spot producer falls back to DB due to out-of-range scaler values: " << spotOnline.x() << ", "
          << spotOnline.y() << ", " << spotOnline.z();
    }
    return true;
  }
  return false;
}

void BeamSpotOnlineProducer::createBeamSpotFromDB(const edm::EventSetup& iSetup,
                                                  reco::BeamSpot& result,
                                                  bool shoutMODE) const {
  edm::ESHandle<BeamSpotObjects> beamhandle = iSetup.getHandle(beamToken_);
  const BeamSpotObjects* spotDB = beamhandle.product();

  reco::BeamSpot::Point apoint(spotDB->x(), spotDB->y(), spotDB->z());

  reco::BeamSpot::CovarianceMatrix matrix;
  for (int i = 0; i < reco::BeamSpot::dimension; ++i) {
    for (int j = 0; j < reco::BeamSpot::dimension; ++j) {
      matrix(i, j) = spotDB->covariance(i, j);
    }
  }

  result = reco::BeamSpot(apoint, spotDB->sigmaZ(), spotDB->dxdz(), spotDB->dydz(), spotDB->beamWidthX(), matrix);
  result.setBeamWidthY(spotDB->beamWidthY());
  result.setEmittanceX(spotDB->emittanceX());
  result.setEmittanceY(spotDB->emittanceY());
  result.setbetaStar(spotDB->betaStar());
  result.setType(reco::BeamSpot::Tracker);

  GlobalError bse(result.rotatedCovariance3D());
  if ((bse.cxx() <= 0.0) || (bse.cyy() <= 0.0) || (bse.czz() <= 0.0)) {
    edm::LogError("UnusableBeamSpot") << "Beamspot from DB fallback has invalid errors: " << result.covariance();
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(BeamSpotOnlineProducer);
