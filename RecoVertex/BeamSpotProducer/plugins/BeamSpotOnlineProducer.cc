/**_________________________________________________________________
   class:   BeamSpotOnlineProducer.h
   package: RecoVertex/BeamSpotProducer



 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)
 modified by: Simone Gennai, INFN MIB


________________________________________________________________**/

#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"
#include "CondFormats/DataRecord/interface/BeamSpotObjectsRcd.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotOnlineObjects.h"
#include "CondFormats/DataRecord/interface/BeamSpotOnlineHLTObjectsRcd.h"
#include "CondFormats/DataRecord/interface/BeamSpotOnlineLegacyObjectsRcd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"
#include "DataFormats/Scalers/interface/BeamSpotOnline.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/TypeDemangler.h"

#include <optional>

class BeamSpotOnlineProducer : public edm::stream::EDProducer<> {
public:
  /// constructor
  explicit BeamSpotOnlineProducer(const edm::ParameterSet& iConf);

  void beginLuminosityBlock(const edm::LuminosityBlock& lumi, const edm::EventSetup& setup) override;

  /// produce a beam spot class
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  ///Fill descriptor
  static void fillDescriptions(edm::ConfigurationDescriptions& iDesc);

private:
  // helper methods
  bool shouldShout(const edm::Event& iEvent) const;
  std::optional<reco::BeamSpot> processRecords(const edm::LuminosityBlock& iLumi,
                                               const edm::EventSetup& iSetup,
                                               bool shoutMODE) const;
  reco::BeamSpot createBeamSpotFromRecord(const BeamSpotObjects& spotDB) const;
  template <typename RecordType, typename TokenType>
  const BeamSpotOnlineObjects* getBeamSpotFromRecord(const TokenType& token,
                                                     const edm::LuminosityBlock& lumi,
                                                     const edm::EventSetup& setup) const;
  const BeamSpotOnlineObjects* chooseBS(const BeamSpotOnlineObjects* bs1, const BeamSpotOnlineObjects* bs2) const;
  std::optional<reco::BeamSpot> processScalers(const edm::Event& iEvent, bool shoutMODE) const;
  reco::BeamSpot createBeamSpotFromScaler(const BeamSpotOnline& spotOnline) const;
  bool isInvalidScaler(const BeamSpotOnline& spotOnline, bool shoutMODE) const;
  reco::BeamSpot createBeamSpotFromDB(const edm::EventSetup& iSetup, bool shoutMODE) const;

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

  std::optional<reco::BeamSpot> lumiResult_;
  const unsigned int theBeamShoutMode;
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
      beamToken_(esConsumes<BeamSpotObjects, BeamSpotObjectsRcd, edm::Transition::BeginLuminosityBlock>()),
      beamTokenLegacy_(
          esConsumes<BeamSpotOnlineObjects, BeamSpotOnlineLegacyObjectsRcd, edm::Transition::BeginLuminosityBlock>()),
      beamTokenHLT_(
          esConsumes<BeamSpotOnlineObjects, BeamSpotOnlineHLTObjectsRcd, edm::Transition::BeginLuminosityBlock>()),
      theBeamShoutMode(iconf.getUntrackedParameter<unsigned int>("beamMode", 11)) {
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

void BeamSpotOnlineProducer::beginLuminosityBlock(const edm::LuminosityBlock& lumi, const edm::EventSetup& setup) {
  /// fetch online records only at the beginning of a lumisection
  if (useBSOnlineRecords_) {
    lumiResult_ = processRecords(lumi, setup, true);
    if (lumiResult_)
      return;
  }
  lumiResult_ = createBeamSpotFromDB(setup, true);
}

void BeamSpotOnlineProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Determine if we should "shout" based on the beam mode
  bool shoutMODE = shouldShout(iEvent);

  std::unique_ptr<reco::BeamSpot> toput;
  if (not useBSOnlineRecords_) {
    // Process online beam spot scalers
    auto bs = processScalers(iEvent, shoutMODE);
    if (bs) {
      toput = std::make_unique<reco::BeamSpot>(std::move(*bs));
    }
  }
  if (not toput) {
    assert(lumiResult_);
    toput = std::make_unique<reco::BeamSpot>(*lumiResult_);
  }
  iEvent.put(std::move(toput));
}

bool BeamSpotOnlineProducer::shouldShout(const edm::Event& iEvent) const {
  edm::Handle<L1GlobalTriggerEvmReadoutRecord> gtEvmReadoutRecord;
  if (iEvent.getByToken(l1GtEvmReadoutRecordToken_, gtEvmReadoutRecord)) {
    return gtEvmReadoutRecord->gtfeWord().beamMode() == theBeamShoutMode;
  }
  return true;  // Default to "shout" if the record is missing
}

std::optional<reco::BeamSpot> BeamSpotOnlineProducer::processRecords(const edm::LuminosityBlock& iLumi,
                                                                     const edm::EventSetup& iSetup,
                                                                     bool shoutMODE) const {
  auto const* spotDBLegacy = getBeamSpotFromRecord<BeamSpotOnlineLegacyObjectsRcd>(beamTokenLegacy_, iLumi, iSetup);
  auto const* spotDBHLT = getBeamSpotFromRecord<BeamSpotOnlineHLTObjectsRcd>(beamTokenHLT_, iLumi, iSetup);
  auto const* spotDB = chooseBS(spotDBLegacy, spotDBHLT);

  if (not spotDB) {
    if (shoutMODE) {
      edm::LogWarning("BeamSpotOnlineProducer") << "None of the online records holds a valid beam spot. The Online "
                                                   "Beam Spot producer falls back to the PCL value.";
    }
    return {};  // Trigger fallback to DB
  }

  // Create BeamSpot from transient record
  return createBeamSpotFromRecord(*spotDB);
}

template <typename RecordType, typename TokenType>
const BeamSpotOnlineObjects* BeamSpotOnlineProducer::getBeamSpotFromRecord(const TokenType& token,
                                                                           const LuminosityBlock& lumi,
                                                                           const EventSetup& setup) const {
  auto const bsRecord = setup.tryToGet<RecordType>();
  if (!bsRecord) {
    const auto& recordTypeName = edm::typeDemangle(typeid(RecordType).name());
    edm::LogWarning("BeamSpotOnlineProducer") << "No " << recordTypeName << " found in the EventSetup";
    return nullptr;
  }
  const auto& bsHandle = setup.getHandle(token);
  if (bsHandle.isValid()) {
    const auto& bs = *bsHandle;
    if (bs.sigmaZ() < sigmaZThreshold_ || bs.beamWidthX() < sigmaXYThreshold_ || bs.beamWidthY() < sigmaXYThreshold_ ||
        bs.beamType() != reco::BeamSpot::Tracker) {
      const auto& recordTypeName = edm::typeDemangle(typeid(RecordType).name());
      edm::LogWarning("BeamSpotOnlineProducer")
          << "The beam spot record does not pass the fit sanity checks (record: " << recordTypeName << ")" << std::endl
          << "sigmaZ: " << bs.sigmaZ() << std::endl
          << "sigmaX: " << bs.beamWidthX() << std::endl
          << "sigmaY: " << bs.beamWidthY() << std::endl
          << "type: " << bs.beamType();
      return nullptr;
    }
    auto lumitime = std::chrono::seconds(lumi.beginTime().unixTime());
    auto bstime = std::chrono::microseconds(bs.creationTime());
    auto threshold = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::hours(timeThreshold_)).count();
    if ((lumitime - bstime).count() > threshold) {
      const auto& recordTypeName = edm::typeDemangle(typeid(RecordType).name());
      edm::LogWarning("BeamSpotOnlineProducer")
          << "The beam spot record is too old. (record: " << recordTypeName << ")" << std::endl
          << " record creation time: " << std::chrono::duration_cast<std::chrono::seconds>(bstime).count()
          << " lumi block time: " << std::chrono::duration_cast<std::chrono::seconds>(lumitime).count();
      return nullptr;
    }
    return &bs;
  } else {
    const auto& recordTypeName = edm::typeDemangle(typeid(RecordType).name());
    edm::LogWarning("BeamSpotOnlineProducer") << "Invalid online beam spot handle for the record: " << recordTypeName;
    return nullptr;
  }
}

const BeamSpotOnlineObjects* BeamSpotOnlineProducer::chooseBS(const BeamSpotOnlineObjects* bs1,
                                                              const BeamSpotOnlineObjects* bs2) const {
  if (bs1 and bs2 and bs1->beamType() == reco::BeamSpot::Tracker && bs2->beamType() == reco::BeamSpot::Tracker) {
    return bs1->sigmaZ() > bs2->sigmaZ() ? bs1 : bs2;
  } else if (bs1 and bs1->beamType() == reco::BeamSpot::Tracker) {
    return bs1;
  } else if (bs2 and bs2->beamType() == reco::BeamSpot::Tracker) {
    return bs2;
  } else {
    return nullptr;
  }
}

reco::BeamSpot BeamSpotOnlineProducer::createBeamSpotFromRecord(const BeamSpotObjects& spotDB) const {
  double f = changeFrame_ ? -1.0 : 1.0;
  reco::BeamSpot::Point apoint(f * spotDB.x(), f * spotDB.y(), f * spotDB.z());

  reco::BeamSpot::CovarianceMatrix matrix;
  for (int i = 0; i < reco::BeamSpot::dimension; ++i) {
    for (int j = 0; j < reco::BeamSpot::dimension; ++j) {
      matrix(i, j) = spotDB.covariance(i, j);
    }
  }

  double sigmaZ = (theSetSigmaZ > 0) ? theSetSigmaZ : spotDB.sigmaZ();
  reco::BeamSpot result(apoint, sigmaZ, spotDB.dxdz(), spotDB.dydz(), spotDB.beamWidthX(), matrix);
  result.setBeamWidthY(spotDB.beamWidthY());
  result.setEmittanceX(spotDB.emittanceX());
  result.setEmittanceY(spotDB.emittanceY());
  result.setbetaStar(spotDB.betaStar());
  result.setType(reco::BeamSpot::Tracker);
  return result;
}

std::optional<reco::BeamSpot> BeamSpotOnlineProducer::processScalers(const edm::Event& iEvent, bool shoutMODE) const {
  edm::Handle<BeamSpotOnlineCollection> handleScaler;
  iEvent.getByToken(scalerToken_, handleScaler);

  if (handleScaler->empty()) {
    if (shoutMODE && iEvent.isRealData()) {
      edm::LogWarning("BeamSpotOnlineProducer") << " Scalers handle is empty. The Online "
                                                   "Beam Spot producer falls back to the PCL value.";
    }
    return {};
  }

  // Extract data from scaler
  BeamSpotOnline spotOnline = *(handleScaler->begin());
  // Validate the scaler data
  if (isInvalidScaler(spotOnline, shoutMODE)) {
    return {};  // Trigger fallback to DB
  }
  return createBeamSpotFromScaler(spotOnline);
}

reco::BeamSpot BeamSpotOnlineProducer::createBeamSpotFromScaler(const BeamSpotOnline& spotOnline) const {
  double f = changeFrame_ ? -1.0 : 1.0;
  reco::BeamSpot::Point apoint(f * spotOnline.x(), spotOnline.y(), f * spotOnline.z());

  reco::BeamSpot::CovarianceMatrix matrix;
  matrix(0, 0) = spotOnline.err_x() * spotOnline.err_x();
  matrix(1, 1) = spotOnline.err_y() * spotOnline.err_y();
  matrix(2, 2) = spotOnline.err_z() * spotOnline.err_z();
  matrix(3, 3) = spotOnline.err_sigma_z() * spotOnline.err_sigma_z();

  double sigmaZ = (theSetSigmaZ > 0) ? theSetSigmaZ : spotOnline.sigma_z();
  reco::BeamSpot result(apoint, sigmaZ, spotOnline.dxdz(), f * spotOnline.dydz(), spotOnline.width_x(), matrix);
  result.setBeamWidthY(spotOnline.width_y());
  result.setEmittanceX(0.0);
  result.setEmittanceY(0.0);
  result.setbetaStar(0.0);
  result.setType(reco::BeamSpot::LHC);
  return result;
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

reco::BeamSpot BeamSpotOnlineProducer::createBeamSpotFromDB(const edm::EventSetup& iSetup, bool shoutMODE) const {
  edm::ESHandle<BeamSpotObjects> beamhandle = iSetup.getHandle(beamToken_);
  const BeamSpotObjects* spotDB = beamhandle.product();

  reco::BeamSpot::Point apoint(spotDB->x(), spotDB->y(), spotDB->z());

  reco::BeamSpot::CovarianceMatrix matrix;
  for (int i = 0; i < reco::BeamSpot::dimension; ++i) {
    for (int j = 0; j < reco::BeamSpot::dimension; ++j) {
      matrix(i, j) = spotDB->covariance(i, j);
    }
  }

  reco::BeamSpot result(apoint, spotDB->sigmaZ(), spotDB->dxdz(), spotDB->dydz(), spotDB->beamWidthX(), matrix);
  result.setBeamWidthY(spotDB->beamWidthY());
  result.setEmittanceX(spotDB->emittanceX());
  result.setEmittanceY(spotDB->emittanceY());
  result.setbetaStar(spotDB->betaStar());
  result.setType(reco::BeamSpot::Tracker);

  GlobalError bse(result.rotatedCovariance3D());
  if ((bse.cxx() <= 0.0) || (bse.cyy() <= 0.0) || (bse.czz() <= 0.0)) {
    edm::LogError("UnusableBeamSpot") << "Beamspot from DB fallback has invalid errors: " << result.covariance();
  }
  return result;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(BeamSpotOnlineProducer);
