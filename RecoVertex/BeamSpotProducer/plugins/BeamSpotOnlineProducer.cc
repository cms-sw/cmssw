/**_________________________________________________________________
   class:   BeamSpotOnlineProducer.h
   package: RecoVertex/BeamSpotProducer



 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)
 modified by: Simone Gennai, INFN MIB


________________________________________________________________**/

#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"
#include "CondFormats/DataRecord/interface/BeamSpotObjectsRcd.h"
#include "CondFormats/DataRecord/interface/BeamSpotTransientObjectsRcd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/Handle.h"
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
  const bool changeFrame_;
  const double theMaxZ, theSetSigmaZ;
  double theMaxR2;
  const bool useTransientRecord_;
  const edm::EDGetTokenT<BeamSpotOnlineCollection> scalerToken_;
  const edm::EDGetTokenT<L1GlobalTriggerEvmReadoutRecord> l1GtEvmReadoutRecordToken_;
  const edm::ESGetToken<BeamSpotObjects, BeamSpotObjectsRcd> beamToken_;
  const edm::ESGetToken<BeamSpotObjects, BeamSpotTransientObjectsRcd> beamTransientToken_;

  // watch IOV transition to emit warnings
  edm::ESWatcher<BeamSpotTransientObjectsRcd> beamTransientRcdESWatcher_;

  const unsigned int theBeamShoutMode;
};

using namespace edm;

BeamSpotOnlineProducer::BeamSpotOnlineProducer(const ParameterSet& iconf)
    : changeFrame_(iconf.getParameter<bool>("changeToCMSCoordinates")),
      theMaxZ(iconf.getParameter<double>("maxZ")),
      theSetSigmaZ(iconf.getParameter<double>("setSigmaZ")),
      useTransientRecord_(iconf.getParameter<bool>("useTransientRecord")),
      scalerToken_(useTransientRecord_ ? edm::EDGetTokenT<BeamSpotOnlineCollection>()
                                       : consumes<BeamSpotOnlineCollection>(iconf.getParameter<InputTag>("src"))),
      l1GtEvmReadoutRecordToken_(consumes<L1GlobalTriggerEvmReadoutRecord>(iconf.getParameter<InputTag>("gtEvmLabel"))),
      beamToken_(esConsumes<BeamSpotObjects, BeamSpotObjectsRcd>()),
      beamTransientToken_(esConsumes<BeamSpotObjects, BeamSpotTransientObjectsRcd>()),
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
  ps.add<bool>("useTransientRecord", false);
  iDesc.addWithDefaultLabel(ps);
}

void BeamSpotOnlineProducer::produce(Event& iEvent, const EventSetup& iSetup) {
  // product is a reco::BeamSpot object
  auto result = std::make_unique<reco::BeamSpot>();
  reco::BeamSpot aSpot;
  //shout MODE only in stable beam
  bool shoutMODE = false;
  edm::Handle<L1GlobalTriggerEvmReadoutRecord> gtEvmReadoutRecord;
  if (iEvent.getByToken(l1GtEvmReadoutRecordToken_, gtEvmReadoutRecord)) {
    if (gtEvmReadoutRecord->gtfeWord().beamMode() == theBeamShoutMode)
      shoutMODE = true;
  } else {
    shoutMODE = true;
  }
  bool fallBackToDB = false;
  if (useTransientRecord_) {
    auto const& spotDB = iSetup.getData(beamTransientToken_);
    if (spotDB.beamType() != 2) {
      if (shoutMODE && beamTransientRcdESWatcher_.check(iSetup)) {
        edm::LogWarning("BeamSpotFromDB")
            << "Online Beam Spot producer falls back to DB value because the ESProducer returned a fake beamspot ";
      }
      fallBackToDB = true;
    } else {
      // translate from BeamSpotObjects to reco::BeamSpot
      // in case we need to switch to LHC reference frame
      // ignore for the moment rotations, and translations
      double f = 1.;
      if (changeFrame_)
        f = -1.;
      reco::BeamSpot::Point apoint(f * spotDB.x(), f * spotDB.y(), f * spotDB.z());

      reco::BeamSpot::CovarianceMatrix matrix;
      for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < 7; ++j) {
          matrix(i, j) = spotDB.covariance(i, j);
        }
      }
      double sigmaZ = spotDB.sigmaZ();
      if (theSetSigmaZ > 0)
        sigmaZ = theSetSigmaZ;

      // this assume beam width same in x and y
      aSpot = reco::BeamSpot(apoint, sigmaZ, spotDB.dxdz(), spotDB.dydz(), spotDB.beamWidthX(), matrix);
      aSpot.setBeamWidthY(spotDB.beamWidthY());
      aSpot.setEmittanceX(spotDB.emittanceX());
      aSpot.setEmittanceY(spotDB.emittanceY());
      aSpot.setbetaStar(spotDB.betaStar());
      aSpot.setType(reco::BeamSpot::Tracker);
    }
  } else {
    // get scalar collection
    Handle<BeamSpotOnlineCollection> handleScaler;
    iEvent.getByToken(scalerToken_, handleScaler);

    // beam spot scalar object
    BeamSpotOnline spotOnline;

    // product is a reco::BeamSpot object
    auto result = std::make_unique<reco::BeamSpot>();

    if (!handleScaler->empty()) {
      // get one element
      spotOnline = *(handleScaler->begin());

      // in case we need to switch to LHC reference frame
      // ignore for the moment rotations, and translations
      double f = 1.;
      if (changeFrame_)
        f = -1.;

      reco::BeamSpot::Point apoint(f * spotOnline.x(), spotOnline.y(), f * spotOnline.z());

      reco::BeamSpot::CovarianceMatrix matrix;
      matrix(0, 0) = spotOnline.err_x() * spotOnline.err_x();
      matrix(1, 1) = spotOnline.err_y() * spotOnline.err_y();
      matrix(2, 2) = spotOnline.err_z() * spotOnline.err_z();
      matrix(3, 3) = spotOnline.err_sigma_z() * spotOnline.err_sigma_z();

      double sigmaZ = spotOnline.sigma_z();
      if (theSetSigmaZ > 0)
        sigmaZ = theSetSigmaZ;

      aSpot = reco::BeamSpot(apoint, sigmaZ, spotOnline.dxdz(), f * spotOnline.dydz(), spotOnline.width_x(), matrix);

      aSpot.setBeamWidthY(spotOnline.width_y());
      aSpot.setEmittanceX(0.);
      aSpot.setEmittanceY(0.);
      aSpot.setbetaStar(0.);
      aSpot.setType(reco::BeamSpot::LHC);  // flag value from scalars

      // check if we have a valid beam spot fit result from online DQM
      if (spotOnline.x() == 0 && spotOnline.y() == 0 && spotOnline.z() == 0 && spotOnline.width_x() == 0 &&
          spotOnline.width_y() == 0) {
        if (shoutMODE) {
          edm::LogWarning("BeamSpotFromDB")
              << "Online Beam Spot producer falls back to DB value because the scaler values are zero ";
        }
        fallBackToDB = true;
      }
      double r2 = spotOnline.x() * spotOnline.x() + spotOnline.y() * spotOnline.y();
      if (std::abs(spotOnline.z()) >= theMaxZ || r2 >= theMaxR2) {
        if (shoutMODE) {
          edm::LogError("BeamSpotFromDB")
              << "Online Beam Spot producer falls back to DB value because the scaler values are too big to be true :"
              << spotOnline.x() << " " << spotOnline.y() << " " << spotOnline.z();
        }
        fallBackToDB = true;
      }
    } else {
      //empty online beamspot collection: FED data was empty
      //the error should probably have been send at unpacker level
      fallBackToDB = true;
    }
  }
  if (fallBackToDB) {
    edm::ESHandle<BeamSpotObjects> beamhandle = iSetup.getHandle(beamToken_);
    const BeamSpotObjects* spotDB = beamhandle.product();

    // translate from BeamSpotObjects to reco::BeamSpot
    reco::BeamSpot::Point apoint(spotDB->x(), spotDB->y(), spotDB->z());

    reco::BeamSpot::CovarianceMatrix matrix;
    for (int i = 0; i < 7; ++i) {
      for (int j = 0; j < 7; ++j) {
        matrix(i, j) = spotDB->covariance(i, j);
      }
    }

    // this assume beam width same in x and y
    aSpot = reco::BeamSpot(apoint, spotDB->sigmaZ(), spotDB->dxdz(), spotDB->dydz(), spotDB->beamWidthX(), matrix);
    aSpot.setBeamWidthY(spotDB->beamWidthY());
    aSpot.setEmittanceX(spotDB->emittanceX());
    aSpot.setEmittanceY(spotDB->emittanceY());
    aSpot.setbetaStar(spotDB->betaStar());
    aSpot.setType(reco::BeamSpot::Tracker);
  }

  *result = aSpot;

  iEvent.put(std::move(result));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(BeamSpotOnlineProducer);
