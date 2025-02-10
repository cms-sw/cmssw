// -*- C++ -*-
//
// Package:    RecoVertex/BeamSpotProducer
// Class:      BeamSpotCompatibilityChecker
//
/**\class BeamSpotCompatibilityChecker BeamSpotCompatibilityChecker.cc RecoVertex/BeamSpotProducer/plugins/BeamSpotCompatibilityChecker.cc

 Description: Class to check the compatibility between the BeamSpot payload in the database and the one in the event

 Implementation:
     Makes use of the Significance struct to establish how compatible are the data members of the two BeamSpots in input
*/
//
// Original Author:  Marco Musich
//         Created:  Thu, 23 Apr 2020 09:00:45 GMT
//
//

// system include files
#include <memory>

// user include files
#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"
#include "CondFormats/DataRecord/interface/BeamSpotObjectsRcd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Scalers/interface/BeamSpotOnline.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
//
// class declaration
//

// ancillary struct to check compatibility
namespace {

  struct Significance {
    Significance(const double& a, const double& b, const double& errA, const double& errB, const std::string& var)
        : m_A(a), m_B(b), m_ErrA(errA), m_ErrB(errB), m_var(var) {
      if (m_ErrA == 0 && m_ErrB == 0) {
        edm::LogError("LogicalError") << "Can't calculate significance when both errors are zero!" << std::endl;
      }
      m_combinedError = std::sqrt((m_ErrA * m_ErrA) + (m_ErrB * m_ErrB));
    }

    float getSig(const bool verbose) {
      if (verbose) {
        edm::LogPrint("BeamSpotCompatibilityChecker")
            << std::fixed << std::setprecision(6)  // Set fixed-point format with 3 decimal places
            << m_var << ": A= " << std::setw(10) << m_A << " +/- " << std::setw(10) << m_ErrA
            << "    B= " << std::setw(5) << m_B << " +/- " << std::setw(10) << m_ErrB
            << "    | delta= " << std::setw(10) << std::abs(m_A - m_B) << " +/- " << std::setw(10) << m_combinedError
            << "    Sig= " << std::setw(10) << std::abs(m_A - m_B) / m_combinedError << std::endl;
      }
      return std::abs(m_A - m_B) / m_combinedError;
    }

  private:
    double m_A;
    double m_B;
    double m_ErrA;
    double m_ErrB;
    double m_ErrAB;
    std::string m_var;
    double m_combinedError;
  };
}  // namespace

class BeamSpotCompatibilityChecker : public edm::global::EDAnalyzer<> {
public:
  explicit BeamSpotCompatibilityChecker(const edm::ParameterSet&);
  ~BeamSpotCompatibilityChecker() override = default;

  void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const final;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  static std::array<float, 6> compareBS(const reco::BeamSpot& BSA, const reco::BeamSpot& BSB, const bool verbose);
  static double computeBeamSpotCompatibility(const reco::BeamSpot& beamSpot1, const reco::BeamSpot& beamSpot2);

private:
  // ----------member data ---------------------------
  static constexpr int cmToum = 10000;
  const double warningThreshold_;
  const double throwingThreshold_;
  const bool verbose_;
  const bool dbFromEvent_;
  const edm::EDGetTokenT<reco::BeamSpot> bsFromEventToken_;             // beamSpot from the event
  edm::ESGetToken<BeamSpotObjects, BeamSpotObjectsRcd> bsFromDBToken_;  // beamSpot from the DB
  edm::EDGetTokenT<reco::BeamSpot> dbBSfromEventToken_;                 // beamSpot from the DB (via event)
};

//
// constructors and destructor
//
BeamSpotCompatibilityChecker::BeamSpotCompatibilityChecker(const edm::ParameterSet& iConfig)
    : warningThreshold_(iConfig.getParameter<double>("warningThr")),
      throwingThreshold_(iConfig.getParameter<double>("errorThr")),
      verbose_(iConfig.getUntrackedParameter<bool>("verbose", false)),
      dbFromEvent_(iConfig.getParameter<bool>("dbFromEvent")),
      bsFromEventToken_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("bsFromEvent"))) {
  //now do what ever initialization is needed
  if (warningThreshold_ > throwingThreshold_) {
    throw cms::Exception("ConfigurationError")
        << __PRETTY_FUNCTION__ << "\n Warning threshold (" << warningThreshold_
        << ") cannot be smaller than the throwing threshold (" << throwingThreshold_ << ")" << std::endl;
  }
  if (dbFromEvent_) {
    edm::LogWarning("BeamSpotCompatibilityChecker")
        << "!!!! Warning !!!\nThe Database Beam Spot is going to be taken from the Event via BeamSpotProducer!";
    dbBSfromEventToken_ = consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("bsFromDB"));
  } else {
    bsFromDBToken_ = esConsumes<BeamSpotObjects, BeamSpotObjectsRcd>();
  }
}

//
// member functions
//

// ------------ method called for each event  ------------
void BeamSpotCompatibilityChecker::analyze(edm::StreamID sid,
                                           edm::Event const& iEvent,
                                           edm::EventSetup const& iSetup) const {
  using namespace edm;
  reco::BeamSpot spotEvent, spotDB;

  edm::Handle<reco::BeamSpot> beamSpotFromEventHandle;
  iEvent.getByToken(bsFromEventToken_, beamSpotFromEventHandle);
  spotEvent = *beamSpotFromEventHandle;

  double evt_BSx0 = spotEvent.x0();
  double evt_BSy0 = spotEvent.y0();
  double evt_BSz0 = spotEvent.z0();
  double evt_Beamsigmaz = spotEvent.sigmaZ();
  double evt_BeamWidthX = spotEvent.BeamWidthX();
  double evt_BeamWidthY = spotEvent.BeamWidthY();

  if (!dbFromEvent_) {
    edm::ESHandle<BeamSpotObjects> beamhandle = iSetup.getHandle(bsFromDBToken_);
    const BeamSpotObjects* aSpot = beamhandle.product();

    // translate from BeamSpotObjects to reco::BeamSpot
    reco::BeamSpot::Point apoint(aSpot->x(), aSpot->y(), aSpot->z());

    reco::BeamSpot::CovarianceMatrix matrix;
    for (int i = 0; i < reco::BeamSpot::dimension; ++i) {
      for (int j = 0; j < reco::BeamSpot::dimension; ++j) {
        matrix(i, j) = aSpot->covariance(i, j);
      }
    }

    // this assume beam width same in x and y
    spotDB = reco::BeamSpot(apoint, aSpot->sigmaZ(), aSpot->dxdz(), aSpot->dydz(), aSpot->beamWidthX(), matrix);
  } else {
    // take the DB beamspot from the event (different label)
    edm::Handle<reco::BeamSpot> beamSpotFromDBHandle;
    iEvent.getByToken(dbBSfromEventToken_, beamSpotFromDBHandle);
    spotDB = *beamSpotFromDBHandle;
  }

  double db_BSx0 = spotDB.x0();
  double db_BSy0 = spotDB.y0();
  double db_BSz0 = spotDB.z0();
  double db_Beamsigmaz = spotDB.sigmaZ();
  double db_BeamWidthX = spotDB.BeamWidthX();
  double db_BeamWidthY = spotDB.BeamWidthY();

  double deltaX0 = evt_BSx0 - db_BSx0;
  double deltaY0 = evt_BSy0 - db_BSy0;
  double deltaZ0 = evt_BSz0 - db_BSz0;
  double deltaSigmaX = evt_BeamWidthX - db_BeamWidthX;
  double deltaSigmaY = evt_BeamWidthY - db_BeamWidthY;
  double deltaSigmaZ = evt_Beamsigmaz - db_Beamsigmaz;

  if (verbose_) {
    edm::LogPrint("BeamSpotCompatibilityChecker") << "BS from Event: \n" << spotEvent << std::endl;
    edm::LogPrint("BeamSpotCompatibilityChecker") << "BS from DB: \n" << spotDB << std::endl;
  }

  auto significances = compareBS(spotDB, spotEvent, verbose_);
  std::vector<std::string> labels = {"x0", "y0", "z0", "sigmaX", "sigmaY", "sigmaZ"};

  std::string msg = " |delta X0|=" + std::to_string(std::abs(deltaX0) * cmToum) +
                    " um |delta Y0|=" + std::to_string(std::abs(deltaY0) * cmToum) +
                    " um |delta Z0|=" + std::to_string(std::abs(deltaZ0) * cmToum) +
                    " um |delta sigmaX|=" + std::to_string(std::abs(deltaSigmaX) * cmToum) +
                    " um |delta sigmaY|=" + std::to_string(std::abs(deltaSigmaY) * cmToum) +
                    " um |delta sigmaZ|=" + std::to_string(std::abs(deltaSigmaZ)) + " cm";
  if (verbose_) {
    edm::LogPrint("BeamSpotCompatibilityChecker") << msg.c_str() << std::endl;
  }

  for (unsigned int i = 0; i < 3; i++) {
    auto sig = significances.at(i);
    if (sig > throwingThreshold_) {
      edm::LogError("BeamSpotCompatibilityChecker") << msg.c_str() << std::endl;
      throw cms::Exception("BeamSpotCompatibilityChecker")
          << "[" << __PRETTY_FUNCTION__ << "] \n DB-Event BeamSpot " << labels.at(i) << " distance sigificance " << sig
          << ", exceeds the threshold of " << throwingThreshold_ << "!" << std::endl;
    } else if (sig > warningThreshold_) {
      edm::LogWarning("BeamSpotCompatibilityChecker") << msg.c_str() << std::endl;
      edm::LogWarning("BeamSpotCompatibilityChecker")
          << "[" << __PRETTY_FUNCTION__ << "]  \n  DB-Event BeamSpot " << labels.at(i) << " distance significance "
          << sig << ", exceeds the threshold of " << warningThreshold_ << "!" << std::endl;
    }
  }
}

std::array<float, 6> BeamSpotCompatibilityChecker::compareBS(const reco::BeamSpot& spotA,
                                                             const reco::BeamSpot& spotB,
                                                             const bool verbose) {
  // Lambda to calculate the significance
  auto calcSignificance = [&](auto a, auto b, auto aErr, auto bErr, auto var) {
    return Significance(a, b, aErr, bErr, var).getSig(verbose);
  };

  // Populate the array using the lambda
  std::array<float, 6> ret = {
      {calcSignificance(spotA.x0(), spotB.x0(), spotA.x0Error(), spotB.x0Error(), "x"),
       calcSignificance(spotA.y0(), spotB.y0(), spotA.y0Error(), spotB.y0Error(), "y"),
       calcSignificance(spotA.z0(), spotB.z0(), spotA.z0Error(), spotB.z0Error(), "z"),
       calcSignificance(
           spotA.BeamWidthX(), spotB.BeamWidthX(), spotA.BeamWidthXError(), spotB.BeamWidthXError(), "widthX"),
       calcSignificance(
           spotA.BeamWidthY(), spotB.BeamWidthY(), spotA.BeamWidthYError(), spotB.BeamWidthYError(), "witdhY"),
       calcSignificance(spotA.sigmaZ(), spotB.sigmaZ(), spotA.sigmaZ0Error(), spotB.sigmaZ0Error(), "witdthZ")}};

  return ret;
}

double BeamSpotCompatibilityChecker::computeBeamSpotCompatibility(const reco::BeamSpot& beamSpot1,
                                                                  const reco::BeamSpot& beamSpot2) {
  reco::Vertex vertex1(
      reco::Vertex::Point(beamSpot1.x0(), beamSpot1.y0(), beamSpot1.z0()), beamSpot1.rotatedCovariance3D(), 0, 0, 0);
  reco::Vertex vertex2(
      reco::Vertex::Point(beamSpot2.x0(), beamSpot2.y0(), beamSpot2.z0()), beamSpot2.rotatedCovariance3D(), 0, 0, 0);

  // Calculate distance and significance using VertexDistance3D
  VertexDistance3D distanceCalculator;
  // double distance = distanceCalculator.distance(vertex1, vertex2).value();  // Euclidean distance
  double significance = distanceCalculator.distance(vertex1, vertex2).significance();  // Distance significance

  // Return the significance as a measure of compatibility
  return significance;  // Smaller values indicate higher compatibility
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void BeamSpotCompatibilityChecker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<double>("warningThr", 1.)->setComment("Threshold on the signficances to emit a warning");
  desc.add<double>("errorThr", 3.)->setComment("Threshold on the signficances to abort the job");
  desc.addUntracked<bool>("verbose", false)->setComment("verbose output");
  desc.add<edm::InputTag>("bsFromEvent", edm::InputTag(""))
      ->setComment("edm::InputTag on the BeamSpot from the Event (Reference)");
  desc.add<bool>("dbFromEvent", false)
      ->setComment("Switch to take the (target) DB beamspot from the event instead of the EventSetup");
  desc.add<edm::InputTag>("bsFromDB", edm::InputTag(""))
      ->setComment("edm::InputTag on the BeamSpot from the Event (Target)\n To be used only if dbFromEvent is True");
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(BeamSpotCompatibilityChecker);
