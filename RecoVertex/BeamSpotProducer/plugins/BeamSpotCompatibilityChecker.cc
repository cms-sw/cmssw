// -*- C++ -*-
//
// Package:    RecoVertex/BeamSpotProducer
// Class:      BeamSpotCompatibilityChecker
//
/**\class BeamSpotCompatibilityChecker BeamSpotCompatibilityChecker.cc RecoVertex/BeamSpotProducer/plugins/BeamSpotCompatibilityChecker.cc

 Description: Class to check the compatibility between the BeamSpot payload in the database and the one in the file

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
#include "CondFormats/DataRecord/interface/BeamSpotTransientObjectsRcd.h"
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

    inline void printBeamSpotComparison(
        const std::string& varName, double A, double ErrA, double B, double ErrB, double combinedError) {
      edm::LogPrint("BeamSpotCompatibilityChecker")
          << std::fixed << std::setprecision(6) << std::left << std::setw(10) << " " + varName << std::right
          << std::setw(12) << A << std::setw(14) << ErrA << std::setw(12) << B << std::setw(14) << ErrB << std::setw(14)
          << std::abs(A - B) << std::setw(14) << combinedError << std::setw(12) << std::abs(A - B) / combinedError
          << std::endl;
    }

    float getSig(const bool verbose) {
      if (verbose) {
        printBeamSpotComparison(m_var, m_A, m_ErrA, m_B, m_ErrB, m_combinedError);
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

  // beamSpot from the file
  const edm::EDGetTokenT<reco::BeamSpot> bsFromFileToken_;

  // switch to decide with record to take
  bool useTransientRecord_;

  // beamSpot from the DB (object record)
  edm::ESGetToken<BeamSpotObjects, BeamSpotObjectsRcd> bsFromDBToken_;

  // beamspot from the DB (transient record)
  edm::ESGetToken<BeamSpotObjects, BeamSpotTransientObjectsRcd> bsFromTransientDBToken_;

  // beamSpot from the DB (via event)
  edm::EDGetTokenT<reco::BeamSpot> dbBSfromEventToken_;

  template <typename RecordT>
  reco::BeamSpot getBeamSpotFromES(edm::EventSetup const& iSetup,
                                   edm::ESGetToken<BeamSpotObjects, RecordT> const& token) const {
    edm::ESHandle<BeamSpotObjects> beamhandle = iSetup.getHandle(token);
    const BeamSpotObjects* aSpot = beamhandle.product();

    // translate from BeamSpotObjects to reco::BeamSpot
    reco::BeamSpot::Point apoint(aSpot->x(), aSpot->y(), aSpot->z());

    reco::BeamSpot::CovarianceMatrix matrix;
    for (int i = 0; i < reco::BeamSpot::dimension; ++i)
      for (int j = 0; j < reco::BeamSpot::dimension; ++j)
        matrix(i, j) = aSpot->covariance(i, j);

    // this assume beam width same in x and y
    return reco::BeamSpot(apoint, aSpot->sigmaZ(), aSpot->dxdz(), aSpot->dydz(), aSpot->beamWidthX(), matrix);
  }
};

//
// constructors and destructor
//
BeamSpotCompatibilityChecker::BeamSpotCompatibilityChecker(const edm::ParameterSet& iConfig)
    : warningThreshold_(iConfig.getParameter<double>("warningThr")),
      throwingThreshold_(iConfig.getParameter<double>("errorThr")),
      verbose_(iConfig.getUntrackedParameter<bool>("verbose", false)),
      dbFromEvent_(iConfig.getParameter<bool>("dbFromEvent")),
      bsFromFileToken_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("bsFromFile"))) {
  //now do what ever initialization is needed
  if (warningThreshold_ > throwingThreshold_) {
    throw cms::Exception("ConfigurationError")
        << __PRETTY_FUNCTION__ << "\n Warning threshold (" << warningThreshold_
        << ") cannot be smaller than the throwing threshold (" << throwingThreshold_ << ")" << std::endl;
  }

  if (dbFromEvent_) {
    edm::LogInfo("BeamSpotCompatibilityChecker")
        << "The Database Beam Spot is going to be taken from the File via BeamSpotProducer!";
    dbBSfromEventToken_ = consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("bsFromDB"));
  } else {
    useTransientRecord_ = iConfig.getParameter<bool>("useTransientRecord");
    if (useTransientRecord_) {
      edm::LogInfo("BeamSpotCompatibilityChecker") << "Using BeamSpot from BeamSpotTransientObjectsRcd.";
      bsFromTransientDBToken_ = esConsumes<BeamSpotObjects, BeamSpotTransientObjectsRcd>();
    } else {
      edm::LogInfo("BeamSpotCompatibilityChecker") << "Using BeamSpot from BeamSpotObjectsRcd.";
      bsFromDBToken_ = esConsumes<BeamSpotObjects, BeamSpotObjectsRcd>();
    }
    dbBSfromEventToken_ = edm::EDGetTokenT<reco::BeamSpot>();
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
  reco::BeamSpot spotFile, spotDB;

  edm::Handle<reco::BeamSpot> beamSpotFromFileHandle;
  iEvent.getByToken(bsFromFileToken_, beamSpotFromFileHandle);
  spotFile = *beamSpotFromFileHandle;

  double file_BSx0 = spotFile.x0();
  double file_BSy0 = spotFile.y0();
  double file_BSz0 = spotFile.z0();
  double file_Beamsigmaz = spotFile.sigmaZ();
  double file_BeamWidthX = spotFile.BeamWidthX();
  double file_BeamWidthY = spotFile.BeamWidthY();

  if (!dbFromEvent_) {
    if (useTransientRecord_) {
      spotDB = getBeamSpotFromES(iSetup, bsFromTransientDBToken_);
    } else {
      spotDB = getBeamSpotFromES(iSetup, bsFromDBToken_);
    }
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

  double deltaX0 = file_BSx0 - db_BSx0;
  double deltaY0 = file_BSy0 - db_BSy0;
  double deltaZ0 = file_BSz0 - db_BSz0;
  double deltaSigmaX = file_BeamWidthX - db_BeamWidthX;
  double deltaSigmaY = file_BeamWidthY - db_BeamWidthY;
  double deltaSigmaZ = file_Beamsigmaz - db_Beamsigmaz;

  if (verbose_) {
    edm::LogPrint("BeamSpotCompatibilityChecker") << "BS from DB:   \n" << spotDB << std::endl;
    edm::LogPrint("BeamSpotCompatibilityChecker") << "BS from File: \n" << spotFile << std::endl;
  }

  auto significances = compareBS(spotDB, spotFile, verbose_);
  std::vector<std::string> labels = {"x0", "y0", "z0", "sigmaX", "sigmaY", "sigmaZ"};

  std::string msg = " |delta X0|     = " + std::to_string(std::abs(deltaX0) * cmToum) +
                    " um\n |delta Y0|     = " + std::to_string(std::abs(deltaY0) * cmToum) +
                    " um\n |delta Z0|     = " + std::to_string(std::abs(deltaZ0)) +
                    " cm\n |delta sigmaX| = " + std::to_string(std::abs(deltaSigmaX) * cmToum) +
                    " um\n |delta sigmaY| = " + std::to_string(std::abs(deltaSigmaY) * cmToum) +
                    " um\n |delta sigmaZ| = " + std::to_string(std::abs(deltaSigmaZ)) + " cm";
  /*
  if (verbose_) {
    edm::LogPrint("BeamSpotCompatibilityChecker") << msg.c_str() << std::endl;
  }
  */

  for (unsigned int i = 0; i < 3; i++) {
    auto sig = significances.at(i);
    if (sig > throwingThreshold_) {
      edm::LogError("BeamSpotCompatibilityChecker") << msg.c_str() << std::endl;
      throw cms::Exception("BeamSpotCompatibilityChecker")
          << "   [" << __PRETTY_FUNCTION__ << "]\n   DB-File BeamSpot " << labels.at(i) << " distance significance is "
          << sig << ", exceeds the threshold of " << throwingThreshold_ << "!" << std::endl;
    } else if (sig > warningThreshold_) {
      edm::LogWarning("BeamSpotCompatibilityChecker") << msg.c_str() << std::endl;
      edm::LogWarning("BeamSpotCompatibilityChecker")
          << "   [" << __PRETTY_FUNCTION__ << "]\n   DB-File BeamSpot " << labels.at(i) << " distance significance is "
          << sig << ", exceeds the threshold of " << warningThreshold_ << "!" << std::endl;
    }
  }
}

std::array<float, 6> BeamSpotCompatibilityChecker::compareBS(const reco::BeamSpot& spotA,
                                                             const reco::BeamSpot& spotB,
                                                             const bool verbose) {
  if (verbose) {
    edm::LogPrint("BeamSpotCompatibilityChecker")
        << std::fixed << std::setprecision(6) << std::left << std::setw(10) << " Var" << std::right << std::setw(12)
        << "DB" << std::setw(14) << "+/-" << std::setw(12) << "File" << std::setw(14) << "+/-" << std::setw(14)
        << "|delta|" << std::setw(14) << "+/-" << std::setw(12) << "Sig";
    edm::LogPrint("BeamSpotCompatibilityChecker") << std::string(102, '-');
  }

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

  if (verbose) {
    edm::LogPrint("BeamSpotCompatibilityChecker") << std::string(102, '-');
  }

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
  desc.add<edm::InputTag>("bsFromFile", edm::InputTag(""))
      ->setComment("edm::InputTag on the BeamSpot from the File (Reference)");

  // Conditional parameters based on dbFromEvent
  desc.ifValue(
      edm::ParameterDescription<bool>(
          "dbFromEvent",
          true,
          true,
          edm::Comment("Switch to take the (target) DB beamspot from the event instead of the EventSetup")),
      true >> edm::ParameterDescription<edm::InputTag>(
                  "bsFromDB",
                  edm::InputTag(""),
                  true,
                  edm::Comment("edm::InputTag on the BeamSpot from the Event (Target) (used if dbFromEvent = true")) or
          false >> edm::ParameterDescription<bool>(
                       "useTransientRecord",
                       false,
                       true,
                       edm::Comment("Use transient BeamSpot record (used if dbFromEvent = false)")));

  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(BeamSpotCompatibilityChecker);
