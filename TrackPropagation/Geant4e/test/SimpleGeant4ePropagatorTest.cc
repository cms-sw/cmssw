#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"  //For define_fwk_module

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

//- Timing
//#include "Utilities/Timing/interface/TimingReport.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"

//- Magnetic field
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "SimG4Core/MagneticField/interface/Field.h"
#include "SimG4Core/MagneticField/interface/FieldBuilder.h"

//- Propagator
#include "TrackPropagation/Geant4e/interface/ConvertFromToCLHEP.h"
#include "TrackPropagation/Geant4e/interface/Geant4ePropagator.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "MagneticField/VolumeGeometry/interface/MagVolumeOutsideValidity.h"
#include "DataFormats/GeometrySurface/interface/PlaneBuilder.h"

//- Geant4
#include "G4TransportationManager.hh"

class SimpleGeant4ePropagatorTest final : public edm::one::EDAnalyzer<> {
public:
  explicit SimpleGeant4ePropagatorTest(const edm::ParameterSet &);
  ~SimpleGeant4ePropagatorTest() override = default;

  void analyze(const edm::Event &, const edm::EventSetup &) override;

protected:
  // tokens
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magFieldToken;

  Propagator *thePropagator;
};

SimpleGeant4ePropagatorTest::SimpleGeant4ePropagatorTest(const edm::ParameterSet &iConfig)
    : magFieldToken(esConsumes()), thePropagator(nullptr) {}

namespace {
  Surface::RotationType rotation(const GlobalVector &zDir) {
    GlobalVector zAxis = zDir.unit();
    GlobalVector yAxis(zAxis.y(), -zAxis.x(), 0);
    GlobalVector xAxis = yAxis.cross(zAxis);
    return Surface::RotationType(xAxis, yAxis, zAxis);
  }
}  // namespace

void SimpleGeant4ePropagatorTest::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  using namespace edm;

  std::cout << "Starting G4e test..." << std::endl;

  ///////////////////////////////////////
  // Construct Magnetic Field
  const ESHandle<MagneticField> bField = iSetup.getHandle(magFieldToken);
  if (bField.isValid())
    std::cout << "G4e -- Magnetic field is valid. Value in (0,0,0): " << bField->inTesla(GlobalPoint(0, 0, 0)).mag()
              << " Tesla " << std::endl;
  else
    LogError("Geant4e") << "G4e -- NO valid Magnetic field" << std::endl;

  // Initialise the propagator
  if (!thePropagator)
    thePropagator = new Geant4ePropagator(bField.product());

  if (thePropagator)
    std::cout << "Propagator built!" << std::endl;
  else
    LogError("Geant4e") << "Could not build propagator!" << std::endl;

  GlobalVector p3T(10., 10., 2.);
  std::cout << "*** Phi (rad): " << p3T.phi() << " - Phi(deg)" << p3T.phi().degrees();
  std::cout << "Track P.: " << p3T << "\nTrack P.: PT=" << p3T.perp() << "\tEta=" << p3T.eta()
            << "\tPhi=" << p3T.phi().degrees() << "--> Rad: Phi=" << p3T.phi() << std::endl;

  GlobalPoint r3T(0., 0., 0.);
  std::cout << "Init point: " << r3T << "\nInit point Ro=" << r3T.perp() << "\tEta=" << r3T.eta()
            << "\tPhi=" << r3T.phi().degrees() << std::endl;

  //- Charge
  int charge = 1;
  std::cout << "Track charge = " << charge << std::endl;

  //- Initial covariance matrix is unity 10-6
  ROOT::Math::SMatrixIdentity id;
  AlgebraicSymMatrix55 C(id);
  C *= 0.01;
  CurvilinearTrajectoryError covT(C);

  PlaneBuilder pb;
  Surface::RotationType rot = rotation(p3T);
  // Define end planes
  for (float d = 50.; d < 700.; d += 50.) {
    float propDistance = d;  // 100 cm
    std::cout << "G4e -- Extrapolatation distance " << d << " cm" << std::endl;
    GlobalPoint targetPos = r3T + propDistance * p3T.unit();
    auto endPlane = pb.plane(targetPos, rot);

    //- Build FreeTrajectoryState
    GlobalTrajectoryParameters trackPars(r3T, p3T, charge, &*bField);
    FreeTrajectoryState ftsTrack(trackPars, covT);

    // Propagate: Need to explicetely
    TrajectoryStateOnSurface tSOSDest = thePropagator->propagate(ftsTrack, *endPlane);
    if (!tSOSDest.isValid()) {
      std::cout << "TSOS not valid? Propagation failed...." << std::endl;
      continue;
    }

    auto posExtrap = tSOSDest.freeState()->position();
    auto momExtrap = tSOSDest.freeState()->momentum();
    std::cout << "G4e -- Extrapolated position:" << posExtrap << " cm\n"
              << "G4e --       (Rho, eta, phi): (" << posExtrap.perp() << " cm, " << posExtrap.eta() << ", "
              << posExtrap.phi() << ')' << std::endl;
    std::cout << "G4e -- Extrapolated momentum:" << momExtrap << " GeV\n"
              << "G4e --       (pt, eta, phi): (" << momExtrap.perp() << " cm, " << momExtrap.eta() << ", "
              << momExtrap.phi() << ')' << std::endl;
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(SimpleGeant4ePropagatorTest);
