#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimG4Core/MagneticField/interface/CMSFieldManager.h"
#include "SimG4Core/MagneticField/interface/Field.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "G4ChordFinder.hh"
#include "G4MagIntegratorStepper.hh"
#include "G4PropagatorInField.hh"
#include "G4Region.hh"
#include "G4RegionStore.hh"
#include "G4Track.hh"

CMSFieldManager::CMSFieldManager()
    : G4FieldManager(),
      m_currChordFinder(nullptr),
      m_chordFinder(nullptr),
      m_chordFinderMonopole(nullptr),
      m_propagator(nullptr),
      m_dChord(0.001),
      m_dChordTracker(0.001),
      m_dOneStep(0.001),
      m_dOneStepTracker(0.0001),
      m_dIntersection(0.0001),
      m_dInterTracker(1e-6),
      m_Rmax2(1.e+6),
      m_Zmax(3.e+3),
      m_stepMax(1000000.),
      m_energyThTracker(1.e+7),
      m_energyThreshold(0.0),
      m_dChordSimple(0.1),
      m_dOneStepSimple(0.1),
      m_dIntersectionSimple(0.01),
      m_stepMaxSimple(1000.),
      m_cfTracker(false),
      m_cfVacuum(false) {}

CMSFieldManager::~CMSFieldManager() {
  if (m_chordFinder != m_currChordFinder) {
    delete m_chordFinder;
  }
  if (m_chordFinderMonopole != m_currChordFinder) {
    delete m_chordFinderMonopole;
  }
}

void CMSFieldManager::InitialiseForVolume(const edm::ParameterSet &p,
                                          sim::Field *field,
                                          G4ChordFinder *cf,
                                          G4ChordFinder *cfmon,
                                          const std::string &vol,
                                          const std::string &type,
                                          const std::string &stepper,
                                          double delta,
                                          G4PropagatorInField *pf) {
  double minstep = p.getParameter<double>("MinStep") * CLHEP::mm;
  double minEpsStep = p.getUntrackedParameter<double>("MinimumEpsilonStep", 0.00001) * CLHEP::mm;
  double maxEpsStep = p.getUntrackedParameter<double>("MaximumEpsilonStep", 0.01) * CLHEP::mm;
  int maxLC = (int)p.getUntrackedParameter<double>("MaximumLoopCounts", 1000.);

  // double
  m_dChord = p.getParameter<double>("DeltaChord") * CLHEP::mm;
  m_dChordTracker = p.getParameter<double>("DeltaChord") * CLHEP::mm;
  m_dOneStep = p.getParameter<double>("DeltaOneStep") * CLHEP::mm;
  m_dOneStepTracker = p.getParameter<double>("DeltaOneStepTracker") * CLHEP::mm;
  m_dIntersection = p.getParameter<double>("DeltaIntersection") * CLHEP::mm;
  m_dInterTracker = p.getParameter<double>("DeltaIntersectionTracker") * CLHEP::mm;
  m_stepMax = p.getParameter<double>("MaxStep") * CLHEP::cm;

  m_energyThreshold = p.getParameter<double>("EnergyThSimple") * CLHEP::GeV;
  m_energyThTracker = p.getParameter<double>("EnergyThTracker") * CLHEP::GeV;

  double rmax = p.getParameter<double>("RmaxTracker") * CLHEP::mm;
  m_Rmax2 = rmax * rmax;
  m_Zmax = p.getParameter<double>("ZmaxTracker") * CLHEP::mm;

  m_dChordSimple = p.getParameter<double>("DeltaChordSimple") * CLHEP::mm;
  m_dOneStepSimple = p.getParameter<double>("DeltaOneStepSimple") * CLHEP::mm;
  m_dIntersectionSimple = p.getParameter<double>("DeltaIntersectionSimple") * CLHEP::mm;
  m_stepMaxSimple = p.getParameter<double>("MaxStepSimple") * CLHEP::cm;

  edm::LogVerbatim("SimG4CoreApplication")
      << " New CMSFieldManager: LogicalVolume:      <" << vol << ">\n"
      << "               Stepper:                   <" << stepper << ">\n"
      << "               Field type                 <" << type << ">\n"
      << "               Field const delta           " << delta << " mm\n"
      << "               MaximumLoopCounts           " << maxLC << "\n"
      << "               MinimumEpsilonStep          " << minEpsStep << "\n"
      << "               MaximumEpsilonStep          " << maxEpsStep << "\n"
      << "               MinStep                     " << minstep << " mm\n"
      << "               MaxStep                     " << m_stepMax / CLHEP::cm << " cm\n"
      << "               DeltaChord                  " << m_dChord << " mm\n"
      << "               DeltaOneStep                " << m_dOneStep << " mm\n"
      << "               DeltaIntersection           " << m_dIntersection << " mm\n"
      << "               DeltaInterTracker           " << m_dInterTracker << " mm\n"
      << "               EnergyThresholdSimple       " << m_energyThreshold / CLHEP::MeV << " MeV\n"
      << "               EnergyThresholdTracker      " << m_energyThTracker / CLHEP::MeV << " MeV\n"
      << "               DeltaChordSimple            " << m_dChordSimple << " mm\n"
      << "               DeltaOneStepSimple          " << m_dOneStepSimple << " mm\n"
      << "               DeltaIntersectionSimple     " << m_dIntersectionSimple << " mm\n"
      << "               MaxStepInVacuum             " << m_stepMaxSimple / CLHEP::cm << " cm";

  // initialisation of chord finders
  m_chordFinder = cf;
  m_chordFinderMonopole = cfmon;

  m_chordFinderMonopole->SetDeltaChord(m_dChord);

  // initialisation of field manager
  theField.reset(field);
  SetDetectorField(field);
  SetMinimumEpsilonStep(minEpsStep);
  SetMaximumEpsilonStep(maxEpsStep);

  // propagater in field
  m_propagator = pf;
  pf->SetMaxLoopCount(maxLC);
  pf->SetMinimumEpsilonStep(minEpsStep);
  pf->SetMaximumEpsilonStep(maxEpsStep);

  // initial initialisation the default chord finder
  setMonopoleTracking(false);

  // define regions
  std::vector<std::string> rnames = p.getParameter<std::vector<std::string>>("VacRegions");
  if (!rnames.empty()) {
    std::stringstream ss;
    std::vector<G4Region *> *rs = G4RegionStore::GetInstance();
    for (auto &regnam : rnames) {
      for (auto &reg : *rs) {
        if (regnam == reg->GetName()) {
          m_regions.push_back(reg);
          ss << "  " << regnam;
        }
      }
    }
    edm::LogVerbatim("SimG4CoreApplication") << "Simple field integration in G4Regions:\n" << ss.str() << "\n";
  }
}

void CMSFieldManager::ConfigureForTrack(const G4Track *track) {
  // run time parameters per track
  if (track->GetKineticEnergy() > m_energyThTracker && isInsideTracker(track)) {
    if (!m_cfTracker) {
      setChordFinderForTracker();
    }

  } else if ((track->GetKineticEnergy() <= m_energyThreshold && track->GetParentID() > 0) || isInsideVacuum(track)) {
    if (!m_cfVacuum) {
      setChordFinderForVacuum();
    }

  } else if (m_cfTracker || m_cfVacuum) {
    // restore defaults
    setDefaultChordFinder();
  }
}

void CMSFieldManager::setMonopoleTracking(G4bool flag) {
  if (flag) {
    if (m_currChordFinder != m_chordFinderMonopole) {
      if (m_cfTracker || m_cfVacuum) {
        setDefaultChordFinder();
      }
      m_currChordFinder = m_chordFinderMonopole;
      SetChordFinder(m_currChordFinder);
    }
  } else {
    setDefaultChordFinder();
  }
  SetFieldChangesEnergy(flag);
  m_currChordFinder->ResetStepEstimate();
}

bool CMSFieldManager::isInsideVacuum(const G4Track *track) {
  if (!m_regions.empty()) {
    const G4Region *reg = track->GetVolume()->GetLogicalVolume()->GetRegion();
    for (auto &areg : m_regions) {
      if (reg == areg) {
        return true;
      }
    }
  }
  return false;
}

bool CMSFieldManager::isInsideTracker(const G4Track *track) {
  const G4ThreeVector &pos = track->GetPosition();
  const double x = pos.x();
  const double y = pos.y();
  return (x * x + y * y < m_Rmax2 && std::abs(pos.z()) < m_Zmax);
}

void CMSFieldManager::setDefaultChordFinder() {
  if (m_currChordFinder != m_chordFinder) {
    m_currChordFinder = m_chordFinder;
    SetChordFinder(m_currChordFinder);
  }
  m_currChordFinder->SetDeltaChord(m_dChord);
  SetDeltaOneStep(m_dOneStep);
  SetDeltaIntersection(m_dIntersection);
  m_propagator->SetLargestAcceptableStep(m_stepMax);
  m_cfVacuum = m_cfTracker = false;
}

void CMSFieldManager::setChordFinderForTracker() {
  if (m_currChordFinder != m_chordFinder) {
    m_currChordFinder = m_chordFinder;
    SetChordFinder(m_currChordFinder);
  }
  m_currChordFinder->SetDeltaChord(m_dChordTracker);
  SetDeltaOneStep(m_dOneStepTracker);
  SetDeltaIntersection(m_dInterTracker);
  m_propagator->SetLargestAcceptableStep(m_stepMax);
  m_cfVacuum = false;
  m_cfTracker = true;
}

void CMSFieldManager::setChordFinderForVacuum() {
  if (m_currChordFinder != m_chordFinder) {
    m_currChordFinder = m_chordFinder;
    SetChordFinder(m_currChordFinder);
  }
  m_currChordFinder->SetDeltaChord(m_dChordSimple);
  SetDeltaOneStep(m_dOneStepSimple);
  SetDeltaIntersection(m_dIntersectionSimple);
  m_propagator->SetLargestAcceptableStep(m_stepMaxSimple);
  m_cfVacuum = true;
  m_cfTracker = false;
}
