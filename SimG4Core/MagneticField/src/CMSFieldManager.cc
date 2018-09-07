#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimG4Core/MagneticField/interface/CMSFieldManager.h"
#include "SimG4Core/MagneticField/interface/Field.h"

#include "G4ChordFinder.hh"
#include "G4Track.hh"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

CMSFieldManager::CMSFieldManager() 
  : G4FieldManager(), currChordFinder(nullptr), chordFinder(nullptr),
    chordFinderMonopole(nullptr), dChord(0.001), dOneStep(0.001),
    dIntersection(0.0001), energyThreshold(0.0), dChordSimple(0.1),
    dOneStepSimple(0.1), dIntersectionSimple(0.01)
{}

CMSFieldManager::~CMSFieldManager()
{}

void CMSFieldManager::InitialiseForVolume(const edm::ParameterSet& p, sim::Field* field,
					  G4ChordFinder* cf, G4ChordFinder* cfmon, 
                                          const std::string& vol, const std::string& type, 
                                          const std::string& stepper,
                                          double delta, double minstep)
{
  dChord = p.getParameter<double>("DeltaChord")*CLHEP::mm;
  dOneStep = p.getParameter<double>("DeltaOneStep")*CLHEP::mm;
  dIntersection = p.getParameter<double>("DeltaIntersection")*CLHEP::mm;
  energyThreshold = p.getParameter<double>("EnergyThSimple")*CLHEP::GeV;
  dChordSimple = p.getParameter<double>("DeltaChordSimple")*CLHEP::mm;
  dOneStepSimple = p.getParameter<double>("DeltaOneStepSimple")*CLHEP::mm;
  dIntersectionSimple = p.getParameter<double>("DeltaIntersectionSimple")*CLHEP::mm;
  int maxLC = (int)p.getUntrackedParameter<double>("MaximumLoopCounts",1000);
  double minEpsStep = 
    p.getUntrackedParameter<double>("MinimumEpsilonStep",0.00001)*CLHEP::mm;
  double maxEpsStep = 
    p.getUntrackedParameter<double>("MaximumEpsilonStep",0.01)*CLHEP::mm;
  edm::LogVerbatim("SimG4CoreApplication") 
    << " New CMSFieldManager: LogicalVolume:      <" << vol << ">\n"
    << "               Stepper:                   <" << stepper << ">\n" 
    << "               Field type                 <" << type<< ">\n"
    << "               Field const delta(mm)       " << delta << "\n"
    << "               MinStep(mm)                 " << minstep<< "\n"
    << "               DeltaChord(mm)              " << dChord<< "\n"
    << "               DeltaOneStep(mm)            " << dOneStep<< "\n"
    << "               DeltaIntersection(mm)       " << dIntersection<< "\n"
    << "               EnergyThreshold(GeV)        " << energyThreshold<< "\n"
    << "               DeltaChordSimple(mm)        " << dChord<< "\n"
    << "               DeltaOneStepSimple(mm)      " << dOneStep<< "\n"
    << "               DeltaIntersectionSimple(mm) " << dIntersection<< "\n"
    << "               MaximumLoopCounts           " << maxLC<< "\n"
    << "               MinimumEpsilonStep          " << minEpsStep<< "\n"
    << "               MaximumEpsilonStep          " << maxEpsStep;

  // initialisation of chord finders
  chordFinder = cf;
  chordFinderMonopole = cfmon;
  chordFinder->SetDeltaChord(dChord);
  chordFinderMonopole->SetDeltaChord(dChord);

  // initialisation of field manager
  theField.reset(field);
  SetDetectorField(field);
  SetDeltaOneStep(dOneStep);
  SetDeltaIntersection(dIntersection);
  SetMinimumEpsilonStep(minEpsStep);
  SetMaximumEpsilonStep(maxEpsStep);

  SetMonopoleTracking(false);
}

void CMSFieldManager::ConfigureForTrack(const G4Track* track)
{
  // run time parameters per track
  if(track->GetKineticEnergy() <= energyThreshold && track->GetParentID() > 0) {
    chordFinder->SetDeltaChord(dChordSimple);
    SetDeltaOneStep(dOneStepSimple);
    SetDeltaIntersection(dIntersectionSimple);
  } else {
    chordFinder->SetDeltaChord(dChord);
    SetDeltaOneStep(dOneStep);
    SetDeltaIntersection(dIntersection);
  }
} 

void CMSFieldManager::SetMonopoleTracking(G4bool flag)
{
  if(flag) {
    currChordFinder = chordFinderMonopole;
    SetFieldChangesEnergy(true);
  } else { 
    currChordFinder = chordFinder;
    SetFieldChangesEnergy(false);
  }
  SetChordFinder(currChordFinder);
}
