#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimG4Core/MagneticField/interface/FieldBuilder.h"
#include "SimG4Core/MagneticField/interface/Field.h"
#include "SimG4Core/MagneticField/interface/FieldStepper.h"
#include "SimG4Core/MagneticField/interface/G4MonopoleEquation.hh"
#include "SimG4Core/MagneticField/interface/ChordFinderSetter.h"

#include "G4Mag_UsualEqRhs.hh"
#include "G4ClassicalRK4.hh"
#include "G4PropagatorInField.hh"
#include "G4FieldManager.hh"
#include "G4TransportationManager.hh"
#include "G4ChordFinder.hh"
#include "G4UniformMagField.hh"

#include "SimG4Core/MagneticField/interface/LocalFieldManager.h"

#include "G4LogicalVolumeStore.hh"

using namespace sim;

FieldBuilder::FieldBuilder(const MagneticField * f, 
			   const edm::ParameterSet & p) 
  : theField(new Field(f, p.getParameter<double>("delta"))),
    theFieldEquation(new G4Mag_UsualEqRhs(theField.get())),
    theTopVolume(nullptr),
    fieldValue(0.), minStep(0.), dChord(0.), dOneStep(0.),
    dIntersection(0.), dIntersectionAndOneStep(0.), 
    maxLoopCount(0), minEpsilonStep(0.), maxEpsilonStep(0.), 
    thePSet(p) 
{
  delta = p.getParameter<double>("delta");
  theField->fieldEquation(theFieldEquation);
}

void FieldBuilder::build( G4FieldManager* fM, G4PropagatorInField* fP, ChordFinderSetter *setter) 
{    
  edm::ParameterSet thePSetForGMFM =
    thePSet.getParameter<edm::ParameterSet>("ConfGlobalMFM");

  std::string volName = thePSetForGMFM.getParameter< std::string >("Volume");
  
  edm::ParameterSet volPSet =
    thePSetForGMFM.getParameter< edm::ParameterSet >( volName );
    
  configureForVolume( volName, volPSet, fM, fP, setter );
    
  if ( thePSet.getParameter<bool>("UseLocalMagFieldManager") )  {

    edm::LogInfo("SimG4CoreApplication") 
      << " FieldBuilder: Local magnetic field is used";

    edm::ParameterSet defpset ;
    edm::ParameterSet thePSetForLMFM = 
      thePSet.getUntrackedParameter<edm::ParameterSet>("ConfLocalMFM", defpset);
    //
    // Patology !!! LocalFM requested but configuration not given ! 
    // In principal, need to throw an exception
    //
    if ( thePSetForLMFM == defpset )  {
      edm::LogError("SimG4CoreApplication") 
	<< " FieldBuilder::build: Patology! Local Mag.Field Manager requested but config not given!";
      return ;
    }
       
    std::vector<std::string> ListOfVolumes = 
      thePSetForLMFM.getParameter< std::vector<std::string> >("ListOfVolumes");
	  
    // creating Local Mag.Field Manager
    for (unsigned int i = 0; i < ListOfVolumes.size(); ++ i )   {
      volPSet = thePSetForLMFM.getParameter< edm::ParameterSet >(ListOfVolumes[i]);
      G4FieldManager* fAltM = new G4FieldManager() ;
      configureForVolume( ListOfVolumes[i], volPSet, fAltM, nullptr, setter ) ;

      LocalFieldManager* fLM = new LocalFieldManager( theField.get(), fM, fAltM ) ;
      fLM->SetVerbosity(thePSet.getUntrackedParameter<bool>("Verbosity",false));
      theTopVolume->SetFieldManager( fLM, true ) ;
    }
  } else {
    edm::LogInfo("SimG4CoreApplication") 
      << " FieldBuilder::build: Global magnetic field is used";
  }
}

void FieldBuilder::configureForVolume( const std::string& volName,
                                       edm::ParameterSet& volPSet,
				       G4FieldManager * fM,
				       G4PropagatorInField * fP,
                                       ChordFinderSetter *setter) 
{
  G4LogicalVolumeStore* theStore = G4LogicalVolumeStore::GetInstance();
  for (unsigned int i=0; i<(*theStore).size(); ++i ) {
    std::string curVolName = ((*theStore)[i])->GetName();
    if ( curVolName == volName ) {
      theTopVolume = (*theStore)[i] ;
    }
  }

  fieldType     = volPSet.getParameter<std::string>("Type") ;
  stepper       = volPSet.getParameter<std::string>("Stepper") ;
  edm::ParameterSet stpPSet = 
    volPSet.getParameter<edm::ParameterSet>("StepperParam") ;
  minStep       = stpPSet.getParameter<double>("MinStep") ;
  dChord        = stpPSet.getParameter<double>("DeltaChord") ;
  dOneStep      = stpPSet.getParameter<double>("DeltaOneStep") ;
  dIntersection = stpPSet.getParameter<double>("DeltaIntersection") ;
  dIntersectionAndOneStep = 
    stpPSet.getUntrackedParameter<double>("DeltaIntersectionAndOneStep",-1.);
  maxLoopCount = 
    stpPSet.getUntrackedParameter<double>("MaximumLoopCounts",1000);
  minEpsilonStep = 
    stpPSet.getUntrackedParameter<double>("MinimumEpsilonStep",0.00001);
  maxEpsilonStep = 
    stpPSet.getUntrackedParameter<double>("MaximumEpsilonStep",0.01);
   
  if (fM!=nullptr) configureFieldManager(fM, setter);
  if (fP!=nullptr) configurePropagatorInField(fP);	

  edm::LogInfo("SimG4CoreApplication") 
    << " FieldBuilder: Selected stepper:     <" << stepper << ">\n" 
    << "               Field type            <" << fieldType<< ">\n"
    << "               Field const delta(mm)= " << delta << "\n"
    << "               MinStep(mm)            " << minStep<< "\n"
    << "               DeltaChord(mm)         " << dChord<< "\n"
    << "               DeltaOneStep(mm)       " << dOneStep<< "\n"
    << "               DeltaIntersection(mm)  " << dIntersection<< "\n"
    << "               IntersectionAndOneStep " << dIntersectionAndOneStep<< "\n"
    << "               MaximumLoopCounts      " << maxLoopCount<< "\n"
    << "               MinimumEpsilonStep     " << minEpsilonStep<< "\n"
    << "               MaximumEpsilonStep     " << maxEpsilonStep;
}

G4LogicalVolume * FieldBuilder::fieldTopVolume() { return theTopVolume; }

void FieldBuilder::configureFieldManager(G4FieldManager * fM, ChordFinderSetter *setter) {

  if (fM!=nullptr) {
    fM->SetDetectorField(theField.get());
    FieldStepper * theStepper = 
      new FieldStepper(theField->fieldEquation(), delta);
    theStepper->select(stepper);
    G4ChordFinder * CF = new G4ChordFinder(theField.get(),minStep,theStepper);
    CF->SetDeltaChord(dChord);
    fM->SetChordFinder(CF);
    fM->SetDeltaOneStep(dOneStep);
    fM->SetDeltaIntersection(dIntersection);
    if (dIntersectionAndOneStep != -1.) 
      fM->SetAccuraciesWithDeltaOneStep(dIntersectionAndOneStep);
  }
  if(setter && !setter->isMonopoleSet()) {
    G4MonopoleEquation* fMonopoleEquation = 
      new G4MonopoleEquation(theField.get());
    G4MagIntegratorStepper* theStepper = 
      new G4ClassicalRK4(fMonopoleEquation,8);
    G4ChordFinder *chordFinderMonopole = 
      new G4ChordFinder(theField.get(),minStep,theStepper);
    chordFinderMonopole->SetDeltaChord(dChord);
    setter->setMonopole(chordFinderMonopole);
  }
}

void FieldBuilder::configurePropagatorInField(G4PropagatorInField * fP) {
  if(fP!=0) {
    fP->SetMaxLoopCount(int(maxLoopCount));
    fP->SetMinimumEpsilonStep(minEpsilonStep);
    fP->SetMaximumEpsilonStep(maxEpsilonStep);
    fP->SetVerboseLevel(0);
  }
}

