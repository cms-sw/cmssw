#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "SimG4Core/MagneticField/interface/FieldBuilder.h"
#include "SimG4Core/MagneticField/interface/Field.h"
#include "SimG4Core/MagneticField/interface/FieldStepper.h"
#include "SimG4Core/Geometry/interface/G4LogicalVolumeToDDLogicalPartMapper.h"
#include "SimG4Core/Notification/interface/SimG4Exception.h"

#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Parser/interface/DDLConfiguration.h"
#include "DetectorDescription/Base/interface/DDException.h"
#include "DetectorDescription/Algorithm/src/AlgoInit.h"

#include "G4Mag_UsualEqRhs.hh"
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
   : theField( new Field(f,p)), 
     theFieldEquation(new G4Mag_UsualEqRhs(theField.get())),
     theTopVolume(0),
     fieldValue(0.), minStep(0.), dChord(0.), dOneStep(0.),
     dIntersection(0.), dIntersectionAndOneStep(0.), 
     maxLoopCount(0), minEpsilonStep(0.), maxEpsilonStep(0.), 
     thePSet(p)
{
    theField->fieldEquation(theFieldEquation);
}

void FieldBuilder::readFieldParameters(DDLogicalPart lp,
				       const std::string& keywordField)
{
    G4LogicalVolumeToDDLogicalPartMapper * m = 
	G4LogicalVolumeToDDLogicalPartMapper::instance();
    int tmp;
    tmp = m->toString(keywordField,lp,fieldType);
    tmp = m->toDouble("FieldValue",lp,fieldValue);
    tmp = m->toString("Stepper",lp,stepper);
    tmp = m->toDouble("MinStep",lp,minStep);
    tmp = m->toDouble("DeltaChord",lp,dChord);
    tmp = m->toDouble("DeltaOneStep",lp,dOneStep);
    tmp = m->toDouble("DeltaIntersection",lp,dIntersection);
    tmp = m->toDouble("DeltaIntersectionAndOneStep",lp,dIntersectionAndOneStep);
    tmp = m->toDouble("MaximumLoopCount",lp,maxLoopCount);
    tmp = m->toDouble("MinimumEpsilonStep",lp,minEpsilonStep);
    tmp = m->toDouble("MaximumEpsilonStep",lp,maxEpsilonStep);
    return;
}

void FieldBuilder::build( G4FieldManager* fM, G4PropagatorInField* fP)
{

    std::cout << " Configuring Global Mag.Field" << std::endl;
    configure( "MagneticFieldType", fM, fP ) ;
       std::cout << " Top Volume: " << theTopVolume->GetName() << std::endl;
       std::cout << " Local Stepper: " << stepper[0] << std::endl;

    if ( thePSet.getParameter<bool>("UseLocalMagFieldManager") )
    {
       edm::ParameterSet defpset ;
       edm::ParameterSet thePSetForLMFM = 
          thePSet.getUntrackedParameter<edm::ParameterSet>("ConfLMFM", defpset);
       //
       // Patology !!! LocalFM requested but configuration not given ! 
       // In principal, need to throw an exception
       //
       if ( thePSetForLMFM == defpset )
       {
          std::cout << " Patology ! Local Mag.Field Manager requested but config not given !" <<
	  std::endl;
	  return ;
       }
       std::vector<std::string> ListOfVolumes = 
          thePSetForLMFM.getParameter< std::vector<std::string> >("ListOfVolumes");
       // creating Local Mag.Field Manager
       std::cout << " Creating Local Mag.Field Manager(s)" << std::endl;
       for ( int i = 0; i < ListOfVolumes.size(); ++ i )
       {
          G4FieldManager* fAltM = new G4FieldManager() ;
          configureLocalFM( ListOfVolumes[i], fAltM ) ;
          std::cout << " Top Volume: " << theTopVolume->GetName() << std::endl;
          std::cout << " Local Stepper: " << stepper << std::endl;
          LocalFieldManager* fLM = new LocalFieldManager( theField.get(), fM, fAltM ) ;
          fLM->SetVerbosity(thePSet.getUntrackedParameter<bool>("Verbosity",false));
          theTopVolume->SetFieldManager( fLM, true ) ;
       }
/*
       G4FieldManager* fAltM = new G4FieldManager() ;
       configure( "LocalField", fAltM ) ; // should I use field propagator as well ?
       std::cout << " Top Volume: " << theTopVolume->GetName() << std::endl;
       std::cout << " Local Stepper: " << stepper << std::endl;
       LocalFieldManager* fLM = new LocalFieldManager( theField.get(), fM, fAltM ) ;
       fLM->SetVerbosity(thePSet.getUntrackedParameter<bool>("Verbosity",false));
       theTopVolume->SetFieldManager( fLM, true ) ;
*/
    }
    return ;
 
}

void FieldBuilder::configure(const std::string& keywordField,
			     G4FieldManager * fM,
			     G4PropagatorInField * fP)
{
    ConcreteG4LogicalVolumeToDDLogicalPartMapper::Vector vec = 
        G4LogicalVolumeToDDLogicalPartMapper::instance()->all(keywordField);
    for (ConcreteG4LogicalVolumeToDDLogicalPartMapper::Vector::iterator 
	     tit = vec.begin(); tit != vec.end(); tit++)            
    {
	theTopVolume = (*tit).first;
	readFieldParameters((*tit).second,keywordField);	
	if (fM!=0) configureFieldManager(fM);
	if (fP!=0) configurePropagatorInField(fP);	
    }
    return ;
}

void FieldBuilder::configureLocalFM( const std::string& volName,
                                     G4FieldManager * fM,
			             G4PropagatorInField * fP)
{

   G4LogicalVolumeStore* theStore = G4LogicalVolumeStore::GetInstance();
   for ( int i=0; i<(*theStore).size(); ++i )
   {
      std::string curVolName = ((*theStore)[i])->GetName();
      if ( curVolName == volName )
      {
         theTopVolume = (*theStore)[i] ;
      }
   }
   
   edm::ParameterSet volPSet = 
      (thePSet.getUntrackedParameter<edm::ParameterSet>("ConfLMFM")).getParameter<edm::ParameterSet>(volName);
   
   
   fieldType     = volPSet.getParameter<std::string>("Type") ;
   stepper       = volPSet.getParameter<std::string>("Stepper") ;
   edm::ParameterSet stpPSet = 
      volPSet.getParameter<edm::ParameterSet>(stepper) ;
   minStep       = stpPSet.getParameter<double>("MinStep") ;
   dChord        = stpPSet.getParameter<double>("DeltaChord") ;
   dOneStep      = stpPSet.getParameter<double>("DeltaOneStep") ;
   dIntersection = stpPSet.getParameter<double>("DeltaIntersection") ;
   
   if (fM!=0) configureFieldManager(fM);
   if (fP!=0) configurePropagatorInField(fP);	

   return ;
}

void FieldBuilder::configureFieldManager(G4FieldManager * fM)
{
    if (fM==0) return;
    fM->SetDetectorField(theField.get());
    FieldStepper * theStepper = new FieldStepper(theField->fieldEquation());
    theStepper->select(stepper);
    G4ChordFinder * CF = new G4ChordFinder(theField.get(),minStep,theStepper);
    CF->SetDeltaChord(dChord);
    fM->SetChordFinder(CF);
    fM->SetDeltaOneStep(dOneStep);
    fM->SetDeltaIntersection(dIntersection);
    if (dIntersectionAndOneStep != -1.) 
	fM->SetAccuraciesWithDeltaOneStep(dIntersectionAndOneStep);
    return;
}

void FieldBuilder::configurePropagatorInField(G4PropagatorInField * fP)
{
    if (fP==0) return;
    fP->SetMaxLoopCount(int(maxLoopCount));
    fP->SetMinimumEpsilonStep(minEpsilonStep);
    fP->SetMaximumEpsilonStep(maxEpsilonStep);
    fP->SetVerboseLevel(0);
    return;
}

G4LogicalVolume * FieldBuilder::fieldTopVolume() { return theTopVolume; }
