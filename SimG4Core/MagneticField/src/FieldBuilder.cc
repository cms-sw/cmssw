// #include "DetectorDescription/Core/interface/DDLogicalPart.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimG4Core/MagneticField/interface/FieldBuilder.h"
#include "SimG4Core/MagneticField/interface/Field.h"
#include "SimG4Core/MagneticField/interface/FieldStepper.h"
#include "SimG4Core/Notification/interface/SimG4Exception.h"

/*
#include "DetectorDescription/Parser/interface/DDLConfiguration.h"
#include "DetectorDescription/Base/interface/DDException.h"
#include "DetectorDescription/Algorithm/src/AlgoInit.h"
*/

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

/*
void FieldBuilder::readFieldParameters(DDLogicalPart lp,
				       const std::string& keywordField)
{
    int tmp;
    tmp = map_.toString(keywordField,lp,fieldType);
    tmp = map_.toDouble("FieldValue",lp,fieldValue);
    tmp = map_.toString("Stepper",lp,stepper);
    tmp = map_.toDouble("MinStep",lp,minStep);
    tmp = map_.toDouble("DeltaChord",lp,dChord);
    tmp = map_.toDouble("DeltaOneStep",lp,dOneStep);
    tmp = map_.toDouble("DeltaIntersection",lp,dIntersection);
    tmp = map_.toDouble("DeltaIntersectionAndOneStep",lp,dIntersectionAndOneStep);
    tmp = map_.toDouble("MaximumLoopCount",lp,maxLoopCount);
    tmp = map_.toDouble("MinimumEpsilonStep",lp,minEpsilonStep);
    tmp = map_.toDouble("MaximumEpsilonStep",lp,maxEpsilonStep);
    LogDebug("MagneticField") << "FieldBuilder " << lp.name() << " " 
			      << fieldType << " " << fieldValue << " " 
			      << stepper << " " << minStep << " " << dChord 
			      << " " << dOneStep << " " << dIntersection << " "
			      << dIntersectionAndOneStep << " "
			      << maxLoopCount << " " << minEpsilonStep << " " 
			      << maxEpsilonStep;
    return;
}
*/

void FieldBuilder::build( G4FieldManager* fM, G4PropagatorInField* fP)
{
    
    edm::ParameterSet thePSetForGMFM =
       thePSet.getParameter<edm::ParameterSet>("ConfGlobalMFM");

    std::string volName = thePSetForGMFM.getParameter< std::string >("Volume");

    edm::ParameterSet volPSet =
       thePSetForGMFM.getParameter< edm::ParameterSet >( volName );
    
    configureForVolume( volName, volPSet, fM, fP );
    
    // configure( "MagneticFieldType", fM, fP ) ;

    if ( thePSet.getParameter<bool>("UseLocalMagFieldManager") )
    {

       edm::ParameterSet defpset ;
       edm::ParameterSet thePSetForLMFM = 
          thePSet.getUntrackedParameter<edm::ParameterSet>("ConfLocalMFM", defpset);
       //
       // Patology !!! LocalFM requested but configuration not given ! 
       // In principal, need to throw an exception
       //
       if ( thePSetForLMFM == defpset )
       {
	  std::cout << " Patology ! Local Mag.Field Manager requested but config not given !\n";
	  return ;
       }
       
       std::vector<std::string> ListOfVolumes = 
          thePSetForLMFM.getParameter< std::vector<std::string> >("ListOfVolumes");
	  
       // creating Local Mag.Field Manager
       for (unsigned int i = 0; i < ListOfVolumes.size(); ++ i )
       {
          volPSet = thePSetForLMFM.getParameter< edm::ParameterSet >(ListOfVolumes[i]);
          G4FieldManager* fAltM = new G4FieldManager() ;
	  configureForVolume( ListOfVolumes[i], volPSet, fAltM ) ;
          //configureLocalFM( ListOfVolumes[i], fAltM ) ;
          LocalFieldManager* fLM = new LocalFieldManager( theField.get(), fM, fAltM ) ;
          fLM->SetVerbosity(thePSet.getUntrackedParameter<bool>("Verbosity",false));
          theTopVolume->SetFieldManager( fLM, true ) ;
       }
    }
    return ;
 
}

/*
void FieldBuilder::configure(const std::string& keywordField,
			     G4FieldManager * fM,
			     G4PropagatorInField * fP)
{
  G4LogicalVolumeToDDLogicalPartMap::Vector vec = map_.all(keywordField);
  for (G4LogicalVolumeToDDLogicalPartMap::Vector::iterator tit = vec.begin();
       tit != vec.end(); tit++) {
    theTopVolume = (*tit).first;
    readFieldParameters((*tit).second,keywordField);	
    if (fM!=0) configureFieldManager(fM);
    if (fP!=0) configurePropagatorInField(fP);	
  }
  return ;
}
*/

void FieldBuilder::configureForVolume( const std::string& volName,
                                       edm::ParameterSet& volPSet,
				       G4FieldManager * fM,
				       G4PropagatorInField * fP )
{

   G4LogicalVolumeStore* theStore = G4LogicalVolumeStore::GetInstance();
   for (unsigned int i=0; i<(*theStore).size(); ++i )
   {
      std::string curVolName = ((*theStore)[i])->GetName();
      if ( curVolName == volName )
      {
         theTopVolume = (*theStore)[i] ;
      }
   }

   fieldType     = volPSet.getParameter<std::string>("Type") ;
   stepper       = volPSet.getParameter<std::string>("Stepper") ;
   edm::ParameterSet stpPSet = 
      volPSet.getParameter<edm::ParameterSet>(stepper) ;
   minStep       = stpPSet.getParameter<double>("MinStep") ;
   dChord        = stpPSet.getParameter<double>("DeltaChord") ;
   dOneStep      = stpPSet.getParameter<double>("DeltaOneStep") ;
   dIntersection = stpPSet.getParameter<double>("DeltaIntersection") ;
   dIntersectionAndOneStep = stpPSet.getUntrackedParameter<double>("DeltaIntersectionAndOneStep",-1.);
   maxLoopCount = stpPSet.getUntrackedParameter<double>("MaximumLoopCounts",1000);
   minEpsilonStep = stpPSet.getUntrackedParameter<double>("MinimumEpsilonStep",0.00001);
   maxEpsilonStep = stpPSet.getUntrackedParameter<double>("MaximumEpsilonStep",0.01);
   
   if (fM!=0) configureFieldManager(fM);
   if (fP!=0) configurePropagatorInField(fP);	

   return;

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
