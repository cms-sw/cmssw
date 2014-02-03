
#include "SimG4Core/MagneticField/interface/LocalFieldManager.h"

#include "G4ChordFinder.hh"
#include "G4Track.hh"

#include <iostream>

using namespace sim;

LocalFieldManager::LocalFieldManager(G4Field* commonField,
                                     G4FieldManager* priFM,
			             G4FieldManager* altFM)
   : G4FieldManager(commonField,0,false),
     fPrimaryFM(priFM), fAlternativeFM(altFM),
     fCurrentFM(0),
     fVerbosity(false)
{
   this->CopyValuesAndChordFinder(priFM);
   fCurrentFM = priFM ;
}

void LocalFieldManager::ConfigureForTrack(const G4Track* trk)
{

   int PID = trk->GetDynamicParticle()->GetDefinition()->GetPDGEncoding();
   
   if ( abs(PID)!=13 ) // maybe also high energy pions ?... what else ?
   {
      if ( fCurrentFM != fAlternativeFM )
      {
         this->CopyValuesAndChordFinder(fAlternativeFM);
	 fCurrentFM = fAlternativeFM;
         if ( fVerbosity) print(trk);
      }
   }
   else
   {
      if ( fCurrentFM != fPrimaryFM )
      {
         this->CopyValuesAndChordFinder(fPrimaryFM);
	 fCurrentFM = fPrimaryFM;
         if ( fVerbosity) print(trk);
      }
   }
   
   return ;

}

const G4FieldManager* LocalFieldManager::CopyValuesAndChordFinder(G4FieldManager * fm)
{

    SetDeltaIntersection(fm->GetDeltaIntersection());
    SetDeltaOneStep(fm->GetDeltaOneStep());
    G4ChordFinder* cf = fm->GetChordFinder();
    cf->SetDeltaChord(cf->GetDeltaChord());
    SetChordFinder(cf);
    
    return fm;

}

void LocalFieldManager::print(const G4Track* trk)
{

  if (fCurrentFM==fAlternativeFM) 
  {
     std::cout << " Alternative field manager with";
  }
  else 
  {
     std::cout << " Global field manager with";
  }
  std::cout << " DeltaIntersection " << G4FieldManager::GetDeltaIntersection()
            << ", DeltaOneStep " << G4FieldManager::GetDeltaOneStep()
            << " and DeltaChord " << G4FieldManager::GetChordFinder()->GetDeltaChord()
            << " for " << trk->GetDynamicParticle()->GetDefinition()->GetPDGEncoding()
            << " with " << trk->GetKineticEnergy()/MeV << " MeV in "
            << trk->GetVolume()->GetName() << std::endl;

   return ;

}
