
#include "SimG4Core/MagneticField/interface/LocalFieldManager.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4ChordFinder.hh"
#include "G4Track.hh"

#include "CLHEP/Units/SystemOfUnits.h"

#include <iostream>

using namespace sim;

LocalFieldManager::LocalFieldManager(G4Field* commonField,
                                     G4FieldManager* priFM,
			             G4FieldManager* altFM)
   : G4FieldManager(commonField,nullptr,false),
     fPrimaryFM(priFM), fAlternativeFM(altFM),
     fCurrentFM(nullptr),
     fVerbosity(false)
{
   this->CopyValuesAndChordFinder(priFM);
   fCurrentFM = priFM ;
}

void LocalFieldManager::ConfigureForTrack(const G4Track* trk)
{

   int PID = trk->GetDynamicParticle()->GetDefinition()->GetPDGEncoding();
   
   if ( std::abs(PID)!=13 ) // maybe also high energy pions ?... what else ?
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
  std::string ss = (fCurrentFM==fAlternativeFM) 
    ? "Alternative field manager with"
    : "Global field manager with";

  edm::LogInfo("SimG4CoreMagneticField") 
    << ss << " DeltaIntersection= " << G4FieldManager::GetDeltaIntersection()
    << ", DeltaOneStep= " << G4FieldManager::GetDeltaOneStep()
    << ", DeltaChord= " << G4FieldManager::GetChordFinder()->GetDeltaChord()
    << " for " << trk->GetDynamicParticle()->GetDefinition()->GetPDGEncoding()
    << " with " << trk->GetKineticEnergy()/CLHEP::GeV << " GeV in "
    << trk->GetVolume()->GetName();
}
