#ifndef SimG4Core_CustomUIsession_H
#define SimG4Core_CustomUIsession_H 

#include "G4UIsession.hh"
#include "G4UImanager.hh"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

class CustomUIsession : public G4UIsession
{

 public:

  CustomUIsession();
  ~CustomUIsession();

  G4int ReceiveG4cout(G4String coutString);
  G4int ReceiveG4cerr(G4String cerrString);

 protected:

  G4UImanager* UI;

};

#endif
