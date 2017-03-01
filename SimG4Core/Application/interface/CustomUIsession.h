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

  G4int ReceiveG4cout(const G4String& coutString) override;
  G4int ReceiveG4cerr(const G4String& cerrString) override;

protected:
  std::string trim(const std::string& str);
};

#endif
