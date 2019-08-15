#ifndef SimG4Core_CustomUIsession_H
#define SimG4Core_CustomUIsession_H

#include "G4UIsession.hh"
#include "G4UImanager.hh"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#include <iostream>
//#include <iomanip>
#include <fstream>

class CustomUIsession : public G4UIsession {
public:
  CustomUIsession();
  ~CustomUIsession() override;

  G4int ReceiveG4cout(const G4String& coutString) override;
  G4int ReceiveG4cerr(const G4String& cerrString) override;

  void sendToFile(std::ofstream*);
  inline void stopSendToFile() { fout = nullptr; }

protected:
  std::string trim(const std::string& str);

private:
  std::ofstream* fout;
};

#endif
