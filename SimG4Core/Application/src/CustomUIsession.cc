#include "SimG4Core/Application/interface/CustomUIsession.h"

CustomUIsession::CustomUIsession()
{
  
  G4UImanager *UI = G4UImanager::GetUIpointer();
  UI->SetCoutDestination(this);

}

CustomUIsession::~CustomUIsession()
{
  
  G4UImanager *UI = G4UImanager::GetUIpointer();
  UI->SetCoutDestination(NULL);

}

G4int CustomUIsession::ReceiveG4cout(const G4String& coutString)
{
  //std::cout << coutString << std::flush;
  edm::LogVerbatim("G4cout") << trim(coutString);
  return 0;
}

G4int CustomUIsession::ReceiveG4cerr(const G4String& cerrString)
{
  //std::cerr << cerrString << std::flush;
  edm::LogWarning("G4cerr") << trim(cerrString);
  return 0;
}

std::string CustomUIsession::trim(const std::string& str) {
  if(!str.empty() && str.back() == '\n')
    return str.substr(0, str.length()-1);
  return str;
}
