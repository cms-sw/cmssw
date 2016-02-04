#include "SimG4Core/Application/interface/CustomUIsession.h"

CustomUIsession::CustomUIsession()
{
  
  UI = G4UImanager::GetUIpointer();
  UI->SetCoutDestination(this);

}

CustomUIsession::~CustomUIsession()
{
  
  UI = G4UImanager::GetUIpointer();
  UI->SetCoutDestination(NULL);

}

G4int CustomUIsession::ReceiveG4cout(G4String coutString)
{
  //std::cout << coutString << std::flush;
  edm::LogInfo("G4cout") << coutString;
  return 0;
}

G4int CustomUIsession::ReceiveG4cerr(G4String cerrString)
{
  //std::cerr << cerrString << std::flush;
  edm::LogWarning("G4cerr") << cerrString;
  return 0;
}
