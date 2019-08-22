#include "SimG4Core/Geometry/interface/CustomUIsession.h"

CustomUIsession::CustomUIsession() : fout(nullptr) {
  G4UImanager* UI = G4UImanager::GetUIpointer();
  UI->SetCoutDestination(this);
}

CustomUIsession::~CustomUIsession() {
  G4UImanager* UI = G4UImanager::GetUIpointer();
  UI->SetCoutDestination(nullptr);
}

G4int CustomUIsession::ReceiveG4cout(const G4String& coutString) {
  //std::cout << coutString << std::flush;
  if (fout) {
    (*fout) << trim(coutString) << "\n";
  } else {
    edm::LogVerbatim("G4cout") << trim(coutString);
  }
  return 0;
}

G4int CustomUIsession::ReceiveG4cerr(const G4String& cerrString) {
  //std::cerr << cerrString << std::flush;
  edm::LogWarning("G4cerr") << trim(cerrString);
  return 0;
}

std::string CustomUIsession::trim(const std::string& str) {
  if (!str.empty() && str.back() == '\n')
    return str.substr(0, str.length() - 1);
  return str;
}

void CustomUIsession::sendToFile(std::ofstream* ptr) {
  if (ptr && !ptr->fail()) {
    fout = ptr;
  }
}
