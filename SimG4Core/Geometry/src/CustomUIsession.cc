#include "SimG4Core/Geometry/interface/CustomUIsession.h"

CustomUIsession::CustomUIsession() : fout(nullptr) {}

CustomUIsession::~CustomUIsession() {}

G4int CustomUIsession::ReceiveG4cout(const G4String& coutString) {
  if (fout) {
    (*fout) << trim(coutString) << "\n";
  } else {
    edm::LogVerbatim("G4cout") << trim(coutString);
  }
  return 0;
}

G4int CustomUIsession::ReceiveG4cerr(const G4String& cerrString) {
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
