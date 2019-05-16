#include "SimG4CMS/Tracker/interface/TkSimHitPrinter.h"

#include <iomanip>
#include <fstream>

#include "G4Track.hh"

std::ofstream* TkSimHitPrinter::theFile(nullptr);

TkSimHitPrinter::TkSimHitPrinter(const std::string& filename) {
  if (theFile)
    return;
  const char* theName = filename.c_str();
  theFile = new std::ofstream(theName, std::ios::out);
}

TkSimHitPrinter::~TkSimHitPrinter() {}

void TkSimHitPrinter::startNewSimHit(const std::string& s, const std::string& d, int i, int j, int k, int eve) {
  (*theFile) << "Event: " << eve << " " << s << " " << d << " " << i << " Track " << j << "  " << k;
}

void TkSimHitPrinter::printLocal(const Local3DPoint& p, const Local3DPoint& e) const {
  (*theFile) << "\n Local(cm):  " << p.x() << " " << p.y() << " " << p.z() << " ; " << e.x() << " " << e.y() << " "
             << e.z();
}

void TkSimHitPrinter::printGlobal(const Local3DPoint& p, const Local3DPoint& e) const {
  (*theFile) << "\n Global(mm): " << p.x() << " " << p.y() << " " << p.z() << " ; " << e.x() << " " << e.y() << " "
             << e.z();
}

void TkSimHitPrinter::printHitData(const std::string& nam, float p, float de, float tof) const {
  (*theFile) << "\n " << nam << " p(GeV): " << p << " Edep(GeV): " << de << " ToF: " << tof;
}
void TkSimHitPrinter::printGlobalMomentum(float px, float py, float pz) const {
  (*theFile) << " Momentum " << px << " " << py << " " << pz << "\n";
}
