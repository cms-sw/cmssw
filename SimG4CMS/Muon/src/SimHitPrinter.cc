#include "SimG4CMS/Muon/interface/SimHitPrinter.h"

#include <G4ios.hh>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>

std::atomic<std::ofstream*> SimHitPrinter::theFile(nullptr);

namespace {
  std::mutex fileMutex;
}

SimHitPrinter::SimHitPrinter(std::string filename) {
  if (theFile)
    return;
  const char* theName = filename.c_str();
  auto f = std::make_unique<std::ofstream>(theName, std::ios::out);

  std::ofstream* previous = nullptr;
  if (theFile.compare_exchange_strong(previous, f.get())) {
    //this thread was the first one to try to set the value
    f.release();
  }
}

SimHitPrinter::~SimHitPrinter() {
  //  theFile->close();
}

void SimHitPrinter::startNewSimHit(std::string s) {
  G4cout.width(10);
  G4cout.setf(std::ios::right, std::ios::adjustfield);
  G4cout.setf(std::ios::scientific, std::ios::floatfield);
  G4cout.precision(6);
  G4cout << "SimHit in " << s << G4endl;

  std::lock_guard<std::mutex> guard{fileMutex};
  (*theFile).width(10);
  (*theFile).setf(std::ios::right, std::ios::adjustfield);
  (*theFile).setf(std::ios::scientific | std::ios::uppercase | std::ios::showpos, std::ios::floatfield);
  (*theFile).precision(5);
  (*theFile) << "SimHit in " << s;
}

void SimHitPrinter::startNewEvent(int num) {
  std::lock_guard<std::mutex> guard{fileMutex};
  (*theFile) << "Event " << num << std::endl;
}

void SimHitPrinter::printId(int id) const {
  G4cout << " Id: " << id << G4endl;
  std::lock_guard<std::mutex> guard{fileMutex};
  (*theFile) << " id ";
  (*theFile).width(10);
  (*theFile).setf(std::ios::right, std::ios::adjustfield);
  (*theFile) << id;
}

void SimHitPrinter::printTrack(int id) const {
  G4cout << " Track: " << id << G4endl;
  std::lock_guard<std::mutex> guard{fileMutex};
  (*theFile) << " trk ";
  (*theFile).width(10);
  (*theFile).setf(std::ios::right, std::ios::adjustfield);
  (*theFile) << id;
}

void SimHitPrinter::printPabs(float pabs) const {
  G4cout << " Pabs: " << pabs << G4endl;
  std::lock_guard<std::mutex> guard{fileMutex};
  (*theFile) << " p " << pabs;
}

void SimHitPrinter::printEloss(float eloss) const {
  G4cout << " Eloss: " << eloss << G4endl;
  std::lock_guard<std::mutex> guard{fileMutex};
  (*theFile) << " e " << eloss;
}

void SimHitPrinter::printLocal(LocalPoint localen, LocalPoint localex) const {
  G4cout << " Local(en/ex): " << localen.x() << " " << localen.y() << " " << localen.z() << " / " << localex.x() << " "
         << localex.y() << " " << localex.z() << G4endl;
  std::lock_guard<std::mutex> guard{fileMutex};
  (*theFile).width(10);
  (*theFile).setf(std::ios::right, std::ios::adjustfield);
  (*theFile).setf(std::ios::floatfield);
  (*theFile).precision(6);
  (*theFile) << " en/ex " << localen.x() << " " << localen.y() << " " << localen.z() << " / " << localex.x() << " "
             << localex.y() << " " << localex.z();
}

void SimHitPrinter::printGlobal(GlobalPoint global) const {
  G4cout << " Global(en): " << global.x() << " " << global.y() << " " << global.z() << G4endl;
  std::lock_guard<std::mutex> guard{fileMutex};
  (*theFile).width(10);
  (*theFile).setf(std::ios::right, std::ios::adjustfield);
  (*theFile).setf(std::ios::floatfield);
  (*theFile).precision(6);
  (*theFile) << " gl " << global.x() << " " << global.y() << " " << global.z() << std::endl;
}
