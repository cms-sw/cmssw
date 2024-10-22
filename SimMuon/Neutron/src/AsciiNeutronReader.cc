#include "SimMuon/Neutron/src/AsciiNeutronReader.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include <sstream>
#include <iostream>

using namespace std;

AsciiNeutronReader::AsciiNeutronReader(string fileNameBase)
    : theFileNameBase(fileNameBase),
      //theStreamPos(nChamberTypes+1, 0) //TODO WON"T WORK!  replace with a map
      theStreamPos(11, 0) {}

void AsciiNeutronReader::readNextEvent(int chamberType, edm::PSimHitContainer& result) {
  stringstream fileName;
  fileName << theFileNameBase << chamberType;
  ifstream fin(fileName.str().c_str(), ios::in);
  if (!fin.is_open()) {
    throw cms::Exception("NeutronReader") << "Muon neutron noise file missing " << fileName.str();
  }

  int nhits = read_nhits(fin, chamberType);
  for (int ihit = 0; ihit < nhits; ++ihit) {
    float entryX, entryY, entryZ, exitX, exitY, exitZ;
    float p, tof, eloss, theta, phi;
    int type, layer, track;

    fin >> entryX >> entryY >> entryZ >> exitX >> exitY >> exitZ >> p >> tof >> eloss >> type >> layer >> track >>
        theta >> phi;
    LocalPoint entry(entryX, entryY, entryZ);
    LocalPoint exit(exitX, exitY, exitZ);
    PSimHit phit(entry, exit, p, tof, eloss, type, layer, track, theta, phi);
    result.push_back(phit);
  }
  theStreamPos[chamberType] = fin.tellg();
}

int AsciiNeutronReader::read_nhits(ifstream& fin, int chamberType) {
  int nhits = 0;
  // go to the last saved place
  fin.seekg(theStreamPos[chamberType]);
  if (fin.eof()) {
    resetStreampos(fin, chamberType);
  }
  LogDebug("NeutronReader") << "starting from pos " << theStreamPos[chamberType] << " EOF " << fin.eof();
  fin >> nhits;
  if (fin.eof()) {
    resetStreampos(fin, chamberType);
    fin >> nhits;
  }
  LogDebug("NeutronReader") << "returning nhits " << nhits;
  return nhits;
}

void AsciiNeutronReader::resetStreampos(ifstream& fin, int chamberType) {
  LogDebug("NeutronReader") << "reached EOF, resetting streampos ";
  theStreamPos[chamberType] = 0;
  fin.clear();
  fin.seekg(0, ios::beg);
}
