#ifndef SimMuon_Neutron_RootChamberReader_h
#define SimMuon_Neutron_RootChamberReader_h

#include <TFile.h>
#include <TTree.h>
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

class RootChamberReader {
public:
  /// default ctor, for STL
  RootChamberReader();
  RootChamberReader(TFile* file, const std::string& treeName);
  /// writes the tree, and deletes everything
  ~RootChamberReader();

  void read(edm::PSimHitContainer& hits);

private:
  TTree* theTree;
  TClonesArray* theHits;
  int thePosition;
  int theSize;
};

#endif
