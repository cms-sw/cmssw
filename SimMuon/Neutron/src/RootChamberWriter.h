#ifndef SimMuon_Neutron_RootChamberWriter_h
#define SimMuon_Neutron_RootChamberWriter_h

#include <TTree.h>
#include <TClonesArray.h>
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

class RootChamberWriter
{
public:
  /// default ctor, for STL
  RootChamberWriter() : theTree(0), theHits(0) {}
  RootChamberWriter(const std::string & treeName);

  /// writes the tree, and deletes everything
  ~RootChamberWriter();

  void write(const edm::PSimHitContainer & hits);

  TTree * tree() {return theTree;}

private:
  TTree * theTree;
  TClonesArray * theHits;
};

#endif

