#ifndef RootNeutronReader_h
#define RootNeutronReader_h

#include "SimMuon/Neutron/src/NeutronReader.h"
#include "SimMuon/Neutron/src/RootChamberReader.h"
#include <TFile.h>

/** This reads patterns of neutron hits in muon chambers
 * from an ROOT database,
 * so they can be superimposed onto signal events.
 * It reads the events sequentially, and loops
 * back to the beginning when it reaches EOF
 */

class RootNeutronReader : public NeutronReader
{
public:
  RootNeutronReader(const std::string & fileName);

  virtual void readNextEvent(int chamberType, edm::PSimHitContainer & result);

  RootChamberReader & chamberReader(int chamberType);

private:
  TFile * theFile;
  std::map<int, RootChamberReader> theChamberReaders;
};

#endif

