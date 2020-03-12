#ifndef AsciiNeutronReader_h
#define AsciiNeutronReader_h

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include "SimMuon/Neutron/src/NeutronReader.h"

/** This reads patterns of neutron hits in muon chambers
 * from an ASCII database,
 * so they can be superimposed onto signal events.
 * It reads the events sequentially, and loops
 * back to the beginning when it reaches EOF
 */

class AsciiNeutronReader : public NeutronReader {
public:
  AsciiNeutronReader(std::string fileNameBase);

  void readNextEvent(int chamberType, edm::PSimHitContainer& result) override;

private:
  int read_nhits(std::ifstream& fin, int chamberType);
  void resetStreampos(std::ifstream& fin, int chamberType);

  std::string theFileNameBase;
  std::vector<std::streampos> theStreamPos;
};

#endif
