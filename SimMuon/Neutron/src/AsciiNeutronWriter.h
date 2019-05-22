#ifndef AsciiNeutronWriter_h
#define AsciiNeutronWriter_h

#include "SimMuon/Neutron/src/NeutronWriter.h"

/**  This writes the fields of a SimHit into an ASCII
 *   file, which can be read out later to add neutron
 *   hits to a muon chamber
 */

class AsciiNeutronWriter : public NeutronWriter {
public:
  AsciiNeutronWriter(std::string fileNameBase);
  ~AsciiNeutronWriter() override;

protected:
  void writeCluster(int chamberType, const edm::PSimHitContainer& hits) override;

private:
  std::string theFileNameBase;
};

#endif
