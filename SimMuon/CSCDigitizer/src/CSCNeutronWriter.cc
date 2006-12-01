#include "SimMuon/CSCDigitizer/src/CSCNeutronWriter.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "Geometry/CSCGeometry/interface/CSCChamberSpecs.h"
#include <iostream>

CSCNeutronWriter::CSCNeutronWriter(edm::ParameterSet const& pset)
: SubsystemNeutronWriter(pset)
{
std::cout << "MAKE NEUTRONWRITER " << std::endl;
}


CSCNeutronWriter::~CSCNeutronWriter() {
std::cout << "DESTRUCT NEUTRONWRITER " << std::endl;
}


int CSCNeutronWriter::localDetId(int globalDetId) const
{
  return CSCDetId(globalDetId).layer();
}


int CSCNeutronWriter::chamberType(int globalDetId) const
{
  CSCDetId id(globalDetId);
  return CSCChamberSpecs::whatChamberType(id.station(), id.ring());
}


int CSCNeutronWriter::chamberId(int globalDetId) const
{
  return CSCDetId(globalDetId).chamberId().rawId();
}

