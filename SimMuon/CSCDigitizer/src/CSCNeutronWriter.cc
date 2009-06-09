#include "SimMuon/CSCDigitizer/src/CSCNeutronWriter.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "Geometry/CSCGeometry/interface/CSCChamberSpecs.h"
#include <iostream>

CSCNeutronWriter::CSCNeutronWriter(edm::ParameterSet const& pset)
: SubsystemNeutronWriter(pset)
{
  for(int i = 1; i <= 10; ++i)
  {
    initialize(i);
  }
}


CSCNeutronWriter::~CSCNeutronWriter() {
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


bool CSCNeutronWriter::accept(const edm::PSimHitContainer & cluster) const
{
  // require at least two layers, to satisfy pretrigger
  if(cluster.size() < 2) 
  {
    unsigned int firstHitDetUnitId = cluster[0].detUnitId();
    for(edm::PSimHitContainer::const_iterator hitItr = cluster.begin()+1;
        hitItr != cluster.end(); ++hitItr)
    {
      if(hitItr->detUnitId() != firstHitDetUnitId)
      {
        return true;
      }
    }
  }
  return false;
}


