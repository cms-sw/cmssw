#ifndef SimMuon_CSCDigitizer_CSCNeutronReader_h
#define SimMuon_CSCDigitizer_CSCNeutronReader_h

#include "SimMuon/Neutron/interface/SubsystemNeutronReader.h"

class CSCNeutronReader : public SubsystemNeutronReader
{
public:
  CSCNeutronReader(const edm::ParameterSet & pset)
  : SubsystemNeutronReader(pset) {}
  virtual ~CSCNeutronReader() {}

  void addHits(std::map<int, edm::PSimHitContainer> & hitMap);

  virtual int detId(int chamberIndex, int localDetId );

  int localDetId(int globalDetId) const;

  int chamberType(int globalDetId) const;

  int chamberId(int globalDetId) const;


};

#endif

