#ifndef SimMuon_CSCDigitizer_CSCNeutronReader_h
#define SimMuon_CSCDigitizer_CSCNeutronReader_h

#include "SimMuon/Neutron/interface/SubsystemNeutronReader.h"

namespace CLHEP {
  class HepRandomEngine;
}

class CSCNeutronReader : public SubsystemNeutronReader {
public:
  CSCNeutronReader(const edm::ParameterSet &pset) : SubsystemNeutronReader(pset) {}
  ~CSCNeutronReader() override {}

  void addHits(std::map<int, edm::PSimHitContainer> &hitMap, CLHEP::HepRandomEngine *);

  int detId(int chamberIndex, int localDetId) override;

  int localDetId(int globalDetId) const;

  int chamberType(int globalDetId) const;

  int chamberId(int globalDetId) const;
};

#endif
