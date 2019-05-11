#ifndef SimMuon_RPCDigitizer_RPCNeutronWriter_h
#define SimMuon_RPCDigitizer_RPCNeutronWriter_h

#include "SimMuon/Neutron/interface/SubsystemNeutronWriter.h"

/**  Writes out the
     database of neutron patterns for
     the CSCs

  \author   Rick Wilkinson, Caltech
*/

class RPCNeutronWriter : public SubsystemNeutronWriter {
public:
  explicit RPCNeutronWriter(edm::ParameterSet const& pset);
  ~RPCNeutronWriter() override;

protected:
  int localDetId(int globalDetId) const override;

  int chamberType(int globalDetId) const override;

  int chamberId(int globalDetId) const override;

  /// decides whether this cluster is good enough to be included
  bool accept(const edm::PSimHitContainer& cluster) const override { return true; }
};

#endif
