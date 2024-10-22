#ifndef SimMuon_DTDigitizer_DTNeutronWriter_h
#define SimMuon_DTDigitizer_DTNeutronWriter_h

#include "SimMuon/Neutron/interface/SubsystemNeutronWriter.h"

/**  Writes out the
     neutron simhits for the DTs

  Original Author:  Vadim Khotilovich
*/

class DTNeutronWriter : public SubsystemNeutronWriter {
public:
  explicit DTNeutronWriter(edm::ParameterSet const &pset);
  ~DTNeutronWriter() override;

protected:
  int localDetId(int globalDetId) const override;

  int chamberType(int globalDetId) const override;

  int chamberId(int globalDetId) const override;

  /// decides whether this cluster is good enough to be included
  bool accept(const edm::PSimHitContainer &cluster) const override { return true; }
};

#endif
