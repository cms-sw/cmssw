#ifndef SimMuon_CSCDigitizer_CSCNeutronWriter_h
#define SimMuon_CSCDigitizer_CSCNeutronWriter_h

#include "SimMuon/Neutron/interface/SubsystemNeutronWriter.h"

/**  Writes out the
     database of neutron patterns for
     the CSCs

  \author   Rick Wilkinson, Caltech
 */

class CSCNeutronWriter : public SubsystemNeutronWriter {
public:
  explicit CSCNeutronWriter(edm::ParameterSet const &pset);
  ~CSCNeutronWriter() override;

protected:
  int localDetId(int globalDetId) const override;

  int chamberType(int globalDetId) const override;

  int chamberId(int globalDetId) const override;

  /// decides whether this cluster is good enough to be included
  bool accept(const edm::PSimHitContainer &cluster) const override { return true; }
};

#endif
