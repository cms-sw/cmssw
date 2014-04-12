#ifndef SimMuon_DTDigitizer_DTNeutronWriter_h
#define SimMuon_DTDigitizer_DTNeutronWriter_h

#include "SimMuon/Neutron/interface/SubsystemNeutronWriter.h"

/**  Writes out the
     neutron simhits for the DTs

  Original Author:  Vadim Khotilovich
*/

class DTNeutronWriter : public SubsystemNeutronWriter
{
 public:
  explicit DTNeutronWriter(edm::ParameterSet const& pset);
  virtual ~DTNeutronWriter();

 protected:
  virtual int localDetId(int globalDetId) const;

  virtual int chamberType(int globalDetId) const;

  virtual int chamberId(int globalDetId) const;

  /// decides whether this cluster is good enough to be included
  virtual bool accept(const edm::PSimHitContainer & cluster) const {return true;}

};

#endif
