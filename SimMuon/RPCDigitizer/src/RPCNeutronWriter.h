#ifndef SimMuon_RPCDigitizer_RPCNeutronWriter_h
#define SimMuon_RPCDigitizer_RPCNeutronWriter_h

#include "SimMuon/Neutron/interface/SubsystemNeutronWriter.h"

/**  Writes out the
     database of neutron patterns for
     the CSCs

  \author   Rick Wilkinson, Caltech
*/

class RPCNeutronWriter : public SubsystemNeutronWriter
{
 public:
  explicit RPCNeutronWriter(edm::ParameterSet const& pset);
  virtual ~RPCNeutronWriter();

 protected:
  virtual int localDetId(int globalDetId) const;

  virtual int chamberType(int globalDetId) const;

  virtual int chamberId(int globalDetId) const;

  /// decides whether this cluster is good enough to be included
  virtual bool accept(const edm::PSimHitContainer & cluster) const {return true;}

};

#endif
