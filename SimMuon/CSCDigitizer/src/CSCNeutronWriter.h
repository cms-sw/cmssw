#ifndef SimMuon_CSCDigitizer_CSCNeutronWriter_h
#define SimMuon_CSCDigitizer_CSCNeutronWriter_h

#include "SimMuon/Neutron/interface/SubsystemNeutronWriter.h"

/**  Writes out the
     database of neutron patterns for
     the CSCs

  \author   Rick Wilkinson, Caltech
 */

class CSCNeutronWriter : public SubsystemNeutronWriter
{
public:
  explicit CSCNeutronWriter(edm::ParameterSet const& pset);
  virtual ~CSCNeutronWriter();

protected:
  virtual int localDetId(int globalDetId) const;

  virtual int chamberType(int globalDetId) const;

  virtual int chamberId(int globalDetId) const;

  /// decides whether this cluster is good enough to be included
  virtual bool accept(const edm::PSimHitContainer & cluster) const {return true;}

};

#endif

