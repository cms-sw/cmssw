#ifndef SimMuon_Neutron_RootNeutronWriter_h
#define SimMuon_Neutron_RootNeutronWriter_h

#include "SimMuon/Neutron/src/NeutronWriter.h"
#include "SimMuon/Neutron/src/RootChamberWriter.h"
#include <TFile.h>
#include <vector>
/**  This writes the fields of a SimHit into an ASCII
 *   file, which can be read out later to add neutron
 *   hits to a muon chamber
 */


class RootNeutronWriter : public NeutronWriter
{
public:
  RootNeutronWriter(const std::string & fileName);
  virtual ~RootNeutronWriter();

  /// users should use this to create chamberwriters
  /// for each chamber type just after creation
  virtual void initialize(int detType);

  RootChamberWriter & chamberWriter(int chamberType);

  virtual void writeCluster(int chamberType, const edm::PSimHitContainer & hits);

private:
  std::map<int, RootChamberWriter> theChamberWriters;
  TFile * theFile;
};

#endif

