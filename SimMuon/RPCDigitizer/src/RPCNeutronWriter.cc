#include "SimMuon/RPCDigitizer/src/RPCNeutronWriter.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"

RPCNeutronWriter::RPCNeutronWriter(edm::ParameterSet const& pset) : SubsystemNeutronWriter(pset) {}

RPCNeutronWriter::~RPCNeutronWriter() {}

int RPCNeutronWriter::localDetId(int globalDetId) const { return RPCDetId(globalDetId).layer(); }

int RPCNeutronWriter::chamberType(int globalDetId) const { return globalDetId; }

int RPCNeutronWriter::chamberId(int globalDetId) const { return RPCDetId(globalDetId).chamberId().rawId(); }
