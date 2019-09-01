#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "SimMuon/DTDigitizer/src/DTNeutronWriter.h"

DTNeutronWriter::DTNeutronWriter(edm::ParameterSet const &pset) : SubsystemNeutronWriter(pset) {}

DTNeutronWriter::~DTNeutronWriter() {}

int DTNeutronWriter::localDetId(int globalDetId) const { return DTLayerId(globalDetId).layer(); }

int DTNeutronWriter::chamberType(int globalDetId) const { return globalDetId; }

int DTNeutronWriter::chamberId(int globalDetId) const { return DTChamberId(globalDetId).rawId(); }
