#ifndef HepPDTProducer_ParticleDataTable_h
#define HepPDTProducer_ParticleDataTable_h
// $Id: ParticleDataTable.h,v 1.3 2006/08/03 08:23:34 fmoortga Exp $

#include "FWCore/Framework/interface/data_default_record_trait.h"
#include <CLHEP/HepPDT/DefaultConfig.hh>
#include <CLHEP/HepPDT/ParticleDataTableT.hh>
#include "SimGeneral/HepPDTRecord/interface/PDTRecord.h"

typedef DefaultConfig::ParticleDataTable ParticleDataTable;
typedef DefaultConfig::ParticleData ParticleData;


EVENTSETUP_DATA_DEFAULT_RECORD( DefaultConfig::ParticleDataTable, PDTRecord )

#endif
