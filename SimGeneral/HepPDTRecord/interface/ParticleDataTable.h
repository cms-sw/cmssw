#ifndef HepPDTProducer_ParticleDataTable_h
#define HepPDTProducer_ParticleDataTable_h
// $Id: ParticleDataTable.h,v 1.1 2006/03/13 18:03:05 llista Exp $

#include "FWCore/Framework/interface/data_default_record_trait.h"
#include <CLHEP/HepPDT/DefaultConfig.hh>
#include <CLHEP/HepPDT/ParticleDataTableT.hh>
#include "PhysicsTools/HepPDTProducer/interface/PDTRecord.h"

EVENTSETUP_DATA_DEFAULT_RECORD( DefaultConfig::ParticleDataTable, PDTRecord );

#endif
