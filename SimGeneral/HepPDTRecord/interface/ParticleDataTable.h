#ifndef HepPDTProducer_ParticleDataTable_h
#define HepPDTProducer_ParticleDataTable_h
// $Id: ParticleDataTable.h,v 1.5 2007/03/20 09:09:29 llista Exp $

#include "FWCore/Framework/interface/data_default_record_trait.h"
#include "HepPDT/ParticleDataTable.hh"
#include "SimGeneral/HepPDTRecord/interface/PDTRecord.h"

typedef HepPDT::ParticleDataTable ParticleDataTable;
typedef HepPDT::ParticleData ParticleData;

EVENTSETUP_DATA_DEFAULT_RECORD( HepPDT::ParticleDataTable, PDTRecord )

#endif
