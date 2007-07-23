#include "SimGeneral/HepPDTRecord/interface/PdtEntry.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

using namespace std;
using namespace edm;

int PdtEntry::pdgId() const {
  if ( pdgId_ == 0 )
    throw cms::Exception( "ConfigError" )
      << "PdtEntry::pdgId was not set.\n"
      << "please, call PdtEntry::setup( const edm::EventSetup & es )";
  return pdgId_;
}

const string & PdtEntry::name() const {
  if ( name_.empty() )
    throw cms::Exception( "ConfigError" )
      << "PdtEntry::name was not set."
      << "please, call PdtEntry::setup( const edm::EventSetup & es )";
  return name_;
}

const HepPDT::ParticleData & PdtEntry::data() const { 
  if( data_ == 0 )
    throw cms::Exception( "ConfigError" )
      << "PdtEntry::name was not set."
      << "please, call PdtEntry::setup( const edm::EventSetup & es )";
  return * data_; 
} 

void PdtEntry::setup( const edm::EventSetup & es ) {
  ESHandle<DefaultConfig::ParticleDataTable> pdt;
  es.getData( pdt );
  const HepPDT::ParticleData * p = 0;
  if ( pdgId_ == 0 ) {
    p = pdt->particle( name_ );
    if ( p == 0 ) 
      throw cms::Exception( "ConfigError" )
	<< "PDT has no entry for " << name_ << "."
	<< "PdtEntry can't be set.";
    pdgId_ = p->pid();
  } else {
    p = pdt->particle( pdgId_ );
    if ( p == 0 ) 
      throw cms::Exception( "ConfigError" )
	<< "PDT has no entry for " << pdgId_ << "."
	<< "PdtEntry can't be set.";
    name_ = p->name();
  }
  data_ = p;
}
