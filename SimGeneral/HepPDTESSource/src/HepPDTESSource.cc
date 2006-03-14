// -*- C++ -*-
//
// Package:    HepPDTESSource
// Class:      HepPDTESSource
// 
/**\class HepPDTESSource HepPDTESSource.h PhysicsTools/HepPDTESSource/interface/HepPDTESSource.h

 Description: HepPDT particle data table ESSource

 Implementation:
    <Notes on implementation>
*/
//
// Original Author:  Luca Lista
//         Created:  Fri Mar 10 15:58:18 CET 2006
// $Id: HepPDTESSource.cc,v 1.1 2006/03/13 18:03:06 llista Exp $
//
#include <memory>
#include <fstream>
#include "boost/shared_ptr.hpp"
#include <CLHEP/HepPDT/DefaultConfig.hh>
#include <CLHEP/HepPDT/TableBuilder.hh>
#include <CLHEP/HepPDT/ParticleDataTableT.hh>
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "SimGeneral/HepPDTRecord/interface/PDTRecord.h"
#include <climits>

class HepPDTESSource : public edm::ESProducer, public  edm::EventSetupRecordIntervalFinder {
public:
  /// constructor from parameter set
  HepPDTESSource( const edm::ParameterSet& );
  /// destructor
  ~HepPDTESSource();
  /// define the particle data table type
  typedef DefaultConfig::ParticleDataTable PDT;
  /// define the return type
  typedef std::auto_ptr<PDT> ReturnType;
  /// return the particle table
  ReturnType produce( const PDTRecord & );
  /// set validity interval
  void setIntervalFor( const edm::eventsetup::EventSetupRecordKey &,
		       const edm::IOVSyncValue &,
		       edm::ValidityInterval & );

private:
  std::string pdtFileName;
};

HepPDTESSource::HepPDTESSource( const edm::ParameterSet& cfg ) :
  pdtFileName( cfg.getParameter<std::string>( "pdtFileName" ) ) {
  setWhatProduced( this );
  findingRecord<PDTRecord>();
}

HepPDTESSource::~HepPDTESSource() {
}

HepPDTESSource::ReturnType
HepPDTESSource::produce( const PDTRecord & iRecord ) {
  using namespace edm::es;
  std::auto_ptr<PDT> pdt( new PDT( "PDG table" ) );
  std::ifstream pdtFile( pdtFileName.c_str() );
  if( ! pdtFile ) 
    throw cms::Exception( "FileNotFound", "can't open pdt file" )
      << "cannot open " << pdtFileName;
  { // notice: the builder has to be destroyed 
    // in order to fill the table!
    HepPDT::TableBuilder builder( * pdt );
    if( ! addPDGParticles( pdtFile, builder ) ) { 
      throw cms::Exception( "ConfigError", "can't read pdt file" )
	<< "wrong format of " << pdtFileName;
    }
  }  
  return pdt;
}

void HepPDTESSource::setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
				     const edm::IOVSyncValue&,
				     edm::ValidityInterval& oInterval ) {
  // the same PDT is valid for any time
  oInterval = edm::ValidityInterval( edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime() );
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_SOURCE( HepPDTESSource )
