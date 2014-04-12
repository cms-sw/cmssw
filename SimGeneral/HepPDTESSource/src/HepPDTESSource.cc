#include "SimGeneral/HepPDTESSource/interface/HepPDTESSource.h"
#include "HepPDT/HeavyIonUnknownID.hh"

HepPDTESSource::HepPDTESSource( const edm::ParameterSet& cfg ) :
  pdtFileName( cfg.getParameter<edm::FileInPath>( "pdtFileName" ) ) {
  setWhatProduced( this );
  findingRecord<PDTRecord>();
}

HepPDTESSource::~HepPDTESSource() {
}

HepPDTESSource::ReturnType
HepPDTESSource::produce( const PDTRecord & iRecord ) {
  using namespace edm::es;
  std::auto_ptr<PDT> pdt( new PDT( "PDG table" , new HepPDT::HeavyIonUnknownID) ); 
  std::ifstream pdtFile( pdtFileName.fullPath().c_str() );
  if( ! pdtFile ) 
    throw cms::Exception( "FileNotFound", "can't open pdt file" )
      << "cannot open " << pdtFileName.fullPath();
  { // notice: the builder has to be destroyed 
    // in order to fill the table!
    HepPDT::TableBuilder builder( * pdt );
    if( ! addParticleTable( pdtFile, builder ) ) { 
      throw cms::Exception( "ConfigError", "can't read pdt file" )
	<< "wrong format of " << pdtFileName.fullPath();
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
//DEFINE_FWK_EVENTSETUP_SOURCE( HepPDTESSource );
