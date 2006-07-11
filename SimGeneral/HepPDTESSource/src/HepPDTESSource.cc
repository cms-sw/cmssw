#include "SimGeneral/HepPDTESSource/interface/HepPDTESSource.h"


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
  std::string HepPDTBase( getenv("HEPPDT_PARAM_PATH") ) ;
  std::string fPDGTablePath = HepPDTBase + "/data/";
  std::string fPDGTableName = fPDGTablePath + pdtFileName;
  std::ifstream pdtFile( fPDGTableName.c_str() );
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
//DEFINE_FWK_EVENTSETUP_SOURCE( HepPDTESSource )
