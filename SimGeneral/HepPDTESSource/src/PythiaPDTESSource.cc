#include "SimGeneral/HepPDTESSource/interface/PythiaPDTESSource.h"


PythiaPDTESSource::PythiaPDTESSource( const edm::ParameterSet& cfg ) :
  pdtFileName( cfg.getParameter<std::string>( "pdtFileName" ) ) {
  setWhatProduced( this );
  findingRecord<PDTRecord>();
}

PythiaPDTESSource::~PythiaPDTESSource() {
}

PythiaPDTESSource::ReturnType
PythiaPDTESSource::produce( const PDTRecord & iRecord ) {
  using namespace edm::es;
  std::auto_ptr<PDT> pdt( new PDT( "Pythia PDT" ) );
  std::string HepPDTBase( getenv("CMSSW_BASE") ) ;
  std::string fPDGTablePath = HepPDTBase + "/src/SimGeneral/HepPDTESSource/data/";
  std::string fPDGTableName = fPDGTablePath + pdtFileName;
  std::ifstream pdtFile( fPDGTableName.c_str() );
  if( ! pdtFile ) 
    throw cms::Exception( "FileNotFound", "can't open pdt file" )
      << "cannot open " << fPDGTableName;
  { // notice: the builder has to be destroyed 
    // in order to fill the table!
    HepPDT::TableBuilder builder( * pdt );
    if( ! addPythiaParticles( pdtFile, builder ) ) { 
      throw cms::Exception( "ConfigError", "can't read pdt file" )
	<< "wrong format of " << pdtFileName;
    }
  }  
  return pdt;
}

void PythiaPDTESSource::setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
				     const edm::IOVSyncValue&,
				     edm::ValidityInterval& oInterval ) {
  // the same PDT is valid for any time
  oInterval = edm::ValidityInterval( edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime() );
}

//define this as a plug-in
//DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE( PythiaPDTESSource )
