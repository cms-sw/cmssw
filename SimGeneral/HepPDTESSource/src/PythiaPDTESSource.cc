#include "SimGeneral/HepPDTESSource/interface/PythiaPDTESSource.h"


PythiaPDTESSource::PythiaPDTESSource( const edm::ParameterSet& cfg ) :
  pdtFileName( cfg.getParameter<edm::FileInPath>( "pdtFileName" ) ) {
  setWhatProduced( this );
  findingRecord<PDTRecord>();
}

PythiaPDTESSource::~PythiaPDTESSource() {
}

PythiaPDTESSource::ReturnType
PythiaPDTESSource::produce( const PDTRecord & iRecord ) {
  using namespace edm::es;
  std::auto_ptr<PDT> pdt( new PDT( "Pythia PDT" ) );   
  std::ifstream pdtFile( pdtFileName.fullPath().c_str() );
  
  if( ! pdtFile ) 
    throw cms::Exception( "FileNotFound", "can't open pdt file" )
      << "cannot open " << pdtFileName.fullPath();
  { // notice: the builder has to be destroyed 
    // in order to fill the table!
    HepPDT::TableBuilder builder( * pdt );
    if( ! cmsaddPythiaParticles( pdtFile, builder ) ) { 
      throw cms::Exception( "ConfigError", "can't read pdt file" )
	<< "wrong format of " << pdtFileName.fullPath();
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

using namespace HepPDT;

// namespace HepPDT {
 
//   bool getPythiaid( int & id, const std::string & pdline );
//   void parsePythiaLine( TempParticleData & tpd, int & anti, std::string & aname, const std::string & pdline );
//   void parsePythiaDecayLine( TempParticleData & tpd, const std::string & pdline );
//   TempDecayData getPythiaDecay( const std::string & pdline );
 
// }

bool  PythiaPDTESSource::cmsaddPythiaParticles( std::istream & pdfile, HepPDT::TableBuilder & tb )
{
  std::string pdline, aname;
  int id, kf;
  int saveid=0;
  int anti=0;
  // read and parse each line
  while( std::getline( pdfile, pdline) ) {
    if( getPythiaid( kf, pdline ) ) {
      if( kf != 0 ) {
          // this is a new particle definition
          saveid = id = kf;
          TempParticleData& tpd = tb.getParticleData( ParticleID( id ) );
          parsePythiaLine( tpd, anti, aname, pdline );
          if( anti > 0 ) {
              // code here to define antiparticles
              TempParticleData& atpd = tb.getAntiParticle( ParticleID( id ), aname );
              // use this variable (fake out the compiler)
              atpd.tempMass = tpd.tempMass;
          }
      } else if( saveid != 0 ) {
          TempParticleData& tpd = tb.getParticleData( ParticleID( saveid ) );
          parsePythiaDecayLine( tpd, pdline );
          if( anti > 0 ) {
              // code here to append antiparticle decays
          }
      }
    }
  }
  std::cout << "found " << tb.size() << " particles" << std::endl;
  return true;
}

//define this as a plug-in
//DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE( PythiaPDTESSource );
