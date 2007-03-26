#ifndef Utilities_PdtEntry_h
#define Utilities_PdtEntry_h
/* \class PdtEntry
 *
 * \author Luca Lista, INFN
 *
 */
#include <string>

namespace edm { class EventSetup; }
namespace HepPDT { class ParticleData; }

class PdtEntry {
public:
  /// construct from PDG id
  explicit PdtEntry( int pdgId ) : pdgId_( pdgId ), data_( 0 ) { }
  /// construct from particle name
  explicit PdtEntry( const std::string & name ) : pdgId_( 0 ), name_( name ), data_( 0 ) { }
  /// PDG id
  int pdgId() const;
  /// particle name
  const std::string & name() const;
  /// particle data
  const HepPDT::ParticleData & data() const; 
  /// fill data from Event Setup
  void setup( const edm::EventSetup & ); 

private:
  /// PDG id
  int pdgId_;
  /// particle name
  std::string name_;
  /// particle data
  const HepPDT::ParticleData * data_;
};

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {
  template<>
    inline PdtEntry ParameterSet::getParameter<PdtEntry>(std::string const& name) const {
    const Entry & e = retrieve(name);
    if ( e.typeCode() == 'I' ) 
      return PdtEntry( e.getInt32() );
    else if( e.typeCode() == 'S' ) 
      return PdtEntry( e.getString() );
    else 
      throw Exception(errors::Configuration, "EntryError")
	<< "can not convert representation of " << name 
	<< " to value of type PdtEntry. "
	<< "Please, provide a parameter either of type int32 or string.";
  }
}

#endif
