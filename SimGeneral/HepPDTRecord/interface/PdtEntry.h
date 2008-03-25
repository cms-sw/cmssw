#ifndef Utilities_PdtEntry_h
#define Utilities_PdtEntry_h
/* \class PdtEntry
 *
 * \author Luca Lista, INFN
 *
 */
#include <string>
#include <iterator>

namespace edm { class EventSetup; }
namespace HepPDT { class ParticleData; }

class PdtEntry {
public:
  /// default construct
  explicit PdtEntry() : pdgId_( 0 ), data_( 0 ) { }
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

  template<>
  inline PdtEntry ParameterSet::getUntrackedParameter<PdtEntry>(std::string const& name) const {
    const Entry * e = getEntryPointerOrThrow_(name);
    if ( e->typeCode() == 'I' ) 
      return PdtEntry( e->getInt32() );
    else if( e->typeCode() == 'S' ) 
      return PdtEntry( e->getString() );
    else 
      throw Exception(errors::Configuration, "EntryError")
	<< "can not convert representation of " << name 
	<< " to value of type PdtEntry. "
	<< "Please, provide a parameter either of type int32 or string.";
  }

  template<>
  inline PdtEntry ParameterSet::getUntrackedParameter<PdtEntry>(std::string const& name, 
								const PdtEntry & defaultValue ) const {
    const Entry * e = retrieveUntracked(name);
    if ( e == 0 ) return defaultValue;
    if ( e->typeCode() == 'I' ) 
      return PdtEntry( e->getInt32() );
    else if( e->typeCode() == 'S' ) 
      return PdtEntry( e->getString() );
    else 
      throw Exception(errors::Configuration, "EntryError")
	<< "can not convert representation of " << name 
	<< " to value of type PdtEntry. "
	<< "Please, provide a parameter either of type int32 or string.";
  }

  template<>
  inline std::vector<PdtEntry> ParameterSet::getParameter<std::vector<PdtEntry> >(std::string const& name) const {
    const Entry & e = retrieve(name);
    std::vector<PdtEntry> ret;
    if ( e.typeCode() == 'i' ) { 
      std::vector<int> v( e.getVInt32() );
      for( std::vector<int>::const_iterator i = v.begin(); i != v.end(); ++ i )
        ret.push_back( PdtEntry( * i ) );
      return ret;
    }
    else if( e.typeCode() == 's' ) {
      std::vector<std::string> v( e.getVString() );
      for( std::vector<std::string>::const_iterator i = v.begin(); i != v.end(); ++ i )
        ret.push_back( PdtEntry( * i ) );
      return ret;
    }
    else 
      throw Exception(errors::Configuration, "EntryError")
	<< "can not convert representation of " << name 
	<< " to value of type PdtEntry. "
	<< "Please, provide a parameter either of type int32 or string.";
  }

  template<>
  inline std::vector<PdtEntry> ParameterSet::getUntrackedParameter<std::vector<PdtEntry> >(std::string const& name) const {
    const Entry * e = getEntryPointerOrThrow_(name);
    std::vector<PdtEntry> ret;
    if ( e->typeCode() == 'i' ) { 
      std::vector<int> v( e->getVInt32() );
      for( std::vector<int>::const_iterator i = v.begin(); i != v.end(); ++ i )
        ret.push_back( PdtEntry( * i ) );
      return ret;
    }
    else if( e->typeCode() == 's' ) {
      std::vector<std::string> v( e->getVString() );
      for( std::vector<std::string>::const_iterator i = v.begin(); i != v.end(); ++ i )
        ret.push_back( PdtEntry( * i ) );
      return ret;
    }
    else 
      throw Exception(errors::Configuration, "EntryError")
	<< "can not convert representation of " << name 
	<< " to value of type PdtEntry. "
	<< "Please, provide a parameter either of type int32 or string.";
  }

  template<>
  inline std::vector<PdtEntry> ParameterSet::getUntrackedParameter<std::vector<PdtEntry> >(std::string const& name,
											   const std::vector<PdtEntry> & defaultValue ) const {
    const Entry * e = retrieveUntracked(name);
    if ( e == 0 ) return defaultValue;
    std::vector<PdtEntry> ret;
    if ( e->typeCode() == 'i' ) { 
      std::vector<int> v( e->getVInt32() );
      for( std::vector<int>::const_iterator i = v.begin(); i != v.end(); ++ i )
        ret.push_back( PdtEntry( * i ) );
      return ret;
    }
    else if( e->typeCode() == 's' ) {
      std::vector<std::string> v( e->getVString() );
      for( std::vector<std::string>::const_iterator i = v.begin(); i != v.end(); ++ i )
        ret.push_back( PdtEntry( * i ) );
      return ret;
    }
    else 
      throw Exception(errors::Configuration, "EntryError")
	<< "can not convert representation of " << name 
	<< " to value of type PdtEntry. "
	<< "Please, provide a parameter either of type int32 or string.";
  }

  template<>
  inline std::vector<std::string> ParameterSet::getParameterNamesForType<PdtEntry>(bool trackiness) const {
    std::vector<std::string> ints = getParameterNamesForType<int>(trackiness);
    std::vector<std::string> strings = getParameterNamesForType<std::string>(trackiness);
    std::copy( strings.begin(), strings.end(), std::back_insert_iterator<std::vector<std::string> >( ints ) ); 
    return ints;
  }
  
  template<>
  inline std::vector<std::string> ParameterSet::getParameterNamesForType<std::vector<PdtEntry> >(bool trackiness) const {
    std::vector<std::string> ints = getParameterNamesForType<std::vector<int> >(trackiness);
    std::vector<std::string> strings = getParameterNamesForType<std::vector<std::string> >(trackiness);
    std::copy( strings.begin(), strings.end(), std::back_insert_iterator<std::vector<std::string> >( ints ) ); 
    return ints;
  }

}
#endif
