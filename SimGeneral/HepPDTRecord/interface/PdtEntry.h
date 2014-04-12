#ifndef Utilities_PdtEntry_h
#define Utilities_PdtEntry_h
/* \class PdtEntry
 *
 * \author Luca Lista, INFN
 *
 */
#include <string>
#include <iterator>
#include <vector>

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

namespace edm {
  namespace pdtentry {
    PdtEntry getPdtEntry(Entry const& e, char const* name);
    std::vector<PdtEntry> getPdtEntryVector(Entry const& e, char const* name);
  }

  template<>
  inline PdtEntry ParameterSet::getParameter<PdtEntry>(std::string const& name) const {
    Entry const& e = retrieve(name);
    return pdtentry::getPdtEntry(e, name.c_str());
  }

  template<>
  inline PdtEntry ParameterSet::getUntrackedParameter<PdtEntry>(std::string const& name) const {
    Entry const* e = getEntryPointerOrThrow_(name);
    return pdtentry::getPdtEntry(*e, name.c_str());
  }

  template<>
  inline PdtEntry ParameterSet::getUntrackedParameter<PdtEntry>(std::string const& name, 
								PdtEntry const& defaultValue) const {
    Entry const* e = retrieveUntracked(name);
    if (e == 0) return defaultValue;
    return pdtentry::getPdtEntry(*e, name.c_str());
  }

  template<>
  inline PdtEntry ParameterSet::getParameter<PdtEntry>(char const* name) const {
    Entry const& e = retrieve(name);
    return pdtentry::getPdtEntry(e, name);
  }

  template<>
  inline PdtEntry ParameterSet::getUntrackedParameter<PdtEntry>(char const* name) const {
    Entry const* e = getEntryPointerOrThrow_(name);
    return pdtentry::getPdtEntry(*e, name);
  }

  template<>
  inline PdtEntry ParameterSet::getUntrackedParameter<PdtEntry>(char const* name, 
								PdtEntry const& defaultValue) const {
    Entry const* e = retrieveUntracked(name);
    if (e == 0) return defaultValue;
    return pdtentry::getPdtEntry(*e, name);
  }

  template<>
  inline std::vector<PdtEntry> ParameterSet::getParameter<std::vector<PdtEntry> >(std::string const& name) const {
    Entry const& e = retrieve(name);
    return pdtentry::getPdtEntryVector(e, name.c_str());
  }

  template<>
  inline std::vector<PdtEntry> ParameterSet::getUntrackedParameter<std::vector<PdtEntry> >(std::string const& name) const {
    Entry const* e = getEntryPointerOrThrow_(name);
    return pdtentry::getPdtEntryVector(*e, name.c_str());
  }

  template<>
  inline std::vector<PdtEntry> ParameterSet::getUntrackedParameter<std::vector<PdtEntry> >(std::string const& name,
								   std::vector<PdtEntry> const& defaultValue) const {
    Entry const* e = retrieveUntracked(name);
    if (e == 0) return defaultValue;
    return pdtentry::getPdtEntryVector(*e, name.c_str());
  }

  template<>
  inline std::vector<PdtEntry> ParameterSet::getParameter<std::vector<PdtEntry> >(char const* name) const {
    Entry const& e = retrieve(name);
    return pdtentry::getPdtEntryVector(e, name);
  }

  template<>
  inline std::vector<PdtEntry> ParameterSet::getUntrackedParameter<std::vector<PdtEntry> >(char const* name) const {
    Entry const* e = getEntryPointerOrThrow_(name);
    return pdtentry::getPdtEntryVector(*e, name);
  }

  template<>
  inline std::vector<PdtEntry> ParameterSet::getUntrackedParameter<std::vector<PdtEntry> >(char const* name,
								   std::vector<PdtEntry> const& defaultValue) const {
    Entry const* e = retrieveUntracked(name);
    if (e == 0) return defaultValue;
    return pdtentry::getPdtEntryVector(*e, name);
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
