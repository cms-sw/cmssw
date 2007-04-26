#ifndef PythiaPDTESSource_h
#define PythiaPDTESSource_h

// -*- C++ -*-
//
// Package:    PythiaPDTESSource
// Class:      PythiaPDTESSource
// 
/**\class PythiaPDTESSource

 Description: Pythia PDT particle data table ESSource

 Implementation:
    <Notes on implementation>
*/

#include <memory>
#include <fstream>
#include "boost/shared_ptr.hpp"
#include "HepPDT/TableBuilder.hh"
#include "HepPDT/ParticleDataTable.hh"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "SimGeneral/HepPDTRecord/interface/PDTRecord.h"
#include <climits>

class PythiaPDTESSource : public edm::ESProducer, public  edm::EventSetupRecordIntervalFinder {
public:
  /// constructor from parameter set
  PythiaPDTESSource( const edm::ParameterSet& );
  /// destructor
  ~PythiaPDTESSource();
  /// define the particle data table type
  typedef HepPDT::ParticleDataTable PDT;
  /// define the return type
  typedef std::auto_ptr<PDT> ReturnType;
  /// return the particle table
  ReturnType produce( const PDTRecord & );
  /// set validity interval
  void setIntervalFor( const edm::eventsetup::EventSetupRecordKey &,
		       const edm::IOVSyncValue &,
		       edm::ValidityInterval & );

  // temporary solution to circumvent internal HepPDT translation

  bool  cmsaddPythiaParticles( std::istream & pdfile, HepPDT::TableBuilder & tb );

                                                                                                                                          
private:
  edm::FileInPath pdtFileName;

};
#endif
