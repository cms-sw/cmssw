#ifndef HepPDTESSource_h
#define HepPDTESSource_h
// -*- C++ -*-
//
// Package:    HepPDTESSource
// Class:      HepPDTESSource
//
/**\class HepPDTESSource HepPDTESSource.h
 PhysicsTools/HepPDTESSource/interface/HepPDTESSource.h

 Description: HepPDT particle data table ESSource

 Implementation:
    <Notes on implementation>
*/
//
// Original Author:  Luca Lista
//         Created:  Fri Mar 10 15:58:18 CET 2006
//
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordInfiniteIntervalFinder.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "HepPDT/ParticleDataTable.hh"
#include "HepPDT/TableBuilder.hh"
#include "SimGeneral/HepPDTRecord/interface/PDTRecord.h"
#include <climits>
#include <fstream>
#include <memory>

class HepPDTESSource : public edm::ESProducer, public edm::EventSetupRecordInfiniteIntervalFinder {
public:
  /// constructor from parameter set
  HepPDTESSource(const edm::ParameterSet &);
  /// destructor
  ~HepPDTESSource() override;
  /// define the particle data table type
  typedef HepPDT::ParticleDataTable PDT;
  /// define the return type
  typedef std::unique_ptr<PDT> ReturnType;
  /// return the particle table
  ReturnType produce(const PDTRecord &);

private:
  edm::FileInPath pdtFileName;
};
#endif
