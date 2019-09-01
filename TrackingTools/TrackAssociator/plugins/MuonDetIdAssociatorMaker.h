#ifndef TrackingTools_TrackAssociator_MuonDetIdAssociatorMaker_h
#define TrackingTools_TrackAssociator_MuonDetIdAssociatorMaker_h
// -*- C++ -*-
//
// Package:     TrackingTools/TrackAssociator
// Class  :     MuonDetIdAssociatorMaker
//
/**\class MuonDetIdAssociatorMaker MuonDetIdAssociatorMaker.h "MuonDetIdAssociatorMaker.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Thu, 30 May 2019 14:59:09 GMT
//

// system include files
#include <memory>

// user include files
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESConsumesCollector.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "DetIdAssociatorMaker.h"

// forward declarations
class DetIdAssociator;
class DetIdAssociatorRecord;
class GlobalTrackingGeometry;
class GlobalTrackingGeometryRecord;
class CSCBadChambers;
class CSCBadChambersRcd;

class MuonDetIdAssociatorMaker : public DetIdAssociatorMaker {
public:
  MuonDetIdAssociatorMaker(edm::ParameterSet const&, edm::ESConsumesCollectorT<DetIdAssociatorRecord>&&);

  // ---------- const member functions ---------------------
  std::unique_ptr<DetIdAssociator> make(const DetIdAssociatorRecord&) const final;

private:
  edm::ESGetToken<GlobalTrackingGeometry, GlobalTrackingGeometryRecord> geomToken_;
  edm::ESGetToken<CSCBadChambers, CSCBadChambersRcd> badChambersToken_;
  const double etaBinSize;
  const int nPhi;
  const int nEta;
  const bool includeBadChambers_;
  const bool includeGEM_;
  const bool includeME0_;
};

#endif
