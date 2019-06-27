#ifndef TrackingTools_TrackAssociator_CaloDetIdAssociatorMaker_h
#define TrackingTools_TrackAssociator_CaloDetIdAssociatorMaker_h
// -*- C++ -*-
//
// Package:     TrackingTools/TrackAssociator
// Class  :     CaloDetIdAssociatorMaker
//
/**\class CaloDetIdAssociatorMaker CaloDetIdAssociatorMaker.h "CaloDetIdAssociatorMaker.h"

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
class CaloGeometry;
class CaloGeometryRecord;

class CaloDetIdAssociatorMaker : public DetIdAssociatorMaker {
public:
  CaloDetIdAssociatorMaker(edm::ParameterSet const&, edm::ESConsumesCollectorT<DetIdAssociatorRecord>&&);

  // ---------- const member functions ---------------------
  std::unique_ptr<DetIdAssociator> make(const DetIdAssociatorRecord&) const final;

private:
  virtual std::unique_ptr<DetIdAssociator> make(CaloGeometry const&, int nPhi, int nEta, double etaBinSize) const;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;
  const double etaBinSize;
  const int nPhi;
  const int nEta;
};

#endif
