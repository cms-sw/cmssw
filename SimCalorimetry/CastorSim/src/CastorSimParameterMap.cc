#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalCastorDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "SimCalorimetry/CastorSim/src/CastorSimParameterMap.h"
#include <iostream>

// some arbitrary numbers for now
CastorSimParameterMap::CastorSimParameterMap() : theCastorParameters(1., 4.3333, 2.09, -4., false) {}
/*
  CaloSimParameters(double photomultiplierGain, double amplifierGain,
                 double samplingFactor, double timePhase,
                 int readoutFrameSize, int binOfMaximum,
                 bool doPhotostatistics);
*/

CastorSimParameterMap::CastorSimParameterMap(const edm::ParameterSet &p)
    : theCastorParameters(p.getParameter<edm::ParameterSet>("castor")) {}

const CaloSimParameters &CastorSimParameterMap::simParameters(const DetId &detId) const {
  HcalGenericDetId genericId(detId);

  //  if(detId.det()==DetId::Calo &&
  //  detId.subdetId()==HcalCastorDetId::SubdetectorId)

  if (genericId.isHcalCastorDetId())
    return theCastorParameters;

  else
    throw cms::Exception("not HcalCastorDetId");
}

void CastorSimParameterMap::setDbService(const CastorDbService *dbService) {
  theCastorParameters.setDbService(dbService);
}
