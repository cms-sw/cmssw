#include "SimCalorimetry/CastorSim/src/CastorSimParameterMap.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalCastorDetId.h"

CastorSimParameterMap::CastorSimParameterMap() :
  theCastorParameters(1., 4.3333,
                   2.09 , -4,
                   6, 4, false)
{
}
/*
  CaloSimParameters(double photomultiplierGain, double amplifierGain,
                 double samplingFactor, double timePhase,
                 int readoutFrameSize, int binOfMaximum,
                 bool doPhotostatistics);
*/

CastorSimParameterMap::CastorSimParameterMap(const edm::ParameterSet & p)
:  theCastorParameters( p.getParameter<edm::ParameterSet>("castor") ) 
{
}

const CaloSimParameters & CastorSimParameterMap::simParameters(const DetId & detId) const {
  if(detId.det()==DetId::Calo && detId.subdetId()==HcalCastorDetId::SubdetectorId)
    return theCastorParameters;
}

void CastorSimParameterMap::setDbService(const HcalDbService * dbService)
{
//  theCastorParameters.setDbService(dbService);
}

