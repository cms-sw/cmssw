#ifndef EcalSimAlgos_CaloHitResponse_h
#define EcalSimAlgos_CaloHitResponse_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitResponse.h"

class ESHitResponse : public CaloHitResponse
{
   public:

      ESHitResponse( const CaloVSimParameterMap* parameterMap,
		     const CaloVShape*           shape) :
	 CaloHitResponse( parameterMap, shape ) {}
      ESHitResponse( const CaloVSimParameterMap* parameterMap,
		     const CaloShapes*           shapes        ) :
	 CaloHitResponse( parameterMap, shapes ) {}

      virtual ~ESHitResponse() {}

      const AnalogSignalMap& signalMap() const { return theAnalogSignalMap ; }
};

#endif


