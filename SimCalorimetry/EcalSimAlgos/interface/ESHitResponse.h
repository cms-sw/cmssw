#ifndef EcalSimAlgos_ESHitResponse_h
#define EcalSimAlgos_ESHitResponse_h

#include "CalibFormats/CaloObjects/interface/CaloTSamples.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalHitResponse.h"

class ESHitResponse : public EcalHitResponse
{
   public:

      typedef CaloTSamples<float,3> ESSamples ;

      ESHitResponse( const CaloVSimParameterMap* parameterMap , 
		     const CaloVShape*           shape          ) ;

      virtual ~ESHitResponse() ;

      virtual bool keepBlank() const { return false ; }

      virtual const EcalSamples* operator[]( unsigned int i ) const ;

   protected:

      virtual EcalSamples* vSam( unsigned int i ) ;

   private:

      std::vector<ESSamples> m_vSam ;
};
#endif


