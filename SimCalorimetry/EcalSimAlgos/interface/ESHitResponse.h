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

      virtual unsigned int samplesSize() const ;

      virtual EcalSamples* operator[]( unsigned int i ) ;

      virtual const EcalSamples* operator[]( unsigned int i ) const ;

   protected:

      virtual unsigned int samplesSizeAll() const ;

      virtual EcalSamples* vSamAll( unsigned int i ) ;

      virtual const EcalSamples* vSamAll( unsigned int i ) const ;

      virtual EcalSamples* vSam( unsigned int i ) ;

   private:

      std::vector<ESSamples> m_vSam ;
};
#endif


