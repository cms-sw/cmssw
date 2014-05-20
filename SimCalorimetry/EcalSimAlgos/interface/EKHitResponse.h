#ifndef EcalSimAlgos_EKHitResponse_h
#define EcalSimAlgos_EKHitResponse_h

#include "CalibFormats/CaloObjects/interface/CaloTSamples.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalHitResponse.h"

class EKHitResponse : public EcalHitResponse
{
   public:

      typedef CaloTSamples<float,10> EKSamples ;

      EKHitResponse( const CaloVSimParameterMap* parameterMap , 
		     const CaloVShape*           shape          ) ;

      virtual ~EKHitResponse() ;

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

      std::vector<EKSamples> m_vSam ;
};
#endif


