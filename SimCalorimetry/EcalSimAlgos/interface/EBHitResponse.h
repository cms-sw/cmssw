#ifndef EcalSimAlgos_EBHitResponse_h
#define EcalSimAlgos_EBHitResponse_h

#include "SimCalorimetry/CaloSimAlgos/interface/CaloHitRespoNew.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstantsMC.h"

/**

 \class EBHitResponse

 \brief Creates electronics signals from EB hits , including APD

*/

class APDSimParameters ;

class EBHitResponse : public CaloHitRespoNew
{
   public:

      typedef CaloHitRespoNew CaloHitResponse ;

      EBHitResponse( const CaloVSimParameterMap* parameterMap , 
		     const CaloVShape*           shape        ,
		     bool                        apdOnly      ,
		     const APDSimParameters*     apdPars      , 
		     const CaloVShape*           apdShape       ) ;

      virtual ~EBHitResponse() ;

      virtual bool keepBlank() const { return false ; }

      void setIntercal( const EcalIntercalibConstantsMC* ical ) ;

   protected:

      virtual void putAnalogSignal( const PCaloHit & inputHit ) ;

   private:

      const APDSimParameters* apdParameters() const ;
      const CaloVShape*       apdShape()      const ;

      double apdSignalAmplitude( const PCaloHit& hit ) const ;

      void findIntercalibConstant( const DetId& detId, 
				   double&      icalconst ) const ;

      const bool                       m_apdOnly  ;
      const APDSimParameters*          m_apdPars  ;
      const CaloVShape*                m_apdShape ;
      const EcalIntercalibConstantsMC* m_intercal ;
};
#endif


