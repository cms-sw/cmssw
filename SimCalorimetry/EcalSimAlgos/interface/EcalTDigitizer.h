#ifndef EcalSimAlgos_EcalTDigitizer_h
#define EcalSimAlgos_EcalTDigitizer_h

/** Turns hits into digis.  Assumes that 
    there's an ElectroncsSim class with the
    interface analogToDigital(const CaloSamples &, Digi &);
*/

#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "DataFormats/DetId/interface/DetId.h"

class EcalHitResponse ;

template< class Traits >
class EcalTDigitizer
{
   public:

      typedef typename Traits::ElectronicsSim ElectronicsSim ;
      typedef typename Traits::Digi           Digi           ;
      typedef typename Traits::DigiCollection DigiCollection ;

//      typedef CaloTSamplesBase<float> EcalSamples ;

      EcalTDigitizer< Traits >( EcalHitResponse* hitResponse    ,
				ElectronicsSim*  electronicsSim ,
				bool             addNoise         ) ;

      ~EcalTDigitizer< Traits >() ;

      void run( MixCollection<PCaloHit>& input ,
		DigiCollection&          output  ) ;

   private:

      EcalHitResponse* m_hitResponse    ;
      ElectronicsSim*  m_electronicsSim ;
      bool             m_addNoise         ;
};

#endif

