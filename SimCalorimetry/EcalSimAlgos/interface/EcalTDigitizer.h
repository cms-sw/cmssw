#ifndef EcalSimAlgos_EcalTDigitizer_h
#define EcalSimAlgos_EcalTDigitizer_h

/** Turns hits into digis.  Assumes that 
    there's an ElectroncsSim class with the
    interface analogToDigital(const CaloSamples &, Digi &);
*/

#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "DataFormats/DetId/interface/DetId.h"

class CaloHitRespoNew ;

template< class Traits >
class EcalTDigitizer
{
   public:
      /// these are the types that need to be defined in the Traits
      /// class.  The ElectronicsSim needs to have an interface
      /// that you'll see in the run() method
      typedef typename Traits::ElectronicsSim ElectronicsSim ;
      typedef typename Traits::Digi           Digi           ;
      typedef typename Traits::DigiCollection DigiCollection ;

      typedef CaloHitRespoNew CaloHitResponse ;

      EcalTDigitizer< Traits >( CaloHitResponse* hitResponse    ,
				ElectronicsSim*  electronicsSim ,
				bool             addNoise         ) ;

      ~EcalTDigitizer< Traits >() ;

      void run( MixCollection<PCaloHit>& input ,
		DigiCollection&          output  ) ;

   private:

      CaloHitResponse* m_hitResponse    ;
      ElectronicsSim*  m_electronicsSim ;
      bool             m_addNoise         ;
};

#endif

