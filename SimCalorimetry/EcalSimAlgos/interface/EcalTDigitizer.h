#ifndef EcalSimAlgos_EcalTDigitizer_h
#define EcalSimAlgos_EcalTDigitizer_h

/** Turns hits into digis.  Assumes that 
    there's an ElectroncsSim class with the
    interface analogToDigital(const CaloSamples &, Digi &);
*/

#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"

class CaloHitResponse ;

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

      EcalTDigitizer< Traits >( CaloHitResponse* hitResponse    ,
				ElectronicsSim*  electronicsSim ,
				bool             addNoise         ) ;

      ~EcalTDigitizer< Traits >() ;

      /// tell the digitizer which cells exist
      void setDetIds( const std::vector<DetId>& detIds ) ;

      /// turns hits into digis
      void run( MixCollection<PCaloHit>& input ,
		DigiCollection&          output  ) ;

   private:

      CaloHitResponse*          theHitResponse    ;
      ElectronicsSim*           theElectronicsSim ;
      const std::vector<DetId>* theDetIds         ;
      bool                      addNoise_         ;

};

#endif

