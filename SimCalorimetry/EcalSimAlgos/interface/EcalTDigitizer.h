#ifndef EcalSimAlgos_EcalTDigitizer_h
#define EcalSimAlgos_EcalTDigitizer_h

/** Turns hits into digis.  Assumes that 
    there's an ElectroncsSim class with the
    interface analogToDigital(const CaloSamples &, Digi &);
*/

#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "CalibFormats/CaloObjects/interface/CaloTSamplesBase.h"

class EcalHitResponse ;

template< class Traits >
class EcalTDigitizer
{
   public:

      typedef typename Traits::ElectronicsSim ElectronicsSim ;
      typedef typename Traits::Digi           Digi           ;
      typedef typename Traits::DigiCollection DigiCollection ;
      typedef typename Traits::EcalSamples    EcalSamples    ;

      EcalTDigitizer< Traits >( EcalHitResponse* hitResponse    ,
				ElectronicsSim*  electronicsSim ,
				bool             addNoise         ) ;

      virtual ~EcalTDigitizer< Traits >() ;

      void add(const std::vector<PCaloHit> & hits, int bunchCrossing);

      virtual void initializeHits();

      virtual void run(DigiCollection&          output  );

      virtual void run( MixCollection<PCaloHit>& input ,
			DigiCollection&          output  ) {
         assert(0);
      }

   protected:

      bool addNoise() const ;

      const EcalHitResponse* hitResponse() const ;

      const ElectronicsSim* elecSim() const ;

   private:

      EcalHitResponse* m_hitResponse    ;
      ElectronicsSim*  m_electronicsSim ;
      bool             m_addNoise       ;
};

#endif

