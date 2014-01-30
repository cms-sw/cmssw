#ifndef EcalSimAlgos_EcalTimeMapDigitizer_h
#define EcalSimAlgos_EcalTimeMapDigitizer_h

/** Turns hits into digis.  Assumes that 
    there's an ElectroncsSim class with the
    interface analogToDigital(const CaloSamples &, Digi &);
*/

#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "CalibFormats/CaloObjects/interface/CaloTSamplesBase.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

class CaloSubdetectorGeometry ;

class EcalTimeMapDigitizer
{
   public:
      EcalTimeMapDigitizer();

      virtual ~EcalTimeMapDigitizer() {};

      void add(const std::vector<PCaloHit> & hits, int bunchCrossing);

      void setGeometry( const CaloSubdetectorGeometry* geometry ) ;

      void initializeMap();

      void run(EcalTimeDigiCollection&          output  );

      double timeOfFlight( const DetId& detId ) const; 

   private:
      
      //time difference between bunches
      static const int BUNCHSPACE=25;
      static const int    m_minBunch=-4;
      static const int    m_maxBunch=4;

      const CaloSubdetectorGeometry* m_geometry ;

      std::vector<float> timeMap;

};

#endif
