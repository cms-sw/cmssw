#ifndef EcalSimAlgos_ESDigitizer_h
#define EcalSimAlgos_ESDigitizer_h

#include "SimCalorimetry/EcalSimAlgos/interface/EcalTDigitizer.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalDigitizerTraits.h"

namespace CLHEP {
   class RandGeneral ;
   class RandPoissonQ ;
   class RandFlat ;
   class HepRandomEngine ; } 

#include <vector>

class ESDigitizer : public EcalTDigitizer< ESDigitizerTraits >
{
   public:

      typedef ESDigitizerTraits::ElectronicsSim ElectronicsSim ;

      ESDigitizer( EcalHitResponse* hitResponse    ,
		   ElectronicsSim*  electronicsSim ,
		   bool             addNoise         ) ;

      virtual ~ESDigitizer() ;

      virtual void run( ESDigiCollection& output ) ;

      void setDetIds( const std::vector<DetId>& detIds ) ;

      void setGain( const int gain ) ;

   private:

      void createNoisyList( std::vector<DetId>& abThreshCh ) ;  

      const std::vector<DetId>* m_detIds      ;
      CLHEP::HepRandomEngine*   m_engine      ;
      CLHEP::RandGeneral*       m_ranGeneral  ;
      CLHEP::RandPoissonQ*      m_ranPois     ;
      CLHEP::RandFlat*          m_ranFlat     ;
      int                       m_ESGain      ;
      double                    m_histoBin    ;
      double                    m_histoInf    ;
      double                    m_histoWid    ;
      double                    m_meanNoisy   ;

      class Triplet
      {
	 public:
	    Triplet() : first ( 0 ), second ( 0 ), third ( 0 ) {}
	    Triplet( uint32_t a0 ,
		     uint32_t a1 ,
		     uint32_t a2  ) :
	       first ( a0 ), second ( a1 ), third ( a2 ) {}
	    ~Triplet() {} ;
	    uint32_t first  ;
	    uint32_t second ;
	    uint32_t third  ;
      };




      std::vector<Triplet> m_trip ;
};

#endif

