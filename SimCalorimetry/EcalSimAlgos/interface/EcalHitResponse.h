#ifndef EcalSimAlgos_EcalHitResponse_h
#define EcalSimAlgos_EcalHitResponse_h

#include "CalibFormats/CaloObjects/interface/CaloTSamplesBase.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbService.h"
#include "DataFormats/Provenance/interface/Timestamp.h"

#include<vector>

typedef unsigned long long TimeValue_t;

class CaloVShape              ;
class CaloVSimParameterMap    ;
class CaloVHitCorrection      ;
class CaloVHitFilter          ;
class CaloSimParameters       ;
class CaloSubdetectorGeometry ;
class CaloVPECorrection       ;
namespace CLHEP 
{ 
   class RandPoissonQ         ; 
   class RandGaussQ           ; 
   class HepRandomEngine      ;
}

class EcalHitResponse 
{
   public:

      typedef CaloTSamplesBase<float> EcalSamples ;

      typedef std::vector< unsigned int > VecInd ;

      enum { BUNCHSPACE = 25 } ;

      EcalHitResponse( const CaloVSimParameterMap* parameterMap ,
		       const CaloVShape*           shape          ) ;

      virtual ~EcalHitResponse() ;

      void setBunchRange( int minBunch ,
			  int maxBunch   ) ;

      void setGeometry( const CaloSubdetectorGeometry* geometry ) ;

      void setPhaseShift( double phaseShift ) ;

      void setHitFilter( const CaloVHitFilter* filter) ;

      void setHitCorrection( const CaloVHitCorrection* hitCorrection) ;

      void setPECorrection( const CaloVPECorrection* peCorrection ) ;

      void setEventTime(const edm::TimeValue_t& iTime);

      void setLaserConstants(const EcalLaserDbService* laser, bool& useLCcorrection);

      void add( const EcalSamples* pSam ) ;

      virtual void add( const PCaloHit&  hit ) ;

      virtual void initializeHits() ;

      virtual void finalizeHits() ;

      virtual void run( MixCollection<PCaloHit>& hits ) ;

      virtual unsigned int samplesSize() const = 0 ;

      virtual EcalSamples* operator[]( unsigned int i ) = 0;

      virtual const EcalSamples* operator[]( unsigned int i ) const = 0;

      const EcalSamples* findDetId( const DetId& detId ) const ;

      bool withinBunchRange(int bunchCrossing) const ;

   protected:

      virtual unsigned int samplesSizeAll() const = 0 ;

      virtual EcalSamples* vSam( unsigned int i ) = 0 ;

      virtual EcalSamples* vSamAll( unsigned int i ) = 0 ;

      virtual const EcalSamples* vSamAll( unsigned int i ) const = 0 ;

      virtual void putAnalogSignal( const PCaloHit& inputHit) ;

      double findLaserConstant(const DetId& detId) const;

      EcalSamples* findSignal( const DetId& detId ) ;

      double analogSignalAmplitude( const DetId& id, float energy ) const;

      double timeOfFlight( const DetId& detId ) const ;

      double phaseShift() const ;

      CLHEP::RandPoissonQ* ranPois() const ;

      CLHEP::RandGaussQ* ranGauss() const ;

      void blankOutUsedSamples() ;

      const CaloSimParameters* params( const DetId& detId ) const ;

      const CaloVShape* shape() const ;

      const CaloSubdetectorGeometry* geometry() const ;

      int minBunch() const ;

      int maxBunch() const ;

      VecInd& index() ;

      const VecInd& index() const ;

      const CaloVHitFilter* hitFilter() const ;

   private:

      const CaloVSimParameterMap*    m_parameterMap  ;
      const CaloVShape*              m_shape         ;
      const CaloVHitCorrection*      m_hitCorrection ;
      const CaloVPECorrection*       m_PECorrection  ;
      const CaloVHitFilter*          m_hitFilter     ;
      const CaloSubdetectorGeometry* m_geometry      ;
      const EcalLaserDbService*      m_lasercals     ;

      mutable CLHEP::RandPoissonQ*   m_RandPoisson   ;
      mutable CLHEP::RandGaussQ*     m_RandGauss     ;

      int    m_minBunch   ;
      int    m_maxBunch   ;
      double m_phaseShift ;

      edm::TimeValue_t               m_iTime;
      bool                           m_useLCcorrection;

      VecInd m_index ;
};

#endif
