#ifndef CaloSimAlgos_CaloHitRespoNew_h
#define CaloSimAlgos_CaloHitRespoNew_h

#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"

#include<vector>

/**

 \class CaloHitRespoNew

 \brief Creates electronics signals from hits 

*/


class CaloVShape              ;
class CaloVSimParameterMap    ;
class CaloVHitCorrection      ;
class CaloVHitFilter          ;
class CaloSimParameters       ;
class CaloSubdetectorGeometry ;
class CaloVPECorrection       ;
namespace CLHEP 
{ 
   class HepRandomEngine      ;
}

class CaloHitRespoNew 
{
   public:

      typedef std::vector< CaloSamples  > VecSam ;
      typedef std::vector< unsigned int > VecInd ;

      enum {BUNCHSPACE=25};

      CaloHitRespoNew( const CaloVSimParameterMap* parameterMap ,
		       const CaloVShape*           shape        ,
		       const DetId                 detId         ) ;

      virtual ~CaloHitRespoNew() ;

      void setBunchRange( int minBunch ,
			  int maxBunch   ) ;

      void setGeometry( const CaloSubdetectorGeometry* geometry ) ;

      void setPhaseShift( double phaseShift ) ;

      void setHitFilter( const CaloVHitFilter* filter) ;

      void setHitCorrection( const CaloVHitCorrection* hitCorrection) ;

      void setPECorrection( const CaloVPECorrection* peCorrection ) ;

      virtual void run( MixCollection<PCaloHit>& hits, CLHEP::HepRandomEngine* ) ;

      virtual void add(const PCaloHit & hit, CLHEP::HepRandomEngine*);

      unsigned int samplesSize() const ;

      const CaloSamples& operator[]( unsigned int i ) const ;

      virtual void initializeHits() {}

      virtual void finalizeHits() {}

      bool withinBunchRange(int bunchCrossing) const {
        return(bunchCrossing >= m_minBunch && bunchCrossing <= m_maxBunch);
      }


   protected:

      CaloSamples* findSignal( const DetId& detId ) ;

      virtual void putAnalogSignal( const PCaloHit& inputHit, CLHEP::HepRandomEngine*) ;

      double analogSignalAmplitude( const DetId& id, float energy, CLHEP::HepRandomEngine* ) const;

      double timeOfFlight( const DetId& detId ) const ;

      double phaseShift() const ;

      void setupSamples( const DetId& detId ) ;

      void blankOutUsedSamples() ;

      const CaloSimParameters* params( const DetId& detId ) const ;

      const CaloVShape* shape() const ;

      const CaloSubdetectorGeometry* geometry() const ;

      int minBunch() const { return m_minBunch ; }

      int maxBunch() const { return m_maxBunch ; }

      VecInd& index() { return m_index ; }

      const CaloVHitFilter* hitFilter() const { return m_hitFilter ; }

   private:

      const CaloVSimParameterMap*    m_parameterMap  ;
      const CaloVShape*              m_shape         ;
      const CaloVHitCorrection*      m_hitCorrection ;
      const CaloVPECorrection*       m_PECorrection  ;
      const CaloVHitFilter*          m_hitFilter     ;
      const CaloSubdetectorGeometry* m_geometry      ;

      int    m_minBunch   ;
      int    m_maxBunch   ;
      double m_phaseShift ;

      VecSam m_vSamp ;
      VecInd m_index ;
};

#endif
