#ifndef EcalSimAlgos_EBHitResponse_h
#define EcalSimAlgos_EBHitResponse_h

#include "CalibFormats/CaloObjects/interface/CaloTSamples.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalHitResponse.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstantsMC.h"

class APDSimParameters ;

namespace CLHEP {
   class HepRandomEngine;
}

class EBHitResponse : public EcalHitResponse
{
   public:

      typedef CaloTSamples<float,10> EBSamples ;

      typedef std::vector<double> VecD ;

      enum { kNOffsets = 2000 } ;

      EBHitResponse( const CaloVSimParameterMap* parameterMap , 
		     const CaloVShape*           shape        ,
		     bool                        apdOnly      ,
		     const APDSimParameters*     apdPars      , 
		     const CaloVShape*           apdShape       ) ;

      virtual ~EBHitResponse() ;

      void initialize(CLHEP::HepRandomEngine*);

      virtual bool keepBlank() const { return false ; }

      void setIntercal( const EcalIntercalibConstantsMC* ical ) ;


      virtual void add( const PCaloHit&  hit, CLHEP::HepRandomEngine* ) override;

      virtual void initializeHits() override;

      virtual void finalizeHits() override;

      virtual void run( MixCollection<PCaloHit>& hits, CLHEP::HepRandomEngine* ) override;

      virtual unsigned int samplesSize() const override;

      virtual EcalSamples* operator[]( unsigned int i ) override;

      virtual const EcalSamples* operator[]( unsigned int i ) const override;

   protected:

      virtual unsigned int samplesSizeAll() const override;

      virtual EcalSamples* vSamAll( unsigned int i ) override;

      virtual const EcalSamples* vSamAll( unsigned int i ) const override;

      virtual EcalSamples* vSam( unsigned int i ) override ;

      void putAPDSignal( const DetId& detId, double npe, double time ) ;

   private:

      const VecD& offsets() const { return m_timeOffVec ; }

      const double nonlFunc( double enr ) const {
	 return ( pelo > enr ? pext :
		  ( pehi > enr ? nonlFunc1( enr ) : 
		    pfac*atan( log10( enr - pehi + 0.00001 ) ) + poff ) ) ; }

      const double nonlFunc1( double energy ) const {
	 const double enr ( log10(energy) ) ;
	 const double enr2 ( enr*enr ) ;
	 const double enr3 ( enr2*enr ) ;
	 return ( pcub*enr3 + pqua*enr2 + plin*enr + pcon ) ; }

      const APDSimParameters* apdParameters() const ;
      const CaloVShape*       apdShape()      const ;

      double apdSignalAmplitude( const PCaloHit& hit, CLHEP::HepRandomEngine* ) const ;

      void findIntercalibConstant( const DetId& detId, 
				   double&      icalconst ) const ;

      const bool                       m_apdOnly  ;
      const APDSimParameters*          m_apdPars  ;
      const CaloVShape*                m_apdShape ;
      const EcalIntercalibConstantsMC* m_intercal ;

      std::vector<double> m_timeOffVec ;

      std::vector<double> m_apdNpeVec ;
      std::vector<double> m_apdTimeVec ;

      const double pcub, pqua, plin, pcon, pelo, pehi, pasy, pext, poff, pfac ;

      std::vector<EBSamples> m_vSam ;

      bool m_isInitialized;
};
#endif


