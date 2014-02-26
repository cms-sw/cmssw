#ifndef EcalSimAlgos_EcalTimeMapDigitizer_h
#define EcalSimAlgos_EcalTimeMapDigitizer_h

/** Turns hits into digis.  Assumes that 
    there's an ElectroncsSim class with the
    interface analogToDigital(const CaloSamples &, Digi &);
*/

#include "CalibFormats/CaloObjects/interface/CaloTSamples.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "CalibFormats/CaloObjects/interface/CaloTSamplesBase.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

class CaloSubdetectorGeometry ;

class EcalTimeMapDigitizer
{
 public:

  typedef CaloTSamples<float,10> TimeSamples ;

  typedef std::vector< unsigned int > VecInd ;

  explicit EcalTimeMapDigitizer(EcalSubdetector myDet);
  
  virtual ~EcalTimeMapDigitizer();
  
  void add(const std::vector<PCaloHit> & hits, int bunchCrossing);
  
  void setGeometry( const CaloSubdetectorGeometry* geometry ) ;

  void initializeMap();
  
  void run(EcalTimeDigiCollection& output  );

  void finalizeHits();
  
  inline void setTimeLayerId( const int& layerId ) { m_timeLayerId = layerId; };
  
  int getTimeLayerId() { return m_timeLayerId; };
  
  EcalSubdetector subdetector() { return m_subDet; };

  int  minBunch() const ;

  int  maxBunch() const  ;

  void blankOutUsedSamples() ; // blank out previously used elements
  

/*   const CaloVHitFilter* hitFilter() const ; */

 private:
  
  TimeSamples* findSignal( const DetId& detId );

  VecInd& index() ;
  
  const VecInd& index() const ;

  unsigned int samplesSize() const;

  unsigned int samplesSizeAll() const;

  const TimeSamples*  operator[]( unsigned int i ) const;

  TimeSamples*  operator[]( unsigned int i );

  TimeSamples*  vSam( unsigned int i );

  TimeSamples*  vSamAll( unsigned int i );

  const TimeSamples*  vSamAll( unsigned int i ) const;


  EcalSubdetector m_subDet;
  //time difference between bunches
  static const int BUNCHSPACE=25;
  
  static const int    m_minBunch=-4;
  static const int    m_maxBunch=5;
  
  int m_timeLayerId;
  
  const CaloSubdetectorGeometry* m_geometry ;
  
  double timeOfFlight( const DetId& detId , int layer) const; 

  std::vector<TimeSamples> m_vSam;

  VecInd m_index;  

};

#endif
