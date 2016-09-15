#include "SimCalorimetry/EcalSimAlgos/interface/EcalTimeMapDigitizer.h" 

#include "Geometry/CaloGeometry/interface/CaloGenericDetId.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"

//#include "FWCore/ServiceRegistry/interface/Service.h"
//#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
//#include "CLHEP/Random/RandPoissonQ.h"
//#include "CLHEP/Random/RandGaussQ.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h" 

#include <iostream>

//#define ecal_time_debug 1

const float EcalTimeMapDigitizer::MIN_ENERGY_THRESHOLD=5e-5; //50 KeV threshold to consider a valid hit in the timing detector

EcalTimeMapDigitizer::EcalTimeMapDigitizer(EcalSubdetector myDet):
  m_subDet(myDet),
  m_geometry(0)
{
//    edm::Service<edm::RandomNumberGenerator> rng ;
//    if ( !rng.isAvailable() ) 
//    {
//       throw cms::Exception("Configuration")
// 	 << "EcalTimeMapDigitizer requires the RandomNumberGeneratorService\n"
// 	 "which is not present in the configuration file.  You must add the service\n"
// 	 "in the configuration file or remove the modules that require it.";
//    }
//    m_RandPoisson = new CLHEP::RandPoissonQ( rng->getEngine() ) ;
//    m_RandGauss   = new CLHEP::RandGaussQ(   rng->getEngine() ) ;

  unsigned int size=0;
  DetId detId(0);

  //Initialize the map
  if (myDet==EcalBarrel)
    {
      size=EBDetId::kSizeForDenseIndexing;
      detId=EBDetId::detIdFromDenseIndex( 0 ) ;
    }
  else if (myDet==EcalEndcap)
    {
      size=EEDetId::kSizeForDenseIndexing;
      detId=EEDetId::detIdFromDenseIndex( 0 ) ;
    }
  else 
    edm::LogError("TimeDigiError") << "[EcalTimeMapDigitizer]::ERROR::This subdetector " << myDet << " is not implemented";


   assert(m_maxBunch-m_minBunch+1<=10);
   assert(m_minBunch<=0);

   m_vSam.reserve( size ) ;

   for( unsigned int i ( 0 ) ; i != size ; ++i )
   {
     //       m_vSam.emplace_back(CaloGenericDetId( detId.det(), detId.subdetId(), i ) ,
     // 			  m_maxBunch-m_minBunch+1, abs(m_minBunch) );
     m_vSam.emplace_back(TimeSamples(CaloGenericDetId( detId.det(), detId.subdetId(), i )));
   }

   
   edm::LogInfo("TimeDigiInfo") << "[EcalTimeDigitizer]::Subdetector " << m_subDet << "::Reserved size for time digis " << m_vSam.size();

#ifdef ecal_time_debug
   std::cout  << "[EcalTimeDigitizer]::Subdetector " << m_subDet << "::Reserved size for time digis " << m_vSam.size() << std::endl;
#endif
   
}

EcalTimeMapDigitizer::~EcalTimeMapDigitizer()
{
}

void
EcalTimeMapDigitizer::add(const std::vector<PCaloHit> & hits, int bunchCrossing) {
  if(bunchCrossing>=m_minBunch && bunchCrossing<=m_maxBunch ) {

    for(std::vector<PCaloHit>::const_iterator it = hits.begin(), itEnd = hits.end(); it != itEnd; ++it) {
      //here goes the map logic

      if (edm::isNotFinite( (*it).time() ) )
	continue;

      //Just consider only the hits belonging to the specified time layer
      if ((*it).depth()!=100+m_timeLayerId) //modified layerId start from 100
	continue;

      if ((*it).energy()<MIN_ENERGY_THRESHOLD) //apply a minimal cut on the hit energy
	continue;
      
       const DetId detId ( (*it).id() ) ;
    
       double time = (*it).time();

       //time of flight is not corrected here for the vertex position 
       const double jitter ( time - timeOfFlight( detId, m_timeLayerId ) ) ;
       
       TimeSamples& result ( *findSignal( detId ) ) ;

       //here fill the result for the given bunch crossing
       result.average_time[bunchCrossing-m_minBunch]+=jitter*(*it).energy();
       result.tot_energy[bunchCrossing-m_minBunch]+=(*it).energy();
       result.nhits[bunchCrossing-m_minBunch]++;

#ifdef ecal_time_debug
       std::cout << (*it).id()  << "\t" << (*it).depth() << "\t" << jitter << "\t" <<  (*it).energy() << "\t" << result.average_time[bunchCrossing-m_minBunch] << "\t" << result.nhits[bunchCrossing-m_minBunch] << "\t" <<  timeOfFlight( detId, m_timeLayerId ) << std::endl;
#endif
    }
  }
}


EcalTimeMapDigitizer::TimeSamples* 
EcalTimeMapDigitizer::findSignal( const DetId& detId )
{
   const unsigned int di ( CaloGenericDetId( detId ).denseIndex() ) ;
   TimeSamples* result ( vSamAll( di ) ) ;
   if( result->zero() ) m_index.push_back( di ) ;
   return result ;
}



void 
EcalTimeMapDigitizer::setGeometry( const CaloSubdetectorGeometry* geometry )
{
   m_geometry = geometry ;
}


void 
EcalTimeMapDigitizer::blankOutUsedSamples()  // blank out previously used elements
{
   const unsigned int size ( m_index.size() ) ;


   for( unsigned int i ( 0 ) ; i != size ; ++i )
   {
#ifdef ecal_time_debug 
     std::cout << "Zeroing " << m_index[i] << std::endl;
#endif
      vSamAll( m_index[i] )->setZero() ;
   }

   m_index.erase( m_index.begin() ,    // done and make ready to start over
		  m_index.end()    ) ;
}


void
EcalTimeMapDigitizer::finalizeHits()
{
  //Getting the size from the actual hits
   const unsigned int size ( m_index.size() ) ;

   //Here just averaging the MC truth
   for( unsigned int i ( 0 ) ; i != size ; ++i )
   {
#ifdef ecal_time_debug 
     std::cout << "Averaging " << m_index[i] << std::endl;
#endif
      vSamAll( m_index[i] )->calculateAverage() ;
#ifdef ecal_time_debug 
      for ( unsigned int j ( 0 ) ; j !=  vSamAll( m_index[i] )->time_average_capacity; ++j )
	std::cout << j << "\t" <<  vSamAll( m_index[i] )->average_time[j] <<  "\t" <<  vSamAll( m_index[i] )->nhits[j] << "\t" <<  vSamAll( m_index[i] )->tot_energy[j] << std::endl;
#endif
   }
   
   //Noise can be added here

}

void
EcalTimeMapDigitizer::initializeMap()
{
#ifdef ecal_time_debug
  std::cout << "[EcalTimeMapDigitizer]::Zeroing the used samples" << std::endl;
#endif
   blankOutUsedSamples();
}


void 
EcalTimeMapDigitizer::run( EcalTimeDigiCollection& output  )
{
#ifdef ecal_time_debug
  std::cout << "[EcalTimeMapDigitizer]::Finalizing hits and fill output collection" << std::endl;
#endif

  //Do the time averages and add noise if simulated
  finalizeHits();

  //Until we do now simulated noise we can get the size from the index vector
  const unsigned int ssize ( m_index.size() ) ;

  output.reserve( ssize ) ;
  
  for( unsigned int i ( 0 ) ; i != ssize ; ++i )
    {

 #ifdef ecal_time_debug
      std::cout << "----- in digi loop " << i << std::endl;
 #endif

      output.push_back( Digi(vSamAll( m_index[i] )->id) );

      unsigned int nTimeHits=0;
      float timeHits[vSamAll( m_index[i] )->time_average_capacity];
      unsigned int timeBX[vSamAll( m_index[i] )->time_average_capacity];

      for ( unsigned int j ( 0 ) ; j !=  vSamAll( m_index[i] )->time_average_capacity; ++j ) //here sampling on the OOTPU 
	{
	  if (vSamAll( m_index[i] )->nhits[j]>0)
	    {
	      timeHits[nTimeHits]= vSamAll( m_index[i] )->average_time[j];
	      timeBX[nTimeHits++]= m_minBunch+j;
	    }
	}

      output.back().setSize(nTimeHits);

      for (unsigned int j ( 0 ) ; j !=  nTimeHits; ++j ) //filling the !zero hits
	{
	  output.back().setSample(j,timeHits[j]);
	  if (timeBX[j]==0)
	    {
#ifdef ecal_time_debug
	      std::cout << "setting interesting sample " << j << std::endl;
#endif
	    output.back().setSampleOfInterest(j);
	    }
	}
      
 #ifdef ecal_time_debug
      std::cout << "digi " << output.back().id().rawId() << "\t" << output.back().size();
       if (output.back().sampleOfInterest()>0)
 	std::cout <<  "\tBX0 time " << output.back().sample(output.back().sampleOfInterest()) << std::endl;
       else
 	std::cout <<  "\tNo in time hits" << std::endl;
 #endif
    }
#ifdef ecal_time_debug
  std::cout << "[EcalTimeMapDigitizer]::Output collection size " << output.size() << std::endl;
#endif

}


double 
EcalTimeMapDigitizer::timeOfFlight( const DetId& detId , int layer) const 
{
  //not using the layer yet
   const CaloCellGeometry* cellGeometry ( m_geometry->getGeometry( detId ) ) ;
   assert( 0 != cellGeometry ) ;
   GlobalPoint layerPos = (dynamic_cast<const TruncatedPyramid*>(cellGeometry))->getPosition( double(layer)+0.5 ); //depth in mm in the middle of the layer position
   return layerPos.mag()*cm/c_light ;
}


unsigned int
EcalTimeMapDigitizer::samplesSize() const
{
   return m_vSam.size() ;
}

unsigned int
EcalTimeMapDigitizer::samplesSizeAll() const
{
   return m_vSam.size() ;
}

const EcalTimeMapDigitizer::TimeSamples* 
EcalTimeMapDigitizer::operator[]( unsigned int i ) const
{
   return &m_vSam[ i ] ;
}

EcalTimeMapDigitizer::TimeSamples* 
EcalTimeMapDigitizer::operator[]( unsigned int i )
{
   return &m_vSam[ i ] ;
}

EcalTimeMapDigitizer::TimeSamples* 
EcalTimeMapDigitizer::vSam( unsigned int i )
{
   return &m_vSam[ i ] ;
}

EcalTimeMapDigitizer::TimeSamples* 
EcalTimeMapDigitizer::vSamAll( unsigned int i )
{
   return &m_vSam[ i ] ;
}

const EcalTimeMapDigitizer::TimeSamples* 
EcalTimeMapDigitizer::vSamAll( unsigned int i ) const
{
   return &m_vSam[ i ] ;
}

int 
EcalTimeMapDigitizer::minBunch() const 
{
  return m_minBunch ; 
}

int 
EcalTimeMapDigitizer::maxBunch() const 
{
  return m_maxBunch ; 
}

EcalTimeMapDigitizer::VecInd& 
EcalTimeMapDigitizer::index() 
{
  return m_index ; 
}

const EcalTimeMapDigitizer::VecInd& 
EcalTimeMapDigitizer::index() const
{
  return m_index ; 
}

// const EcalTimeMapDigitizer::TimeSamples* 
// EcalTimeMapDigitizer::findDetId( const DetId& detId ) const
// {
//    const unsigned int di ( CaloGenericDetId( detId ).denseIndex() ) ;
//    return vSamAll( di ) ;
// }
