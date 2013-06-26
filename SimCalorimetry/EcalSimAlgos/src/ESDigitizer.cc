#include "SimCalorimetry/EcalSimAlgos/interface/ESDigitizer.h"
#include "SimCalorimetry/EcalSimAlgos/interface/ESElectronicsSimFast.h"
#include "SimCalorimetry/EcalSimAlgos/interface/ESHitResponse.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "CLHEP/Random/RandGeneral.h"
#include "CLHEP/Random/RandPoissonQ.h"
#include "CLHEP/Random/RandFlat.h"

#include <gsl/gsl_sf_erf.h>
#include <gsl/gsl_sf_result.h>

ESDigitizer::ESDigitizer( EcalHitResponse*      hitResponse    , 
			  ESElectronicsSimFast* electronicsSim ,
			  bool                  addNoise         ) :
   EcalTDigitizer< ESDigitizerTraits >( hitResponse, electronicsSim, addNoise ) ,
   m_detIds      ( 0              ) ,
   m_engine      ( 0              ) ,
   m_ranGeneral  ( 0              ) ,
   m_ranPois     ( 0              ) ,
   m_ranFlat     ( 0              ) ,
   m_ESGain      ( 0              ) ,
   m_histoBin    ( 0              ) ,
   m_histoInf    ( 0              ) ,
   m_histoWid    ( 0              ) ,
   m_meanNoisy   ( 0              ) ,
   m_trip        (                )
{
   m_trip.reserve( 2500 ) ; 

   edm::Service<edm::RandomNumberGenerator> rng ;
   if( !rng.isAvailable() )
   {
      throw cms::Exception( "Configuration" )
	 << "ESDigitizer requires the RandomNumberGeneratorService\n"
	 "which is not present in the configuration file.  You must add the service\n"
	 "in the configuration file or remove the modules that require it.";
   }
   m_engine = &rng->getEngine() ;
}

ESDigitizer::~ESDigitizer() 
{
   delete m_ranGeneral ;
   delete m_ranPois    ;
   delete m_ranFlat    ;
}

/// tell the digitizer which cells exist; cannot change during a run
void 
ESDigitizer::setDetIds( const std::vector<DetId>& detIds )
{
   assert( 0       == m_detIds ||
	   &detIds == m_detIds    ) ; // sanity check; don't allow to change midstream
   m_detIds = &detIds ;
}

void 
ESDigitizer::setGain( const int gain ) 
{
   if( 0 != m_ESGain )
   {
      assert( gain == m_ESGain ) ; // only allow one value
   }
   else
   {
      assert( 0 != m_detIds &&
	      0 != m_detIds->size() ) ; // detIds must already be set as we need size

      assert( 1 == gain ||
	      2 == gain    ) ; // legal values
      
      m_ESGain = gain ;
      
      if( addNoise() ) 
      {
	 assert( 0 != m_engine ) ; // sanity check

	 double zsThresh ( 0. ) ;
	 std::string refFile ;

	 if( 1 == m_ESGain ) 
	 {
	    zsThresh = 3 ;
	    refFile = "SimCalorimetry/EcalSimProducers/data/esRefHistosFile_LG.txt";
	 }
	 else
	 {
	    zsThresh = 4 ;
	    refFile = "SimCalorimetry/EcalSimProducers/data/esRefHistosFile_HG.txt";
	 }

	 gsl_sf_result result ;
	 int status  = gsl_sf_erf_Q_e( zsThresh, &result ) ;
	 if( status != 0 ) std::cerr << "ESDigitizer::could not compute gaussian tail probability for the threshold chosen" << std::endl ;

	 const double probabilityLeft ( result.val ) ;
	 m_meanNoisy = probabilityLeft * m_detIds->size() ;

	 m_ranPois = new CLHEP::RandPoissonQ( *m_engine, m_meanNoisy      ) ;
	 m_ranFlat = new CLHEP::RandFlat(     *m_engine, m_detIds->size() ) ;

	 std::ifstream histofile ( edm::FileInPath( refFile ).fullPath().c_str() ) ;
	 if( !histofile.good() )
	 { 
	    throw edm::Exception(edm::errors::InvalidReference,"NullPointer")
	       << "Reference histos file not opened" ;
	 }
	 else
	 {
	    // number of bins
	    char buffer[200] ;
	    int thisLine = 0 ;
	    while( 0 == thisLine ) 
	    {
	       histofile.getline( buffer, 200 ) ;
	       if( !strstr(buffer,"#")  && 
		   !(strspn(buffer," ") == strlen(buffer) ) )
	       {	
		  float histoBin ; 
		  sscanf( buffer, "%f" , &histoBin ) ; 
		  m_histoBin = (double) histoBin ;
		  ++thisLine ;
	       }
	    }
	    const uint32_t histoBin1 ( (int) m_histoBin    ) ;
	    const uint32_t histoBin2 ( histoBin1*histoBin1 ) ;

	    double t_histoSup ( 0 ) ;

	    std::vector<double> t_refHistos ;
	    t_refHistos.reserve( 2500 ) ;

	    int thisBin = -2 ;
	    while( !( histofile.eof() ) )
	    {
	       histofile.getline( buffer, 200 ) ;
	       if( !strstr( buffer, "#" ) &&
		   !( strspn( buffer, " " ) == strlen( buffer ) ) )
	       {
		  if( -2 == thisBin )
		  {
		     float histoInf ;
		     sscanf( buffer, "%f" , &histoInf ) ;
		     m_histoInf = (double) histoInf ;
		  }
		  if( -1 == thisBin  )
		  {
		     float histoSup ;
		     sscanf( buffer, "%f" , &histoSup ) ;
		     t_histoSup = (double) histoSup ;
		  }
		  if( 0 <= thisBin )
		  { 
		     float refBin ; 
		     sscanf( buffer, "%f", &refBin ) ;
		     if( 0.5 < refBin ) 
		     {
			t_refHistos.push_back( (double) refBin ) ;
			const uint32_t i2 ( thisBin/histoBin2 ) ;
			const uint32_t off ( i2*histoBin2 ) ;
			const uint32_t i1 ( ( thisBin - off )/histoBin1 ) ;
			const uint32_t i0 ( thisBin - off - i1*histoBin1 ) ;
			m_trip.emplace_back(i0, i1, i2) ;
		     }
		  }
		  ++thisBin ;
	       }
	    }
	    m_histoWid = ( t_histoSup - m_histoInf )/m_histoBin ;

	    m_histoInf -= 1000. ;

	    // creating the reference distribution to extract random numbers
	    m_ranGeneral = new CLHEP::RandGeneral( *m_engine            ,
						   &t_refHistos.front() ,
						   t_refHistos.size()   ,
						   0             ) ;
	    histofile.close();
	 }
      }
   }
}

/// turns hits into digis
void 
ESDigitizer::run( ESDigiCollection&        output   )
{
    assert( 0 != m_detIds         &&
	    0 != m_detIds->size() &&
	    ( !addNoise()         ||
	      ( 0 != m_engine     &&
		0 != m_ranPois    &&
		0 != m_ranFlat    &&
		0 != m_ranGeneral        ) ) ) ; // sanity check

    // reserve space for how many digis we expect, with some cushion
    output.reserve( 2*( (int) m_meanNoisy ) + hitResponse()->samplesSize() ) ;

    EcalTDigitizer< ESDigitizerTraits >::run( output ) ;

    // random generation of channel above threshold
    std::vector<DetId> abThreshCh ;
    if( addNoise() ) createNoisyList( abThreshCh ) ;

    // first make a raw digi for every cell where we have noise
    for( std::vector<DetId>::const_iterator idItr ( abThreshCh.begin() ) ;
	 idItr != abThreshCh.end(); ++idItr ) 
    {
       if( hitResponse()->findDetId( *idItr )->zero() ) // only if no true hit!
       {
	  ESHitResponse::ESSamples analogSignal ( *idItr, 3 ) ; // space for the noise hit
	  uint32_t myBin ( (uint32_t) m_trip.size()*m_ranGeneral->fire() ) ;
	  if( myBin == m_trip.size() ) --myBin ; // guard against roundup
	  assert( myBin < m_trip.size() ) ;
	  const Triplet& trip ( m_trip[ myBin ] ) ;
	  analogSignal[ 0 ] = m_histoInf + m_histoWid*trip.first  ;
	  analogSignal[ 1 ] = m_histoInf + m_histoWid*trip.second ;
	  analogSignal[ 2 ] = m_histoInf + m_histoWid*trip.third  ;
	  ESDataFrame digi( *idItr ) ;
	  const_cast<ESElectronicsSimFast*>(elecSim())->
	     analogToDigital( analogSignal ,
			      digi         ,
			      true           ) ;	
	  output.push_back( std::move(digi) ) ;  
       }
    }
}

// preparing the list of channels where the noise has to be generated
void 
ESDigitizer::createNoisyList( std::vector<DetId>& abThreshCh )
{
   const unsigned int nChan ( m_ranPois->fire() ) ;
   abThreshCh.reserve( nChan ) ;

   for( unsigned int i ( 0 ) ; i != nChan ; ++i )
   {
      std::vector<DetId>::const_iterator idItr ( abThreshCh.end() ) ;
      uint32_t iChan ( 0 ) ;
      DetId id ;
      do 
      {
	 iChan = (uint32_t) m_ranFlat->fire() ;
	 if( iChan == m_detIds->size() ) --iChan ; //protect against roundup at end
	 assert( m_detIds->size() > iChan ) ;      // sanity check
	 id = (*m_detIds)[ iChan ] ;
	 idItr = find( abThreshCh.begin() ,
		       abThreshCh.end()   ,
		       id                  ) ;
      }
      while( idItr != abThreshCh.end() ) ;

      abThreshCh.push_back( std::move(id) ) ;
   }
}
