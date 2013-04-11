
#include "RecoVertex/TertiaryTracksVertexFinder/interface/Flight2DSvFilter.h"

#include "RecoVertex/TertiaryTracksVertexFinder/interface/DistanceOfVertices2D.h"


Flight2DSvFilter::Flight2DSvFilter () {
  // init. data members
  maxFlightDist2D                = 2.5  ;  // cm             
  minFlightDist2D                = 0.01 ;  // cm           
  minFlightDistSignificance2D    = 3.0  ;
  minTracksInVertex              = 2    ;
  usePrimaryVertex = false;
}


Flight2DSvFilter::Flight2DSvFilter ( double maxDist2D=2.5 , double minDist2D=0.01 , double  minSign2D=3.0 , int  minTracks=2 ) {
  // init. data members
  maxFlightDist2D                = maxDist2D ;  // cm             
  minFlightDist2D                = minDist2D ;  // cm           
  minFlightDistSignificance2D    = minSign2D ;
  minTracksInVertex              = minTracks ;
  usePrimaryVertex = false; 
}

Flight2DSvFilter::Flight2DSvFilter ( const TransientVertex &primaryVertex , double maxDist2D=2.5 , double minDist2D=0.01 , double  minSign2D=3.0 , int  minTracks=2 ) {
  // init. data members
  maxFlightDist2D                = maxDist2D ;  // cm             
  minFlightDist2D                = minDist2D ;  // cm           
  minFlightDistSignificance2D    = minSign2D ;
  minTracksInVertex              = minTracks ;
  if(primaryVertex.isValid()) {
    PrimaryVertex = primaryVertex;  
    usePrimaryVertex = true;
  }
  else
    usePrimaryVertex = false;
}

bool Flight2DSvFilter::operator () ( const TransientVertex & VertexToFilter ) const {
  bool passed = true ;
 
  double flightDist=0;       
  double SignificanceForFlightDist=0; 
  if( usePrimaryVertex ) { 
    if (debug) std::cout <<"[Flight2DSvFilter] using PV\n";
    DistanceOfVertices2D MyDistanceOfVertices2D;  
    Measurement1D DistanceMeasurement = MyDistanceOfVertices2D.distance( PrimaryVertex , VertexToFilter ); 
    flightDist = DistanceMeasurement.value();  
    SignificanceForFlightDist = DistanceMeasurement.significance();  
  }
  else {   
    if (debug) std::cout <<"[Flight2DSvFilter] NOT using PV\n";
    flightDist = DistanceToBeamLine( VertexToFilter ); 
    SignificanceForFlightDist = DistanceSignificance2DToBeamLine( VertexToFilter );
  }

  int NrTracksInVertex = VertexToFilter.originalTracks().size();        

  if (debug) std::cout <<"[Flight2DSvFilter] ntracks,flightdist,flightdistsign: "<<NrTracksInVertex<<","<<flightDist<<","<<SignificanceForFlightDist<<std::endl;
            
  // check condidtions 1 by 1 
  
  if ( flightDist                > maxFlightDist2D )              passed = false ;
  if ( flightDist                < minFlightDist2D )              passed = false ;   
  if ( SignificanceForFlightDist < minFlightDistSignificance2D )  passed = false ;
  if ( NrTracksInVertex < minTracksInVertex )                     passed = false ;

  if (debug && !passed) std::cout <<"[Flight2DSvFilter] failed!\n";

  return passed ;
}


double Flight2DSvFilter::DistanceToBeamLine( const TransientVertex & theTransientVertex) const {
  double Distance = sqrt (  theTransientVertex.position().x() * theTransientVertex.position().x() 
		      + theTransientVertex.position().y() * theTransientVertex.position().y() );
  return Distance;
}


double Flight2DSvFilter::DistanceSignificance2DToBeamLine( const TransientVertex & theTransientVertex ) const {
  
  double sigmaX = 0.0015 ;  double covXp2 = sigmaX * sigmaX;   // error on beam 
  double sigmaY = 0.0015 ;  double covYp2 = sigmaY * sigmaY;   // error on beam 
  
  double xS = ( theTransientVertex.position().x() );  double xS2 = xS*xS ;
  double yS = ( theTransientVertex.position().y() );  double yS2 = yS*yS ;     
  double covXs2 = theTransientVertex.positionError().cxx();
  double covYs2 = theTransientVertex.positionError().cyy();
  double covXsYs = theTransientVertex.positionError().cyx();

  double DistanceToBeam = DistanceToBeamLine( theTransientVertex ); 

  double sigmaDistance = 0.0; 
   if(DistanceToBeam > 0.0) sigmaDistance = ( 1 / (DistanceToBeam * DistanceToBeam) ) *
			      ( xS2 * covXs2          + 
				xS2 * covXp2          + 
				yS2 * covYs2          + 
				yS2 * covYp2          + 
				2 * xS * yS * covXsYs );
  
   double SignificanceForSv=0.0; 
   if( DistanceToBeam - minFlightDist2D >0 && sigmaDistance > 0.0) SignificanceForSv = (DistanceToBeam) / sqrt(sigmaDistance);
     
   return(SignificanceForSv);
}


void Flight2DSvFilter::setPrimaryVertex( const TransientVertex & primaryVertex ) {

  PrimaryVertex = primaryVertex;  
  usePrimaryVertex = true;

}



  



