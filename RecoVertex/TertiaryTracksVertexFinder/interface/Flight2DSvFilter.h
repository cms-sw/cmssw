#ifndef FLIGHT2DSVFILTER_H  
#define FLIGHT2DSVFILTER_H 

#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"

// class to filter individual secondary vertex candidates

class Flight2DSvFilter {

public:

  Flight2DSvFilter () ;
  Flight2DSvFilter ( double  , double  , double  , int ) ; 
  Flight2DSvFilter ( TransientVertex , double  , double  , double  , int ) ; 
  ~Flight2DSvFilter () {}

  bool operator () ( const TransientVertex & ) const ;

  double DistanceToBeamLine( const TransientVertex & ) const ;
  double DistanceSignificance2DToBeamLine( const TransientVertex & ) const ;

  inline TransientVertex getPrimaryVertex() const { return PrimaryVertex; } 

  void setPrimaryVertex( const TransientVertex & ) ;
  void useNotPV() { usePrimaryVertex = false; }

private:

  // data members are the quantities to cut on for selection
  double maxFlightDist2D ;                 // maximum flight distance in 2 dim. in cm (reject material interactions)
  double minFlightDist2D ;                 // minimum flight distance in 2 dim. in cm  
  double minFlightDistSignificance2D ;     // the name says it
  int    minTracksInVertex;                // the name says it 
  TransientVertex PrimaryVertex;
  bool usePrimaryVertex;
  // now a data member to use the usual Filter<T> interface

  static const bool debug = true; 
  
};
#endif

