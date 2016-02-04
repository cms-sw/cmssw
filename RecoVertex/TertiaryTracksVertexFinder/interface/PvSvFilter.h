
#ifndef PVSVFILTER_H  
#define PVSVFILTER_H 

class TransientVertex;

// class to filter secondary vertices if too many tracks from
// reconstructed primary vertex

class PvSvFilter {

public:

  PvSvFilter ( double , TransientVertex & ) ; 

  ~PvSvFilter ( ) { } ;

  bool operator () ( const TransientVertex & ) const ;
  
private:

  double maxFractionPv ; // max. allowd fraction of tracks in reconstructed PV

  TransientVertex * thePrimaryVertex ;

  static const bool debug = false;
  
};
#endif

