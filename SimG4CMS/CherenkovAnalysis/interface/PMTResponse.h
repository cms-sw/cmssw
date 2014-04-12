#ifndef SimG4CMS_PMTResponse_h
#define SimG4CMS_PMTResponse_h
///
///  \class PMTResponse
///  
///   Encodes the PMT response function
///

#include "FWCore/MessageLogger/interface/MessageLogger.h"

class PMTResponse {

public:    

  /// Default constructor
  PMTResponse() {}

  /// Return efficiency for given photon wavelength (in nm)
  static const double  getEfficiency( const double& waveLengthNm );

private:    

};

#endif // DreamSD_h
