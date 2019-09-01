#ifndef SimG4CMS_PMTResponse_h
#define SimG4CMS_PMTResponse_h
///
///  \class PMTResponse
///
///   Encodes the PMT response function
///

class PMTResponse {
public:
  /// Return efficiency for given photon wavelength (in nm)
  static double getEfficiency(const double &waveLengthNm);
};

#endif  // DreamSD_h
