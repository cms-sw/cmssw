///////////////////////////////////////////////////////////////////////////////
// File: HFFibre.h
// Description: Calculates attenuation length
///////////////////////////////////////////////////////////////////////////////
#ifndef HFFibre_h
#define HFFibre_h 1

#include "DetectorDescription/Core/interface/DDsvalues.h"

#include <vector>
#include <string>

class DDCompactView;    

class HFFibre {
  
public:
  
  //Constructor and Destructor
  HFFibre(int iv, const DDCompactView & cpv);
  ~HFFibre();

  double              attLength(double lambda);

protected:

  std::vector<double> getDDDArray(const std::string&, 
				  const DDsvalues_type&, int&);

private:

  int                         verbosity;
  std::vector<double>         attL;
  int                         nBinAtt;
  double                      lambLim[2];

};
#endif
