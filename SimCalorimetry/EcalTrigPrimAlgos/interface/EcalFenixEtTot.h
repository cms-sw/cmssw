#ifndef ECAL_FENIX_ET_TOT_H
#define ECAL_FENIX_ET_TOT_H

#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalVAdder.h>
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include <vector>

namespace tpg {

// global type definitions for header defined by Tag entries in ArgoUML
// Result: typedef <typedef_global_header> <tag_value>;


  /** 
   \class EcalFenixEtTot

   \brief class for calculation of Et for Fenix tcp
   *  calculates the sum
   *  
   *  input: 5x 12 bits (12 premiers bits du output bu by passlin ou alors on prend les 13 et dans la classe on selectionne les 12 premiers...a voir ) 
   *  
   *  output :12 bits
   *  
   *  in case of overflow, result is set to (12**2)-1
   */
class EcalFenixEtTot : public EcalVAdder {


 public:
  EcalFenixEtTot();
  virtual ~EcalFenixEtTot();
  virtual std::vector<int> process(std::vector<EBDataFrame> ); //UB FIXME: used??
    
    std::vector<int> process(std::vector<std::vector <int> >);

  private:
    enum { SIZEMAX = 10}; 
  };

} /* End of namespace tpg */

#endif
