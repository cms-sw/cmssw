#ifndef ECAL_FENIX_FGVB_EB_H
#define ECAL_FENIX_FGVB_EB_H

#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalVFgvb.h>
#include <vector>

class DBInterface ;

// global type definitions for header defined by Tag entries in ArgoUML
// Result: typedef <typedef_global_header> <tag_value>;


/** 
    \class EcalFenixFgvbEB
    \brief calculation of Fgvb for Fenix Tcp, format barrel
    *  calculates fgvb for the barrel
    *  
    *  
    *  input: 2X12 bits ( 12 bits Ettot + 12 bits maxof2)
    *  output: 1 bit 
    *  
    *  
    *  makes comparisons between maxof2 and 2 fractions of Ettot and  uses this comparison to decide ---> needs to get some values from outside
    */
class EcalFenixFgvbEB : public EcalVFgvb {

 private:
    DBInterface * db_ ;
    std::vector<unsigned int> params_ ;

 public:
  EcalFenixFgvbEB(DBInterface * db) ;
  virtual ~EcalFenixFgvbEB();
  int process() {return 0;} //FIXME: find better base methods
  void setParameters(int SM, int towNum);

  std::vector<int> process( std::vector<int> add_out, std::vector<int> maxof2_out);
};


#endif
