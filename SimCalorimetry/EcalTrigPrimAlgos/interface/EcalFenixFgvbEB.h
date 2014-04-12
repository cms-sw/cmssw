#ifndef ECAL_FENIX_FGVB_EB_H
#define ECAL_FENIX_FGVB_EB_H

#include <vector>
#include <stdint.h>

class EcalTPGFineGrainEBGroup;
class EcalTPGFineGrainEBIdMap;

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
class EcalFenixFgvbEB {

 private:
  uint32_t ETlow_,  EThigh_,  Ratlow_,  Rathigh_, lut_;
  //    std::vector<int> adder_out_;
  //    std::vector<int> maxOf2_out_;
  //    std::vector<int> fgvb_out_;
    std::vector<int> add_out_8_;


 public:
    EcalFenixFgvbEB(int maxNrSamples) ;
    virtual ~EcalFenixFgvbEB();
    void setParameters(uint32_t towid,const EcalTPGFineGrainEBGroup *ecaltpgFgEBGroup,const EcalTPGFineGrainEBIdMap *ecaltpgFineGrainEB );
    
    void process( std::vector<int> &add_out, std::vector<int> &maxof2_out, std::vector<int> & output);
};


#endif
