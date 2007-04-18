#ifndef ECAL_FENIX_LINEARIZER_H
#define ECAL_FENIX_LINEARIZER_H

#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalVLinearizer.h>
#include <DataFormats/EcalDigi/interface/EBDataFrame.h>
#include <vector> 

class DBInterface ;

// global type definitions for header defined by Tag entries in ArgoUML
// Result: typedef <typedef_global_header> <tag_value>;


  /** 
   \class EcalFenixLinearizer
   \brief Linearisation for Fenix strip
   *  input: 16 bits  corresponding to input EBDataFrame
   *  output: 18 bits 
   *  
   */

  class EcalFenixLinearizer : public EcalVLinearizer {


  private:
    /** maximum number of samples per frame */
    //temporary, waiting for changes in EBDataFrame
    enum { SIZEMAX = 10}; 
    DBInterface * db_ ;
    bool off;
    int uncorrectedSample_;
    int gainID_;
    int base_;
    int mult_;
    int shift_;
    int strip_;
    std::vector<unsigned int> params_ ;

    int setInput(EcalMGPASample RawSam) ;
    int process() ;


  public:
    //    EcalFenixLinearizer(EcalBarrelTopology *);
    EcalFenixLinearizer(DBInterface * db);
    virtual ~EcalFenixLinearizer();


    virtual EBDataFrame process(EBDataFrame&) {EBDataFrame df;return df;} //for base class

    std::vector<int>  process(const EBDataFrame &); 
    void setParameters(int SM, int towNum, int stripNum,int XtalNumberInStrip) ;


  };


#endif
