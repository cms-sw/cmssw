#ifndef ECAL_FENIX_LINEARIZER_H
#define ECAL_FENIX_LINEARIZER_H

#include <DataFormats/EcalDigi/interface/EcalMGPASample.h>
#include <vector> 
#include <iostream>

class EcalTPParameters;

// global type definitions for header defined by Tag entries in ArgoUML
// Result: typedef <typedef_global_header> <tag_value>;


  /** 
   \class EcalFenixLinearizer
   \brief Linearisation for Fenix strip
   *  input: 16 bits  corresponding to input EBDataFrame
   *  output: 18 bits 
   *  
   */

  class EcalFenixLinearizer  {


  private:
    const EcalTPParameters * ecaltpp_ ;
    bool famos_;
    int uncorrectedSample_;
    int gainID_;
    int base_;
    int mult_;
    int shift_;
    int strip_;
    std::vector<unsigned int> const * params_ ;

    int setInput(const EcalMGPASample &RawSam) ;
    int process() ;


  public:
    EcalFenixLinearizer(const EcalTPParameters *,bool famos);
    virtual ~EcalFenixLinearizer();

    template <class T>  void process(const T &, std::vector<int>&); 
    void setParameters(int SM, int towNum, int stripNum,int XtalNumberInStrip) ;
  };

    template <class T> void EcalFenixLinearizer::process(const T&df, std::vector<int> & output_percry)
{

//We know a tower numbering is:
// S1 S2 S3 S4 S5
//
// 4  5  14 15 24
// 3  6  13 16 23
// 2  7  12 17 22
// 1  8  11 18 21
// 0  9  10 19 20
  for (int i=0;i<df.size();i++) {
    setInput(df[i]);
    output_percry[i]=process();
  }

  return;
}

#endif
