#ifndef ECAL_FENIX_LINEARIZER_H
#define ECAL_FENIX_LINEARIZER_H

#include <DataFormats/EcalDigi/interface/EcalMGPASample.h>
#include "SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixChip.h"
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
   *  Question:
   *  - est-ce que ca correspond a la transformation en Et?
   *           -----> oui (avec possibilite de disable) 
   *  - faut-il chercher qqpart des constantes de calibration?
   *           -----> oui (une methode set est envisageable au debut remplir les valeurs a la main puis plus tard aller les chercher dans une database, ou eventuellement dans un .orcarc)
   *           -----> de meme pour le theta et pour les autres valeurs needed.
   *   
   *  
   *  Attention: 
   *  La multplication par un float est equivalente a la multiplication par un int suivie de n decalage (division par 2^n)
   *   
   *  Cette classe doit etre configuree cristal par cristal avec 9 parametres entiers chacun.
   */

//  class EcalFenixLinearizer : public EcalVLinearizer {
  class EcalFenixLinearizer  {


  private:
    /** maximum number of samples per frame */
    //temporary, waiting for changes in EBDataFrame
    const EcalTPParameters * ecaltpp_ ;
    bool off;
    int uncorrectedSample_;
    int gainID_;
    int base_;
    int mult_;
    int shift_;
    int strip_;
    std::vector<unsigned int> params_ ;

    int setInput(const EcalMGPASample &RawSam) ;
    int process() ;


  public:
    EcalFenixLinearizer(const EcalTPParameters *);
    virtual ~EcalFenixLinearizer();


    //    virtual EBDataFrame process(EBDataFrame&) {EBDataFrame df;return df;} //for base class
    //    template <class T> void  process(const T &, T *out); 
    template <class T>  void process(const T &, std::vector<int>&); 
    void setParameters(int SM, int towNum, int stripNum,int XtalNumberInStrip) ;
  };

// template <class T> std::vector<int> EcalFenixLinearizer::process(const T&df)
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
//  std::vector<int> output_percry;
  // cout<<"dfsize input lin= "<<df.size()<<endl;
  for (int i=0;i<df.size();i++) {
    setInput(df[i]);
    output_percry.push_back(process());
  }

  return;
//  return(output_percry);
}

#endif
