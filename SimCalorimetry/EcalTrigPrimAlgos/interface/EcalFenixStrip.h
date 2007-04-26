#ifndef ECAL_FENIXSTRIP_H
#define ECAL_FENIXSTRIP_H

#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixChip.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixLinearizer.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixAmplitudeFilter.h>
//#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixStripFormatEB.h>
//#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixStripFormatEE.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixPeakFinder.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixEtStrip.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixLinearizer.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixEtStrip.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixStripFgvbEE.h>

#include <DataFormats/EcalDigi/interface/EBDataFrame.h>
#include <DataFormats/EcalDigi/interface/EEDataFrame.h>


class TTree;
class EBDataFrame;
class EcalTriggerPrimitiveSample;
class EcalTPParameters;
class EcalFenixStripFgvbEE;
class EcalFenixStripFormatEB;
class EcalFenixStripFormatEE;
class EcalElectronicsMapping;

  /** 
      \class EcalFenixStrip
      \brief class representing the Fenix chip, format strip
  */
// ENDCAP::MODIF  : enlever l'heritage EcalFenixChip et EcalVFormatter right ??
//  class EcalFenixStrip : public EcalFenixChip {

class EcalFenixStrip {
  private:
    const EcalElectronicsMapping* theMapping_;

    bool debug_;
    enum {numberOfCrystalsInStrip = 5};//FIXME

    
    EcalFenixLinearizer *linearizer_[nCrystalsPerStrip_];

    EcalFenixAmplitudeFilter *amplitude_filter_; 

    EcalFenixPeakFinder *peak_finder_; 
    
    // ENDCAP::MODIF  ????????????? formatter_
    EcalFenixStripFormatEB *fenixFormatterEB_;   
    
    EcalFenixStripFormatEE *fenixFormatterEE_;
    

    EcalFenixEtStrip *adder_;
    
    //ENDCAP:MODIF
    EcalFenixStripFgvbEE *fgvbEE_;


  public:

    // constructor, destructor
    EcalFenixStrip(TTree *tree, const EcalTPParameters * ecaltpp, const EcalElectronicsMapping* theMapping,bool debug);
    virtual ~EcalFenixStrip() ;

    // main methods
    // process method is splitted in 2 parts:
    //   the first one is templated, the same except input
    //   the second part is slightly different for barrel/endcap
    template <class T> std::vector<int> process(std::vector<const T *> &, int stripnr,int townr,int smnr);
    //  template <class T> std::vector<int> process_part1(std::vector<const T *> &,std::vector<const T *> &, std::vector<int> &, int smnr, int stripnr,int townr);//FIXME: many transmissions per value...
    std::vector<int> process_part2_barrel(std::vector<int> &, std::vector<int> &, int smnr, int stripnr,int townr);
    std::vector<int> process_part2_endcap(std::vector<const EEDataFrame *> &,std::vector<int> &, std::vector<int> &, int smnr, int stripnr,int townr); // FIXME: to be filled


    //template <class T> int getCrystalNumberInStrip (const T*) ;
    //ENDCAP: Missing geometry. Template int getCrystalNumberInStrip  for later
    int getCrystalNumberInStrip(const EBDataFrame *frame,int crystalIndexinStrip); //2nd argument dummy forEB 
    int getCrystalNumberInStrip(const EEDataFrame *frame,int crystalIndexinStrip); //2nd argument dummy forEB 
       

    // getters for the algorithms  ;

    EcalFenixLinearizer *getLinearizer (int i) const { return linearizer_[i];}
    EcalFenixEtStrip *getAdder() const { return  adder_;}
    EcalFenixAmplitudeFilter *getFilter() const { return amplitude_filter_;}
    EcalFenixPeakFinder *getPeakFinder() const { return peak_finder_;}
    //ENDCAP:MODIF
    //EcalFenixStripFormat *getFormatter() const { return dynamic_cast<EcalFenixStripFormat *> (formatter_);}
    
    EcalFenixStripFormatEB *getFormatterEB() const { return fenixFormatterEB_;}
    EcalFenixStripFormatEE *getFormatterEE() const { return fenixFormatterEE_;}
    
    EcalFenixStripFgvbEE *getFGVB()      const { return fgvbEE_;}

// ========================= implementations ==============================================================
std::vector<int> process(std::vector<const EBDataFrame *> &samples, int stripnr,int townr, int &smnr){
  //ENDCAP:MODIF I need the  linearizer output for endcap Fgvb 
  //std::vector<T * > lin_out;
  std::vector<const EBDataFrame * > lin_out;
  std::vector<int> filt_out;
  
  //  smnr=samples[0]->id().ism();  //FIXME: not efficient.Perhaps transmit directly EcalTrigTowerDetId to DB in the future? 
  //ENCAP:MODIF : added lin_out parameter . needed for EE in process_2
  std::vector<int> res=process_part1(samples,filt_out, smnr,stripnr,townr);//templated part
  for (unsigned int i=0;i<lin_out.size();++i) delete lin_out[i];
  return process_part2_barrel(res,filt_out,smnr,stripnr,townr);//part different for barrel/endcap
}

std::vector<int>  process(std::vector<const EEDataFrame *> &samples, int stripnr,int townr, int &sectorNr){
  //ENDCAP:MODIF  
  //std::vector<T * > lin_out;
  std::vector<const EEDataFrame * > lin_out;
  std::vector<int> filt_out;
  
  //ENCAP:MODIF : added lin_out parameter . needed for EE in process_2
  std::vector<int> res=process_part1(samples,filt_out,sectorNr,stripnr,townr); //templated part
  std::vector<int> format_out= process_part2_endcap(lin_out,res,filt_out,sectorNr,stripnr,townr);//FinegrainVB  avant l'entree format
  //ENDCAP:MODIF
  for (unsigned int i=0;i<lin_out.size();++i) delete lin_out[i];
  return format_out; 
}

//template <class T> std::vector<int>  process_part1(std::vector<const T *> & df,std::vector<const T *> & lin_out,std::vector<int> & filt_out, int smnr, int stripnr, int townr)
 template <class T> std::vector<int>  process_part1(std::vector<const T *> & df,std::vector<int> & filt_out, int smnr, int stripnr, int townr)
   {
  
     if(debug_)  std::cout<<"EcalFenixStrip input is a vector of size: "<<df.size()<< std::endl;
     std::vector<std::vector<int> > lin_out;

     //loop over crystals
     for (unsigned int cryst=0;cryst<df.size();cryst++) {
       if(debug_){
	 std::cout<<std::endl;
	 std::cout <<"cryst= "<<cryst<<" EBDataFrame/EEDataFrame is: "<<std::endl; 
	 for ( int i = 0; i<df[cryst]->size();i++){
       	   std::cout <<" "<<std::dec<<(*df[cryst])[i].adc();
	 }
	 std::cout<<std::endl;
       }
       // call linearizer
       int crystalNumberInStrip=getCrystalNumberInStrip(df[cryst],cryst);
       this->getLinearizer(cryst)->setParameters(smnr, townr, stripnr, crystalNumberInStrip) ; 
       std::vector<int> linout_percry;
       this->getLinearizer(cryst)->process(*(df[cryst]),linout_percry);
       lin_out.push_back(linout_percry);
     }

     if(debug_){
       std::cout<< "output of linearizer is a vector of size: "<<std::dec<<lin_out.size()<<std::endl; 
       for (unsigned int ix=0;ix<lin_out.size();ix++){
	 std::cout<< "cryst: "<<ix<<"  value : "<<std::dec<<std::endl;
	 std::cout<<" lin_out[ix].size()= "<<std::dec<<lin_out[ix].size()<<std::endl;
	 for (unsigned int i =0; i<lin_out[ix].size();i++){
	   std::cout <<" "<<std::dec<<(lin_out[ix])[i];
	 }
	 std::cout<<std::endl;
       }
    
       std::cout<<std::endl;
     }
  
     // call adder
     std::vector<int> add_out;
     add_out = this->getAdder()->process(lin_out);
  
     if(debug_){
       std::cout<< "output of adder is a vector of size: "<<std::dec<<add_out.size()<<std::endl; 
       for (unsigned int ix=0;ix<add_out.size();ix++){
	 std::cout<< "cryst: "<<ix<<"  value : "<<std::dec<<add_out[ix]<<std::endl;
       }
       std::cout<<std::endl;
     }
 
     //ENDCAP:MODIF -I still need lin_out for EE
     //for (unsigned int i=0;i<lin_out.size();++i) delete lin_out[i];

 
     // call amplitudefilter
     this->getFilter()->setParameters(smnr, townr,stripnr) ; 
     filt_out= this->getFilter()->process(add_out); 

     if(debug_){
       std::cout<< "output of filter is a vector of size: "<<std::dec<<filt_out.size()<<std::endl; 
       for (unsigned int ix=0;ix<filt_out.size();ix++){
	 std::cout<< "cryst: "<<ix<<"  value : "<<std::dec<<filt_out[ix]<<std::endl;
       }
       std::cout<<std::endl;
     }

     // call peakfinder
     std::vector<int> peak_out;
     peak_out =this->getPeakFinder()->process(filt_out);
     if(debug_){
       std::cout<< "output of peakfinder is a vector of size: "<<peak_out.size()<<std::endl; 
       for (unsigned int ix=0;ix<peak_out.size();ix++){
	 std::cout<< "cryst: "<<ix<<"  value : "<<peak_out[ix]<<std::endl;
       }
       std::cout<<std::endl;
     }

  
     //FIXME for speed !!!!!!!!!!
     return peak_out;
   }

};
#endif

