#ifndef ECAL_FENIXSTRIP_H
#define ECAL_FENIXSTRIP_H

#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixLinearizer.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixAmplitudeFilter.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixPeakFinder.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixEtStrip.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixLinearizer.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixEtStrip.h>
#include <SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixStripFgvbEE.h>

#include <DataFormats/EcalDigi/interface/EBDataFrame.h>
#include <DataFormats/EcalDigi/interface/EEDataFrame.h>

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
class EcalFenixStrip {
 private:
  const EcalElectronicsMapping* theMapping_;

  bool debug_;
  bool famos_;
    
  std::vector <EcalFenixLinearizer *> linearizer_; 

  EcalFenixAmplitudeFilter *amplitude_filter_; 

  EcalFenixPeakFinder *peak_finder_; 
    
  EcalFenixStripFormatEB *fenixFormatterEB_;   
    
  EcalFenixStripFormatEE *fenixFormatterEE_;
    

  EcalFenixEtStrip *adder_;
    
  //ENDCAP:MODIF
  EcalFenixStripFgvbEE *fgvbEE_;

  // data formats for each event
  std::vector<std::vector<int> > lin_out_;
  std::vector<int> add_out_;
  std::vector<int> filt_out_;
  std::vector<int> peak_out_;
  std::vector<int> format_out_;
  std::vector<int> fgvb_out_;

 public:

  // constructor, destructor
  EcalFenixStrip(const EcalTPParameters * ecaltpp, const EcalElectronicsMapping* theMapping,bool debug,bool famos,int maxNrSamples);
  virtual ~EcalFenixStrip() ;

  // main methods
  // process method is splitted in 2 parts:
  //   the first one is templated, the same except input
  //   the second part is slightly different for barrel/endcap
  template <class T> void process(std::vector<const T *> &, int nrxtals, int stripnr,int townr,int smnr,std::vector<int> & out);
  void process_part2_barrel(int smnr, int stripnr,int townr);

  void process_part2_endcap(int smnr, int stripnr,int townr);

  int getCrystalNumberInStrip(const EBDataFrame *frame,int crystalIndexinStrip); //2nd argument dummy forEB 
  int getCrystalNumberInStrip(const EEDataFrame *frame,int crystalIndexinStrip); 
       

  // getters for the algorithms  ;

  EcalFenixLinearizer *getLinearizer (int i) const { return linearizer_[i];}
  EcalFenixEtStrip *getAdder() const { return  adder_;}
  EcalFenixAmplitudeFilter *getFilter() const { return amplitude_filter_;}
  EcalFenixPeakFinder *getPeakFinder() const { return peak_finder_;}
    
  EcalFenixStripFormatEB *getFormatterEB() const { return fenixFormatterEB_;}
  EcalFenixStripFormatEE *getFormatterEE() const { return fenixFormatterEE_;}
    
  EcalFenixStripFgvbEE *getFGVB()      const { return fgvbEE_;}

  // ========================= implementations ==============================================================
  void process(std::vector<const EBDataFrame *> &samples, int nrXtals, int stripnr,int townr, int smnr,std::vector<int> &out){
    process_part1(samples,nrXtals,smnr,stripnr,townr);//templated part
    process_part2_barrel(smnr,stripnr,townr);//part different for barrel/endcap
    out=format_out_;
  }
  void  process(std::vector<const EEDataFrame *> &samples, int nrXtals, int stripnr,int townr, int &sectorNr, std::vector<int> & out){

    process_part1(samples,nrXtals,sectorNr,stripnr,townr); //templated part
    process_part2_endcap(sectorNr,stripnr,townr);
    out=format_out_;
    return;
  }

  template <class T> void  process_part1(std::vector<const T *> & df,int nrXtals, int smnr, int stripnr, int townr)
    {
  
      if(debug_)  std::cout<<"\n\nEcalFenixStrip input is a vector of size: "<<nrXtals<< std::endl;

      //loop over crystals
      for (int cryst=0;cryst<nrXtals;cryst++) {
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
	this->getLinearizer(cryst)->process(*(df[cryst]),lin_out_[cryst]);
      }

      if(debug_){
	std::cout<< "output of linearizer is a vector of size: "<<std::dec<<lin_out_.size()<<" of which used "<<nrXtals<<std::endl; 
	for (int ix=0;ix<nrXtals;ix++){
	  std::cout<< "cryst: "<<ix<<"  value : "<<std::dec<<std::endl;
	  std::cout<<" lin_out[ix].size()= "<<std::dec<<lin_out_[ix].size()<<std::endl;
	  for (unsigned int i =0; i<lin_out_[ix].size();i++){
	    std::cout <<" "<<std::dec<<(lin_out_[ix])[i];
	  }
	  std::cout<<std::endl;
	}
    
	std::cout<<std::endl;
      }
  
      // call adder
      this->getAdder()->process(lin_out_,nrXtals,add_out_);  //add_out is of size SIZEMAX=maxNrSamples
 
      if(debug_){
	std::cout<< "output of adder is a vector of size: "<<std::dec<<add_out_.size()<<std::endl; 
	for (unsigned int ix=0;ix<add_out_.size();ix++){
	  std::cout<< "cryst: "<<ix<<"  value : "<<std::dec<<add_out_[ix]<<std::endl;
	}
	std::cout<<std::endl;
      }
 

      if (famos_) {
	filt_out_[0]= add_out_[0];
	peak_out_[0]= add_out_[0];
	return;
      }else {
	// call amplitudefilter
	this->getFilter()->setParameters(smnr, townr,stripnr) ; 
	this->getFilter()->process(add_out_,filt_out_); 

	if(debug_){
	  std::cout<< "output of filter is a vector of size: "<<std::dec<<filt_out_.size()<<std::endl; 
	  for (unsigned int ix=0;ix<filt_out_.size();ix++){
	    std::cout<< "cryst: "<<ix<<"  value : "<<std::dec<<filt_out_[ix]<<std::endl;
	  }
	  std::cout<<std::endl;
	}

	// call peakfinder
	this->getPeakFinder()->process(filt_out_,peak_out_);
	if(debug_){
	  std::cout<< "output of peakfinder is a vector of size: "<<peak_out_.size()<<std::endl; 
	  for (unsigned int ix=0;ix<peak_out_.size();ix++){
	    std::cout<< "cryst: "<<ix<<"  value : "<<peak_out_[ix]<<std::endl;
	  }
	  std::cout<<std::endl;
	}
	return;
      }
    }

};
#endif

