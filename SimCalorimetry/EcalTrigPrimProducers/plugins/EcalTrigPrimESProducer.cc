// user include files
#include "EcalTrigPrimESProducer.h"

#include <iostream>
#include <fstream>
#include <TMath.h>
#include <sstream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// input stream from a gz file
//

struct GzInputStream
 {
  typedef voidp gzFile ;
  gzFile gzf ;
  char buffer[256] ;
  std::istringstream iss ;
  bool eof ;
  GzInputStream( const char * file )
   : eof(false)
   {
    gzf = gzopen(file,"rb") ;
    if (gzf==Z_NULL)
     {
      eof = true ;
      edm::LogWarning("EcalTPG") <<"Database file "<<file<<" not found!!!";
     }
    else readLine() ;
   }
  void readLine()
   {
    char * res = gzgets(gzf,buffer,256) ;
    eof = (res==Z_NULL) ;
    if (!eof)
     {
      iss.clear() ;
      iss.str(buffer) ;
     }
   }
  ~GzInputStream()
   { gzclose(gzf) ; }
  operator void*()
   { return ((eof==true)?((void*)0):iss) ; }
 } ;

template <typename T>
GzInputStream & operator>>( GzInputStream & gis, T & var )
 {
  while ((gis)&&(!(gis.iss>>var)))
   { gis.readLine() ; }
  return gis ;
 }


//
// constructors and destructor
//

EcalTrigPrimESProducer::EcalTrigPrimESProducer(const edm::ParameterSet& iConfig) :
  dbFilename_(iConfig.getUntrackedParameter<std::string>("DatabaseFile",""))
{
  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this, &EcalTrigPrimESProducer::producePedestals) ;
  setWhatProduced(this, &EcalTrigPrimESProducer::produceLinearizationConst) ;
  setWhatProduced(this, &EcalTrigPrimESProducer::produceSlidingWindow) ;
  setWhatProduced(this, &EcalTrigPrimESProducer::produceFineGrainEB) ;
  setWhatProduced(this, &EcalTrigPrimESProducer::produceFineGrainEEstrip) ;
  setWhatProduced(this, &EcalTrigPrimESProducer::produceFineGrainEEtower) ;
  setWhatProduced(this, &EcalTrigPrimESProducer::produceLUT) ;
  setWhatProduced(this, &EcalTrigPrimESProducer::produceWeight) ;
  setWhatProduced(this, &EcalTrigPrimESProducer::produceWeightGroup) ;
  setWhatProduced(this, &EcalTrigPrimESProducer::produceLutGroup) ;
  setWhatProduced(this, &EcalTrigPrimESProducer::produceFineGrainEBGroup) ;
  setWhatProduced(this, &EcalTrigPrimESProducer::producePhysicsConst) ;
  setWhatProduced(this, &EcalTrigPrimESProducer::produceBadX) ;
  setWhatProduced(this, &EcalTrigPrimESProducer::produceBadTT) ;
  //now do what ever other initialization is needed
}


EcalTrigPrimESProducer::~EcalTrigPrimESProducer()
{ 
}

//
// member functions
//

// ------------ method called to produce the data  ------------


std::auto_ptr<EcalTPGPedestals> EcalTrigPrimESProducer::producePedestals(const EcalTPGPedestalsRcd & iRecord)
{
  std::auto_ptr<EcalTPGPedestals> prod(new EcalTPGPedestals());
  parseTextFile() ;
  std::map<uint32_t, std::vector<uint32_t> >::const_iterator it ;
  for (it = mapXtal_.begin() ; it != mapXtal_.end() ; it++) {
    EcalTPGPedestal item ;
    item.mean_x12 = (it->second)[0] ;
    item.mean_x6  = (it->second)[3] ;
    item.mean_x1  = (it->second)[6] ;
    prod->setValue(it->first,item) ;
  }
  return prod;
}

std::auto_ptr<EcalTPGLinearizationConst> EcalTrigPrimESProducer::produceLinearizationConst(const EcalTPGLinearizationConstRcd & iRecord)
{
  std::auto_ptr<EcalTPGLinearizationConst> prod(new EcalTPGLinearizationConst());
  parseTextFile() ;
  std::map<uint32_t, std::vector<uint32_t> >::const_iterator it ;
  for (it = mapXtal_.begin() ; it != mapXtal_.end() ; it++) {
    EcalTPGLinearizationConstant item ;
    item.mult_x12 = (it->second)[1] ;
    item.mult_x6  = (it->second)[4] ;
    item.mult_x1  = (it->second)[7] ;
    item.shift_x12 = (it->second)[2] ;
    item.shift_x6  = (it->second)[5] ;
    item.shift_x1  = (it->second)[8] ;
    prod->setValue(it->first,item) ;
  }
  return prod;
}

std::auto_ptr<EcalTPGSlidingWindow> EcalTrigPrimESProducer::produceSlidingWindow(const EcalTPGSlidingWindowRcd & iRecord)
{
  std::auto_ptr<EcalTPGSlidingWindow> prod(new EcalTPGSlidingWindow());
  parseTextFile() ;
  for (int subdet=0 ; subdet<2 ; subdet++) {
    std::map<uint32_t, std::vector<uint32_t> >::const_iterator it ;
    for (it = mapStrip_[subdet].begin() ; it != mapStrip_[subdet].end() ; it++) {
      prod->setValue(it->first,(it->second)[0]) ;
    }
  }
  return prod;
}

std::auto_ptr<EcalTPGFineGrainEBIdMap> EcalTrigPrimESProducer::produceFineGrainEB(const EcalTPGFineGrainEBIdMapRcd & iRecord)
{
  std::auto_ptr<EcalTPGFineGrainEBIdMap> prod(new EcalTPGFineGrainEBIdMap());
  parseTextFile() ;
  EcalTPGFineGrainConstEB fg ;
  std::map<uint32_t, std::vector<uint32_t> >::const_iterator it ;
  for (it = mapFg_.begin() ; it != mapFg_.end() ; it++) {
    fg.setValues((it->second)[0], (it->second)[1], (it->second)[2], (it->second)[3], (it->second)[4]) ;
    prod->setValue(it->first,fg) ;
  }
  return prod;
}

std::auto_ptr<EcalTPGFineGrainStripEE> EcalTrigPrimESProducer::produceFineGrainEEstrip(const EcalTPGFineGrainStripEERcd & iRecord)
{
  std::auto_ptr<EcalTPGFineGrainStripEE> prod(new EcalTPGFineGrainStripEE());
  parseTextFile() ;
  std::map<uint32_t, std::vector<uint32_t> >::const_iterator it ;
  for (it = mapStrip_[1].begin() ; it != mapStrip_[1].end() ; it++) {
    EcalTPGFineGrainStripEE::Item item ;
    item.threshold = (it->second)[2] ;
    item.lut  = (it->second)[3] ;
    prod->setValue(it->first,item) ;
  }
  return prod;
}

std::auto_ptr<EcalTPGFineGrainTowerEE> EcalTrigPrimESProducer::produceFineGrainEEtower(const EcalTPGFineGrainTowerEERcd & iRecord)
{
  std::auto_ptr<EcalTPGFineGrainTowerEE> prod(new EcalTPGFineGrainTowerEE());
  parseTextFile() ;
  std::map<uint32_t, std::vector<uint32_t> >::const_iterator it ;
  for (it = mapTower_[1].begin() ; it != mapTower_[1].end() ; it++) {
    prod->setValue(it->first,(it->second)[1]) ;
  }
  return prod;
}

std::auto_ptr<EcalTPGLutIdMap> EcalTrigPrimESProducer::produceLUT(const EcalTPGLutIdMapRcd & iRecord)
{
  std::auto_ptr<EcalTPGLutIdMap> prod(new EcalTPGLutIdMap());
  parseTextFile() ;
  EcalTPGLut lut ;
  std::map<uint32_t, std::vector<uint32_t> >::const_iterator it ;
  for (it = mapLut_.begin() ; it != mapLut_.end() ; it++) {
    unsigned int lutArray[1024] ;
    for (int i=0 ; i <1024 ; i++) lutArray[i] = (it->second)[i] ;
    lut.setLut(lutArray) ;
    prod->setValue(it->first,lut) ;
  }
  return prod;
}

std::auto_ptr<EcalTPGWeightIdMap> EcalTrigPrimESProducer::produceWeight(const EcalTPGWeightIdMapRcd & iRecord)
{
  std::auto_ptr<EcalTPGWeightIdMap> prod(new EcalTPGWeightIdMap());
  parseTextFile() ;
  EcalTPGWeights weights ;
  std::map<uint32_t, std::vector<uint32_t> >::const_iterator it ;
  for (it = mapWeight_.begin() ; it != mapWeight_.end() ; it++) {
    weights.setValues((it->second)[0], 
		      (it->second)[1],
		      (it->second)[2], 
		      (it->second)[3],
		      (it->second)[4]) ;
    prod->setValue(it->first,weights) ;
  }
  return prod;
}

std::auto_ptr<EcalTPGWeightGroup> EcalTrigPrimESProducer::produceWeightGroup(const EcalTPGWeightGroupRcd & iRecord)
{
  std::auto_ptr<EcalTPGWeightGroup> prod(new EcalTPGWeightGroup());
  parseTextFile() ;
  for (int subdet=0 ; subdet<2 ; subdet++) {
    std::map<uint32_t, std::vector<uint32_t> >::const_iterator it ;
    for (it = mapStrip_[subdet].begin() ; it != mapStrip_[subdet].end() ; it++) {
      prod->setValue(it->first,(it->second)[1]) ;
    }
  }
  return prod;
}

std::auto_ptr<EcalTPGLutGroup> EcalTrigPrimESProducer::produceLutGroup(const EcalTPGLutGroupRcd & iRecord)
{
  std::auto_ptr<EcalTPGLutGroup> prod(new EcalTPGLutGroup());
  parseTextFile() ;
  for (int subdet=0 ; subdet<2 ; subdet++) {
    std::map<uint32_t, std::vector<uint32_t> >::const_iterator it ;
    for (it = mapTower_[subdet].begin() ; it != mapTower_[subdet].end() ; it++) {
      prod->setValue(it->first,(it->second)[0]) ;
    }
  }
  return prod;
}

std::auto_ptr<EcalTPGFineGrainEBGroup> EcalTrigPrimESProducer::produceFineGrainEBGroup(const EcalTPGFineGrainEBGroupRcd & iRecord)
{
  std::auto_ptr<EcalTPGFineGrainEBGroup> prod(new EcalTPGFineGrainEBGroup());
  parseTextFile() ;
  std::map<uint32_t, std::vector<uint32_t> >::const_iterator it ;
  for (it = mapTower_[0].begin() ; it != mapTower_[0].end() ; it++) {
    prod->setValue(it->first,(it->second)[1]) ;
  }
  return prod;
}

std::auto_ptr<EcalTPGPhysicsConst> EcalTrigPrimESProducer::producePhysicsConst(const EcalTPGPhysicsConstRcd & iRecord)
{
  std::auto_ptr<EcalTPGPhysicsConst> prod(new EcalTPGPhysicsConst());
  parseTextFile() ;
  std::map<uint32_t, std::vector<float> >::const_iterator it ;
  for (it = mapPhys_.begin() ; it != mapPhys_.end() ; it++) {
    EcalTPGPhysicsConst::Item item ;
    item.EtSat = (it->second)[0] ;
    item.ttf_threshold_Low = (it->second)[1] ;
    item.ttf_threshold_High = (it->second)[2] ;
    item.FG_lowThreshold = (it->second)[3] ;
    item.FG_highThreshold = (it->second)[4] ;
    item.FG_lowRatio = (it->second)[5] ;
    item.FG_highRatio = (it->second)[6] ;
    prod->setValue(it->first,item) ;
  }
  return prod;
}

std::auto_ptr<EcalTPGCrystalStatus> EcalTrigPrimESProducer::produceBadX(const EcalTPGCrystalStatusRcd & iRecord)
{
  std::auto_ptr<EcalTPGCrystalStatus> prod(new EcalTPGCrystalStatus());
  parseTextFile() ;
  std::map<uint32_t, std::vector<uint32_t> >::const_iterator it ;
  for (it = mapXtal_.begin() ; it != mapXtal_.end() ; it++) {
    
    EcalTPGCrystalStatusCode badXValue;
    badXValue.setStatusCode(0);
    prod->setValue(it->first,badXValue) ;
  }
  return prod;
  
}

std::auto_ptr<EcalTPGTowerStatus> EcalTrigPrimESProducer::produceBadTT(const EcalTPGTowerStatusRcd & iRecord)
{
  std::auto_ptr<EcalTPGTowerStatus> prod(new EcalTPGTowerStatus());
  parseTextFile() ;
  std::map<uint32_t, std::vector<uint32_t> >::const_iterator it ;
  //Barrel
  for (it = mapTower_[0].begin() ; it != mapTower_[0].end() ; it++) {
    //set the BadTT status to 0
    prod->setValue(it->first,0) ;
  }
  //Endcap
  for (it = mapTower_[1].begin() ; it != mapTower_[1].end() ; it++) {
    //set the BadTT status to 0
    prod->setValue(it->first,0) ;
  }
  
  return prod; 
}

void EcalTrigPrimESProducer::parseTextFile()
{
  if (mapXtal_.size() != 0) return ; // just parse the file once!

  uint32_t id ;
  std::string dataCard ;
  std::string line;
  std::ifstream infile ; 
  std::vector<unsigned int> param ;
  std::vector<float> paramF ;
  int NBstripparams[2] = {2, 4} ;
  unsigned int data ;
  float dataF ;

  std::string bufString;
  std::string iString;
  std::string fString;
  std::string filename = "SimCalorimetry/EcalTrigPrimProducers/data/"+dbFilename_;
  std::string finalFileName ;
  size_t slash=dbFilename_.find("/");
  if (slash!=0) {
    edm::FileInPath fileInPath(filename);
    finalFileName = fileInPath.fullPath() ;
  }
  else {
    finalFileName = dbFilename_.c_str() ;
    edm::LogWarning("EcalTPG") <<"Couldnt find database file via fileinpath, trying with pathname directly!!";
  }


  GzInputStream gis(finalFileName.c_str()) ;
  while (gis>>dataCard) {

    if (dataCard == "PHYSICS_EB" || dataCard == "PHYSICS_EE") {
      gis>>std::dec>>id ;
      //std::cout<<dataCard<<" "<<std::dec<<id ;
      paramF.clear() ;
      for (int i=0 ; i <7 ; i++) {
        gis>>std::dec>>dataF ;
        paramF.push_back(dataF) ;
        //std::cout<<", "<<std::dec<<dataF ;
      }
      //std::cout<<std::endl ;
      mapPhys_[id] = paramF ;
    }

    if (dataCard == "CRYSTAL") {
      gis>>std::dec>>id ;
      //std::cout<<dataCard<<" "<<std::dec<<id ;
      param.clear() ;
      for (int i=0 ; i <9 ; i++) {
        gis>>std::hex>>data ;
        //std::cout<<", "<<std::hex<<data ;
        param.push_back(data) ;
      }
      //std::cout<<std::endl ;
      mapXtal_[id] = param ;
    }

    if (dataCard == "STRIP_EB") {
      gis>>std::dec>>id ;
      //std::cout<<dataCard<<" "<<std::dec<<id ;
      param.clear() ;
      for (int i=0 ; i <NBstripparams[0] ; i++) {
        gis>>std::hex>>data ;
        //std::cout<<", "<<std::hex<<data ;
        param.push_back(data) ;
      }
      //std::cout<<std::endl ;
      mapStrip_[0][id] = param ;
    }

    if (dataCard == "STRIP_EE") {
      gis>>std::dec>>id ;
      //std::cout<<dataCard<<" "<<std::dec<<id ;
      param.clear() ;
      for (int i=0 ; i <NBstripparams[1] ; i++) {
        gis>>std::hex>>data ;
        //std::cout<<", "<<std::hex<<data ;
        param.push_back(data) ;
      }
      //std::cout<<std::endl ;
      mapStrip_[1][id] = param ;
    }

    if (dataCard == "TOWER_EB" || dataCard == "TOWER_EE") {
      gis>>std::dec>>id ;
      //std::cout<<dataCard<<" "<<std::dec<<id ;
      param.clear() ;
      for (int i=0 ; i <2 ; i++) {
        gis>>std::hex>>data ;
        //std::cout<<", "<<std::hex<<data ;
        param.push_back(data) ;
      }
      //std::cout<<std::endl ;
      if (dataCard == "TOWER_EB") mapTower_[0][id] = param ;
      if (dataCard == "TOWER_EE") mapTower_[1][id] = param ;
    }

    if (dataCard == "WEIGHT") {
      gis>>std::hex>>id ;
      //std::cout<<dataCard<<" "<<std::dec<<id ;
      param.clear() ;
      for (int i=0 ; i <5 ; i++) {
        gis>>std::hex>>data ;
        //std::cout<<", "<<std::hex<<data ;
        param.push_back(data) ;
      }
      //std::cout<<std::endl ;
      mapWeight_[id] = param ;
    }

    if (dataCard == "FG") {
      gis>>std::hex>>id ;
      //std::cout<<dataCard<<" "<<std::dec<<id ;
      param.clear() ;
      for (int i=0 ; i <5 ; i++) {
        gis>>std::hex>>data ;
        //std::cout<<", "<<std::hex<<data ;
        param.push_back(data) ;
      }
      //std::cout<<std::endl ;
      mapFg_[id] = param ;
    }
 
    if (dataCard == "LUT") {
      gis>>std::hex>>id ;
      //std::cout<<dataCard<<" "<<std::dec<<id ;
      param.clear() ;
      for (int i=0 ; i <1024 ; i++) {
        gis>>std::hex>>data ;
        //std::cout<<", "<<std::hex<<data ;
        param.push_back(data) ;
      }
      //std::cout<<std::endl ;
      mapLut_[id] = param ;
    }
  }
}

std::vector<int> EcalTrigPrimESProducer::getRange(int subdet, int tccNb, int towerNbInTcc, int stripNbInTower, int xtalNbInStrip)
{
  std::vector<int> range ;
  if (subdet == 0) { 
    // Barrel
    range.push_back(37)  ; // stccNbMin
    range.push_back(73) ; // tccNbMax
    range.push_back(1)  ; // towerNbMin
    range.push_back(69) ; // towerNbMax
    range.push_back(1)  ; // stripNbMin
    range.push_back(6)  ; // stripNbMax
    range.push_back(1)  ; // xtalNbMin
    range.push_back(6)  ; // xtalNbMax
  } else {
    // Endcap eta >0
    if (subdet >0 ) {
      range.push_back(73) ; // tccNbMin
      range.push_back(109) ; // tccNbMax
    } else { //endcap eta <0
      range.push_back(1) ; // tccNbMin
      range.push_back(37) ; // tccNbMax
    }
    range.push_back(1)  ; // towerNbMin
    range.push_back(29) ; // towerNbMax
    range.push_back(1)  ; // stripNbMin
    range.push_back(6)  ; // stripNbMax
    range.push_back(1)  ; // xtalNbMin
    range.push_back(6)  ; // xtalNbMax
  }

  if (tccNb>0) {
    range[0] = tccNb ; 
    range[1] = tccNb+1 ;
  }
  if (towerNbInTcc>0) {
    range[2] = towerNbInTcc ; 
    range[3] = towerNbInTcc+1 ;
  }
  if (stripNbInTower>0) {
    range[4] = stripNbInTower ; 
    range[5] = stripNbInTower+1 ;
  }
  if (xtalNbInStrip>0) {
    range[6] = xtalNbInStrip ; 
    range[7] = xtalNbInStrip+1 ;
  }

  return range ;
}

// bool EcalTrigPrimESProducer::getNextString(gzFile &gzf){
//   size_t blank;
//   if (bufpos_==0) {
//     gzgets(gzf,buf_,80);
//     if (gzeof(gzf)) return true;
//     bufString_=std::string(buf_);
//   }
//   int  pos=0;
//   pos =bufpos_;
//   // look for next non-blank
//   while (pos<bufString_.size()) {
//     if (!bufString_.compare(pos,1," ")) pos++;
//     else break;
//   }
//   blank=bufString_.find(" ",pos);
//   size_t end = blank;
//   if (blank==std::string::npos) end=bufString_.size();
//   sub_=bufString_.substr(pos,end-pos);
//   bufpos_= blank;
//   if (blank==std::string::npos) bufpos_=0;
//   return false;
// }
//  
// int EcalTrigPrimESProducer::converthex() {
//   // converts hex dec string sub to hexa
//   //FIXME:: find something better (istrstream?)!!!!
// 
//   std::string chars("0123456789abcdef");
//   int hex=0;
//   for (size_t i=2;i<sub_.length();++i) {
//     size_t f=chars.find(sub_[i]);
//     if (f==std::string::npos) break;  //FIXME: length is 4 for 0x3!!
//     hex=hex*16+chars.find(sub_[i]);
//   }
//   return hex;
// }
