// File: DataMixingHcalDigiWorker.cc
// Description:  see DataMixingHcalDigiWorker.h
// Author:  Mike Hildreth, University of Notre Dame
//
//--------------------------------------------

#include <map>
#include <memory>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
// calibration headers, for future reference 
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"   
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h" 
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"  
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"   
//
//
#include "DataMixingHcalDigiWorker.h"

using namespace std;

namespace {

  typedef std::multimap<DetId, CaloSamples> HcalDigiMap;

  template <class DIGI>
  void convertFc2adc (const  CaloSamples& fC, const HcalDbService& conditions, DIGI& digi, int capIdOffset = 0) {
    HcalDetId id = fC.id();
    const HcalQIECoder* channelCoder = conditions.getHcalCoder (id);
    const HcalQIEShape* shape = conditions.getHcalShape (channelCoder);
    HcalCoderDb coder (*channelCoder, *shape);
    coder.fC2adc(fC, digi, capIdOffset);
  }

  template <class DIGI>
  void convertAdc2fChelper (const DIGI& digi, const HcalDbService& conditions, CaloSamples& fC, const HcalDetId& id) {
    const HcalCalibrations& calib = conditions.getHcalCalibrations(id);
    for (int i = 0; i < digi.size(); ++i) {
      int capId (digi.sample(i).capid());
      fC[i] -= calib.pedestal (capId);
    }
  }
  
  template <>
  void convertAdc2fChelper<QIE10DataFrame> (const QIE10DataFrame& digi, const HcalDbService& conditions, CaloSamples& fC, const HcalDetId& id) {
    const HcalCalibrations& calib = conditions.getHcalCalibrations(id);
    for (int i = 0; i < digi.samples(); ++i) {
      int capId (digi[i].capid());
      fC[i] -= calib.pedestal (capId);
    }
  }
  
  template <>
  void convertAdc2fChelper<QIE11DataFrame> (const QIE11DataFrame& digi, const HcalDbService& conditions, CaloSamples& fC, const HcalDetId& id) {
    const HcalCalibrations& calib = conditions.getHcalCalibrations(id);
    for (int i = 0; i < digi.samples(); ++i) {
      int capId (digi[i].capid());
      fC[i] -= calib.pedestal (capId);
    }
  }
  
  template <class DIGI>
  void convertAdc2fC (const DIGI& digi, const HcalDbService& conditions, bool keepPedestals, CaloSamples& fC) {
    HcalDetId id (digi.id());
    const HcalQIECoder* channelCoder = conditions.getHcalCoder (id);
    const HcalQIEShape* shape = conditions.getHcalShape (channelCoder);
    HcalCoderDb coder (*channelCoder, *shape);
    coder.adc2fC(digi, fC);
    if (!keepPedestals) { // subtract pedestals
      convertAdc2fChelper(digi,conditions,fC,id);
    }
  }

  template <class DIGIS>
   void convertHcalDigis (const DIGIS& digis, const HcalDbService& conditions, bool keepPedestals, HcalDigiMap& map) {
    for (auto digi = digis.begin(); digi != digis.end(); ++digi) {
      CaloSamples fC;
      convertAdc2fC (*digi, conditions, keepPedestals, fC);
      if (!keepPedestals && map.find(digi->id()) == map.end()) {
        edm::LogWarning("DataMixingHcalDigiWorker")<<"No signal hits found for HCAL cell "<<digi->id()<<" Pedestals may be lost for mixed hit";
      }
      map.insert(HcalDigiMap::value_type (digi->id(), fC));
    }
  }

  template <>
   void convertHcalDigis<QIE10DigiCollection> (const QIE10DigiCollection& digis, const HcalDbService& conditions, bool keepPedestals, HcalDigiMap& map) {
    for (auto digiItr = digis.begin(); digiItr != digis.end(); ++digiItr) {
      QIE10DataFrame digi(*digiItr);
	  CaloSamples fC;
      convertAdc2fC (digi, conditions, keepPedestals, fC);
      if (!keepPedestals && map.find(digi.id()) == map.end()) {
        edm::LogWarning("DataMixingHcalDigiWorker")<<"No signal hits found for HCAL cell "<<digi.id()<<" Pedestals may be lost for mixed hit";
      }
      map.insert(HcalDigiMap::value_type (digi.id(), fC));
    }
  }
  
  template <>
   void convertHcalDigis<QIE11DigiCollection> (const QIE11DigiCollection& digis, const HcalDbService& conditions, bool keepPedestals, HcalDigiMap& map) {
    for (auto digiItr = digis.begin(); digiItr != digis.end(); ++digiItr) {
      QIE11DataFrame digi(*digiItr);
	  CaloSamples fC;
      convertAdc2fC (digi, conditions, keepPedestals, fC);
      if (!keepPedestals && map.find(digi.id()) == map.end()) {
        edm::LogWarning("DataMixingHcalDigiWorker")<<"No signal hits found for HCAL cell "<<digi.id()<<" Pedestals may be lost for mixed hit";
      }
      map.insert(HcalDigiMap::value_type (digi.id(), fC));
    }
  }
  
  template <class DIGIS>
  bool convertSignalHcalDigis (const edm::Event &e, const edm::EDGetTokenT<DIGIS>& token,
                               const HcalDbService& conditions, HcalDigiMap& map) {
    edm::Handle<DIGIS> digis;
    if (!e.getByToken (token, digis)) return false;
    convertHcalDigis (*digis, conditions, true, map); // keep pedestals
    return true;
  }

  template <class DIGIS>
  bool convertPileupHcalDigis (const edm::EventPrincipal& ep, 
                               const edm::InputTag& tag, const edm::ModuleCallingContext* mcc,
                               const HcalDbService& conditions, HcalDigiMap& map) {
    auto digis = edm::getProductByTag<DIGIS>(ep, tag, mcc);
    if (!digis) return false;
    convertHcalDigis (*(digis->product()), conditions, false, map); // subtract pedestals
    return true;
  }

  template <class DIGIS>
  void buildHcalDigisHelper (DIGIS& digis, const DetId& formerID, CaloSamples& resultSample, const HcalDbService& conditions){
    // make new digi
    digis.push_back(typename DIGIS::value_type(formerID));
    convertFc2adc (resultSample, conditions, digis.back(), 0); // FR guess: simulation starts with 0
  }
  
  template <>
  void buildHcalDigisHelper<QIE10DigiCollection> (QIE10DigiCollection& digis, const DetId& formerID, CaloSamples& resultSample, const HcalDbService& conditions){
    // make new digi
    digis.push_back(formerID.rawId());
	QIE10DataFrame digi( digis.back() );
    convertFc2adc (resultSample, conditions, digi, 0); // FR guess: simulation starts with 0
  }
  
  template <>
  void buildHcalDigisHelper<QIE11DigiCollection> (QIE11DigiCollection& digis, const DetId& formerID, CaloSamples& resultSample, const HcalDbService& conditions){
    // make new digi
    digis.push_back(formerID.rawId());
	QIE11DataFrame digi( digis.back() );
    convertFc2adc (resultSample, conditions, digi, 0); // FR guess: simulation starts with 0
  }
  
  template <class DIGIS>
  std::unique_ptr<DIGIS> buildHcalDigis (const HcalDigiMap& map, const HcalDbService& conditions) {
    std::unique_ptr<DIGIS> digis( new DIGIS );
    // loop over the maps we have, re-making individual hits or digis if necessary.
    DetId formerID = 0;
    CaloSamples resultSample;

    for(auto hit = map.begin(); hit != map.end(); ++hit) {
      DetId currentID = hit->first; 
      const CaloSamples& hitSample = hit->second;

      if (currentID == formerID) { // accumulating hits
        //loop over digi samples in each CaloSample                                                  
        unsigned int sizenew = (hitSample).size();
        unsigned int sizeold = resultSample.size();
        if (sizenew > sizeold) { // extend sample
          resultSample.setSize (sizenew);
        }
        for(unsigned int isamp = 0; isamp<sizenew; isamp++) { // add new values
          resultSample[isamp] += hitSample[isamp]; // for debugging below
        }
      }
      auto hit1 = hit;
      bool lastEntry = (++hit1 ==  map.end());
      if (currentID != formerID || lastEntry) { // store current digi
        if (formerID>0 || lastEntry) {
          // make new digi
          buildHcalDigisHelper(*digis,formerID,resultSample,conditions);
        }
        //reset pointers for next iteration                                                                 
        formerID = currentID;
        resultSample = hitSample;
      }
    }
    return digis;
  }
  
  

} // namespace {}

namespace edm
{

  // Virtual constructor

  DataMixingHcalDigiWorker::DataMixingHcalDigiWorker() { }

  // Constructor 
  DataMixingHcalDigiWorker::DataMixingHcalDigiWorker(const edm::ParameterSet& ps, edm::ConsumesCollector && iC) : 
                                                            label_(ps.getParameter<std::string>("Label"))
  {                                                         

    // get the subdetector names
    //    this->getSubdetectorNames();  //something like this may be useful to check what we are supposed to do...

    // declare the products to produce

    // Hcal 

    HBHEdigiCollectionSig_  = ps.getParameter<edm::InputTag>("HBHEdigiCollectionSig");
    HOdigiCollectionSig_    = ps.getParameter<edm::InputTag>("HOdigiCollectionSig");
    HFdigiCollectionSig_    = ps.getParameter<edm::InputTag>("HFdigiCollectionSig");
    ZDCdigiCollectionSig_   = ps.getParameter<edm::InputTag>("ZDCdigiCollectionSig");
    QIE10digiCollectionSig_   = ps.getParameter<edm::InputTag>("QIE10digiCollectionSig");
    QIE11digiCollectionSig_   = ps.getParameter<edm::InputTag>("QIE11digiCollectionSig");

    HBHEPileInputTag_ = ps.getParameter<edm::InputTag>("HBHEPileInputTag");
    HOPileInputTag_ = ps.getParameter<edm::InputTag>("HOPileInputTag");
    HFPileInputTag_ = ps.getParameter<edm::InputTag>("HFPileInputTag");
    ZDCPileInputTag_ = ps.getParameter<edm::InputTag>("ZDCPileInputTag");
    QIE10PileInputTag_ = ps.getParameter<edm::InputTag>("QIE10PileInputTag");
    QIE11PileInputTag_ = ps.getParameter<edm::InputTag>("QIE11PileInputTag");

    HBHEDigiToken_ = iC.consumes<HBHEDigiCollection>(HBHEdigiCollectionSig_);
    HODigiToken_ = iC.consumes<HODigiCollection>(HOdigiCollectionSig_);
    HFDigiToken_ = iC.consumes<HFDigiCollection>(HFdigiCollectionSig_);
    QIE10DigiToken_ = iC.consumes<QIE10DigiCollection>(QIE10digiCollectionSig_);
    QIE11DigiToken_ = iC.consumes<QIE11DigiCollection>(QIE11digiCollectionSig_);

    HBHEDigiPToken_ = iC.consumes<HBHEDigiCollection>(HBHEPileInputTag_);
    HODigiPToken_ = iC.consumes<HODigiCollection>(HOPileInputTag_);
    HFDigiPToken_ = iC.consumes<HFDigiCollection>(HFPileInputTag_);
    QIE10DigiPToken_ = iC.consumes<QIE10DigiCollection>(QIE10PileInputTag_);
    QIE11DigiPToken_ = iC.consumes<QIE11DigiCollection>(QIE11PileInputTag_);

    DoZDC_ = false;
    if(!ZDCPileInputTag_.label().empty()) DoZDC_ = true;

    if(DoZDC_) { 
      ZDCDigiToken_ = iC.consumes<ZDCDigiCollection>(ZDCdigiCollectionSig_);
      ZDCDigiPToken_ = iC.consumes<ZDCDigiCollection>(ZDCPileInputTag_);
    }


    HBHEDigiCollectionDM_ = ps.getParameter<std::string>("HBHEDigiCollectionDM");
    HODigiCollectionDM_   = ps.getParameter<std::string>("HODigiCollectionDM");
    HFDigiCollectionDM_   = ps.getParameter<std::string>("HFDigiCollectionDM");
    ZDCDigiCollectionDM_  = ps.getParameter<std::string>("ZDCDigiCollectionDM");
    QIE10DigiCollectionDM_  = ps.getParameter<std::string>("QIE10DigiCollectionDM");
    QIE11DigiCollectionDM_  = ps.getParameter<std::string>("QIE11DigiCollectionDM");


  }

  // Virtual destructor needed.
  DataMixingHcalDigiWorker::~DataMixingHcalDigiWorker() { 
  }  

  void DataMixingHcalDigiWorker::addHcalSignals(const edm::Event &e,const edm::EventSetup& ES) { 
    // Calibration stuff will look like this:                                                 
    // get conditions                                                                         
    edm::ESHandle<HcalDbService> conditions;                                                
    ES.get<HcalDbRecord>().get(conditions);                                         


    // fill in maps of hits

    LogInfo("DataMixingHcalDigiWorker")<<"===============> adding MC signals for "<<e.id();
    convertSignalHcalDigis<HBHEDigiCollection> (e, HBHEDigiToken_, *conditions, HBHEDigiStorage_);
    convertSignalHcalDigis<HODigiCollection> (e, HODigiToken_, *conditions, HODigiStorage_);
    convertSignalHcalDigis<HFDigiCollection> (e, HFDigiToken_, *conditions, HFDigiStorage_);
    convertSignalHcalDigis<QIE10DigiCollection> (e, QIE10DigiToken_, *conditions, QIE10DigiStorage_);
    convertSignalHcalDigis<QIE11DigiCollection> (e, QIE11DigiToken_, *conditions, QIE11DigiStorage_);



   // ZDC next

   if(DoZDC_){

     Handle< ZDCDigiCollection > pZDCDigis;

     const ZDCDigiCollection*  ZDCDigis = nullptr;

     if( e.getByToken( ZDCDigiToken_, pZDCDigis) ) {
       ZDCDigis = pZDCDigis.product(); // get a ptr to the product
#ifdef DEBUG
       LogDebug("DataMixingHcalDigiWorker") << "total # ZDC digis: " << ZDCDigis->size();
#endif
     } 
   
 
     if (ZDCDigis)
       {
         // loop over digis, storing them in a map so we can add pileup later
         for(ZDCDigiCollection::const_iterator it  = ZDCDigis->begin(); it != ZDCDigis->end(); ++it) {

           // calibration, for future reference:  (same block for all Hcal types) ZDC is different                               
           HcalZDCDetId cell = it->id();
           //         const HcalCalibrations& calibrations=conditions->getHcalCalibrations(cell);                
           const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
           const HcalQIEShape* shape = conditions->getHcalShape (channelCoder); // this one is generic         
           HcalCoderDb coder (*channelCoder, *shape);

           CaloSamples tool;
           coder.adc2fC((*it),tool);

           ZDCDigiStorage_.insert(ZDCDigiMap::value_type( ( it->id() ), tool ));

#ifdef DEBUG
           // Commented out because this does not compile anymore	 
           // LogDebug("DataMixingHcalDigiWorker") << "processed ZDCDigi with rawId: "
           //                                      << it->id() << "\n"
           //                                      << " digi energy: " << it->energy();
#endif

         }
       }
   }
    
  } // end of addHCalSignals

  void DataMixingHcalDigiWorker::addHcalPileups(const int bcr, const EventPrincipal *ep, unsigned int eventNr,const edm::EventSetup& ES,
                                                ModuleCallingContext const* mcc) {
  
    LogDebug("DataMixingHcalDigiWorker") <<"\n===============> adding pileups from event  "<<ep->id()<<" for bunchcrossing "<<bcr;

    // get conditions                                                                                                             
    edm::ESHandle<HcalDbService> conditions;
    ES.get<HcalDbRecord>().get(conditions);

    convertPileupHcalDigis<HBHEDigiCollection> (*ep, HBHEPileInputTag_, mcc, *conditions, HBHEDigiStorage_);
    convertPileupHcalDigis<HODigiCollection> (*ep, HOPileInputTag_, mcc, *conditions, HODigiStorage_);
    convertPileupHcalDigis<HFDigiCollection> (*ep, HFPileInputTag_, mcc, *conditions, HFDigiStorage_);
    convertPileupHcalDigis<QIE10DigiCollection> (*ep, QIE10PileInputTag_, mcc, *conditions, QIE10DigiStorage_);
    convertPileupHcalDigis<QIE11DigiCollection> (*ep, QIE11PileInputTag_, mcc, *conditions, QIE11DigiStorage_);


    // ZDC Next

    if(DoZDC_) {


      std::shared_ptr<Wrapper<ZDCDigiCollection>  const> ZDCDigisPTR = getProductByTag<ZDCDigiCollection>(*ep, ZDCPileInputTag_, mcc);
 
      if(ZDCDigisPTR ) {

        const ZDCDigiCollection*  ZDCDigis = const_cast< ZDCDigiCollection * >(ZDCDigisPTR->product());

        LogDebug("DataMixingHcalDigiWorker") << "total # ZDC digis: " << ZDCDigis->size();

        // loop over digis, adding these to the existing maps
        for(ZDCDigiCollection::const_iterator it  = ZDCDigis->begin(); it != ZDCDigis->end(); ++it) {

          // calibration, for future reference:  (same block for all Hcal types) ZDC is different
          HcalZDCDetId cell = it->id();
          const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
          const HcalQIEShape* shape = conditions->getHcalShape (channelCoder); // this one is generic
          HcalCoderDb coder (*channelCoder, *shape);

          CaloSamples tool;
          coder.adc2fC((*it),tool);

          ZDCDigiStorage_.insert(ZDCDigiMap::value_type( (it->id()), tool ));

#ifdef DEBUG
          // Commented out because this does not compile anymore
          // LogDebug("DataMixingHcalDigiWorker") << "processed ZDCDigi with rawId: "
          //                                      << it->id() << "\n"
          //                                      << " digi energy: " << it->energy();
#endif
        }
      }
    }


  }
 
  void DataMixingHcalDigiWorker::putHcal(edm::Event &e,const edm::EventSetup& ES) {
    edm::ESHandle<HcalDbService> conditions;
    ES.get<HcalDbRecord>().get(conditions);

    // collection of digis to put in the event
    std::unique_ptr< HBHEDigiCollection > HBHEdigis = buildHcalDigis<HBHEDigiCollection> (HBHEDigiStorage_, *conditions);
    std::unique_ptr< HODigiCollection > HOdigis = buildHcalDigis<HODigiCollection> (HODigiStorage_, *conditions);
    std::unique_ptr< HFDigiCollection > HFdigis = buildHcalDigis<HFDigiCollection> (HFDigiStorage_, *conditions);
    std::unique_ptr< QIE10DigiCollection > QIE10digis = buildHcalDigis<QIE10DigiCollection> (QIE10DigiStorage_, *conditions);
    std::unique_ptr< QIE11DigiCollection > QIE11digis = buildHcalDigis<QIE11DigiCollection> (QIE11DigiStorage_, *conditions);
    std::unique_ptr< ZDCDigiCollection > ZDCdigis( new ZDCDigiCollection );

    // loop over the maps we have, re-making individual hits or digis if necessary.
    DetId formerID = 0;
    DetId currentID;

    double fC_new;
    double fC_old;
    double fC_sum;


    // ZDC next...

    // loop over the maps we have, re-making individual hits or digis if necessary.
    formerID = 0;
    CaloSamples ZDC_old;

    ZDCDigiMap::const_iterator iZDCchk;

    for(ZDCDigiMap::const_iterator iZDC  = ZDCDigiStorage_.begin(); iZDC != ZDCDigiStorage_.end(); ++iZDC) {

      currentID = iZDC->first; 

      if (currentID == formerID) { // we have to add these digis together

        //loop over digi samples in each CaloSample                                                           
        unsigned int sizenew = (iZDC->second).size();
        unsigned int sizeold = ZDC_old.size();

        unsigned int max_samp = std::max(sizenew, sizeold);

        CaloSamples ZDC_bigger(currentID,max_samp);

        bool usenew = false;

        if(sizenew > sizeold) usenew = true;

        // samples from different events can be of different lengths - sum all                               
        // that overlap.                                                                                     

        for(unsigned int isamp = 0; isamp<max_samp; isamp++) {
          if(isamp < sizenew) {
            fC_new = (iZDC->second)[isamp];
          }
          else { fC_new = 0;}

          if(isamp < sizeold) {
            fC_old = ZDC_old[isamp];
          }
          else { fC_old = 0;}

          // add values                                                                                      
          fC_sum = fC_new + fC_old;

          if(usenew) {ZDC_bigger[isamp] = fC_sum; }
          else { ZDC_old[isamp] = fC_sum; }  // overwrite old sample, adding new info     

        }
        if(usenew) ZDC_old = ZDC_bigger; // save new, larger sized sample in "old" slot
      
      }
      else {
        if(formerID>0) {
          // make new digi
          ZDCdigis->push_back(ZDCDataFrame(formerID));	  

          // set up information to convert back

          HcalZDCDetId cell = ZDC_old.id();
          const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
          const HcalQIEShape* shape = conditions->getHcalShape (channelCoder); // this one is generic         
          HcalCoderDb coder (*channelCoder, *shape);

          unsigned int sizeold = ZDC_old.size();
          for(unsigned int isamp = 0; isamp<sizeold; isamp++) {
            coder.fC2adc(ZDC_old,(ZDCdigis->back()), 1 );   // as per simulation, capid=1
          }
        }
        //save pointers for next iteration                                                                 
        formerID = currentID;
        ZDC_old = iZDC->second;
      }

      iZDCchk = iZDC;
      if((++iZDCchk) == ZDCDigiStorage_.end()) {  //make sure not to lose the last one                         
          // make new digi
          ZDCdigis->push_back(ZDCDataFrame(currentID));	  

          // set up information to convert back

          HcalZDCDetId cell = (iZDC->second).id();
          const HcalQIECoder* channelCoder = conditions->getHcalCoder (cell);
          const HcalQIEShape* shape = conditions->getHcalShape (channelCoder); // this one is generic         
          HcalCoderDb coder (*channelCoder, *shape);

          unsigned int sizeold = (iZDC->second).size();
          for(unsigned int isamp = 0; isamp<sizeold; isamp++) {
            coder.fC2adc(ZDC_old,(ZDCdigis->back()), 1 );   // as per simulation, capid=1
          }
      }
    }


  
   //done merging

    // put the collection of recunstructed hits in the event   
    LogInfo("DataMixingHcalDigiWorker") << "total # HBHE Merged digis: " << HBHEdigis->size() ;
    LogInfo("DataMixingHcalDigiWorker") << "total # HO Merged digis: " << HOdigis->size() ;
    LogInfo("DataMixingHcalDigiWorker") << "total # HF Merged digis: " << HFdigis->size() ;
    LogInfo("DataMixingHcalDigiWorker") << "total # QIE10 Merged digis: " << QIE10digis->size() ;
    LogInfo("DataMixingHcalDigiWorker") << "total # QIE11 Merged digis: " << QIE11digis->size() ;
    LogInfo("DataMixingHcalDigiWorker") << "total # ZDC Merged digis: " << ZDCdigis->size() ;


    e.put(std::move(HBHEdigis), HBHEDigiCollectionDM_);
    e.put(std::move(HOdigis), HODigiCollectionDM_);
    e.put(std::move(HFdigis), HFDigiCollectionDM_);
    e.put(std::move(QIE10digis), QIE10DigiCollectionDM_);
    e.put(std::move(QIE11digis), QIE11DigiCollectionDM_);
    e.put(std::move(ZDCdigis), ZDCDigiCollectionDM_);

    // clear local storage after this event
    HBHEDigiStorage_.clear();
    HODigiStorage_.clear();
    HFDigiStorage_.clear();
    QIE10DigiStorage_.clear();
    QIE11DigiStorage_.clear();
    ZDCDigiStorage_.clear();

  }

} //edm
