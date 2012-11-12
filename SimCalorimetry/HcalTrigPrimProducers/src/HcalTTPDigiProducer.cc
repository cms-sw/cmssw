#include "SimCalorimetry/HcalTrigPrimProducers/src/HcalTTPDigiProducer.h"

#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "CalibFormats/HcalObjects/interface/HcalTPGRecord.h"
#include "CalibFormats/HcalObjects/interface/HcalTPGCoder.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <cstdio>

// DO NOT MODIFY: Mapping between iphi (array index) and TTP input (value) for HF
const int HcalTTPDigiProducer::inputs_[] = { 30,66,4,44,4,44,0,68,
                                             0,68,16,48,16,48,6,46,
                                             6,46,2,70,2,70,18,50,
                                             18,50,12,40,12,40,8,52,
                                             8,52,20,36,20,36,14,42,
                                             14,42,10,54,10,54,22,38,
                                             22,38,24,56,24,56,32,60,
                                             32,60,28,64,28,64,26,58,
                                             26,58,34,62,34,62,30,66 } ;

HcalTTPDigiProducer::HcalTTPDigiProducer(const edm::ParameterSet& ps) 
{
    hfDigis_        = ps.getParameter<edm::InputTag>("HFDigiCollection") ; 
    maskedChannels_ = ps.getParameter< std::vector<unsigned int> >("maskedChannels") ;
    bit_[0] = ps.getParameter<std::string>("defTT8") ; 
    bit_[1] = ps.getParameter<std::string>("defTT9") ; 
    bit_[2] = ps.getParameter<std::string>("defTT10") ; 
    bit_[3] = ps.getParameter<std::string>("defTTLocal") ; 

    for (int i=0; i<4; i++) { 
        nHits_[i] = -1 ; nHFp_[i] = -1 ; nHFm_[i] = -1 ; 
        pReq_[i] = ' ' ; mReq_[i] = ' ' ; pmLogic_[i] = ' ' ;
        calc_[i] = sscanf(bit_[i].c_str(),"hits>=%d:hfp%c=%d%chfm%c=%d",
                          &(nHits_[i]),&(pReq_[i]),&(nHFp_[i]),
                          &(pmLogic_[i]),&(mReq_[i]),&(nHFm_[i])) ;
        if ( calc_[i] == 1 ) {
            if ( nHits_[i] < 0 )
                throw cms::Exception("HcalTTPDigiProducer")
                    << "Unable to read logic for technical trigger" ;
        } else if ( calc_[i] == 6 ) {
            if ( nHits_[i] < 0 || nHFp_[i] < 0 || nHFm_[i] < 0 )
                throw cms::Exception("HcalTTPDigiProducer")
                    << "Unable to read logic for technical trigger" ;
            if ( (pReq_[i] != '>' && pReq_[i] != '<') || 
                 (mReq_[i] != '>' && mReq_[i] != '<') ||
                 (pmLogic_[i] != ':' && pmLogic_[i] != '|') )
                throw cms::Exception("HcalTTPDigiProducer")
                    << "Technical Trigger logic must obey the following format:\n"
                    "\"hits>=[A1]:hfp[B1]=[A2][C]hfm[B2]=[A3]\",\n"
                    "or \"hits>=[A1]\",\n"
                    "with A# >= 0, B# = (</>) and C = (:/|)" ;
        } else {
            throw cms::Exception("HcalTTPDigiProducer")
                << "Unable to read logic for technical trigger" ;
        }
    }

    id_         = ps.getUntrackedParameter<int>("id",-1) ; 
    samples_    = ps.getParameter<int>("samples") ; 
    presamples_ = ps.getParameter<int>("presamples") ; 
    iEtaMin_    = ps.getParameter<int>("iEtaMin") ; 
    iEtaMax_    = ps.getParameter<int>("iEtaMax") ;
    threshold_  = ps.getParameter<unsigned int>("threshold") ;
    fwAlgo_     = ps.getParameter<int>("fwAlgorithm") ;

    SoI_ = ps.getParameter<int>("HFSoI") ;

    if ( samples_ > 8 ) {
        samples_ = 8 ;
        edm::LogWarning("HcalTTPDigiProducer") << "Samples forced to maximum value of 8" ; 
    }
    if ( presamples_ - SoI_ > 0 ) { // Too many presamples
        presamples_ = SoI_ ; 
        edm::LogWarning("HcalTTPDigiProducer") << "Presamples reset to HF SoI value" ; 
    }
        
    produces<HcalTTPDigiCollection>();
}


HcalTTPDigiProducer::~HcalTTPDigiProducer() {
}

bool HcalTTPDigiProducer::isMasked(HcalDetId id) {

    for ( unsigned int i=0; i<maskedChannels_.size(); i++ ) 
        if ( id.rawId() == maskedChannels_.at(i) ) return true ;
    return false ; 
}

bool HcalTTPDigiProducer::decision(int nP, int nM, int bit) {

    bool pOK = false ; bool mOK = false ;    
    if ( (nP + nM) < nHits_[bit] ) return false ;
    if ( calc_[bit] == 1 ) return ( (nP + nM) >= nHits_[bit] ) ;

    if ( pReq_[bit] == '>' ) pOK = ( nP >= nHFp_[bit] ) ; 
    else if ( pReq_[bit] == '<' ) pOK = ( nP <= nHFp_[bit] ) ; 

    if ( mReq_[bit] == '>' ) mOK = ( nM >= nHFm_[bit] ) ; 
    else if ( mReq_[bit] == '<' ) mOK = ( nM <= nHFm_[bit] ) ; 

    if ( pmLogic_[bit] == ':' ) return ( pOK && mOK ) ;
    else if ( pmLogic_[bit] == '|' ) return ( pOK || mOK ) ;
    
    // Should not ever get here...need to create a warning message
    edm::LogWarning("HcalTTPDigiProducer") << "Trigger logic exhausted.  Returning false" ; 
    return false ; 
}

void HcalTTPDigiProducer::produce(edm::Event& e, const edm::EventSetup& eventSetup) {
    
    // Step A: Get Inputs
    edm::Handle<HFDigiCollection> hfDigiCollection ; 
    e.getByLabel(hfDigis_,hfDigiCollection) ;
    edm::ESHandle<HcalTPGCoder> inputCoder ;
    eventSetup.get<HcalTPGRecord>().get(inputCoder) ;

    // Step B: Create empty output
    std::auto_ptr<HcalTTPDigiCollection> ttpResult(new HcalTTPDigiCollection()) ; 
    
    // Step C: Compute TTP inputs
    uint16_t trigInputs[40] ;
    int nP[8] ; int nM[8] ;
    for (int i=0; i<8; i++) {
        nP[i] = 0 ; nM[i] = 0 ; 
        for (int j=0; j<5; j++) trigInputs[j*8+i] = 0 ;
    }
    for (HFDigiCollection::const_iterator theDigi=hfDigiCollection->begin();
         theDigi!=hfDigiCollection->end(); theDigi++) {
        HcalDetId id = HcalDetId(theDigi->id()) ;
        if ( isMasked(id) ) continue ;
        if ( id.ietaAbs() < iEtaMin_ || id.ietaAbs() > iEtaMax_ ) continue ; 

        IntegerCaloSamples samples(id,theDigi->size()) ; 
        inputCoder->adc2Linear(*theDigi,samples) ;

        for (int relSample=-presamples_; relSample<(samples_-presamples_); relSample++) {
            if ( samples[SoI_+relSample] >= threshold_ ) {
                int linSample = presamples_ + relSample ;
                int offset = (-1+id.zside())/2 ; 
                int shift = inputs_[id.iphi()+offset] ;
                int group = 0 ;
                while ( shift >= 16 ) { shift -= 16 ; group++ ; }
                if ( !(trigInputs[(linSample*8)+group]&(1<<shift)) ) 
                    ( id.ieta() > 0 ) ? ( nP[linSample]++) : ( nM[linSample]++ ) ; 
                trigInputs[(linSample*8)+group] |= (1<<shift) ;
            }
        }
    }

    // Step D: Compute trigger decision and fill TTP digi 
    uint8_t trigOutput[8] ;
    uint32_t algoDepBits[8] ;
    HcalTTPDigi ttpDigi(id_,samples_,presamples_,0,fwAlgo_,0) ;
    for (int linSample=0; linSample<8; linSample++) {
        trigOutput[linSample] = 0 ; algoDepBits[linSample] = 0 ;
        if ( linSample<samples_) {
            for (int j=0; j<4; j++) 
                trigOutput[linSample] |= (decision(nP[linSample],nM[linSample],j)<<j) ;
            int nT = nP[linSample] + nM[linSample] ;

            // Algorithm Dependent bits for FW flavor = 1
            // NOTE: this disagrees with the fw var. names that implies (LSB) T,M,P (MSB)
            if ( fwAlgo_ == 1 ) algoDepBits[linSample] = (nT&0x7F) | ((nP[linSample]&0x3F)<<7) | ((nM[linSample]&0x3F)<<13) ; 
            ttpDigi.setSample((linSample-presamples_),&trigInputs[linSample*8],algoDepBits[linSample],trigOutput[linSample]) ;
        }
    }    
    ttpResult->push_back( ttpDigi ) ;
    
    // Step E: Put outputs into event
    e.put(ttpResult);
}

