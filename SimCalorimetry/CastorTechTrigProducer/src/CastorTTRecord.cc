#include "SimCalorimetry/CastorTechTrigProducer/src/CastorTTRecord.h"

#include "DataFormats/HcalDigi/interface/HcalTTPDigi.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtTechnicalTriggerRecord.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CalibFormats/CastorObjects/interface/CastorCoderDb.h"
#include "CalibFormats/CastorObjects/interface/CastorCalibrations.h"
#include "CalibFormats/CastorObjects/interface/CastorDbService.h"
#include "CalibFormats/CastorObjects/interface/CastorDbRecord.h"

CastorTTRecord::CastorTTRecord(const edm::ParameterSet& ps) 
{
    CastorDigiColl_ = consumes<CastorDigiCollection>(ps.getParameter<edm::InputTag>("CastorDigiCollection")) ;
    CastorSignalTS_ = ps.getParameter< unsigned int >("CastorSignalTS") ;

    ttpBits_        = ps.getParameter< std::vector<unsigned int> >("ttpBits");
    TrigNames_      = ps.getParameter< std::vector<std::string> >("TriggerBitNames");
    TrigThresholds_ = ps.getParameter< std::vector<double> >("TriggerThresholds");

    reweighted_gain = 1.0;

    produces<L1GtTechnicalTriggerRecord>();
}


CastorTTRecord::~CastorTTRecord() {
}

void CastorTTRecord::produce(edm::Event& e, const edm::EventSetup& eventSetup) {

    // std::cerr << "**** RUNNING THROUGH CastorTTRecord::produce" << std::endl;

    std::vector<L1GtTechnicalTrigger> vecTT(ttpBits_.size()) ;

    // Get Inputs
    edm::Handle<CastorDigiCollection> CastorDigiColl ; 
    e.getByToken(CastorDigiColl_,CastorDigiColl) ;

    if ( !CastorDigiColl.failedToGet() ) { 

        double cas_efC[16][14];
        getEnergy_fC(cas_efC,CastorDigiColl,e,eventSetup);


        std::vector<bool> decision(ttpBits_.size());

        getTriggerDecisions(decision,cas_efC);

        for(unsigned int i=0; i<ttpBits_.size(); i++) {
            // if( decision.at(i) ) std::cerr << "**** Something Triggered" << std::endl;
            // std::cout << "Run CastorTTRecord::produce. TriggerBit = " << ttpBits_.at(i) << "; TriggerName = " << TrigNames_.at(i) << "; Decision = " << decision[i] << std::endl;
            vecTT.at(i) = L1GtTechnicalTrigger(TrigNames_.at(i), ttpBits_.at(i), 0, decision.at(i)) ;
        }

    } else {
        vecTT.clear() ;
    }

    // Put output into event
    std::auto_ptr<L1GtTechnicalTriggerRecord> output(new L1GtTechnicalTriggerRecord()) ;
    output->setGtTechnicalTrigger(vecTT) ;    
    e.put(output) ;
}


void CastorTTRecord::getEnergy_fC(double energy[16][14], edm::Handle<CastorDigiCollection>& CastorDigiColl,
                                  edm::Event& e, const edm::EventSetup& eventSetup)
{
    // std::cerr << "**** RUNNING THROUGH CastorTTRecord::getEnergy_fC" << std::endl;

    // Get Conditions
    edm::ESHandle<CastorDbService> conditions ;
    eventSetup.get<CastorDbRecord>().get(conditions) ;
    const CastorQIEShape* shape = conditions->getCastorShape () ; // this one is generic

    for(int isec=0; isec<16; isec++) for(int imod=0; imod<14; imod++) energy[isec][imod] = 0;

    // Loop over digis
    CastorDigiCollection::const_iterator idigi ;
    for (idigi=CastorDigiColl->begin(); idigi!=CastorDigiColl->end(); idigi++) {
        const CastorDataFrame & digi = (*idigi) ;
        HcalCastorDetId cell = digi.id() ;

        // Get Castor Coder
        const CastorQIECoder* channelCoder = conditions->getCastorCoder(cell);
        CastorCoderDb coder (*channelCoder, *shape);

        // Get Castor Calibration
        const CastorCalibrations& calibrations=conditions->getCastorCalibrations(cell);

        // convert adc to fC
        CaloSamples tool ;
        coder.adc2fC(digi,tool) ;

        // pedestal substraction
        int capid=digi[CastorSignalTS_].capid();
        double fC = tool[CastorSignalTS_] - calibrations.pedestal(capid);

        // to correct threshold levels in fC for different gains
        reweighted_gain = calibrations.gain(capid) / 0.015;
        
        energy[digi.id().sector()-1][digi.id().module()-1] = fC;
    }
}

void CastorTTRecord::getTriggerDecisions(std::vector<bool>& decision, double energy[16][14]) const
{
    // std::cerr << "**** RUNNING THROUGH CastorTTRecord::getTriggerDecisions" << std::endl;

    // check if number of bits is at least four
    if( decision.size() < 4 ) return;

    std::vector<bool> tdpo[8]; // TriggerDecisionsPerOctant
    getTriggerDecisionsPerOctant(tdpo,energy);


    // preset trigger decisions
    decision.at(0) = true;
    decision.at(1) = false;
    decision.at(2) = false;
    decision.at(3) = false;

    // canceld for low pt jet
    // bool EM_decision = false;
    // bool HAD_decision = false;
    // loop over castor octants
    for(int ioct=0; ioct<8; ioct++) {
        int next_oct = (ioct+1)%8;
        int prev_oct = (ioct+8-1)%8;

        // gap Trigger
        if( !tdpo[ioct].at(0) ) decision.at(0) = false;
        if( !tdpo[ioct].at(1) ) decision.at(0) = false;

        // jet Trigger
        if( tdpo[ioct].at(2) ) decision.at(1) = true;

        // electron
        // canceld for low pt jet
        // if( tdpo[ioct].at(3) ) EM_decision = true;
        // if( tdpo[ioct].at(4) ) HAD_decision = true;

        // iso muon
        if( tdpo[ioct].at(5) ) {
            // was one of the other sectors 
            // in the octant empty ?
            if( tdpo[ioct].at(0) ) {
                if( tdpo[prev_oct].at(1) && 
                    tdpo[next_oct].at(0) && 
                    tdpo[next_oct].at(1) )
                    decision.at(3) = true;
            }
            else if( tdpo[ioct].at(1) ) {
                if( tdpo[prev_oct].at(0) && 
                    tdpo[prev_oct].at(1) && 
                    tdpo[next_oct].at(0) )
                    decision.at(3) = true;
            }
            // when not no iso muon
        }

        // low pt jet Trigger
        if( tdpo[ioct].at(6) ) decision.at(2) = true;
    }

    // for EM Trigger whole castor not hadronic and somewhere EM
    // canceld for low pt jet
    // decision.at(2) = EM_decision && !HAD_decision;
}

void CastorTTRecord::getTriggerDecisionsPerOctant(std::vector<bool> tdpo[8], double energy[16][14]) const
{
    // std::cerr << "**** RUNNING THROUGH CastorTTRecord::getTriggerDecisionsPerOctant" << std::endl;

    // loop over octatants
    for(int ioct=0; ioct<8; ioct++)
    {
        // six bits from HTR card
        // 0. first sector empty
        // 1. second sector empty
        // 2. jet any sector
        // 3. EM any sector
        // 4. HAD any sector
        // 5. muon any sector
        // add instead of EM Trigger (not bit 6 in real)
        // 6. low pt jet any sector
        tdpo[ioct].resize(7);

        for(int ibit=0; ibit<7; ibit++)
            tdpo[ioct].at(ibit) = false;

        // loop over castor sectors in octant
        for(int ioctsec=0; ioctsec<2; ioctsec++)
        {
            // absolute sector number
            int isec = 2*ioct+ioctsec;

            // init module sums for every sector
            double fCsum_mod = 0;
            double fCsum_em = 0, fCsum_ha = 0;
            double fCsum_jet_had = 0;
            double fCsum_col[3] = { 0, 0, 0 };
            
            // loop over modules
            for(int imod=0; imod<14; imod++) { 
                // total sum
                fCsum_mod += energy[isec][imod];
                
                // EM & HAD sum
                if( imod < 2 ) fCsum_em += energy[isec][imod];
                if( imod > 2 && imod < 12 ) fCsum_ha += energy[isec][imod];
                
                // sum over three sector parts
                if( imod < 4 )       fCsum_col[0] += energy[isec][imod];
                else if( imod < 8 )  fCsum_col[1] += energy[isec][imod];
                else if( imod < 12 ) fCsum_col[2] += energy[isec][imod];

                // HAD sum for jet trigger v2
                if( imod > 1 && imod < 5 ) fCsum_jet_had += energy[isec][imod];
            }

            // gap Trigger
            if( fCsum_mod < TrigThresholds_.at(0) ) {
                if( ioctsec == 0 ) tdpo[ioct].at(0) = true;
                else if( ioctsec == 1 ) tdpo[ioct].at(1) = true;
            }

            // jet Trigger 
            // with gain correction
            /* old version of jet trigger ( deprecated because of saturation )
            if( fCsum_mod > TrigThresholds_.at(1) / reweighted_gain ) 
                tdpo[ioct].at(2) = true; 
            */
            if( fCsum_jet_had > TrigThresholds_.at(1) / reweighted_gain )
            // additional high threshold near saturation for EM part
                if( energy[isec][0] > 26000 / reweighted_gain && energy[isec][1] > 26000 / reweighted_gain )
                    tdpo[ioct].at(2) = true;

            // low pt jet Trigger
            if( fCsum_mod > TrigThresholds_.at(5) / reweighted_gain ) 
                tdpo[ioct].at(6) = true; 

            // egamma Trigger
            // with gain correction only in the EM threshold
            if( fCsum_em > TrigThresholds_.at(2) / reweighted_gain )
                tdpo[ioct].at(3) = true;
            if( fCsum_ha > TrigThresholds_.at(3) ) 
                tdpo[ioct].at(4) = true; 

            // muon Trigger
            int countColumns = 0;
            for( int icol=0; icol<3; icol++ ) 
                if( fCsum_col[icol] > TrigThresholds_.at(4) ) 
                    countColumns++;
            if( countColumns >= 2 )
                tdpo[ioct].at(5) = true;
        }
    }
}
