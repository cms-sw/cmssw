#ifndef CastorTechTrigProducer_CastorTTRecord_h
#define CastorTechTrigProducer_CastorTTRecord_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

class CastorTTRecord : public edm::EDProducer
{
public:

    explicit CastorTTRecord(const edm::ParameterSet& ps);
    virtual ~CastorTTRecord();
    
    virtual void produce(edm::Event& e, const edm::EventSetup& c);

    // get fC from digis and save it to array double energy[16 sectors][14 modules]
    void getEnergy_fC(double energy[16][14], edm::Handle<CastorDigiCollection>& CastorDigiColl,
    				  edm::Event& e, const edm::EventSetup& c);

    // get Trigger decisions | vector needs same SIZE and ORDER as in 'ttpBits_'
    void getTriggerDecisions(std::vector<bool>& decision, double energy[16][14]) const;

    // get Trigger decisions for every octant | vector has size of 6 -> 6 HTR card bits
    void getTriggerDecisionsPerOctant(std::vector<bool> tdps[16], double energy[16][14]) const;

private:
    
    edm::EDGetTokenT<CastorDigiCollection> CastorDigiColl_;
    unsigned int CastorSignalTS_;

    std::vector<unsigned int> ttpBits_ ;
    std::vector<std::string> TrigNames_ ; 
   	std::vector<double> TrigThresholds_ ;

    double reweighted_gain;
};

#endif


