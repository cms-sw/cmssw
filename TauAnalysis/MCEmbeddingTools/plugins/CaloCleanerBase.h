#ifndef CaloCleanerBase_h
#define CaloCleanerBase_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/DetId/interface/DetId.h"

#include <map>
class CaloCleanerBase 
{

  public:
    virtual ~CaloCleanerBase(){};
    void setEvent(edm::Event& iEvent);

    virtual std::string name() = 0;

    virtual float cleanRH(DetId det, float energy) = 0;


    typedef std::map< uint32_t , float> TLenMap;



  protected:
    edm::Event * iEvent_;
    CaloCleanerBase(const edm::ParameterSet& config);

    edm::InputTag col1;
    edm::InputTag col2;
    edm::Handle< TLenMap > h1;
    edm::Handle< TLenMap > h2;

};


#endif
