#ifndef CaloCleanerBase_h
#define CaloCleanerBase_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "DataFormats/Candidate/interface/CompositeCandidate.h"


#include <map>
class CaloCleanerBase 
{

  public:
    virtual ~CaloCleanerBase(){};
    virtual void setEvent(edm::Event& iEvent);

    virtual std::string name() = 0;

    /// returns energy correction for rechit. Negative value - remove rh completely.
    virtual float cleanRH(DetId det, float energy) = 0;


    typedef std::map< uint32_t , float> TLenMap;



  protected:
    edm::Event * iEvent_;
    CaloCleanerBase(const edm::ParameterSet& config);

    edm::InputTag colPlus;
    edm::InputTag colMinus;
    edm::Handle< TLenMap > hPlus;
    edm::Handle< TLenMap > hMinus;

    edm::InputTag muons_;
    edm::Handle< std::vector< reco::CompositeCandidate > > ZmumuHandle_;

};


#endif
