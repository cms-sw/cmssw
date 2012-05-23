#ifndef MCTauValidation_h
#define MCTauValidation_h

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/JetReco/interface/GenJet.h"

#include "PhysicsTools/JetMCUtils/interface/JetMCTag.h"

#include <iostream>

class MCTauValidation {

public:
	MCTauValidation(const reco::Candidate*                                 ,
                    std::map<std::string,MonitorElement *>::const_iterator ,
                    std::string                                            ,
                    double                                                 ,
                    std::map<std::string,MonitorElement *>                 
	                );
	~MCTauValidation();

private:
	std::string                                            currentDiscriminatorLabel_ ;
	std::map<std::string,MonitorElement *>::const_iterator element_                   ;
	double                                                 tauPtRes_                  ;
    std::map<std::string,MonitorElement *>                 plotMapPrivate_            ;
};

MCTauValidation::MCTauValidation(const reco::Candidate                                   *gen_particle_             , 
                                 std::map<std::string,MonitorElement *>::const_iterator  element_                   , 
                                 std::string                                             currentDiscriminatorLabel_ ,
                                 double                                                  tauPtRes_                  ,
                                 std::map<std::string,MonitorElement *>                  plotMapPrivate_            
                                )
{
    std::cout << __PRETTY_FUNCTION__ << "]\t" << "[" << __LINE__ << "]\t" << "sono dentro" << std::endl ;
	const reco::GenJet* tauGenJet_ = dynamic_cast<const reco::GenJet*>(gen_particle_);
	if(tauGenJet_!=0){
		std::string genTauDecayMode_ =  JetMCTagUtils::genTauDecayMode(*tauGenJet_); // gen_particle is the tauGenJet matched to the reconstructed tau
		element_ = plotMapPrivate_.find( currentDiscriminatorLabel_ + "_pTRatio_" + genTauDecayMode_ );
		if( element_ != plotMapPrivate_.end() ) element_->second->Fill(tauPtRes_);
	}
	else{
	}

}

MCTauValidation::~MCTauValidation()
{
  
}

#endif
