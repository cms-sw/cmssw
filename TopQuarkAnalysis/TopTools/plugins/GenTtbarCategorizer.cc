// -*- C++ -*-
//
// Package:    TopQuarkAnalysis/TopTools
// Class:      GenTtbarCategorizer
// 
/**\class GenTtbarCategorizer GenTtbarCategorizer.cc TopQuarkAnalysis/TopTools/plugins/GenTtbarCategorizer.cc

 Description: Categorization of different tt+xx processes, returning unique ID for each process as e.g. tt+bb, tt+b, tt+2b, tt+cc, ...

 Implementation:
     
     The classification scheme returns an ID per event, and works as follows:
     
     All jets in the following need to be in the acceptance as given by the config parameters |eta|, pt.
     A c jet must contain at least one c hadron and should contain no b hadrons
     
     First, b jets from top are identified, i.e. jets containing a b hadron from t->b decay
     They are encoded in the ID as numberOfBjetsFromTop*100, i.e.
     0xx: no b jets from top in acceptance
     1xx: 1 b jet from top in acceptance
     2xx: both b jets from top in acceptance
     
     Then, b jets from W are identified, i.e. jets containing a b hadron from W->b decay
     They are encoded in the ID as numberOfBjetsFromW*1000, i.e.
     0xxx: no b jets from W in acceptance
     1xxx: 1 b jet from W in acceptance
     2xxx: 2 b jets from W in acceptance
     
     Then, c jets from W are identified, i.e. jets containing a c hadron from W->c decay, but no b hadrons
     They are encoded in the ID as numberOfCjetsFromW*10000, i.e.
     0xxxx: no c jets from W in acceptance
     1xxxx: 1 c jet from W in acceptance
     2xxxx: 2 c jets from W in acceptance
     
     From the remaining jets, the ID is formed based on the additional b jets (IDs x5x) and c jets (IDs x4x) in the following order:
     x55: at least 2 additional b jets with at least two of them having >= 2 b hadrons in each
     x54: at least 2 additional b jets with one of them having >= 2 b hadrons, the others having =1 b hadron
     x53: at least 2 additional b jets with all having =1 b hadron
     x52: exactly 1 additional b jet having >=2 b hadrons
     x51: exactly 1 additional b jet having =1 b hadron
     x45: at least 2 additional c jets with at least two of them having >= 2 c hadrons in each
     x44: at least 2 additional c jets with one of them having >= 2 c hadrons, the others having =1 c hadron
     x43: at least 2 additional c jets with all having =1 c hadron
     x42: exactly 1 additional c jet having >=2 c hadrons
     x41: exactly 1 additional c jet having =1 c hadron
     x00: No additional b or c jet, i.e. only light flavour jets or no additional jets
*/
//
// Original Author:  Johannes Hauk, Nazar Bartosik
//         Created:  Sun, 14 Jun 2015 19:42:58 GMT
//
//


// system include files
#include <memory>
#include <algorithm>
#include <functional>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"


//
// class declaration
//

class GenTtbarCategorizer : public edm::EDProducer {
    public:
        explicit GenTtbarCategorizer(const edm::ParameterSet&);
        ~GenTtbarCategorizer();
        
        static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
        
    private:
        virtual void beginJob() override;
        virtual void produce(edm::Event&, const edm::EventSetup&) override;
        virtual void endJob() override;
        
        //virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
        //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
        //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
        //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
        
        std::vector<int> nHadronsOrderedJetIndices(const std::map<int, int>& m_jetIndex);
        
        // ----------member data ---------------------------
        
        // Jet configuration
        const double genJetPtMin_;
        const double genJetAbsEtaMax_;
        
        // Input tags
        const edm::EDGetTokenT<reco::GenJetCollection> genJetsToken_;
        
        const edm::EDGetTokenT<std::vector<int> > genBHadJetIndexToken_;
        const edm::EDGetTokenT<std::vector<int> > genBHadFlavourToken_;
        const edm::EDGetTokenT<std::vector<int> > genBHadFromTopWeakDecayToken_;
        const edm::EDGetTokenT<std::vector<reco::GenParticle> > genBHadPlusMothersToken_;
        const edm::EDGetTokenT<std::vector<std::vector<int> > > genBHadPlusMothersIndicesToken_;
        const edm::EDGetTokenT<std::vector<int> > genBHadIndexToken_;
        const edm::EDGetTokenT<std::vector<int> > genBHadLeptonHadronIndexToken_;
        const edm::EDGetTokenT<std::vector<int> > genBHadLeptonViaTauToken_;
        
        const edm::EDGetTokenT<std::vector<int> > genCHadJetIndexToken_;
        const edm::EDGetTokenT<std::vector<int> > genCHadFlavourToken_;
        const edm::EDGetTokenT<std::vector<int> > genCHadFromTopWeakDecayToken_;
        const edm::EDGetTokenT<std::vector<int> > genCHadBHadronIdToken_;
        
        
};

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
GenTtbarCategorizer::GenTtbarCategorizer(const edm::ParameterSet& iConfig):
genJetPtMin_(iConfig.getParameter<double>("genJetPtMin")),
genJetAbsEtaMax_(iConfig.getParameter<double>("genJetAbsEtaMax")),
genJetsToken_(consumes<reco::GenJetCollection>(iConfig.getParameter<edm::InputTag>("genJets"))),
genBHadJetIndexToken_(consumes<std::vector<int> >(iConfig.getParameter<edm::InputTag>("genBHadJetIndex"))),
genBHadFlavourToken_(consumes<std::vector<int> >(iConfig.getParameter<edm::InputTag>("genBHadFlavour"))),
genBHadFromTopWeakDecayToken_(consumes<std::vector<int> >(iConfig.getParameter<edm::InputTag>("genBHadFromTopWeakDecay"))),
genBHadPlusMothersToken_(consumes<std::vector<reco::GenParticle> >(iConfig.getParameter<edm::InputTag>("genBHadPlusMothers"))),
genBHadPlusMothersIndicesToken_(consumes<std::vector<std::vector<int> > >(iConfig.getParameter<edm::InputTag>("genBHadPlusMothersIndices"))),
genBHadIndexToken_(consumes<std::vector<int> >(iConfig.getParameter<edm::InputTag>("genBHadIndex"))),
genBHadLeptonHadronIndexToken_(consumes<std::vector<int> >(iConfig.getParameter<edm::InputTag>("genBHadLeptonHadronIndex"))),
genBHadLeptonViaTauToken_(consumes<std::vector<int> >(iConfig.getParameter<edm::InputTag>("genBHadLeptonViaTau"))),
genCHadJetIndexToken_(consumes<std::vector<int> >(iConfig.getParameter<edm::InputTag>("genCHadJetIndex"))),
genCHadFlavourToken_(consumes<std::vector<int> >(iConfig.getParameter<edm::InputTag>("genCHadFlavour"))),
genCHadFromTopWeakDecayToken_(consumes<std::vector<int> >(iConfig.getParameter<edm::InputTag>("genCHadFromTopWeakDecay"))),
genCHadBHadronIdToken_(consumes<std::vector<int> >(iConfig.getParameter<edm::InputTag>("genCHadBHadronId")))
{
    produces<int>("genTtbarId");
}


GenTtbarCategorizer::~GenTtbarCategorizer()
{}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
GenTtbarCategorizer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    // Access gen jets
    edm::Handle<reco::GenJetCollection> genJets;
    iEvent.getByToken(genJetsToken_, genJets);
    
    
    // Access B hadrons information
    edm::Handle<std::vector<int> > genBHadFlavour;
    iEvent.getByToken(genBHadFlavourToken_, genBHadFlavour);
    
    edm::Handle<std::vector<int> > genBHadJetIndex;
    iEvent.getByToken(genBHadJetIndexToken_, genBHadJetIndex);
    
    edm::Handle<std::vector<int> > genBHadFromTopWeakDecay;
    iEvent.getByToken(genBHadFromTopWeakDecayToken_, genBHadFromTopWeakDecay);
    
    edm::Handle<std::vector<reco::GenParticle> > genBHadPlusMothers;
    iEvent.getByToken(genBHadPlusMothersToken_, genBHadPlusMothers);
    
    edm::Handle<std::vector<std::vector<int> > > genBHadPlusMothersIndices;
    iEvent.getByToken(genBHadPlusMothersIndicesToken_, genBHadPlusMothersIndices);
    
    edm::Handle<std::vector<int> > genBHadIndex;
    iEvent.getByToken(genBHadIndexToken_, genBHadIndex);
    
    edm::Handle<std::vector<int> > genBHadLeptonHadronIndex;
    iEvent.getByToken(genBHadLeptonHadronIndexToken_, genBHadLeptonHadronIndex);
    
    edm::Handle<std::vector<int> > genBHadLeptonViaTau;
    iEvent.getByToken(genBHadLeptonViaTauToken_, genBHadLeptonViaTau);
    
    
    // Access C hadrons information
    edm::Handle<std::vector<int> > genCHadFlavour;
    iEvent.getByToken(genCHadFlavourToken_, genCHadFlavour);
    
    edm::Handle<std::vector<int> > genCHadJetIndex;
    iEvent.getByToken(genCHadJetIndexToken_, genCHadJetIndex);
    
    edm::Handle<std::vector<int> > genCHadFromTopWeakDecay;
    iEvent.getByToken(genCHadFromTopWeakDecayToken_, genCHadFromTopWeakDecay);
    
    edm::Handle<std::vector<int> > genCHadBHadronId;
    iEvent.getByToken(genCHadBHadronIdToken_, genCHadBHadronId);
    
    
    // Map <jet index, number of specific hadrons in jet>
    // B jets with b hadrons directly from t->b decay
    std::map<int, int> bJetFromTopIds;
    // B jets with b hadrons from W->b decay
    std::map<int, int> bJetFromWIds;
    // C jets with c hadrons from W->c decay
    std::map<int, int> cJetFromWIds;
    // B jets with b hadrons before top quark decay chain
    std::map<int, int> bJetAdditionalIds;
    // C jets with c hadrons before top quark decay chain
    std::map<int, int> cJetAdditionalIds;
    
    
    // Count number of specific b hadrons in each jet
    for(size_t hadronId = 0; hadronId < genBHadIndex->size(); ++hadronId) {
        // Index of jet associated to the hadron
        const int jetIndex = genBHadJetIndex->at(hadronId);
        // Skip hadrons which have no associated jet
        if(jetIndex < 0) continue;
        // Skip if jet is not in acceptance
        if(genJets->at(jetIndex).pt() < genJetPtMin_) continue;
        if(std::fabs(genJets->at(jetIndex).eta()) > genJetAbsEtaMax_) continue;
        // Flavour of the hadron's origin
        const int flavour = genBHadFlavour->at(hadronId);
        // Jet from t->b decay [pdgId(top)=6]
        if(std::abs(flavour) == 6) {
            if(bJetFromTopIds.count(jetIndex) < 1) bJetFromTopIds[jetIndex] = 1;
            else bJetFromTopIds[jetIndex]++;
            continue;
        }
        // Jet from W->b decay [pdgId(W)=24]
        if(std::abs(flavour) == 24) {
            if(bJetFromWIds.count(jetIndex) < 1) bJetFromWIds[jetIndex] = 1;
            else bJetFromWIds[jetIndex]++;
            continue;
        }
        // Identify jets with b hadrons not from top-quark or W-boson decay
        if(bJetAdditionalIds.count(jetIndex) < 1) bJetAdditionalIds[jetIndex] = 1;
        else bJetAdditionalIds[jetIndex]++;
    }
    
    // Cleaning up b jets from W->b decays
    for(std::map<int, int>::iterator it = bJetFromWIds.begin(); it != bJetFromWIds.end(); ) {
        // Cannot be a b jet from t->b decay
        if(bJetFromTopIds.count(it->first) > 0) bJetFromWIds.erase(it++);
        else ++it;
    }
    
    // Cleaning up additional b jets
    for(std::map<int, int>::iterator it = bJetAdditionalIds.begin(); it != bJetAdditionalIds.end(); ) {
        // Cannot be a b jet from t->b decay
        if(bJetFromTopIds.count(it->first) > 0) bJetAdditionalIds.erase(it++);
        // Cannot be a b jet from W->b decay
        else if(bJetFromWIds.count(it->first) > 0) bJetAdditionalIds.erase(it++);
        else ++it;
    }
    
    // Count number of specific c hadrons in each c jet
    for(size_t hadronId = 0; hadronId < genCHadJetIndex->size(); ++hadronId) {
        // Index of jet associated to the hadron
        const int jetIndex = genCHadJetIndex->at(hadronId);
        // Skip hadrons which have no associated jet
        if(jetIndex < 0) continue;
        // Skip c hadrons that are coming from b hadrons
        if(genCHadBHadronId->at(hadronId) >= 0) continue;
        // Skip if jet is not in acceptance
        if(genJets->at(jetIndex).pt() < genJetPtMin_) continue;
        if(std::fabs(genJets->at(jetIndex).eta()) > genJetAbsEtaMax_) continue;
        // Skip if jet contains a b hadron
        if(bJetFromTopIds.count(jetIndex) > 0) continue;
        if(bJetFromWIds.count(jetIndex) > 0) continue;
        if(bJetAdditionalIds.count(jetIndex) > 0) continue;
        // Flavour of the hadron's origin
        const int flavour = genCHadFlavour->at(hadronId);
        // Jet from W->c decay [pdgId(W)=24]
        if(std::abs(flavour) == 24) {
            if(cJetFromWIds.count(jetIndex) < 1) cJetFromWIds[jetIndex] = 1;
            else cJetFromWIds[jetIndex]++;
            continue;
        }
        // Identify jets with c hadrons not from W-boson decay
        if(cJetAdditionalIds.count(jetIndex) < 1) cJetAdditionalIds[jetIndex] = 1;
        else cJetAdditionalIds[jetIndex]++;
    }

    // Cleaning up additional c jets
    for(std::map<int, int>::iterator it = cJetAdditionalIds.begin(); it != cJetAdditionalIds.end(); ) {
        // Cannot be a c jet from W->c decay
        if(cJetFromWIds.count(it->first) > 0) cJetAdditionalIds.erase(it++);
        else ++it;
    }
    
    // Categorize event based on number of additional b/c jets
    // and number of corresponding hadrons in each of them
    int additionalJetEventId = bJetFromTopIds.size()*100 + bJetFromWIds.size()*1000 + cJetFromWIds.size()*10000;
    // tt + 1 additional b jet
    if(bJetAdditionalIds.size() == 1){
        const int nHadronsInJet = bJetAdditionalIds.begin()->second;
        // tt + 1 additional b jet from 1 additional b hadron
        if(nHadronsInJet == 1) additionalJetEventId += 51;
        // tt + 1 additional b jet from >=2 additional b hadrons
        else additionalJetEventId += 52;
    }
    // tt + >=2 additional b jets
    else if(bJetAdditionalIds.size() > 1){
        // Check first two additional b jets (rare cases could have more)
        const std::vector<int> orderedJetIndices = nHadronsOrderedJetIndices(bJetAdditionalIds);
        int nHadronsInJet1 = bJetAdditionalIds.at(orderedJetIndices.at(0));
        int nHadronsInJet2 = bJetAdditionalIds.at(orderedJetIndices.at(1));
        // tt + >=2 additional b jets each from 1 additional b hadron
        if(std::max(nHadronsInJet1, nHadronsInJet2) == 1) additionalJetEventId += 53;
        // tt + >=2 additional b jets one of which from >=2 additional b hadrons
        else if(std::min(nHadronsInJet1, nHadronsInJet2) == 1 && std::max(nHadronsInJet1, nHadronsInJet2) > 1) additionalJetEventId += 54;
        // tt + >=2 additional b jets each from >=2 additional b hadrons
        else if(std::min(nHadronsInJet1, nHadronsInJet2) > 1) additionalJetEventId += 55;
    }
    // tt + no additional b jets
    else{
        // tt + 1 additional c jet
        if(cJetAdditionalIds.size() == 1){
            const int nHadronsInJet = cJetAdditionalIds.begin()->second;
            // tt + 1 additional c jet from 1 additional c hadron
            if(nHadronsInJet == 1) additionalJetEventId += 41;
            // tt + 1 additional c jet from >=2 additional c hadrons
            else additionalJetEventId += 42;
        }
        // tt + >=2 additional c jets
        else if(cJetAdditionalIds.size() > 1){
            // Check first two additional c jets (rare cases could have more)
            const std::vector<int> orderedJetIndices = nHadronsOrderedJetIndices(cJetAdditionalIds);
            int nHadronsInJet1 = cJetAdditionalIds.at(orderedJetIndices.at(0));
            int nHadronsInJet2 = cJetAdditionalIds.at(orderedJetIndices.at(1));
            // tt + >=2 additional c jets each from 1 additional c hadron
            if(std::max(nHadronsInJet1, nHadronsInJet2) == 1) additionalJetEventId += 43;
            // tt + >=2 additional c jets one of which from >=2 additional c hadrons
            else if(std::min(nHadronsInJet1, nHadronsInJet2) == 1 && std::max(nHadronsInJet1, nHadronsInJet2) > 1) additionalJetEventId += 44;
            // tt + >=2 additional c jets each from >=2 additional c hadrons
            else if(std::min(nHadronsInJet1, nHadronsInJet2) > 1) additionalJetEventId += 45;
        }
        // tt + no additional c jets
        else{
            // tt + light jets
            additionalJetEventId += 0;
        }
    }
    
    std::auto_ptr<int> ttbarId(new int);
    *ttbarId = additionalJetEventId;
    iEvent.put(ttbarId, "genTtbarId");
}

// ------------ method called once each job just before starting event loop  ------------
void 
GenTtbarCategorizer::beginJob()
{}

// ------------ method called once each job just after ending the event loop  ------------
void 
GenTtbarCategorizer::endJob()
{}

// ------------ method called when starting to processes a run  ------------
/*
void
GenTtbarCategorizer::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when ending the processing of a run  ------------
/*
void
GenTtbarCategorizer::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when starting to processes a luminosity block  ------------
/*
void
GenTtbarCategorizer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
GenTtbarCategorizer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method returns a vector of jet indices from the given map, sorted by N hadrons in descending order  ------------
std::vector<int> GenTtbarCategorizer::nHadronsOrderedJetIndices(const std::map<int, int>& m_jetIndex)
{
    const int nElements = m_jetIndex.size();
    std::vector<std::pair<int, int> > v_jetNhadIndexPair;
    v_jetNhadIndexPair.reserve(nElements);
    for(std::map<int, int>::const_iterator it = m_jetIndex.begin(); it != m_jetIndex.end(); ++it) {
        const int jetIndex = it->first;
        const int nHadrons = it->second;
        v_jetNhadIndexPair.push_back( std::pair<int, int>(nHadrons, jetIndex) );
    }
    // Sorting the vector of pairs by their key value
    std::sort(v_jetNhadIndexPair.begin(), v_jetNhadIndexPair.end(), std::greater<std::pair<int, int> >());
    // Building the vector of indices in the proper order
    std::vector<int> v_orderedJetIndices;
    v_orderedJetIndices.reserve(nElements);
    for(std::vector<std::pair<int, int> >::const_iterator it = v_jetNhadIndexPair.begin(); it != v_jetNhadIndexPair.end(); ++it) {
        v_orderedJetIndices.push_back(it->second);
    }
    
    return v_orderedJetIndices;
}
 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
GenTtbarCategorizer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(GenTtbarCategorizer);
