// -*- C++ -*-
//
// Package:    test/EmbeddingLHEProducer
// Class:      EmbeddingLHEProducer
// 
/**\class EmbeddingLHEProducer EmbeddingLHEProducer.cc test/EmbeddingLHEProducer/plugins/EmbeddingLHEProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Stefan Wayand
//         Created:  Wed, 13 Jan 2016 08:15:01 GMT
//
//


// system include files
#include <algorithm>
#include <iostream>
#include <iterator>
#include <fstream>
#include <string>
#include <memory>
#include "TLorentzVector.h"
#include "boost/ptr_container/ptr_deque.hpp"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "SimDataFormats/GeneratorProducts/interface/LesHouches.h"
#include "SimDataFormats/GeneratorProducts/interface/LHECommonBlocks.h"
#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEXMLStringProduct.h"

#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"
#include "GeneratorInterface/LHEInterface/interface/LHEReader.h"



//
// class declaration
//


class EmbeddingLHEProducer : public edm::one::EDProducer<edm::BeginRunProducer,
                                                        edm::EndRunProducer> {
   public:
      explicit EmbeddingLHEProducer(const edm::ParameterSet&);
      ~EmbeddingLHEProducer();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void beginJob() override;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override;
      
      virtual void beginRunProduce(edm::Run& run, edm::EventSetup const& es) override;
      virtual void endRunProduce(edm::Run&, edm::EventSetup const&) override;

      void fill_lhe_from_mumu(TLorentzVector &positiveLepton, TLorentzVector &negativeLepton, lhef::HEPEUP &outlhe);
      void fill_lhe_with_particle(TLorentzVector &particle, double spin, int motherindex, int pdgid, int status, lhef::HEPEUP &outlhe);
      
      void transform_mumu_to_tautau(TLorentzVector &positiveLepton, TLorentzVector &negativeLepton);
      const reco::Candidate* find_original_muon(const reco::Candidate* muon);
      void assign_4vector(TLorentzVector &Lepton, const pat::Muon* muon, std::string FSRmode);
      void mirror(TLorentzVector &positiveLepton, TLorentzVector &negativeLepton);
      void rotate180(TLorentzVector &positiveLepton, TLorentzVector &negativeLepton);
      
      // ----------member data ---------------------------
      boost::shared_ptr<lhef::LHERunInfo>	runInfoLast;
      boost::shared_ptr<lhef::LHERunInfo>	runInfo;
      boost::shared_ptr<lhef::LHEEvent>	partonLevel;
      boost::ptr_deque<LHERunInfoProduct>	runInfoProducts;
      
      edm::EDGetTokenT<edm::View<pat::Muon>> muonsCollection_;
      edm::EDGetTokenT<reco::VertexCollection> vertexCollection_;
      bool switchToMuonEmbedding_;
      bool mirror_,rotate180_;
      const double tauMass_ = 1.77682;
      
      std::ofstream file;
      bool write_lheout;
      
      // instead of reconstruted 4vectors of muons generated are taken to study FSR. Possible modes:
      // afterFSR - muons without FSR photons taken into account
      // beforeFSR - muons with FSR photons taken into account
      std::string studyFSRmode_;
};

//
// constructors and destructor
//
EmbeddingLHEProducer::EmbeddingLHEProducer(const edm::ParameterSet& iConfig)
{
   //register your products
   produces<LHEEventProduct>();
   produces<LHERunInfoProduct, edm::InRun>();
   produces<math::XYZTLorentzVectorD>("vertexPosition");

   muonsCollection_ = consumes<edm::View<pat::Muon>>(iConfig.getParameter<edm::InputTag>("src"));
   vertexCollection_ = consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices"));
   switchToMuonEmbedding_ = iConfig.getParameter<bool>("switchToMuonEmbedding");
   mirror_ = iConfig.getParameter<bool>("mirror");
   rotate180_ = iConfig.getParameter<bool>("rotate180");
   studyFSRmode_ = iConfig.getUntrackedParameter<std::string>("studyFSRmode","");
   
   write_lheout=false;
   std::string lhe_ouputfile = iConfig.getUntrackedParameter<std::string>("lhe_outputfilename","");
   if (lhe_ouputfile !=""){
     write_lheout=true;
     file.open(lhe_ouputfile, std::fstream::out | std::fstream::trunc);
   }
}


EmbeddingLHEProducer::~EmbeddingLHEProducer()
{
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
EmbeddingLHEProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    using namespace edm;
    
    
    edm::Handle< edm::View<pat::Muon> > muonHandle;
    iEvent.getByToken(muonsCollection_, muonHandle);
    edm::View<pat::Muon> coll_muons = *muonHandle;
    
    Handle<std::vector<reco::Vertex>> coll_vertices;
    iEvent.getByToken(vertexCollection_ , coll_vertices);
    
    TLorentzVector positiveLepton, negativeLepton;
    bool mu_plus_found = false;
    bool mu_minus_found = false;
    lhef::HEPEUP hepeup;
    hepeup.IDPRUP = 0;
    hepeup.XWGTUP = 1;
    hepeup.SCALUP = -1;
    hepeup.AQEDUP = -1;
    hepeup.AQCDUP = -1;
    // Assuming Pt-Order
    for (edm::View<pat::Muon>::const_iterator muon=  coll_muons.begin(); muon!= coll_muons.end();  ++muon)
    {
      if (muon->charge() == 1 && !mu_plus_found)
      {
        assign_4vector(positiveLepton, &(*muon), studyFSRmode_);
        mu_plus_found = true;
      }
      else if (muon->charge() == -1 && !mu_minus_found)
      {
        assign_4vector(negativeLepton, &(*muon), studyFSRmode_);
        mu_minus_found = true;
      }
      else if (mu_minus_found && mu_plus_found) break;
    }
    mirror(positiveLepton,negativeLepton); // if no mirror, function does nothing.
    rotate180(positiveLepton,negativeLepton); // if no rotate180, function does nothing
    transform_mumu_to_tautau(positiveLepton,negativeLepton); // if MuonEmbedding, function does nothing.
    fill_lhe_from_mumu(positiveLepton,negativeLepton,hepeup);
    
    double originalXWGTUP_ = 1.;
    std::unique_ptr<LHEEventProduct> product( new LHEEventProduct(hepeup,originalXWGTUP_) );
    
    if (write_lheout) std::copy(product->begin(), product->end(), std::ostream_iterator<std::string>(file));
    
    iEvent.put(std::move(product));
    // Saving vertex position
    std::unique_ptr<math::XYZTLorentzVectorD> vertex_position (new math::XYZTLorentzVectorD(coll_vertices->at(0).x(),coll_vertices->at(0).y(),coll_vertices->at(0).z(),0.0));
    iEvent.put(std::move(vertex_position), "vertexPosition");

}

// ------------ method called once each job just before starting event loop  ------------
void 
EmbeddingLHEProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
EmbeddingLHEProducer::endJob() {
}

// ------------ method called when starting to processes a run  ------------

void
EmbeddingLHEProducer::beginRunProduce(edm::Run &run, edm::EventSetup const&)
{
    // fill HEPRUP common block and store in edm::Run
    lhef::HEPRUP heprup;
    
    // set number of processes: 1 for Z to tau tau
    heprup.resize(1);
    
    //Process independent information
    
    //beam particles ID (two protons)
    //heprup.IDBMUP.first = 2212;
    //heprup.IDBMUP.second = 2212;
    
    //beam particles energies (both 6.5 GeV)
    //heprup.EBMUP.first = 6500.;
    //heprup.EBMUP.second = 6500.;
    
    //take default pdf group for both beamparticles
    //heprup.PDFGUP.first = -1;
    //heprup.PDFGUP.second = -1;
    
    //take certan pdf set ID (same as in officially produced DYJets LHE files)
    //heprup.PDFSUP.first = -1;
    //heprup.PDFSUP.second = -1;
    
    //master switch for event weight iterpretation (same as in officially produced DYJets LHE files) 
    heprup.IDWTUP = 3;
    
    //Information for first process (Z to tau tau), for now only placeholder:
    heprup.XSECUP[0] = 1.;
    heprup.XERRUP[0] = 0;
    heprup.XMAXUP[0] = 1;
    heprup.LPRUP[0]= 0;
    
    std::unique_ptr<LHERunInfoProduct> runInfo(new LHERunInfoProduct(heprup));
    if (write_lheout)std::copy(runInfo->begin(), runInfo->end(),std::ostream_iterator<std::string>(file));
    
    run.put(std::move(runInfo));

}


void 
EmbeddingLHEProducer::endRunProduce(edm::Run& run, edm::EventSetup const& es)
{
    if (!runInfoProducts.empty()) {
        std::unique_ptr<LHERunInfoProduct> product(runInfoProducts.pop_front().release());
        run.put(std::move(product));
    }
    if (write_lheout) {
      file << LHERunInfoProduct::endOfFile();
      file.close();
    }
}

void 
EmbeddingLHEProducer::fill_lhe_from_mumu(TLorentzVector &positiveLepton, TLorentzVector &negativeLepton, lhef::HEPEUP &outlhe)
{
    TLorentzVector Z = positiveLepton + negativeLepton;
    int leptonPDGID = switchToMuonEmbedding_ ? 13 : 15;
    fill_lhe_with_particle(Z,9.0,0,23,2,outlhe);
    fill_lhe_with_particle(positiveLepton,1.0,1,-leptonPDGID,1,outlhe);
    fill_lhe_with_particle(negativeLepton,-1.0,1,leptonPDGID,1,outlhe);
    
    return;
}

void EmbeddingLHEProducer::fill_lhe_with_particle(TLorentzVector &particle, double spin, int motherindex, int pdgid, int status, lhef::HEPEUP &outlhe)
{
    // Pay attention to different index conventions:
    // 'particleindex' follows usual C++ index conventions starting at 0 for a list.
    // 'motherindex' follows the LHE index conventions: 0 is for 'not defined', so the listing starts at 1.
    // That means: LHE index 1 == C++ index 0.
    int particleindex = outlhe.NUP;
    outlhe.resize(outlhe.NUP+1);
    
    outlhe.PUP[particleindex][0] = particle.Px();
    outlhe.PUP[particleindex][1] = particle.Py();
    outlhe.PUP[particleindex][2] = particle.Pz();
    outlhe.PUP[particleindex][3] = particle.E();
    outlhe.PUP[particleindex][4] = particle.M();
    outlhe.SPINUP[particleindex] = spin;
    outlhe.ICOLUP[particleindex].first = 0;
    outlhe.ICOLUP[particleindex].second = 0;
    
    outlhe.MOTHUP[particleindex].first = motherindex;
    outlhe.MOTHUP[particleindex].second = motherindex;
    outlhe.IDUP[particleindex] = pdgid;
    outlhe.ISTUP[particleindex] = status;
    
    return;
}




void EmbeddingLHEProducer::transform_mumu_to_tautau(TLorentzVector &positiveLepton, TLorentzVector &negativeLepton)
{
    // No corrections applied for muon embedding
    if (switchToMuonEmbedding_) return;

    TLorentzVector Z = positiveLepton + negativeLepton;

    TVector3 boost_from_Z_to_LAB = Z.BoostVector();
    TVector3 boost_from_LAB_to_Z = -Z.BoostVector();

    // Boosting the two leptons to Z restframe, then both are back to back. This means, same 3-momentum squared
    positiveLepton.Boost(boost_from_LAB_to_Z);
    negativeLepton.Boost(boost_from_LAB_to_Z);

    // Energy of tau = 0.5*Z-mass
    double tau_mass_squared = tauMass_*tauMass_;
    double tau_energy_squared = 0.25*Z.M2();
    double tau_3momentum_squared = tau_energy_squared - tau_mass_squared;
    if (tau_3momentum_squared < 0)
    {
        std::cout << "3-Momentum squared is negative" << std::endl;
        return;
    }
    
    //Computing scale, applying it on the 3-momenta and building new 4 momenta of the taus
    double scale = std::sqrt(tau_3momentum_squared/positiveLepton.Vect().Mag2());
    positiveLepton.SetPxPyPzE(scale*positiveLepton.Px(),scale*positiveLepton.Py(),scale*positiveLepton.Pz(),std::sqrt(tau_energy_squared));
    negativeLepton.SetPxPyPzE(scale*negativeLepton.Px(),scale*negativeLepton.Py(),scale*negativeLepton.Pz(),std::sqrt(tau_energy_squared));

    //Boosting the new taus back to LAB frame
    positiveLepton.Boost(boost_from_Z_to_LAB);
    negativeLepton.Boost(boost_from_Z_to_LAB);

    return;
}

void EmbeddingLHEProducer::assign_4vector(TLorentzVector &Lepton, const pat::Muon* muon, std::string FSRmode)
{
    if("afterFSR" == FSRmode && muon->genParticle() != 0)
    {
        const reco::GenParticle* afterFSRMuon = muon->genParticle();
        Lepton.SetPxPyPzE(afterFSRMuon->p4().px(),afterFSRMuon->p4().py(),afterFSRMuon->p4().pz(), afterFSRMuon->p4().e());
    }
    else if ("beforeFSR" == FSRmode && muon->genParticle() != 0)
    {
        const reco::Candidate* beforeFSRMuon = find_original_muon(muon->genParticle());
        Lepton.SetPxPyPzE(beforeFSRMuon->p4().px(),beforeFSRMuon->p4().py(),beforeFSRMuon->p4().pz(), beforeFSRMuon->p4().e());
    }
    else
    {
        Lepton.SetPxPyPzE(muon->p4().px(),muon->p4().py(),muon->p4().pz(), muon->p4().e());
    }
    return;
}

const reco::Candidate* EmbeddingLHEProducer::find_original_muon(const reco::Candidate* muon)
{
    if(muon->mother(0) == 0) return muon;
    if(muon->pdgId() == muon->mother(0)->pdgId()) return find_original_muon(muon->mother(0));
    else return muon;
}

void EmbeddingLHEProducer::rotate180(TLorentzVector &positiveLepton, TLorentzVector &negativeLepton)
{
    if (!rotate180_) return;
    std::cout << "Applying 180° rotation" << std::endl;
    // By construction, the 3-momenta of mu-, mu+ and Z are in one plane. 
    // That means, one vector for perpendicular projection can be used for both leptons.
    TLorentzVector Z = positiveLepton + negativeLepton;

    std::cout << "MuMinus before. Pt: " << negativeLepton.Pt() << " Eta: " << negativeLepton.Eta() << " Phi: " << negativeLepton.Phi() << " Mass: " << negativeLepton.M() << std::endl;

    TVector3 Z3 = Z.Vect();
    TVector3 positiveLepton3 = positiveLepton.Vect();
    TVector3 negativeLepton3 = negativeLepton.Vect();

    TVector3 p3_perp = positiveLepton3 - positiveLepton3.Dot(Z3)/Z3.Dot(Z3)*Z3;
    p3_perp = p3_perp.Unit();

    positiveLepton3 -= 2*positiveLepton3.Dot(p3_perp)*p3_perp;
    negativeLepton3 -= 2*negativeLepton3.Dot(p3_perp)*p3_perp;

    positiveLepton.SetVect(positiveLepton3);
    negativeLepton.SetVect(negativeLepton3);

    std::cout << "MuMinus after. Pt: " << negativeLepton.Pt() << " Eta: " << negativeLepton.Eta() << " Phi: " << negativeLepton.Phi() << " Mass: " << negativeLepton.M() << std::endl;

    return;
}

void EmbeddingLHEProducer::mirror(TLorentzVector &positiveLepton, TLorentzVector &negativeLepton)
{
    if(!mirror_) return;
    std::cout << "Applying mirroring" << std::endl;
    TLorentzVector Z = positiveLepton + negativeLepton;

    std::cout << "MuMinus before. Pt: " << negativeLepton.Pt() << " Eta: " << negativeLepton.Eta() << " Phi: " << negativeLepton.Phi() << " Mass: " << negativeLepton.M() << std::endl;

    TVector3 Z3 = Z.Vect();
    TVector3 positiveLepton3 = positiveLepton.Vect();
    TVector3 negativeLepton3 = negativeLepton.Vect();

    TVector3 beam(0.,0.,1.);
    TVector3 perpToZandBeam = Z3.Cross(beam).Unit();

    positiveLepton3 -= 2*positiveLepton3.Dot(perpToZandBeam)*perpToZandBeam;
    negativeLepton3 -= 2*negativeLepton3.Dot(perpToZandBeam)*perpToZandBeam;

    positiveLepton.SetVect(positiveLepton3);
    negativeLepton.SetVect(negativeLepton3);

    std::cout << "MuMinus after. Pt: " << negativeLepton.Pt() << " Eta: " << negativeLepton.Eta() << " Phi: " << negativeLepton.Phi() << " Mass: " << negativeLepton.M() << std::endl;

    return;
}


// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
EmbeddingLHEProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(EmbeddingLHEProducer);
