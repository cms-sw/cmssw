#include "TauAnalysis/MCEmbeddingTools/plugins/L1ExtraMEtMixerPlugin.h"
#include "TauAnalysis/MCEmbeddingTools/interface/embeddingAuxFunctions.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"
#include "DataFormats/Common/interface/Handle.h"

#include <TMath.h>

#include <vector>
#include <set>
#include <algorithm>

L1ExtraMEtMixerPlugin::L1ExtraMEtMixerPlugin(const edm::ParameterSet& cfg)
  : L1ExtraMixerPluginBase(cfg),
    srcMuons_(cfg.getParameter<edm::InputTag>("srcMuons")),
    srcDistanceMapMuPlus_(cfg.getParameter<edm::InputTag>("distanceMapMuPlus")),
    srcDistanceMapMuMinus_(cfg.getParameter<edm::InputTag>("distanceMapMuMinus")),
    sfAbsEtaLt12_(cfg.getParameter<double>("H_Calo_AbsEtaLt12")),
    sfAbsEta12to17_(cfg.getParameter<double>("H_Calo_AbsEta12to17")),
    sfAbsEtaGt17_(cfg.getParameter<double>("H_Calo_AbsEtaGt17"))
{}

void L1ExtraMEtMixerPlugin::registerProducts(edm::EDProducer& producer)
{
  producer.produces<l1extra::L1EtMissParticleCollection>(instanceLabel_); 
}

void L1ExtraMEtMixerPlugin::produce(edm::Event& evt, const edm::EventSetup& es)
{
  //std::cout << "<L1ExtraMEtMixerPlugin::produce>:" << std::endl;
  //std::cout << " src1 = " << src1_ << std::endl;
  //std::cout << " src2 = " << src2_ << std::endl;
  //std::cout << " instanceLabel = " << instanceLabel_ << std::endl;

  edm::Handle<l1extra::L1EtMissParticleCollection> met1;
  evt.getByLabel(src1_, met1);
  
  edm::Handle<l1extra::L1EtMissParticleCollection> met2;
  evt.getByLabel(src2_, met2);

  edm::Handle<reco::CandidateCollection> muons;
  evt.getByLabel(srcMuons_, muons);

  edm::Handle<detIdToFloatMap> distanceMapMuPlus;
  evt.getByLabel(srcDistanceMapMuPlus_, distanceMapMuPlus);
  edm::Handle<detIdToFloatMap> distanceMapMuMinus;
  evt.getByLabel(srcDistanceMapMuMinus_, distanceMapMuMinus);

  std::auto_ptr<l1extra::L1EtMissParticleCollection> metSum(new l1extra::L1EtMissParticleCollection());

  // CV: entries in MET collections refer to different bunch-crossings.
  //     The number of entries in the two MET collections is not necessarily the same
  //    --> match entries by bunch-crossing number
  std::set<int> bx_set;
  for ( l1extra::L1EtMissParticleCollection::const_iterator met1_i = met1->begin();
	met1_i != met1->end(); ++met1_i ) {
    bx_set.insert(met1_i->bx());
  }
  for ( l1extra::L1EtMissParticleCollection::const_iterator met2_i = met2->begin();
	met2_i != met2->end(); ++met2_i ) {
    bx_set.insert(met2_i->bx());
  }

  std::vector<int> bx_vector;
  for ( std::set<int>::const_iterator bx = bx_set.begin();
	bx != bx_set.end(); ++bx ) {
    bx_vector.push_back(*bx);
  }
  std::sort(bx_vector.begin(), bx_vector.end());

  for ( std::vector<int>::const_iterator bx = bx_vector.begin();
	bx != bx_vector.end(); ++bx ) {
    bool errorFlag = false;

    const l1extra::L1EtMissParticle* met1_bx = 0;
    for ( l1extra::L1EtMissParticleCollection::const_iterator met1_i = met1->begin();
	  met1_i != met1->end(); ++met1_i ) {
      if ( met1_i->bx() == (*bx) ) {
	if ( met1_bx ) errorFlag = true;
	met1_bx = &(*met1_i);
      }
    }

    const l1extra::L1EtMissParticle* met2_bx = 0;
    for ( l1extra::L1EtMissParticleCollection::const_iterator met2_i = met2->begin();
	  met2_i != met2->end(); ++met2_i ) {
      if ( met2_i->bx() == (*bx) ) {
	if ( met2_bx ) errorFlag = true;
	met2_bx = &(*met2_i);
      }
    }

    if ( errorFlag )
      throw cms::Exception("L1ExtraMEtMixer::produce")
	<< " Failed to find unique match of MET objects for BX = " << (*bx) << " !!\n";
    assert(met1_bx || met2_bx);
    
     // CV: check that both MET objects are of the same type
    if ( met1_bx && met2_bx && met1_bx->type() != met2_bx->type() )
      throw cms::Exception("L1ExtraMEtMixer::produce")
	<< " Mismatch in type between MET objects stored in collections 'src1' and 'src2' !!\n";

    double metSumPx = 0.;
    double metSumPy = 0.;
    double metSumEt = 0.;
    int type = -1;
    if ( met1_bx ) {
      //std::cout << "met1 (BX = " << (*bx) << "): Px = " << met1_bx->px() << ", Py = " << met1_bx->py() 
      //	  << " (Et = " << met1_bx->etMiss() << ", Pt = " << met1_bx->pt() << ", phi = " << met1_bx->phi() << ", sumEt = " << met1_bx->etTotal() << ")" << std::endl;
      metSumPx += met1_bx->px();
      metSumPy += met1_bx->py();
      metSumEt += met1_bx->etTotal();
      type = met1_bx->type();
    }
    if ( met2_bx ) {
      //std::cout << "met2 (BX = " << (*bx) << "): Px = " << met2_bx->px() << ", Py = " << met2_bx->py() 
      //	  << " (Et = " << met2_bx->etMiss() << ", Pt = " << met2_bx->pt() << ", phi = " << met2_bx->phi() << ", sumEt = " << met2_bx->etTotal() << ")" << std::endl;
      metSumPx += met2_bx->px();
      metSumPy += met2_bx->py();
      metSumEt += met2_bx->etTotal();
      type = met2_bx->type();
    }

    // Subtract contribution of muons that were replaced by simulated taus
    if ( (*bx) == 0 ) {
      for ( reco::CandidateCollection::const_iterator muon = muons->begin(); 
	    muon != muons->end(); ++muon ) {
	//std::cout << "muon: Pt = " << muon->pt() << ", eta = " << muon->eta() << ", phi = " << muon->phi() 
	//	    << " (Px = " << muon->px() << ", Py = " << muon->py() << ")" << std::endl;

	double distance_ecal = 0.;
	double distance_hcal = 0.;
	const detIdToFloatMap& distanceMap = ( muon->charge() > 0 ) ? 
	  (*distanceMapMuPlus) : (*distanceMapMuMinus);
	for ( detIdToFloatMap::const_iterator dist_iter = distanceMap.begin(); 
	      dist_iter != distanceMap.end(); ++dist_iter ) {
	  const DetId& id = dist_iter->first;
	  if ( id.det() == DetId::Ecal                              ) distance_ecal += dist_iter->second;
	  if ( id.det() == DetId::Hcal && id.subdetId() != HcalOuter) distance_hcal += dist_iter->second;
	}
	//std::cout << "distance_ecal = " << distance_ecal << std::endl;
	//std::cout << "distance_hcal = " << distance_hcal << std::endl;

	// This is what we would expect from theory:
	double dedx = getDeDxForPbWO4(muon->p());
	//std::cout << "dedx = " << dedx << std::endl;
	double muonEnLoss_theory = dedx*(distance_ecal*DENSITY_PBWO4 + distance_hcal*DENSITY_BRASS);

	// We correct the muon energy loss computed from theory
	// by the ratio of average reconstructed L1Extra MET (projected in direction of the muon)
	// in single muon gun Monte Carlo events to the theory expecation.
	// The ratio depends on eta:
	double muonEnLoss = muonEnLoss_theory;
	//std::cout << "muonEnLoss (before eta-correction) = " << muonEnLoss << std::endl;
	if      ( fabs(muon->eta()) < 1.2 ) muonEnLoss *= sfAbsEtaLt12_;
	else if ( fabs(muon->eta()) < 1.7 ) muonEnLoss *= sfAbsEta12to17_;
	else                                muonEnLoss *= sfAbsEtaGt17_;
	//std::cout << "muonEnLoss (after eta-correction) = " << muonEnLoss << std::endl;

	double muonEnLoss_transverse = muonEnLoss*sin(muon->theta());
	if ( muonEnLoss_transverse > metSumEt ) muonEnLoss_transverse = metSumEt;
	//std::cout << "muonEnLoss_transverse = " << muonEnLoss_transverse << std::endl;

	// The direction is given by the muon direction
	double muonMetCorrPx    = muonEnLoss_transverse*cos(muon->phi());
	double muonMetCorrPy    = muonEnLoss_transverse*sin(muon->phi());
	double muonMetCorrSumEt = muonEnLoss_transverse;
	//std::cout << "muonMetCorr: Px = " << muonMetCorrPx << ", Py = " << muonMetCorrPy << " (sumEt = " << muonMetCorrSumEt << ")" << std::endl;
	metSumPx += muonMetCorrPx;
	metSumPy += muonMetCorrPy;
	metSumEt -= muonMetCorrSumEt;
      }
    }

    const double metSumPt = TMath::Sqrt(metSumPx*metSumPx + metSumPy*metSumPy);
    // CV: setting edm::Refs to L1Gct objects not implemented yet
    l1extra::L1EtMissParticle metSum_bx(
      reco::Candidate::LorentzVector(metSumPx, metSumPy, 0., metSumPt),
      (l1extra::L1EtMissParticle::EtMissType)type,
      metSumEt,
      edm::Ref<L1GctEtMissCollection>(),
      edm::Ref<L1GctEtTotalCollection>(),
      edm::Ref<L1GctHtMissCollection>(),
      edm::Ref<L1GctEtHadCollection>(),
      (*bx));
    //std::cout << "metSum (BX = " << (*bx) << "): Px = " << metSum_bx.px() << ", Py = " << metSum_bx.py() 
    //	        << " (Et = " << metSum_bx.etMiss() << ", Pt = " << metSum_bx.pt() << ", phi = " << metSum_bx.phi() << ")" << std::endl;
    metSum->push_back(metSum_bx);					       
  }
  
  evt.put(metSum, instanceLabel_);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_EDM_PLUGIN(L1ExtraMixerPluginFactory, L1ExtraMEtMixerPlugin, "L1ExtraMEtMixerPlugin");

