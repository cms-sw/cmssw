#ifndef TauAnalysis_MCEmbeddingTools_MCEmbeddingValidationAnalyzer_h
#define TauAnalysis_MCEmbeddingTools_MCEmbeddingValidationAnalyzer_h

/** \class MCEmbeddingValidationAnalyzer
 *
 * Compare Ztautau events produced via MCEmbedding 
 * to Ztautau events produced via direct Monte Carlo production in terms of:
 *   o kinematic distributions of electrons, muons, taus and MET
 *   o electron, muon and tau reconstruction and identification efficiencies
 *   o MET resolution
 *   o trigger efficiencies
 * 
 * \author Christian Veelken, LLR
 *
 * \version $Revision: 1.20 $
 *
 * $Id: MCEmbeddingValidationAnalyzer.h,v 1.20 2013/06/04 13:44:44 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "PhysicsTools/JetMCUtils/interface/JetMCTag.h"
#include "DataFormats/TauReco/interface/PFTauDecayMode.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"

/*#include "EGamma/EGammaAnalysisTools/interface/EGammaMvaEleEstimator.h"*/
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"

#include <TString.h>
#include <TArrayD.h>

#include <string>
#include <vector>

namespace
{
  void countDecayProducts(const reco::GenParticle* genParticle,
			  int& numElectrons, int& numElecNeutrinos, int& numMuons, int& numMuNeutrinos, 
			  int& numChargedHadrons, int& numPi0s, int& numOtherNeutralHadrons, int& numPhotons)
  {
    //std::cout << " genParticle: pdgId = " << genParticle->pdgId() << std::endl;

    int absPdgId = TMath::Abs(genParticle->pdgId());
    int status   = genParticle->status();
    int charge   = genParticle->charge();

    if      ( absPdgId == 111 ) ++numPi0s;
    else if ( status   ==   1 ) {
      if      ( absPdgId == 11 ) ++numElectrons;
      else if ( absPdgId == 12 ) ++numElecNeutrinos;
      else if ( absPdgId == 13 ) ++numMuons;
      else if ( absPdgId == 14 ) ++numMuNeutrinos;
      else if ( absPdgId == 15 ) { 
	edm::LogError ("countDecayProducts")
	  << "Found tau lepton with status code 1 !!";
	return; 
      }
      else if ( absPdgId == 16 ) return; // no need to count tau neutrinos
      else if ( absPdgId == 22 ) ++numPhotons;
      else if ( charge   !=  0 ) ++numChargedHadrons;
      else                       ++numOtherNeutralHadrons;
    } else {
      unsigned numDaughters = genParticle->numberOfDaughters();
      for ( unsigned iDaughter = 0; iDaughter < numDaughters; ++iDaughter ) {
	const reco::GenParticle* daughter = genParticle->daughterRef(iDaughter).get();
	
	countDecayProducts(daughter, 
			   numElectrons, numElecNeutrinos, numMuons, numMuNeutrinos,
			   numChargedHadrons, numPi0s, numOtherNeutralHadrons, numPhotons);
      }
    }
  }
  
  std::string getGenTauDecayMode(const reco::GenParticle* genParticle) 
  {
//--- determine generator level tau decay mode
//
//    NOTE: 
//        (1) function implements logic defined in PhysicsTools/JetMCUtils/src/JetMCTag::genTauDecayMode
//            for different type of argument 
//        (2) this implementation should be more robust to handle cases of tau --> tau + gamma radiation
//
  
    //std::cout << "<MCEmbeddingValidationAnalyzer::getGenTauDecayMode>:" << std::endl;

    int numElectrons           = 0;
    int numElecNeutrinos       = 0;
    int numMuons               = 0;
    int numMuNeutrinos         = 0; 
    int numChargedHadrons      = 0;
    int numPi0s                = 0; 
    int numOtherNeutralHadrons = 0;
    int numPhotons             = 0;
    
    countDecayProducts(genParticle,
		       numElectrons, numElecNeutrinos, numMuons, numMuNeutrinos,
		       numChargedHadrons, numPi0s, numOtherNeutralHadrons, numPhotons);
    
    if      ( numElectrons == 1 && numElecNeutrinos == 1 ) return std::string("electron");
    else if ( numMuons     == 1 && numMuNeutrinos   == 1 ) return std::string("muon");
    
    switch ( numChargedHadrons ) {
    case 1 : 
      if ( numOtherNeutralHadrons != 0 ) return std::string("oneProngOther");
      switch ( numPi0s ) {
      case 0:
	return std::string("oneProng0Pi0");
      case 1:
	return std::string("oneProng1Pi0");
      case 2:
	return std::string("oneProng2Pi0");
      default:
	return std::string("oneProngOther");
      }
    case 3 : 
      if ( numOtherNeutralHadrons != 0 ) return std::string("threeProngOther");
      switch ( numPi0s ) {
      case 0:
	return std::string("threeProng0Pi0");
      case 1:
	return std::string("threeProng1Pi0");
      default:
	return std::string("threeProngOther");
      }
    default:
      return std::string("rare");
    }
  }
  
  MonitorElement* bookHistogram2D(DQMStore& dqmStore, const std::string& name, const std::string& title, int numBinsX, float* binningX, int numBinsY, float yMin, float yMax)
  {
    assert(numBinsY >= 1);
    TArrayF binningY(numBinsY + 1);
    float dy = (yMax - yMin)/numBinsY;
    for ( int iBinY = 0; iBinY <= numBinsY; ++iBinY ) {
      binningY[iBinY] = yMin + iBinY*dy;
    }
    MonitorElement* histogram = dqmStore.book2D(name, title, numBinsX, binningX, numBinsY, binningY.GetArray());
    return histogram;
  }

  std::pair<double, double> compMEtProjU(const reco::Candidate::LorentzVector& zP4, double metPx, double metPy, int& errorFlag, bool subtract_qT)
  {
    if ( zP4.pt() == 0. ) {
      edm::LogWarning ("compMEtProjU")
	<< " Failed to compute projection, because Z0 candidate has zero Pt --> returning dummy solution !!";
      errorFlag = 1;
      return std::pair<double, double>(0., 0.);
    }
  
    double qX = zP4.px();
    double qY = zP4.py();
    double qT = TMath::Sqrt(qX*qX + qY*qY);
  
    double uX = -metPx;
    double uY = -metPy;
    if ( subtract_qT ) {
      uX -= qX;
      uY -= qY;
    }
  
    double u1 = (uX*qX + uY*qY)/qT;
    double u2 = (uX*qY - uY*qX)/qT;
  
    return std::pair<double, double>(u1, u2);
  }
}

class MCEmbeddingValidationAnalyzer : public edm::EDAnalyzer 
{
 public:
  explicit MCEmbeddingValidationAnalyzer(const edm::ParameterSet&);
  ~MCEmbeddingValidationAnalyzer();
    
  void analyze(const edm::Event&, const edm::EventSetup&);
  void beginJob();

 private:
  std::string moduleLabel_;

  std::string dqmDirectory_full(const std::string& dqmSubDirectory)
  {
    TString dqmDirectory_full = dqmDirectory_.data();
    if ( !dqmDirectory_full.EndsWith("/") ) dqmDirectory_full.Append("/");
    dqmDirectory_full.Append(dqmSubDirectory);
    return dqmDirectory_full.Data();
  }
  
  edm::InputTag srcReplacedMuons_;
  edm::InputTag srcRecMuons_;
  edm::InputTag srcRecTracks_;
  edm::InputTag srcCaloTowers_;
  edm::InputTag srcRecPFCandidates_;
  edm::InputTag srcRecJets_;
  edm::InputTag srcTheRecVertex_;
  edm::InputTag srcRecVertices_;
  edm::InputTag srcRecVerticesWithBS_;
  edm::InputTag srcBeamSpot_;
  edm::InputTag srcGenDiTaus_;
  double dRminSeparation_; // CV: minimum separation in dR between replaced muons and embedded tau leptons
  edm::InputTag srcGenLeg1_;
  edm::InputTag srcRecLeg1_;
  edm::InputTag srcGenLeg2_;
  edm::InputTag srcRecLeg2_;
  edm::InputTag srcGenParticles_;
  edm::InputTag srcL1ETM_;
  edm::InputTag srcGenCaloMEt_;
  edm::InputTag srcGenPFMEt_;
  edm::InputTag srcRecCaloMEt_;
  edm::InputTag srcRecPFMEt_;
  edm::InputTag srcMuonsBeforeRad_;
  edm::InputTag srcMuonsAfterRad_;
  edm::InputTag srcMuonRadCorrWeight_;
  edm::InputTag srcMuonRadCorrWeightUp_;
  edm::InputTag srcMuonRadCorrWeightDown_;
  typedef std::vector<edm::InputTag> vInputTag;
  vInputTag srcOtherWeights_;
  edm::InputTag srcGenFilterInfo_;

  std::string dqmDirectory_;

  MonitorElement* histogramEventCounter_;

  MonitorElement* histogramGenFilterEfficiency_;

  MonitorElement* histogramRotationAngleMatrix_;
  double replacedMuonPtThresholdHigh_;
  double replacedMuonPtThresholdLow_;
  MonitorElement* histogramRotationLegPlusDeltaR_;
  MonitorElement* histogramRotationLegMinusDeltaR_;
  MonitorElement* histogramPhiRotLegPlus_;
  MonitorElement* histogramPhiRotLegMinus_;

  MonitorElement* histogramNumTracksPtGt5_;
  MonitorElement* histogramNumTracksPtGt10_;
  MonitorElement* histogramNumTracksPtGt20_;
  MonitorElement* histogramNumTracksPtGt30_;
  MonitorElement* histogramNumTracksPtGt40_;

  MonitorElement* histogramNumGlobalMuons_;
  MonitorElement* histogramNumStandAloneMuons_;
  MonitorElement* histogramNumPFMuons_;

  MonitorElement* histogramNumChargedPFCandsPtGt5_;
  MonitorElement* histogramNumChargedPFCandsPtGt10_;
  MonitorElement* histogramNumChargedPFCandsPtGt20_;
  MonitorElement* histogramNumChargedPFCandsPtGt30_;
  MonitorElement* histogramNumChargedPFCandsPtGt40_;

  MonitorElement* histogramNumNeutralPFCandsPtGt5_;
  MonitorElement* histogramNumNeutralPFCandsPtGt10_;
  MonitorElement* histogramNumNeutralPFCandsPtGt20_;
  MonitorElement* histogramNumNeutralPFCandsPtGt30_;
  MonitorElement* histogramNumNeutralPFCandsPtGt40_;
    
  MonitorElement* histogramRawJetPt_;
  MonitorElement* histogramRawJetPtAbsEtaLt2_5_;
  MonitorElement* histogramRawJetPtAbsEta2_5to4_5_;
  MonitorElement* histogramRawJetEtaPtGt20_;
  MonitorElement* histogramRawJetEtaPtGt30_;
  MonitorElement* histogramNumJetsRawPtGt20_;
  MonitorElement* histogramNumJetsRawPtGt20AbsEtaLt2_5_;
  MonitorElement* histogramNumJetsRawPtGt20AbsEta2_5to4_5_;
  MonitorElement* histogramNumJetsRawPtGt30_;
  MonitorElement* histogramNumJetsRawPtGt30AbsEtaLt2_5_;
  MonitorElement* histogramNumJetsRawPtGt30AbsEta2_5to4_5_;
  MonitorElement* histogramCorrJetPt_;
  MonitorElement* histogramCorrJetPtAbsEtaLt2_5_;
  MonitorElement* histogramCorrJetPtAbsEta2_5to4_5_;
  MonitorElement* histogramCorrJetEtaPtGt20_;
  MonitorElement* histogramCorrJetEtaPtGt30_;
  MonitorElement* histogramNumJetsCorrPtGt20_;
  MonitorElement* histogramNumJetsCorrPtGt20AbsEtaLt2_5_;
  MonitorElement* histogramNumJetsCorrPtGt20AbsEta2_5to4_5_;
  MonitorElement* histogramNumJetsCorrPtGt30_;
  MonitorElement* histogramNumJetsCorrPtGt30AbsEtaLt2_5_;
  MonitorElement* histogramNumJetsCorrPtGt30AbsEta2_5to4_5_;

  MonitorElement* histogramTheRecVertexX_;
  MonitorElement* histogramTheRecVertexY_;
  MonitorElement* histogramTheRecVertexZ_;
  MonitorElement* histogramRecVertexX_;
  MonitorElement* histogramRecVertexY_;
  MonitorElement* histogramRecVertexZ_;
  MonitorElement* histogramNumRecVertices_;
  MonitorElement* histogramRecVertexWithBSx_;
  MonitorElement* histogramRecVertexWithBSy_;
  MonitorElement* histogramRecVertexWithBSz_;
  MonitorElement* histogramNumRecVerticesWithBS_;

  MonitorElement* histogramBeamSpotX_;
  MonitorElement* histogramBeamSpotY_;

  MonitorElement* histogramGenDiTauPt_;
  MonitorElement* histogramGenDiTauEta_;
  MonitorElement* histogramGenDiTauPhi_;
  MonitorElement* histogramGenDiTauMass_;
  MonitorElement* histogramGenDeltaPhiLeg1Leg2_;
  MonitorElement* histogramGenDiTauDecayAngle_;

  MonitorElement* histogramGenVisDiTauPt_;
  MonitorElement* histogramGenVisDiTauEta_;
  MonitorElement* histogramGenVisDiTauPhi_;
  MonitorElement* histogramGenVisDiTauMass_;
  MonitorElement* histogramGenVisDeltaPhiLeg1Leg2_;
  
  MonitorElement* histogramRecVisDiTauPt_;
  MonitorElement* histogramRecVisDiTauEta_;
  MonitorElement* histogramRecVisDiTauPhi_;
  MonitorElement* histogramRecVisDiTauMass_;
  MonitorElement* histogramRecVisDeltaPhiLeg1Leg2_;

  MonitorElement* histogramGenTau1Pt_;
  MonitorElement* histogramGenTau1Eta_;
  MonitorElement* histogramGenTau1Phi_;
  MonitorElement* histogramGenLeg1Pt_;
  MonitorElement* histogramGenLeg1Eta_;
  MonitorElement* histogramGenLeg1Phi_;
  MonitorElement* histogramGenLeg1X_;
  MonitorElement* histogramGenLeg1XforGenLeg2X0_00to0_25_;
  MonitorElement* histogramGenLeg1XforGenLeg2X0_25to0_50_;
  MonitorElement* histogramGenLeg1XforGenLeg2X0_50to0_75_;
  MonitorElement* histogramGenLeg1XforGenLeg2X0_75to1_00_;
  MonitorElement* histogramGenLeg1Mt_;
  MonitorElement* histogramRecLeg1X_;  
  MonitorElement* histogramRecLeg1PFMt_;
  MonitorElement* histogramGenTau2Pt_;
  MonitorElement* histogramGenTau2Eta_;
  MonitorElement* histogramGenTau2Phi_;
  MonitorElement* histogramGenLeg2Pt_;
  MonitorElement* histogramGenLeg2Eta_;
  MonitorElement* histogramGenLeg2Phi_;
  MonitorElement* histogramGenLeg2X_;
  MonitorElement* histogramGenLeg2XforGenLeg1X0_00to0_25_;
  MonitorElement* histogramGenLeg2XforGenLeg1X0_25to0_50_;
  MonitorElement* histogramGenLeg2XforGenLeg1X0_50to0_75_;
  MonitorElement* histogramGenLeg2XforGenLeg1X0_75to1_00_;
  MonitorElement* histogramGenLeg2Mt_;
  MonitorElement* histogramRecLeg2X_;  
  MonitorElement* histogramRecLeg2PFMt_;

  MonitorElement* histogramSumGenParticlePt_;
  MonitorElement* histogramSumGenParticlePt_charged_;
  MonitorElement* histogramGenCaloMEt_;
  MonitorElement* histogramGenPFMEt_;

  MonitorElement* histogramRecCaloMEtECAL_;
  MonitorElement* histogramRecCaloSumEtECAL_;
  MonitorElement* histogramRecCaloMEtHCAL_;
  MonitorElement* histogramRecCaloSumEtHCAL_;
  MonitorElement* histogramRecCaloMEtHF_;
  MonitorElement* histogramRecCaloSumEtHF_;
  MonitorElement* histogramRecCaloMEtHO_;
  MonitorElement* histogramRecCaloSumEtHO_;  

  MonitorElement* histogramWarning_recTrackNearReplacedMuon_;
  MonitorElement* histogramWarning_recPFCandNearReplacedMuon_;
  MonitorElement* histogramWarning_recMuonNearReplacedMuon_;

  struct plotEntryTypeEvtWeight
  {
    plotEntryTypeEvtWeight(const edm::InputTag& srcWeight,
			   const std::string& dqmDirectory)
      : srcWeight_(srcWeight),
	dqmDirectory_(dqmDirectory)
    {
      evtWeightKey_ = Form("%s_%s", srcWeight_.label().data(), srcWeight_.instance().data());
    }
    ~plotEntryTypeEvtWeight() {}
    void bookHistograms(DQMStore& dqmStore)
    {
      dqmStore.setCurrentFolder(dqmDirectory_.data());
      std::string histogramWeightName = Form("weight_%s_%s", srcWeight_.label().data(), srcWeight_.instance().data());
      histogramWeight_ = dqmStore.book1D(histogramWeightName.data(), histogramWeightName.data(), 1001, -0.005, 10.005);
    }
    void fillHistograms(const edm::Event& evt, const std::map<std::string, double>& evtWeightMap_other)
    {
      edm::Handle<double> weight;
      evt.getByLabel(srcWeight_, weight);
      double evtWeightOther = 1.;
      for ( std::map<std::string, double>::const_iterator evtWeightEntry_other = evtWeightMap_other.begin();
	    evtWeightEntry_other != evtWeightMap_other.end(); ++evtWeightEntry_other ) {
	if ( evtWeightEntry_other->first != evtWeightKey_ ) evtWeightOther *= evtWeightEntry_other->second;
      }
      histogramWeight_->Fill(*weight, evtWeightOther);
    }
    edm::InputTag srcWeight_;
    std::string dqmDirectory_;
    MonitorElement* histogramWeight_;
    std::string evtWeightKey_;
  };

  std::vector<plotEntryTypeEvtWeight*> evtWeightPlotEntries_;

  struct plotEntryTypeMuonRadCorrUncertainty
  {
    plotEntryTypeMuonRadCorrUncertainty(int minJets, int maxJets, 
					const std::string& dqmDirectory)
      : minJets_(minJets),
	maxJets_(maxJets)
    {
      dqmDirectory_ = dqmDirectory;
      if      ( minJets_ < 0 && maxJets_ < 0 ) dqmDirectory_.append("");
      else if (                 maxJets_ < 0 ) dqmDirectory_.append(Form("_numJetsGe%i", minJets_));
      else if ( minJets_ < 0                 ) dqmDirectory_.append(Form("_numJetsLe%i", maxJets_));
      else if ( maxJets_     == minJets_     ) dqmDirectory_.append(Form("_numJetsEq%i", minJets_));
      else                                     dqmDirectory_.append(Form("_numJets%ito%i", minJets_, maxJets_));
    }
    ~plotEntryTypeMuonRadCorrUncertainty() {}
    void bookHistograms(DQMStore& dqmStore)
    {
      dqmStore.setCurrentFolder(dqmDirectory_.data());
      histogramMuonPlusPt_unweighted_ = dqmStore.book1D("muonPlusPt_unweighted", "muonPlusPt_unweighted", 250, 0., 250.);
      histogramMuonPlusPt_weighted_ = dqmStore.book1D("muonPlusPt_weighted", "muonPlusPt_weighted", 250, 0., 250.);
      histogramMuonPlusPt_weightedUp_ = dqmStore.book1D("muonPlusPt_weightedUp", "muonPlusPt_weightedUp", 250, 0., 250.);
      histogramMuonPlusPt_weightedDown_ = dqmStore.book1D("muonPlusPt_weightedDown", "muonPlusPt_weightedDown", 250,  0., 250.);
      histogramMuonPlusEta_unweighted_ = dqmStore.book1D("muonPlusEta_unweighted", "muonPlusEta_unweighted", 198, -9.9, +9.9);
      histogramMuonPlusEta_weighted_ = dqmStore.book1D("muonPlusEta_weighted", "muonPlusEta_weighted", 198, -9.9, +9.9);
      histogramMuonPlusEta_weightedUp_ = dqmStore.book1D("muonPlusEta_weightedUp", "muonPlusEta_weightedUp", 198, -9.9, +9.9);
      histogramMuonPlusEta_weightedDown_ = dqmStore.book1D("muonPlusEta_weightedDown", "muonPlusEta_weightedDown", 198, -9.9, +9.9);
      histogramMuonMinusPt_unweighted_ = dqmStore.book1D("muonMinusPt_unweighted", "muonMinusPt_unweighted", 250, 0., 250.);
      histogramMuonMinusPt_weighted_ = dqmStore.book1D("muonMinusPt_weighted", "muonMinusPt_weighted", 250, 0., 250.);
      histogramMuonMinusPt_weightedUp_ = dqmStore.book1D("muonMinusPt_weightedUp", "muonMinusPt_weightedUp", 250, 0., 250.);
      histogramMuonMinusPt_weightedDown_ = dqmStore.book1D("muonMinusPt_weightedDown", "muonMinusPt_weightedDown", 250, 0., 250.);
      histogramMuonMinusEta_unweighted_ = dqmStore.book1D("muonMinusEta_unweighted", "muonMinusEta_unweighted", 198, -9.9, +9.9);
      histogramMuonMinusEta_weighted_ = dqmStore.book1D("muonMinusEta_weighted", "muonMinusEta_weighted", 198, -9.9, +9.9);
      histogramMuonMinusEta_weightedUp_ = dqmStore.book1D("muonMinusEta_weightedUp", "muonMinusEta_weightedUp", 198, -9.9, +9.9);
      histogramMuonMinusEta_weightedDown_ = dqmStore.book1D("muonMinusEta_weightedDown", "muonMinusEta_weightedDown", 198, -9.9, +9.9);
      histogramDiMuonMass_unweighted_ = dqmStore.book1D("diMuonMass_unweighted", "diMuonMass_unweighted", 250, 0., 250.);
      histogramDiMuonMass_weighted_ = dqmStore.book1D("diMuonMass_weighted", "diMuonMass_weighted", 250, 0., 250.);
      histogramDiMuonMass_weightedUp_ = dqmStore.book1D("diMuonMass_weightedUp", "diMuonMass_weightedUp", 250, 0., 250.);
      histogramDiMuonMass_weightedDown_ = dqmStore.book1D("diMuonMass_weightedDown", "diMuonMass_weightedDown", 250, 0., 250.);
      histogramMuonRadCorrWeight_ = dqmStore.book1D("muonRadCorrWeight", "muonRadCorrWeight", 1001, -0.005, 10.005);
      histogramMuonRadCorrWeightUp_ = dqmStore.book1D("muonRadCorrWeightUp", "muonRadCorrWeightUp", 1001, -0.005, 10.005);
      histogramMuonRadCorrWeightDown_ = dqmStore.book1D("muonRadCorrWeightDown", "muonRadCorrWeightDown", 1001, -0.005, 10.005);
    }
    void fillHistograms(int numJets,
			const reco::Candidate::LorentzVector& muonPlusP4, const reco::Candidate::LorentzVector& muonMinusP4,
			double evtWeight_others, double muonRadCorrWeight, double muonRadCorrWeightUp, double muonRadCorrWeightDown)
    {
      if ( (minJets_ == -1 || numJets >= minJets_) &&
	   (maxJets_ == -1 || numJets <= maxJets_) ) {
	histogramMuonPlusPt_unweighted_->Fill(muonPlusP4.pt(), evtWeight_others);
	histogramMuonPlusPt_weighted_->Fill(muonPlusP4.pt(), evtWeight_others*muonRadCorrWeight);
	histogramMuonPlusPt_weightedUp_->Fill(muonPlusP4.pt(), evtWeight_others*muonRadCorrWeightUp);
	histogramMuonPlusPt_weightedDown_->Fill(muonPlusP4.pt(), evtWeight_others*muonRadCorrWeightDown);
	histogramMuonPlusEta_unweighted_->Fill(muonPlusP4.eta(), evtWeight_others);
	histogramMuonPlusEta_weighted_->Fill(muonPlusP4.eta(), evtWeight_others*muonRadCorrWeight);
	histogramMuonPlusEta_weightedUp_->Fill(muonPlusP4.eta(), evtWeight_others*muonRadCorrWeightUp);
	histogramMuonPlusEta_weightedDown_->Fill(muonPlusP4.eta(), evtWeight_others*muonRadCorrWeightDown);
	histogramMuonMinusPt_unweighted_->Fill(muonMinusP4.pt(), evtWeight_others);
	histogramMuonMinusPt_weighted_->Fill(muonMinusP4.pt(), evtWeight_others*muonRadCorrWeight);
	histogramMuonMinusPt_weightedUp_->Fill(muonMinusP4.pt(), evtWeight_others*muonRadCorrWeightUp);
	histogramMuonMinusPt_weightedDown_->Fill(muonMinusP4.pt(), evtWeight_others*muonRadCorrWeightDown);
	histogramMuonMinusEta_unweighted_->Fill(muonMinusP4.eta(), evtWeight_others);
	histogramMuonMinusEta_weighted_->Fill(muonMinusP4.eta(), evtWeight_others*muonRadCorrWeight);
	histogramMuonMinusEta_weightedUp_->Fill(muonMinusP4.eta(), evtWeight_others*muonRadCorrWeightUp);
	histogramMuonMinusEta_weightedDown_->Fill(muonMinusP4.eta(), evtWeight_others*muonRadCorrWeightDown);
	double diMuonMass = (muonPlusP4 + muonMinusP4).mass();
	histogramDiMuonMass_unweighted_->Fill(diMuonMass, evtWeight_others);
	histogramDiMuonMass_weighted_->Fill(diMuonMass, evtWeight_others*muonRadCorrWeight);
	histogramDiMuonMass_weightedUp_->Fill(diMuonMass, evtWeight_others*muonRadCorrWeightUp);
	histogramDiMuonMass_weightedDown_->Fill(diMuonMass, evtWeight_others*muonRadCorrWeightDown);
	histogramMuonRadCorrWeight_->Fill(muonRadCorrWeight, evtWeight_others);
	histogramMuonRadCorrWeightUp_->Fill(muonRadCorrWeightUp, evtWeight_others);
	histogramMuonRadCorrWeightDown_->Fill(muonRadCorrWeightDown, evtWeight_others);
      }
    }
    int minJets_;
    int maxJets_;
    std::string dqmDirectory_;
    MonitorElement* histogramMuonPlusPt_unweighted_;
    MonitorElement* histogramMuonPlusPt_weighted_;
    MonitorElement* histogramMuonPlusPt_weightedUp_;
    MonitorElement* histogramMuonPlusPt_weightedDown_;
    MonitorElement* histogramMuonPlusEta_unweighted_;
    MonitorElement* histogramMuonPlusEta_weighted_;
    MonitorElement* histogramMuonPlusEta_weightedUp_;
    MonitorElement* histogramMuonPlusEta_weightedDown_;
    MonitorElement* histogramMuonMinusPt_unweighted_;
    MonitorElement* histogramMuonMinusPt_weighted_;
    MonitorElement* histogramMuonMinusPt_weightedUp_;
    MonitorElement* histogramMuonMinusPt_weightedDown_;
    MonitorElement* histogramMuonMinusEta_unweighted_;
    MonitorElement* histogramMuonMinusEta_weighted_;
    MonitorElement* histogramMuonMinusEta_weightedUp_;
    MonitorElement* histogramMuonMinusEta_weightedDown_;
    MonitorElement* histogramDiMuonMass_unweighted_;
    MonitorElement* histogramDiMuonMass_weighted_;
    MonitorElement* histogramDiMuonMass_weightedUp_;
    MonitorElement* histogramDiMuonMass_weightedDown_;
    MonitorElement* histogramMuonRadCorrWeight_;
    MonitorElement* histogramMuonRadCorrWeightUp_;
    MonitorElement* histogramMuonRadCorrWeightDown_;
  };

  std::vector<plotEntryTypeMuonRadCorrUncertainty*> muonRadCorrUncertaintyPlotEntries_beforeRad_;
  std::vector<plotEntryTypeMuonRadCorrUncertainty*> muonRadCorrUncertaintyPlotEntries_afterRad_;
  std::vector<plotEntryTypeMuonRadCorrUncertainty*> muonRadCorrUncertaintyPlotEntries_afterRadAndCorr_;  
  int muonRadCorrUncertainty_numWarnings_;
  int muonRadCorrUncertainty_maxWarnings_;

  struct plotEntryTypeL1ETM
  {
    plotEntryTypeL1ETM(const std::string& genTauDecayMode, const std::string& dqmDirectory)
      : genTauDecayMode_(genTauDecayMode)
    {
      dqmDirectory_ = dqmDirectory;
      if ( genTauDecayMode != "" ) dqmDirectory_.append("_").append(genTauDecayMode);
    }
    ~plotEntryTypeL1ETM() {}
    void bookHistograms(DQMStore& dqmStore)
    {
      dqmStore.setCurrentFolder(dqmDirectory_.data());
      const int qTnumBins = 34;
      float qTbinning[qTnumBins + 1] = { 
	0., 2.5, 5., 7.5, 10., 12.5, 15., 17.5, 20., 22.5, 25., 27.5, 30., 35., 40., 45., 50., 
	60., 70., 80., 90., 100., 110., 120., 130., 140., 150., 160., 170., 180., 200., 220., 240., 260., 300.
      };
      histogramL1ETM_                              = dqmStore.book1D(          "L1ETM",                              "L1ETM",                                    250, 0., 250.);
      histogramL1ETM_vs_genCaloMEt_                = dqmStore.book2D(          "L1ETM_vs_genCaloMEt",                "L1ETM_vs_genCaloMEt",                      250, 0., 250.,  250,    0.,  250.);
      histogramL1ETM_vs_recCaloMEt_                = dqmStore.book2D(          "L1ETM_vs_recCaloMEt",                "L1ETM_vs_recCaloMEt",                      250, 0., 250.,  250,    0.,  250.);
      histogramL1ETMparlZdivQtVsQt_                = bookHistogram2D(dqmStore, "L1ETMparlZdivQtVsQt",                "L1ETMparlZdivQtVsQt",                qTnumBins, qTbinning, 400,   -5.,   +5.);
      histogramL1ETMparlZvsQt_                     = bookHistogram2D(dqmStore, "L1ETMparlZvsQt",                     "L1ETMparlZvsQt",                     qTnumBins, qTbinning, 400, -500., +500.);
      histogramL1ETMperpZvsQt_                     = bookHistogram2D(dqmStore, "L1ETMperpZvsQt",                     "L1ETMperpZvsQt",                     qTnumBins, qTbinning,  60,  -75.,  +75.);
      histogramRecCaloMEtParlZdivQtVsQt_           = bookHistogram2D(dqmStore, "recCaloMEtParlZdivQtVsQt",           "recCaloMEtParlZdivQtVsQt",           qTnumBins, qTbinning, 400,   -5.,   +5.);
      histogramRecCaloMEtParlZvsQt_                = bookHistogram2D(dqmStore, "recCaloMEtParlZvsQt",                "recCaloMEtParlZvsQt",                qTnumBins, qTbinning, 400, -500., +500.);
      histogramRecCaloMEtPerpZvsQt_                = bookHistogram2D(dqmStore, "recCaloMEtPerpZvsQt",                "recCaloMEtPerpZvsQt",                qTnumBins, qTbinning,  60,  -75.,  +75.);
      histogramL1ETMminusGenCaloMEtParlZdivQtVsQt_ = bookHistogram2D(dqmStore, "L1ETMminusGenCaloMEtParlZdivQtVsQt", "L1ETMminusGenCaloMEtParlZdivQtVsQt", qTnumBins, qTbinning, 400,   -5.,   +5.);
      histogramL1ETMminusGenCaloMEtParlZvsQt_      = bookHistogram2D(dqmStore, "L1ETMminusGenCaloMEtParlZvsQt",      "L1ETMminusGenCaloMEtParlZvsQt",      qTnumBins, qTbinning, 400, -500., +500.);
      histogramL1ETMminusGenCaloMEtPerpZvsQt_      = bookHistogram2D(dqmStore, "L1ETMminusGenCaloMEtPerpZvsQt",      "L1ETMminusGenCaloMEtPerpZvsQt",      qTnumBins, qTbinning,  60,  -75.,  +75.);
      histogramL1ETMminusRecCaloMEtParlZdivQtVsQt_ = bookHistogram2D(dqmStore, "L1ETMminusRecCaloMEtParlZdivQtVsQt", "L1ETMminusRecCaloMEtParlZdivQtVsQt", qTnumBins, qTbinning, 400,   -5.,   +5.);
      histogramL1ETMminusRecCaloMEtParlZvsQt_      = bookHistogram2D(dqmStore, "L1ETMminusRecCaloMEtParlZvsQt",      "L1ETMminusRecCaloMEtParlZvsQt",      qTnumBins, qTbinning, 400, -500., +500.);
      histogramL1ETMminusRecCaloMEtPerpZvsQt_      = bookHistogram2D(dqmStore, "L1ETMminusRecCaloMEtPerpZvsQt",      "L1ETMminusRecCaloMEtPerpZvsQt",      qTnumBins, qTbinning,  60,  -75.,  +75.);
      histogramQt_                                 = dqmStore.book1D(          "qT",                                 "qT",                                       600, 0., 300.);
    }
    void fillHistograms(const std::string& genTauDecayMode_ref,
			const reco::Candidate::LorentzVector& l1MEtP4, const reco::Candidate::LorentzVector& genCaloMEtP4, const reco::Candidate::LorentzVector& recCaloMEtP4, 
			const reco::Candidate::LorentzVector& genDiTauP4, double evtWeight)
    {
      if ( genTauDecayMode_ == "" || genTauDecayMode_ref == genTauDecayMode_ ) {
	histogramL1ETM_->Fill(l1MEtP4.pt(), evtWeight);
	histogramL1ETM_vs_genCaloMEt_->Fill(genCaloMEtP4.pt(), l1MEtP4.pt(), evtWeight);
	histogramL1ETM_vs_recCaloMEt_->Fill(recCaloMEtP4.pt(), l1MEtP4.pt(), evtWeight);
	double qT = genDiTauP4.pt();
	int errorFlag = 0;
	std::pair<double, double> uT = compMEtProjU(genDiTauP4, l1MEtP4.px(), l1MEtP4.py(), errorFlag, false);
	if ( !errorFlag ) {
	  double uParl = uT.first;
	  double uPerp = uT.second;
	  if ( qT > 0. ) histogramL1ETMparlZdivQtVsQt_->Fill(qT, uParl/qT, evtWeight);
	  histogramL1ETMparlZvsQt_->Fill(qT, uParl, evtWeight);
	  histogramL1ETMperpZvsQt_->Fill(qT, uPerp, evtWeight);
	}
	errorFlag = 0;
	uT = compMEtProjU(genDiTauP4, recCaloMEtP4.px(), recCaloMEtP4.py(), errorFlag, false);
	if ( !errorFlag ) {
	  double uParl = uT.first;
	  double uPerp = uT.second;
	  if ( qT > 0. ) histogramRecCaloMEtParlZdivQtVsQt_->Fill(qT, uParl/qT, evtWeight);
	  histogramRecCaloMEtParlZvsQt_->Fill(qT, uParl, evtWeight);
	  histogramRecCaloMEtPerpZvsQt_->Fill(qT, uPerp, evtWeight);
	}
        errorFlag = 0;
	uT = compMEtProjU(genDiTauP4, l1MEtP4.px() - genCaloMEtP4.px(), l1MEtP4.py() - genCaloMEtP4.py(), errorFlag, false);
	if ( !errorFlag ) {
	  double uParl = uT.first;
	  double uPerp = uT.second;
	  if ( qT > 0. ) histogramL1ETMminusGenCaloMEtParlZdivQtVsQt_->Fill(qT, uParl/qT, evtWeight);
	  histogramL1ETMminusGenCaloMEtParlZvsQt_->Fill(qT, uParl, evtWeight); 
	  histogramL1ETMminusGenCaloMEtPerpZvsQt_->Fill(qT, uPerp, evtWeight); 
	}
	errorFlag = 0;
	uT = compMEtProjU(genDiTauP4, l1MEtP4.px() - recCaloMEtP4.px(), l1MEtP4.py() - recCaloMEtP4.py(), errorFlag, false);
	if ( !errorFlag ) {
	  double uParl = uT.first;
	  double uPerp = uT.second;
	  if ( qT > 0. ) histogramL1ETMminusRecCaloMEtParlZdivQtVsQt_->Fill(qT, uParl/qT, evtWeight);
	  histogramL1ETMminusRecCaloMEtParlZvsQt_->Fill(qT, uParl, evtWeight); 
	  histogramL1ETMminusRecCaloMEtPerpZvsQt_->Fill(qT, uPerp, evtWeight); 
	}
	histogramQt_->Fill(qT, evtWeight); 
      }
    }
    std::string genTauDecayMode_;
    std::string dqmDirectory_;
    MonitorElement* histogramL1ETM_;
    MonitorElement* histogramL1ETM_vs_genCaloMEt_;   
    MonitorElement* histogramL1ETM_vs_recCaloMEt_; 
    MonitorElement* histogramL1ETMparlZdivQtVsQt_;
    MonitorElement* histogramL1ETMparlZvsQt_;
    MonitorElement* histogramL1ETMperpZvsQt_;
    MonitorElement* histogramRecCaloMEtParlZdivQtVsQt_;
    MonitorElement* histogramRecCaloMEtParlZvsQt_;
    MonitorElement* histogramRecCaloMEtPerpZvsQt_;
    MonitorElement* histogramL1ETMminusGenCaloMEtParlZdivQtVsQt_; 
    MonitorElement* histogramL1ETMminusGenCaloMEtParlZvsQt_; 
    MonitorElement* histogramL1ETMminusGenCaloMEtPerpZvsQt_; 
    MonitorElement* histogramL1ETMminusRecCaloMEtParlZdivQtVsQt_; 
    MonitorElement* histogramL1ETMminusRecCaloMEtParlZvsQt_; 
    MonitorElement* histogramL1ETMminusRecCaloMEtPerpZvsQt_; 
    MonitorElement* histogramQt_; 
  };

  std::vector<plotEntryTypeL1ETM*> l1ETMplotEntries_;
  
  template <typename T>
  struct leptonDistributionT
  {
    leptonDistributionT(int minJets, int maxJets, 
			const edm::InputTag& srcGen, const std::string& cutGen, const edm::InputTag& srcRec, const std::string& cutRec, double dRmatch, const std::string& dqmDirectory)
      : minJets_(minJets),
	maxJets_(maxJets),
	srcGen_(srcGen),
	cutGen_(0),
	srcRec_(srcRec),
	cutRec_(0),
	dRmatch_(dRmatch),
	dqmDirectory_(dqmDirectory)
    {
      if ( cutGen != "" ) cutGen_ = new StringCutObjectSelector<reco::Candidate>(cutGen);
      if ( cutRec != "" ) cutRec_ = new StringCutObjectSelector<T>(cutRec);
      dqmDirectory_ = dqmDirectory;
      if      ( minJets_ < 0 && maxJets_ < 0 ) dqmDirectory_.append("");
      else if (                 maxJets_ < 0 ) dqmDirectory_.append(Form("_numJetsGe%i", minJets_));
      else if ( minJets_ < 0                 ) dqmDirectory_.append(Form("_numJetsLe%i", maxJets_));
      else if ( maxJets_     == minJets_     ) dqmDirectory_.append(Form("_numJetsEq%i", minJets_));
      else                                     dqmDirectory_.append(Form("_numJets%ito%i", minJets_, maxJets_));
    }
    ~leptonDistributionT() 
    {
      delete cutGen_;
      delete cutRec_;
    }
    void bookHistograms(DQMStore& dqmStore)
    {
      dqmStore.setCurrentFolder(dqmDirectory_.data());
      histogramNumGenLeptons_ = dqmStore.book1D("numGenLeptons", "numGenLeptons", 10, -0.5, 9.5);      
      histogramGenLeptonPt_ = dqmStore.book1D("genLeptonPt", "genLeptonPt", 250, 0., 250.);
      histogramGenLeptonEta_ = dqmStore.book1D("genLeptonEta", "genLeptonEta", 198, -9.9, +9.9);
      histogramGenLeptonPhi_ = dqmStore.book1D("genLeptonPhi", "genLeptonPhi", 72, -TMath::Pi(), +TMath::Pi());
      histogramNumRecLeptons_ = dqmStore.book1D("numRecLeptons", "numRecLeptons", 10, -0.5, 9.5);
      histogramRecLeptonPt_ = dqmStore.book1D("recLeptonPt", "recLeptonPt", 250, 0., 250.);
      histogramRecLeptonEta_ = dqmStore.book1D("recLeptonEta", "recLeptonEta", 198, -9.9, +9.9);
      histogramRecLeptonPhi_ = dqmStore.book1D("recLeptonPhi", "recLeptonPhi", 72, -TMath::Pi(), +TMath::Pi());
      histogramRecMinusGenLeptonPt_ = dqmStore.book1D("recMinusGenLeptonPt", "recMinusGenLeptonPt", 200, -100., +100.);
      histogramRecMinusGenLeptonPt_div_genLeptonPt_ = dqmStore.book1D("recMinusGenLeptonPt_div_genLeptonPt", "recMinusGenLeptonPt_div_genLeptonPt", 200, 0., 2.);
      histogramRecMinusGenLeptonEta_ = dqmStore.book1D("recMinusGenLeptonEta", "recMinusGenLeptonEta", 100, -0.5, +0.5);
      histogramRecMinusGenLeptonPhi_ = dqmStore.book1D("recMinusGenLeptonPhi", "recMinusGenLeptonPhi", 100, -0.5, +0.5);
    }
    void fillHistograms(int numJets, 
			const edm::Event& evt, double evtWeight)
    {
      if ( (minJets_ == -1 || numJets >= minJets_) &&
	   (maxJets_ == -1 || numJets <= maxJets_) ) {
	typedef edm::View<reco::Candidate> CandidateView;
	edm::Handle<CandidateView> genLeptons;
	evt.getByLabel(srcGen_, genLeptons);
	histogramNumGenLeptons_->Fill(genLeptons->size(), evtWeight);
	typedef std::vector<T> recLeptonCollection;
	edm::Handle<recLeptonCollection> recLeptons;
	evt.getByLabel(srcRec_, recLeptons);
	histogramNumRecLeptons_->Fill(recLeptons->size(), evtWeight);
	for ( CandidateView::const_iterator genLepton = genLeptons->begin();
	      genLepton != genLeptons->end(); ++genLepton ) {
	  if ( cutGen_ && !(*cutGen_)(*genLepton) ) continue;
	  for ( typename recLeptonCollection::const_iterator recLepton = recLeptons->begin();
		recLepton != recLeptons->end(); ++recLepton ) {
	    if ( cutRec_ && !(*cutRec_)(*recLepton) ) continue;
	    double dR = deltaR(genLepton->p4(), recLepton->p4());
	    if ( dR < dRmatch_ ) {
	      histogramGenLeptonPt_->Fill(genLepton->pt(), evtWeight);
	      histogramGenLeptonEta_->Fill(genLepton->eta(), evtWeight);
	      histogramGenLeptonPhi_->Fill(genLepton->phi(), evtWeight);
	      histogramRecLeptonPt_->Fill(recLepton->pt(), evtWeight);
	      histogramRecLeptonEta_->Fill(recLepton->eta(), evtWeight);
	      histogramRecLeptonPhi_->Fill(recLepton->phi(), evtWeight);
	      histogramRecMinusGenLeptonPt_->Fill(recLepton->pt() - genLepton->pt(), evtWeight);
	      if ( genLepton->pt() > 0. ) histogramRecMinusGenLeptonPt_div_genLeptonPt_->Fill((recLepton->pt() - genLepton->pt())/genLepton->pt(), evtWeight);
	      histogramRecMinusGenLeptonEta_->Fill(recLepton->eta() - genLepton->eta(), evtWeight);
	      histogramRecMinusGenLeptonPhi_->Fill(recLepton->phi() - genLepton->phi(), evtWeight);
	    }	     
	  }
	}
      }
    }
    int minJets_;
    int maxJets_;
    edm::InputTag srcGen_;
    StringCutObjectSelector<reco::Candidate>* cutGen_;
    edm::InputTag srcRec_;
    StringCutObjectSelector<T>* cutRec_;
    double dRmatch_;
    std::string dqmDirectory_;
    MonitorElement* histogramNumGenLeptons_;
    MonitorElement* histogramGenLeptonPt_;
    MonitorElement* histogramGenLeptonEta_;
    MonitorElement* histogramGenLeptonPhi_;
    MonitorElement* histogramNumRecLeptons_;
    MonitorElement* histogramRecLeptonPt_;
    MonitorElement* histogramRecLeptonEta_;
    MonitorElement* histogramRecLeptonPhi_;
    MonitorElement* histogramRecMinusGenLeptonPt_;
    MonitorElement* histogramRecMinusGenLeptonPt_div_genLeptonPt_;
    MonitorElement* histogramRecMinusGenLeptonEta_;
    MonitorElement* histogramRecMinusGenLeptonPhi_;
  };

  struct electronDistributionExtra
  {
    electronDistributionExtra(int minJets, int maxJets, 
			      const edm::InputTag& srcGen, const std::string& cutGen, const edm::InputTag& srcRec, const std::string& cutRec, double dRmatch, const std::string& dqmDirectory,
			      const edm::InputTag& srcTheRecVertex)
      : minJets_(minJets),
	maxJets_(maxJets),
	srcGen_(srcGen),
	cutGen_(0),
	srcRec_(srcRec),
	cutRec_(0),
        dRmatch_(dRmatch),
        srcTheRecVertex_(srcTheRecVertex)
    {
      if ( cutGen != "" ) cutGen_ = new StringCutObjectSelector<reco::Candidate>(cutGen);
      if ( cutRec != "" ) cutRec_ = new StringCutObjectSelector<pat::Electron>(cutRec);
      dqmDirectory_ = dqmDirectory;
      if      ( minJets_ < 0 && maxJets_ < 0 ) dqmDirectory_.append("");
      else if (                 maxJets_ < 0 ) dqmDirectory_.append(Form("_numJetsGe%i", minJets_));
      else if ( minJets_ < 0                 ) dqmDirectory_.append(Form("_numJetsLe%i", maxJets_));
      else if ( maxJets_     == minJets_     ) dqmDirectory_.append(Form("_numJetsEq%i", minJets_));
      else                                     dqmDirectory_.append(Form("_numJets%ito%i", minJets_, maxJets_));
      /*if ( !fMVA_isInitialized_ ) {
	std::vector<std::string> mvaWeightFiles;
	mvaWeightFiles.push_back("EGamma/EGammaAnalysisTools/data/Electrons_BDTG_NonTrigV0_Cat1.weights.xml");
	mvaWeightFiles.push_back("EGamma/EGammaAnalysisTools/data/Electrons_BDTG_NonTrigV0_Cat2.weights.xml");
	mvaWeightFiles.push_back("EGamma/EGammaAnalysisTools/data/Electrons_BDTG_NonTrigV0_Cat3.weights.xml");
	mvaWeightFiles.push_back("EGamma/EGammaAnalysisTools/data/Electrons_BDTG_NonTrigV0_Cat4.weights.xml");
	mvaWeightFiles.push_back("EGamma/EGammaAnalysisTools/data/Electrons_BDTG_NonTrigV0_Cat5.weights.xml");
	mvaWeightFiles.push_back("EGamma/EGammaAnalysisTools/data/Electrons_BDTG_NonTrigV0_Cat6.weights.xml");
	std::vector<std::string> mvaWeightFiles_full;
	for ( std::vector<std::string>::const_iterator mvaWeightFile = mvaWeightFiles.begin();
	      mvaWeightFile != mvaWeightFiles.end(); ++mvaWeightFile ) {
	  edm::FileInPath mvaWeightFile_full(*mvaWeightFile);
	  if ( mvaWeightFile_full.location() == edm::FileInPath::Unknown ) 
	    throw cms::Exception("MCEmbeddingValidationAnalyzer")
	      << " Failed to find File = " << (*mvaWeightFile) << " !!\n";
	  mvaWeightFiles_full.push_back(mvaWeightFile_full.fullPath());
	}
	fMVA_ = new EGammaMvaEleEstimator();
	fMVA_->initialize("BDT",
			  EGammaMvaEleEstimator::kNonTrig,
			  true, 
			  mvaWeightFiles_full);
	fMVA_isInitialized_ = true;
      }*/
    }
    ~electronDistributionExtra() 
    {
      delete cutGen_;
      delete cutRec_;
      /*if ( fMVA_isInitialized_ ) {
	delete fMVA_;
	fMVA_ = 0;
	fMVA_isInitialized_ = false;
      }*/
    }
    void bookHistograms(DQMStore& dqmStore)
    {
      dqmStore.setCurrentFolder(dqmDirectory_.data());
      histogramMVAptLt20AbsEtaLt0_8_      = dqmStore.book1D("MVAptLt20AbsEtaLt0_8",      "MVAptLt20AbsEtaLt0_8",       201, -1.005,   +1.005);  
      histogramMVAptLt20AbsEta0_8to1_479_ = dqmStore.book1D("MVAptLt20AbsEta0_8to1_479", "MVAptLt20AbsEta0_8to1_479",  201, -1.005,   +1.005);  
      histogramMVAptLt20AbsEtaGt1_479_    = dqmStore.book1D("MVAptLt20AbsEtaGt1_479",    "MVAptLt20AbsEtaGt1_479",     201, -1.005,   +1.005);  
      histogramMVAptGt20AbsEtaLt0_8_      = dqmStore.book1D("MVAptGt20AbsEtaLt0_8",      "MVAptGt20AbsEtaLt0_8",       201, -1.005,   +1.005);  
      histogramMVAptGt20AbsEta0_8to1_479_ = dqmStore.book1D("MVAptGt20AbsEta0_8to1_479", "MVAptGt20AbsEta0_8to1_479",  201, -1.005,   +1.005);  
      histogramMVAptGt20AbsEtaGt1_479_    = dqmStore.book1D("MVAptGt20AbsEtaGt1_479",    "MVAptGt20AbsEtaGt1_479",     201, -1.005,   +1.005); 
      histogramFBrem_                     = dqmStore.book1D("fBrem",                     "fBrem",                     1100, -1.,     +10.);  
      histogramKFchi2_                    = dqmStore.book1D("kfChi2",                    "kfChi2",                    1001, -0.05,  +100.05);  
      histogramNumKFhits_                 = dqmStore.book1D("numKFhits",                 "numKFhits",                  25,  -0.5,    +24.5); 
      histogramGSFchi2_                   = dqmStore.book1D("gsfChi2",                   "gsfChi2",                   1001, -0.05,  +100.05);
      histogramDEta_                      = dqmStore.book1D("dEta",                      "dEta",                      1000, -0.5,     +0.5); 
      histogramDPhi_                      = dqmStore.book1D("dPhi",                      "dPhi",                      1000, -0.5,     +0.5);
      histogramDEtaCalo_                  = dqmStore.book1D("dEtaCalo",                  "dEtaCalo",                  1000, -0.5,     +0.5); 
      histogramSee_                       = dqmStore.book1D("See",                       "See",                        500,  0.,       0.5);
      histogramSpp_                       = dqmStore.book1D("Spp",                       "Spp",                        500,  0.,       0.5);
      histogramEtaWidth_                  = dqmStore.book1D("etaWidth",                  "etaWidth",                   500,  0.,       0.5);
      histogramPhiWidth_                  = dqmStore.book1D("phiWidth",                  "phiWidth",                   500,  0.,       0.5);
      histogramOneMinusE1x5E5x5_          = dqmStore.book1D("oneMinusE1x5E5x5",          "oneMinusE1x5E5x5",           201, -1.005,   +1.005);
      histogramR9_                        = dqmStore.book1D("R9",                        "R9",                        1000,  0.,      10.);
      histogramHoE_                       = dqmStore.book1D("HoE",                       "HoE",                        200, -1.,      +1.);
      histogramEoP_                       = dqmStore.book1D("EoP",                       "EoP",                       1000,  0.,      10.);
      histogramIoEmIoP_                   = dqmStore.book1D("IoEmIoP",                   "IoEmIoP",                   1000,  0.,      10.);
      histogramEleEoPout_                 = dqmStore.book1D("EleEoPout",                 "EleEoPout",                 1000,  0.,      10.);
      histogramPreShowerOverRaw_          = dqmStore.book1D("PreShowerOverRaw",          "PreShowerOverRaw",          1000,  0.,      10.);
      histogramD0_                        = dqmStore.book1D("D0",                        "D0",                        1000,  0.,       0.1);
      histogramIP3d_                      = dqmStore.book1D("IP3d",                      "IP3d",                      5000,  0.,       0.5);
      histogramEta_                       = dqmStore.book1D("eta",                       "eta",                        198, -9.9,     +9.9); 
      histogramPt_                        = dqmStore.book1D("pt",                        "pt",                         250,  0.,     250.); 
      histogramCharge_                    = dqmStore.book1D("charge",                    "charge",                       3, -1.5,     +1.5); 
      histogramFlags_                     = dqmStore.book1D("flags",                     "flags",                       12, -0.5,    +11.5); 
    }
    void fillHistograms(int numJets, 
			const edm::Event& evt, const edm::EventSetup& es, double evtWeight)
    {
      if ( (minJets_ == -1 || numJets >= minJets_) &&
	   (maxJets_ == -1 || numJets <= maxJets_) ) {
	typedef edm::View<reco::Candidate> CandidateView;
	edm::Handle<CandidateView> genLeptons;
	evt.getByLabel(srcGen_, genLeptons);
	typedef std::vector<pat::Electron> recLeptonCollection;
	edm::Handle<recLeptonCollection> recLeptons;
	evt.getByLabel(srcRec_, recLeptons);
	for ( CandidateView::const_iterator genLepton = genLeptons->begin();
	      genLepton != genLeptons->end(); ++genLepton ) {
	  if ( cutGen_ && !(*cutGen_)(*genLepton) ) continue;
	  for ( recLeptonCollection::const_iterator recLepton = recLeptons->begin();
		recLepton != recLeptons->end(); ++recLepton ) {
	    if ( cutRec_ && !(*cutRec_)(*recLepton) ) continue;
	    double dR = deltaR(genLepton->p4(), recLepton->p4());
	    if ( dR < dRmatch_ ) {
	      edm::Handle<reco::VertexCollection> theVertex;
	      evt.getByLabel(srcTheRecVertex_, theVertex);
	      if ( !(theVertex->size() >= 1) ) return;
	      edm::ESHandle<TransientTrackBuilder> trackBuilderHandle;
	      es.get<TransientTrackRecord>().get("TransientTrackBuilder", trackBuilderHandle);
	      const TransientTrackBuilder* trackBuilder = trackBuilderHandle.product();
	      if ( !trackBuilder ) 
		throw cms::Exception("MCEmbeddingValidationAnalyzer")
		  << " Failed to access TransientTrackBuilder !!\n";
	      EcalClusterLazyTools myEcalCluster(evt, es, edm::InputTag("reducedEcalRecHitsEB"), edm::InputTag("reducedEcalRecHitsEE"));
	      /*double mva = fMVA_->mvaValue(*recLepton, theVertex->front(), *trackBuilder, myEcalCluster);
	      if ( recLepton->pt() < 20. ) {
		if      ( TMath::Abs(recLepton->eta()) < 0.8   ) histogramMVAptLt20AbsEtaLt0_8_->Fill(mva, evtWeight);
		else if ( TMath::Abs(recLepton->eta()) < 1.479 ) histogramMVAptLt20AbsEta0_8to1_479_->Fill(mva, evtWeight);
		else                                             histogramMVAptLt20AbsEtaGt1_479_->Fill(mva, evtWeight);
	      } else {
		if      ( TMath::Abs(recLepton->eta()) < 0.8   ) histogramMVAptGt20AbsEtaLt0_8_->Fill(mva, evtWeight);
		else if ( TMath::Abs(recLepton->eta()) < 1.479 ) histogramMVAptGt20AbsEta0_8to1_479_->Fill(mva, evtWeight);
		else                                             histogramMVAptGt20AbsEtaGt1_479_->Fill(mva, evtWeight);
	      }
	      histogramFBrem_->Fill(fMVA_->fBrem(), evtWeight);
	      histogramKFchi2_->Fill(fMVA_->kfChi2(), evtWeight);
	      histogramNumKFhits_->Fill(fMVA_->numKFhits(), evtWeight);
	      histogramGSFchi2_->Fill(fMVA_->gsfChi2(), evtWeight);
	      histogramDEta_->Fill(fMVA_->dEta(), evtWeight);
	      histogramDPhi_->Fill(fMVA_->dPhi(), evtWeight);
	      histogramDEtaCalo_->Fill(fMVA_->dEtaCalo(), evtWeight);
	      histogramSee_->Fill(fMVA_->See(), evtWeight);
	      histogramSpp_->Fill(fMVA_->Spp(), evtWeight);
	      histogramEtaWidth_->Fill(fMVA_->etaWidth(), evtWeight);
	      histogramPhiWidth_->Fill(fMVA_->phiWidth(), evtWeight);
	      histogramOneMinusE1x5E5x5_->Fill(fMVA_->oneMinusE1x5E5x5(), evtWeight);
	      histogramR9_->Fill(fMVA_->R9(), evtWeight);
	      histogramHoE_->Fill(fMVA_->HoE(), evtWeight);  
	      histogramEoP_->Fill(fMVA_->EoP(), evtWeight);
	      histogramIoEmIoP_->Fill(fMVA_->IoEmIoP(), evtWeight);
	      histogramEleEoPout_->Fill(fMVA_->eleEoPout(), evtWeight);
	      histogramPreShowerOverRaw_->Fill(fMVA_->preShowerOverRaw(), evtWeight);
	      histogramD0_->Fill(fMVA_->d0(), evtWeight);
	      //std::cout << "d0 = " << fMVA_->d0() << std::endl;
    	      histogramIP3d_->Fill(fMVA_->ip3d(), evtWeight);
	      //std::cout << "ip3d = " << fMVA_->ip3d() << std::endl;
	      histogramEta_->Fill(fMVA_->eta(), evtWeight);
	      histogramPt_->Fill(fMVA_->pt(), evtWeight);*/
	      histogramCharge_->Fill(recLepton->charge(), evtWeight); 
	      histogramFlags_->Fill(0, evtWeight);   
	      if ( recLepton->isEB() ) histogramFlags_->Fill(1, evtWeight);   
	      if ( recLepton->isEE() ) histogramFlags_->Fill(2, evtWeight);  
	      if ( recLepton->isEBEEGap() ) histogramFlags_->Fill(3, evtWeight);  
	      if ( recLepton->isEBEtaGap() ) histogramFlags_->Fill(4, evtWeight);
	      if ( recLepton->isEBPhiGap() ) histogramFlags_->Fill(5, evtWeight);
	      if ( recLepton->isEEDeeGap() ) histogramFlags_->Fill(6, evtWeight);
	      if ( recLepton->isEERingGap() ) histogramFlags_->Fill(7, evtWeight);
	      if ( recLepton->ecalDrivenSeed() ) histogramFlags_->Fill(8, evtWeight);
	      if ( recLepton->trackerDrivenSeed() ) histogramFlags_->Fill(9, evtWeight);
	      if ( recLepton->superCluster().isNonnull() || recLepton->superCluster().id() != recLepton->parentSuperCluster().id() ) histogramFlags_->Fill(10, evtWeight);
	      if ( recLepton->parentSuperCluster().isNonnull() ) histogramFlags_->Fill(11, evtWeight);
	    }
	  }	
	}
      }
    }
    int minJets_;
    int maxJets_;
    edm::InputTag srcGen_;
    StringCutObjectSelector<reco::Candidate>* cutGen_;
    edm::InputTag srcRec_;
    StringCutObjectSelector<pat::Electron>* cutRec_;
    double dRmatch_;
    std::string dqmDirectory_;
    /*static EGammaMvaEleEstimator* fMVA_;
    static bool fMVA_isInitialized_;*/
    edm::InputTag srcTheRecVertex_;
    MonitorElement* histogramMVAptLt20AbsEtaLt0_8_;
    MonitorElement* histogramMVAptLt20AbsEta0_8to1_479_;
    MonitorElement* histogramMVAptLt20AbsEtaGt1_479_;
    MonitorElement* histogramMVAptGt20AbsEtaLt0_8_;
    MonitorElement* histogramMVAptGt20AbsEta0_8to1_479_;
    MonitorElement* histogramMVAptGt20AbsEtaGt1_479_;
    MonitorElement* histogramFBrem_;
    MonitorElement* histogramKFchi2_;
    MonitorElement* histogramNumKFhits_;
    MonitorElement* histogramGSFchi2_;
    MonitorElement* histogramDEta_;
    MonitorElement* histogramDPhi_;
    MonitorElement* histogramDEtaCalo_;
    MonitorElement* histogramSee_;
    MonitorElement* histogramSpp_;
    MonitorElement* histogramEtaWidth_;
    MonitorElement* histogramPhiWidth_;
    MonitorElement* histogramOneMinusE1x5E5x5_;
    MonitorElement* histogramR9_;
    MonitorElement* histogramHoE_;
    MonitorElement* histogramEoP_;
    MonitorElement* histogramIoEmIoP_;
    MonitorElement* histogramEleEoPout_;
    MonitorElement* histogramPreShowerOverRaw_;
    MonitorElement* histogramD0_;
    MonitorElement* histogramIP3d_;
    MonitorElement* histogramEta_;
    MonitorElement* histogramPt_;
    MonitorElement* histogramCharge_;
    MonitorElement* histogramFlags_;
  };

  struct tauDistributionExtra
  {
    tauDistributionExtra(int minJets, int maxJets, 
			 const edm::InputTag& srcGen, const std::string& cutGen, const edm::InputTag& srcRec, const std::string& cutRec, double dRmatch, const std::string& dqmDirectory)
      : minJets_(minJets),
	maxJets_(maxJets),
	srcGen_(srcGen),
	cutGen_(0),
	srcRec_(srcRec),
	cutRec_(0),
	dRmatch_(dRmatch)
    {
      if ( cutGen != "" ) cutGen_ = new StringCutObjectSelector<reco::Candidate>(cutGen);
      if ( cutRec != "" ) cutRec_ = new StringCutObjectSelector<pat::Tau>(cutRec);
      dqmDirectory_ = dqmDirectory;
      if      ( minJets_ < 0 && maxJets_ < 0 ) dqmDirectory_.append("");
      else if (                 maxJets_ < 0 ) dqmDirectory_.append(Form("_numJetsGe%i", minJets_));
      else if ( minJets_ < 0                 ) dqmDirectory_.append(Form("_numJetsLe%i", maxJets_));
      else if ( maxJets_     == minJets_     ) dqmDirectory_.append(Form("_numJetsEq%i", minJets_));
      else                                     dqmDirectory_.append(Form("_numJets%ito%i", minJets_, maxJets_));
    }
    ~tauDistributionExtra() 
    {
      delete cutGen_;
      delete cutRec_;
    }
    void bookHistograms(DQMStore& dqmStore)
    {
      dqmStore.setCurrentFolder(dqmDirectory_.data());
      histogramGenTauDecayMode_ = dqmStore.book1D("genTauDecayMode", "genTauDecayMode", 21, -1.5, +19.5);      
      histogramRecTauDecayMode_ = dqmStore.book1D("recTauDecayMode", "recTauDecayMode", 21, -1.5, +19.5);  
    }
    void fillHistograms(int numJets, 
			const edm::Event& evt, double evtWeight)
    {
      if ( (minJets_ == -1 || numJets >= minJets_) &&
	   (maxJets_ == -1 || numJets <= maxJets_) ) {
	typedef edm::View<reco::Candidate> CandidateView;
	edm::Handle<CandidateView> genLeptons;
	evt.getByLabel(srcGen_, genLeptons);
	typedef std::vector<pat::Tau> recLeptonCollection;
	edm::Handle<recLeptonCollection> recLeptons;
	evt.getByLabel(srcRec_, recLeptons);
	for ( CandidateView::const_iterator genLepton = genLeptons->begin();
	      genLepton != genLeptons->end(); ++genLepton ) {
	  if ( cutGen_ && !(*cutGen_)(*genLepton) ) continue;
	  const reco::CompositePtrCandidate* genLepton_composite = dynamic_cast<const reco::CompositePtrCandidate*>(&(*genLepton));
	  for ( recLeptonCollection::const_iterator recLepton = recLeptons->begin();
		recLepton != recLeptons->end(); ++recLepton ) {
	    if ( cutRec_ && !(*cutRec_)(*recLepton) ) continue;
	    double dR = deltaR(genLepton->p4(), recLepton->p4());
	    if ( dR < dRmatch_ ) {
	      std::string genTauDecayMode_string  = JetMCTagUtils::genTauDecayMode(*genLepton_composite);	    
	      //std::cout << "--> genTauDecayMode: = " << genTauDecayMode_string << std::endl;
	      int genTauDecayMode = -1;
	      if      ( genTauDecayMode_string == "electron"        ) genTauDecayMode = reco::PFTauDecayMode::tauDecaysElectron;
	      else if ( genTauDecayMode_string == "muon"            ) genTauDecayMode = reco::PFTauDecayMode::tauDecayMuon;
	      else if ( genTauDecayMode_string == "oneProng0Pi0"    ) genTauDecayMode = reco::PFTauDecayMode::tauDecay1ChargedPion0PiZero;
	      else if ( genTauDecayMode_string == "oneProng1Pi0"    ) genTauDecayMode = reco::PFTauDecayMode::tauDecay1ChargedPion1PiZero;
	      else if ( genTauDecayMode_string == "oneProng2Pi0"    ) genTauDecayMode = reco::PFTauDecayMode::tauDecay1ChargedPion2PiZero;
	      else if ( genTauDecayMode_string == "oneProngOther"   ) genTauDecayMode = reco::PFTauDecayMode::tauDecay1ChargedPion3PiZero;
	      else if ( genTauDecayMode_string == "threeProng0Pi0"  ) genTauDecayMode = reco::PFTauDecayMode::tauDecay3ChargedPion0PiZero;
	      else if ( genTauDecayMode_string == "threeProng1Pi0"  ) genTauDecayMode = reco::PFTauDecayMode::tauDecay3ChargedPion1PiZero;
	      else if ( genTauDecayMode_string == "threeProngOther" ) genTauDecayMode = reco::PFTauDecayMode::tauDecay3ChargedPion2PiZero;
	      else if ( genTauDecayMode_string == "rare"            ) genTauDecayMode = reco::PFTauDecayMode::tauDecayOther;
	      histogramGenTauDecayMode_->Fill(genTauDecayMode, evtWeight);
	      int recTauDecayMode = recLepton->decayMode();
	      histogramRecTauDecayMode_->Fill(recTauDecayMode, evtWeight);
	    }
	  }	
	}
	//edm::Handle<reco::GenParticleCollection> genParticles;
	//evt.getByLabel("genParticles", genParticles);     
	//for ( reco::GenParticleCollection::const_iterator genParticle = genParticles->begin();
	//      genParticle != genParticles->end(); ++genParticle ) {
	//  if ( TMath::Abs(genParticle->pdgId()) == 15 ) {
	//    for ( recLeptonCollection::const_iterator recLepton = recLeptons->begin();
	//          recLepton != recLeptons->end(); ++recLepton ) {
	//      if ( cutRec_ && !(*cutRec_)(*recLepton) ) continue;		    
	//      double dR = deltaR(genParticle->p4(), recLepton->p4());
	//      if ( dR < dRmatch_ ) {
	//        std::string genTauDecayMode_string = getGenTauDecayMode(&(*genParticle));
	//        std::cout << "--> genTauDecayMode: = " << genTauDecayMode_string << std::endl;
	//      }
	//    }
	//  }
	//}
      }
    }
    int minJets_;
    int maxJets_;
    edm::InputTag srcGen_;
    StringCutObjectSelector<reco::Candidate>* cutGen_;
    edm::InputTag srcRec_;
    StringCutObjectSelector<pat::Tau>* cutRec_;
    double dRmatch_;
    std::string dqmDirectory_;
    MonitorElement* histogramGenTauDecayMode_;
    MonitorElement* histogramRecTauDecayMode_;
  };

  template <typename T>
  void setupLeptonDistribution(int, int, const edm::ParameterSet&, const std::string&, std::vector<leptonDistributionT<T>*>&);
  void setupElectronDistributionExtra(int, int, const edm::ParameterSet&, const std::string&, std::vector<electronDistributionExtra*>&);
  void setupTauDistributionExtra(int, int, const edm::ParameterSet&, const std::string&, std::vector<tauDistributionExtra*>&);

  std::vector<leptonDistributionT<pat::Electron>*> electronDistributions_;
  std::vector<electronDistributionExtra*> electronDistributionsExtra_;
  std::vector<leptonDistributionT<pat::Muon>*> muonDistributions_;
  std::vector<leptonDistributionT<pat::Tau>*> tauDistributions_;
  std::vector<tauDistributionExtra*> tauDistributionsExtra_;

  template <typename T>
  struct leptonEfficiencyT
  {
    leptonEfficiencyT(int minJets, int maxJets, 
		      const edm::InputTag& srcGen, const std::string& cutGen, const edm::InputTag& srcRec, const std::string& cutRec, double dRmatch, const std::string& dqmDirectory)
      : minJets_(minJets),
	maxJets_(maxJets),
	srcGen_(srcGen),
	cutGen_(0),
	srcRec_(srcRec),
	cutRec_(0),
	dRmatch_(dRmatch)
    {
      if ( cutGen != "" ) cutGen_ = new StringCutObjectSelector<reco::Candidate>(cutGen);
      if ( cutRec != "" ) cutRec_ = new StringCutObjectSelector<T>(cutRec);
      dqmDirectory_ = dqmDirectory;
      if      ( minJets_ < 0 && maxJets_ < 0 ) dqmDirectory_.append("");
      else if (                 maxJets_ < 0 ) dqmDirectory_.append(Form("_numJetsGe%i", minJets_));
      else if ( minJets_ < 0                 ) dqmDirectory_.append(Form("_numJetsLe%i", maxJets_));
      else if ( maxJets_     == minJets_     ) dqmDirectory_.append(Form("_numJetsEq%i", minJets_));
      else                                     dqmDirectory_.append(Form("_numJets%ito%i", minJets_, maxJets_));
    }
    ~leptonEfficiencyT() 
    {
      delete cutGen_;
      delete cutRec_;
    }
    void bookHistograms(DQMStore& dqmStore)
    {
      dqmStore.setCurrentFolder(dqmDirectory_.data());
      histogramNumGenLeptons_ = dqmStore.book1D("numGenLeptons", "numGenLeptons", 10, -0.5, 9.5);
      histogramNumRecLeptons_ = dqmStore.book1D("numRecLeptons", "numRecLeptons", 10, -0.5, 9.5);
      histogramNumeratorPt_ = dqmStore.book1D("numeratorPt", "numeratorPt", 250, 0., 250.);
      histogramDenominatorPt_ = dqmStore.book1D("denominatorPt", "denominatorPt", 250, 0., 250.);
      histogramNumeratorEta_ = dqmStore.book1D("numeratorEta", "numeratorEta", 198, -9.9, +9.9);
      histogramDenominatorEta_ = dqmStore.book1D("denominatorEta", "denominatorEta", 198, -9.9, +9.9);
      histogramNumeratorPhi_ = dqmStore.book1D("numeratorPhi", "numeratorPhi", 72, -TMath::Pi(), +TMath::Pi());
      histogramDenominatorPhi_ = dqmStore.book1D("denominatorPhi", "denominatorPhi", 72, -TMath::Pi(), +TMath::Pi());
    }
    void fillHistograms(int numJets, 
			const edm::Event& evt, double evtWeight)
    {
      if ( (minJets_ == -1 || numJets >= minJets_) &&
	   (maxJets_ == -1 || numJets <= maxJets_) ) {
	typedef edm::View<reco::Candidate> CandidateView;
	edm::Handle<CandidateView> genLeptons;
	evt.getByLabel(srcGen_, genLeptons);
	histogramNumGenLeptons_->Fill(genLeptons->size(), evtWeight);
	typedef std::vector<T> recLeptonCollection;
	edm::Handle<recLeptonCollection> recLeptons;
	evt.getByLabel(srcRec_, recLeptons);
	histogramNumRecLeptons_->Fill(recLeptons->size(), evtWeight);
	for ( CandidateView::const_iterator genLepton = genLeptons->begin();
	      genLepton != genLeptons->end(); ++genLepton ) {
	  if ( cutGen_ && !(*cutGen_)(*genLepton) ) continue;	
	  bool isMatched = false;
	  for ( typename recLeptonCollection::const_iterator recLepton = recLeptons->begin();
		recLepton != recLeptons->end(); ++recLepton ) {
	    if ( cutRec_ && !(*cutRec_)(*recLepton) ) continue;
	    double dR = deltaR(genLepton->p4(), recLepton->p4());
	    if ( dR < dRmatch_ ) isMatched = true;
	  }
	  histogramDenominatorPt_->Fill(genLepton->pt(), evtWeight);
	  histogramDenominatorEta_->Fill(genLepton->eta(), evtWeight);
	  histogramDenominatorPhi_->Fill(genLepton->phi(), evtWeight);
	  if ( isMatched ) {
	    histogramNumeratorPt_->Fill(genLepton->pt(), evtWeight);
	    histogramNumeratorEta_->Fill(genLepton->eta(), evtWeight);
	    histogramNumeratorPhi_->Fill(genLepton->phi(), evtWeight);
	  }
	}
      }
    }
    int minJets_;
    int maxJets_;
    edm::InputTag srcGen_;
    StringCutObjectSelector<reco::Candidate>* cutGen_;
    edm::InputTag srcRec_;
    StringCutObjectSelector<T>* cutRec_;
    double dRmatch_;
    std::string dqmDirectory_;
    MonitorElement* histogramNumGenLeptons_;
    MonitorElement* histogramNumRecLeptons_;
    MonitorElement* histogramNumeratorPt_;
    MonitorElement* histogramDenominatorPt_;
    MonitorElement* histogramNumeratorEta_;
    MonitorElement* histogramDenominatorEta_;
    MonitorElement* histogramNumeratorPhi_;
    MonitorElement* histogramDenominatorPhi_;
  };

  template <typename T>
  void setupLeptonEfficiency(int, int, const edm::ParameterSet&, const std::string&, std::vector<leptonEfficiencyT<T>*>&);

  std::vector<leptonEfficiencyT<reco::GsfElectron>*> gsfElectronEfficiencies_;
  std::vector<leptonEfficiencyT<pat::Electron>*> electronEfficiencies_;
  std::vector<leptonEfficiencyT<pat::Muon>*> muonEfficiencies_;
  std::vector<leptonEfficiencyT<pat::Tau>*> tauEfficiencies_;

  template <typename T1, typename T2>
  struct leptonL1TriggerEfficiencyT1T2
  {
    leptonL1TriggerEfficiencyT1T2(int minJets, int maxJets, 
				  const edm::InputTag& srcRef, const std::string& cutRef, const edm::InputTag& srcL1, const std::string& cutL1, double dRmatch, const std::string& dqmDirectory)
      : minJets_(minJets),
	maxJets_(maxJets),
	srcRef_(srcRef),
	cutRef_(0),
	srcL1_(srcL1),
	cutL1_(0),
	dRmatch_(dRmatch)
    {
      if ( cutRef != "" ) cutRef_ = new StringCutObjectSelector<T1>(cutRef);
      if ( cutL1  != "" ) cutL1_  = new StringCutObjectSelector<T2>(cutL1);
      dqmDirectory_ = dqmDirectory;
      if      ( minJets_ < 0 && maxJets_ < 0 ) dqmDirectory_.append("");
      else if (                 maxJets_ < 0 ) dqmDirectory_.append(Form("_numJetsGe%i", minJets_));
      else if ( minJets_ < 0                 ) dqmDirectory_.append(Form("_numJetsLe%i", maxJets_));
      else if ( maxJets_     == minJets_     ) dqmDirectory_.append(Form("_numJetsEq%i", minJets_));
      else                                     dqmDirectory_.append(Form("_numJets%ito%i", minJets_, maxJets_));
    }
    ~leptonL1TriggerEfficiencyT1T2() 
    {
      delete cutRef_;
      delete cutL1_;
    }
    void bookHistograms(DQMStore& dqmStore)
    {
      dqmStore.setCurrentFolder(dqmDirectory_.data());
      histogramNumeratorPt_ = dqmStore.book1D("numeratorPt", "numeratorPt", 250, 0., 250.);
      histogramDenominatorPt_ = dqmStore.book1D("denominatorPt", "denominatorPt", 250, 0., 250.);
      histogramNumeratorEta_ = dqmStore.book1D("numeratorEta", "numeratorEta", 198, -9.9, +9.9);
      histogramDenominatorEta_ = dqmStore.book1D("denominatorEta", "denominatorEta", 198, -9.9, +9.9);
      histogramNumeratorPhi_ = dqmStore.book1D("numeratorPhi", "numeratorPhi", 72, -TMath::Pi(), +TMath::Pi());
      histogramDenominatorPhi_ = dqmStore.book1D("denominatorPhi", "denominatorPhi", 72, -TMath::Pi(), +TMath::Pi());
    }
    void fillHistograms(int numJets, 
			const edm::Event& evt, double evtWeight)
    {
      if ( (minJets_ == -1 || numJets >= minJets_) &&
	   (maxJets_ == -1 || numJets <= maxJets_) ) {
	typedef std::vector<T1> refLeptonCollection;
	edm::Handle<refLeptonCollection> refLeptons;
	evt.getByLabel(srcRef_, refLeptons);
	typedef std::vector<T2> l1LeptonCollection;
	edm::Handle<l1LeptonCollection> l1Leptons;
	evt.getByLabel(srcL1_, l1Leptons);
	for ( typename refLeptonCollection::const_iterator refLepton = refLeptons->begin();
	      refLepton != refLeptons->end(); ++refLepton ) {
	  if ( cutRef_ && !(*cutRef_)(*refLepton) ) continue;
	  bool isMatched = false;
	  for ( typename l1LeptonCollection::const_iterator l1Lepton = l1Leptons->begin();
		l1Lepton != l1Leptons->end(); ++l1Lepton ) {
	    if ( cutL1_ && !(*cutL1_)(*l1Lepton) ) continue;
	    double dR = deltaR(refLepton->p4(), l1Lepton->p4());
	    if ( dR < dRmatch_ ) isMatched = true;
	  }
	  histogramDenominatorPt_->Fill(refLepton->pt(), evtWeight);
	  histogramDenominatorEta_->Fill(refLepton->eta(), evtWeight);
	  histogramDenominatorPhi_->Fill(refLepton->phi(), evtWeight);
	  if ( isMatched ) {
	    histogramNumeratorPt_->Fill(refLepton->pt(), evtWeight);
	    histogramNumeratorEta_->Fill(refLepton->eta(), evtWeight);
	    histogramNumeratorPhi_->Fill(refLepton->phi(), evtWeight);
	  }
	}
      }
    }
    int minJets_;
    int maxJets_;
    edm::InputTag srcRef_;
    StringCutObjectSelector<T1>* cutRef_;
    edm::InputTag srcL1_;
    StringCutObjectSelector<T2>* cutL1_;
    double dRmatch_;
    std::string dqmDirectory_;
    MonitorElement* histogramNumeratorPt_;
    MonitorElement* histogramDenominatorPt_;
    MonitorElement* histogramNumeratorEta_;
    MonitorElement* histogramDenominatorEta_;
    MonitorElement* histogramNumeratorPhi_;
    MonitorElement* histogramDenominatorPhi_;
  };

  template <typename T1, typename T2>
  void setupLeptonL1TriggerEfficiency(int, int, const edm::ParameterSet&, const std::string&, std::vector<leptonL1TriggerEfficiencyT1T2<T1,T2>*>&);

  std::vector<leptonL1TriggerEfficiencyT1T2<pat::Electron, l1extra::L1EmParticle>*> electronL1TriggerEfficiencies_;
  std::vector<leptonL1TriggerEfficiencyT1T2<pat::Muon, l1extra::L1MuonParticle>*> muonL1TriggerEfficiencies_;

  template <typename T>
  struct l1ExtraObjectDistributionT
  {
    l1ExtraObjectDistributionT(int minJets, int maxJets, 
			       const edm::InputTag& src, const std::string& cut, const std::string& dqmDirectory)
      : minJets_(minJets),
	maxJets_(maxJets),
	src_(src),
	cut_(0)
    {
      if ( cut != "" ) cut_ = new StringCutObjectSelector<T>(cut);
      dqmDirectory_ = dqmDirectory;
      if      ( minJets_ < 0 && maxJets_ < 0 ) dqmDirectory_.append("");
      else if (                 maxJets_ < 0 ) dqmDirectory_.append(Form("_numJetsGe%i", minJets_));
      else if ( minJets_ < 0                 ) dqmDirectory_.append(Form("_numJetsLe%i", maxJets_));
      else if ( maxJets_     == minJets_     ) dqmDirectory_.append(Form("_numJetsEq%i", minJets_));
      else                                     dqmDirectory_.append(Form("_numJets%ito%i", minJets_, maxJets_));
    }
    ~l1ExtraObjectDistributionT() 
    {
      delete cut_;
    }
    void bookHistograms(DQMStore& dqmStore)
    {
      dqmStore.setCurrentFolder(dqmDirectory_.data());
      histogramNumL1ExtraObjects_ = dqmStore.book1D("numL1ExtraObjects", "numL1ExtraObjects", 10, -0.5, 9.5);      
      histogramL1ExtraObjectPt_ = dqmStore.book1D("l1ExtraObjectPt", "l1ExtraObjectPt", 250, 0., 250.);
      histogramL1ExtraObjectEta_ = dqmStore.book1D("l1ExtraObjectEta", "l1ExtraObjectEta", 198, -9.9, +9.9);
      histogramL1ExtraObjectPhi_ = dqmStore.book1D("l1ExtraObjectPhi", "l1ExtraObjectPhi", 72, -TMath::Pi(), +TMath::Pi());
    }
    void fillHistograms(int numJets, 
			const edm::Event& evt, double evtWeight)
    {
      if ( (minJets_ == -1 || numJets >= minJets_) &&
	   (maxJets_ == -1 || numJets <= maxJets_) ) {
	typedef std::vector<T> l1ExtraObjectCollection;
	edm::Handle<l1ExtraObjectCollection> l1ExtraObjects;
	evt.getByLabel(src_, l1ExtraObjects);
	int numL1ExtraObjects = 0;
	for ( typename l1ExtraObjectCollection::const_iterator l1ExtraObject = l1ExtraObjects->begin();
	      l1ExtraObject != l1ExtraObjects->end(); ++l1ExtraObject ) {
	  if ( cut_ && !(*cut_)(*l1ExtraObject) ) continue;
	  ++numL1ExtraObjects;
	  histogramL1ExtraObjectPt_->Fill(l1ExtraObject->pt(), evtWeight);
	  histogramL1ExtraObjectEta_->Fill(l1ExtraObject->eta(), evtWeight);
	  histogramL1ExtraObjectPhi_->Fill(l1ExtraObject->phi(), evtWeight);
	}
	histogramNumL1ExtraObjects_->Fill(numL1ExtraObjects, evtWeight);
      }
    }
    int minJets_;
    int maxJets_;
    edm::InputTag src_;
    StringCutObjectSelector<T>* cut_;
    std::string dqmDirectory_;
    MonitorElement* histogramNumL1ExtraObjects_;
    MonitorElement* histogramL1ExtraObjectPt_;
    MonitorElement* histogramL1ExtraObjectEta_;
    MonitorElement* histogramL1ExtraObjectPhi_;
  };

  template <typename T>
  void setupL1ExtraObjectDistribution(int, int, const edm::ParameterSet&, const std::string&, std::vector<l1ExtraObjectDistributionT<T>*>&);

  std::vector<l1ExtraObjectDistributionT<l1extra::L1EmParticle>*> l1ElectronDistributions_;
  std::vector<l1ExtraObjectDistributionT<l1extra::L1MuonParticle>*> l1MuonDistributions_;
  std::vector<l1ExtraObjectDistributionT<l1extra::L1JetParticle>*> l1TauDistributions_;
  std::vector<l1ExtraObjectDistributionT<l1extra::L1JetParticle>*> l1CentralJetDistributions_;
  std::vector<l1ExtraObjectDistributionT<l1extra::L1JetParticle>*> l1ForwardJetDistributions_;

  struct metDistributionType
  {
    metDistributionType(int minJets, int maxJets, 
			const edm::InputTag& srcGen, const edm::InputTag& srcRec, const edm::InputTag& srcGenZs, const std::string& dqmDirectory)
      : minJets_(minJets),
	maxJets_(maxJets),
	srcGen_(srcGen),
	srcRec_(srcRec),
	srcGenZs_(srcGenZs)
    {
      dqmDirectory_ = dqmDirectory;
      if      ( minJets_ < 0 && maxJets_ < 0 ) dqmDirectory_.append("");
      else if (                 maxJets_ < 0 ) dqmDirectory_.append(Form("_numJetsGe%i", minJets_));
      else if ( minJets_ < 0                 ) dqmDirectory_.append(Form("_numJetsLe%i", maxJets_));
      else if ( maxJets_     == minJets_     ) dqmDirectory_.append(Form("_numJetsEq%i", minJets_));
      else                                     dqmDirectory_.append(Form("_numJets%ito%i", minJets_, maxJets_));
    }
    ~metDistributionType() {}
    void bookHistograms(DQMStore& dqmStore)
    {
      dqmStore.setCurrentFolder(dqmDirectory_.data());
      histogramGenMEtPt_  = dqmStore.book1D("genMEtPt", "genMEtPt", 250, 0., 250.);
      histogramGenMEtPhi_ = dqmStore.book1D("genMEtPhi", "genMEtPhi", 72, -TMath::Pi(), +TMath::Pi());
      histogramRecMEtPt_  = dqmStore.book1D("recMEtPt", "recMEtPt", 250, 0., 250.);
      histogramRecMEtPhi_ = dqmStore.book1D("recMEtPhi", "recMEtPhi", 72, -TMath::Pi(), +TMath::Pi());
      histogramRecMinusGenMEtParlZ_ = dqmStore.book1D("recMinusGenMEtParlZ", "recMinusGenMEtParlZ", 200, -100., +100.);
      histogramRecMinusGenMEtPerpZ_ = dqmStore.book1D("recMinusGenMEtPerpZ", "recMinusGenMEtPerpZ", 100, 0., 100.);
    }
    void fillHistograms(int numJets, 
			const edm::Event& evt, double evtWeight)
    {
      if ( (minJets_ == -1 || numJets >= minJets_) &&
	   (maxJets_ == -1 || numJets <= maxJets_) ) {
	typedef edm::View<reco::MET> METView;
	edm::Handle<METView> genMETs;
	evt.getByLabel(srcGen_, genMETs);
	const reco::Candidate::LorentzVector& genMEtP4 = genMETs->front().p4();
	edm::Handle<METView> recMETs;
	evt.getByLabel(srcRec_, recMETs);
	const reco::Candidate::LorentzVector& recMEtP4 = recMETs->front().p4();
	//std::cout << "<MCEmbeddingValidationAnalyzer>:" << std::endl;
	//std::cout << " recMEt(" << srcRec_.label() << "): Pt = " << recMEtP4.pt() << ", phi = " << recMEtP4.phi() 
	//	    << " (Px = " << recMEtP4.px() << ", Py = " << recMEtP4.py() << ")" << std::endl;
	typedef edm::View<reco::Candidate> CandidateView;
	edm::Handle<CandidateView> genZs;
	evt.getByLabel(srcGenZs_, genZs);
	if ( !(genZs->size() >= 1) ) return;
	const reco::Candidate::LorentzVector& genZp4 = genZs->front().p4();
	histogramGenMEtPt_->Fill(genMEtP4.pt(), evtWeight);
	histogramGenMEtPhi_->Fill(genMEtP4.phi(), evtWeight);
	histogramRecMEtPt_->Fill(recMEtP4.pt(), evtWeight);
	histogramRecMEtPhi_->Fill(recMEtP4.phi(), evtWeight);
	if ( genZp4.pt() > 0. ) {
	  double qX = genZp4.px();
	  double qY = genZp4.py();
	  double qT = TMath::Sqrt(qX*qX + qY*qY);
	  double dX = recMEtP4.px() - genMEtP4.px();
	  double dY = recMEtP4.py() - genMEtP4.py();
	  double dParl = (dX*qX + dY*qY)/qT;
	  double dPerp = (dX*qY - dY*qX)/qT;
	  histogramRecMinusGenMEtParlZ_->Fill(dParl, evtWeight);
	  histogramRecMinusGenMEtPerpZ_->Fill(TMath::Abs(dPerp), evtWeight);
	}
      }
    }
    int minJets_;
    int maxJets_;
    edm::InputTag srcGen_;
    edm::InputTag srcRec_;
    edm::InputTag srcGenZs_;
    std::string dqmDirectory_;
    MonitorElement* histogramGenMEtPt_;
    MonitorElement* histogramGenMEtPhi_;
    MonitorElement* histogramRecMEtPt_;
    MonitorElement* histogramRecMEtPhi_;
    MonitorElement* histogramRecMinusGenMEtParlZ_;
    MonitorElement* histogramRecMinusGenMEtPerpZ_;
  };

  void setupMEtDistribution(int, int, const edm::ParameterSet&, const std::string&, std::vector<metDistributionType*>&);

  std::vector<metDistributionType*> metDistributions_;

  struct metL1TriggerEfficiencyType
  {
    metL1TriggerEfficiencyType(int minJets, int maxJets, 
			       const edm::InputTag& srcRef, const edm::InputTag& srcL1, double cutL1Et, double cutL1Pt, const std::string& dqmDirectory)
      : minJets_(minJets),
	maxJets_(maxJets),
	srcRef_(srcRef),
	srcL1_(srcL1),
	cutL1Et_(cutL1Et),
	cutL1Pt_(cutL1Pt)
    {
      dqmDirectory_ = dqmDirectory;
      if      ( minJets_ < 0 && maxJets_ < 0 ) dqmDirectory_.append("");
      else if (                 maxJets_ < 0 ) dqmDirectory_.append(Form("_numJetsGe%i", minJets_));
      else if ( minJets_ < 0                 ) dqmDirectory_.append(Form("_numJetsLe%i", maxJets_));
      else if ( maxJets_     == minJets_     ) dqmDirectory_.append(Form("_numJetsEq%i", minJets_));
      else                                     dqmDirectory_.append(Form("_numJets%ito%i", minJets_, maxJets_));
    }
    ~metL1TriggerEfficiencyType() {}
    void bookHistograms(DQMStore& dqmStore)
    {
      dqmStore.setCurrentFolder(dqmDirectory_.data());
      histogramNumeratorPt_ = dqmStore.book1D("numeratorPt", "numeratorPt", 250, 0., 250.);
      histogramDenominatorPt_ = dqmStore.book1D("denominatorPt", "denominatorPt", 250, 0., 250.);
      histogramNumeratorPhi_ = dqmStore.book1D("numeratorPhi", "numeratorPhi", 72, -TMath::Pi(), +TMath::Pi());
      histogramDenominatorPhi_ = dqmStore.book1D("denominatorPhi", "denominatorPhi", 72, -TMath::Pi(), +TMath::Pi());
    }
    void fillHistograms(int numJets,
			const edm::Event& evt, double evtWeight)
    {
      if ( (minJets_ == -1 || numJets >= minJets_) &&
	   (maxJets_ == -1 || numJets <= maxJets_) ) {
	typedef edm::View<reco::MET> METView;
	edm::Handle<METView> refMETs;
	evt.getByLabel(srcRef_, refMETs);
	const reco::Candidate::LorentzVector& refMEtP4 = refMETs->front().p4();
	edm::Handle<l1extra::L1EtMissParticleCollection> l1METs;
	evt.getByLabel(srcL1_, l1METs);
	for ( l1extra::L1EtMissParticleCollection::const_iterator l1MEt = l1METs->begin();
	      l1MEt != l1METs->end(); ++l1MEt ) {
	  if ( !(l1MEt->bx() == 0) ) continue;
	  double l1MEt_et = l1METs->front().etMiss();
	  double l1MEt_pt = l1METs->front().pt();
	  histogramDenominatorPt_->Fill(refMEtP4.pt(), evtWeight);
	  histogramDenominatorPhi_->Fill(refMEtP4.phi(), evtWeight);
	  if ( l1MEt_et > cutL1Et_ && l1MEt_pt > cutL1Pt_ ) {
	    histogramNumeratorPt_->Fill(refMEtP4.pt(), evtWeight);
	    histogramNumeratorPhi_->Fill(refMEtP4.phi(), evtWeight);
	  }
	}
      }
    }
    int minJets_;
    int maxJets_;
    edm::InputTag srcRef_;
    edm::InputTag srcL1_;
    double cutL1Et_;
    double cutL1Pt_;
    std::string dqmDirectory_;
    MonitorElement* histogramNumeratorPt_;
    MonitorElement* histogramDenominatorPt_;
    MonitorElement* histogramNumeratorPhi_;
    MonitorElement* histogramDenominatorPhi_;
  };
  
  void setupMEtL1TriggerEfficiency(int, int, const edm::ParameterSet&, const std::string&, std::vector<metL1TriggerEfficiencyType*>&);

  std::vector<metL1TriggerEfficiencyType*> metL1TriggerEfficiencies_;

  template <typename T>
  void cleanCollection(std::vector<T*> collection)
  {
    for ( typename std::vector<T*>::iterator object = collection.begin();
	  object != collection.end(); ++object ) {
      delete (*object);
    }
  }

  template <typename T>
  void bookHistograms(std::vector<T*> collection, DQMStore& dqmStore)
  {
    for ( typename std::vector<T*>::iterator object = collection.begin();
	  object != collection.end(); ++object ) {
      (*object)->bookHistograms(dqmStore);
    }
  } 

  template <typename T>
  void fillHistograms(std::vector<T*> collection, int numJets, const edm::Event& evt, double evtWeight)
  {
    for ( typename std::vector<T*>::iterator object = collection.begin();
	  object != collection.end(); ++object ) {
      (*object)->fillHistograms(numJets, evt, evtWeight);
    }
  } 
  template <typename T>
  void fillHistograms(std::vector<T*> collection, int numJets, const edm::Event& evt, const edm::EventSetup& es, double evtWeight)
  {
    for ( typename std::vector<T*>::iterator object = collection.begin();
	  object != collection.end(); ++object ) {
      (*object)->fillHistograms(numJets, evt, es, evtWeight);
    }
  } 

  int verbosity_;
};

#endif
