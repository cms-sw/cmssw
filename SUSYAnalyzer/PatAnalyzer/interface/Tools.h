#ifndef Tools_H
#define Tools_H

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/PatCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"

#include "RecoEgamma/EgammaTools/interface/ConversionTools.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

//Root Classes

#include "TH1F.h"
#include "TH2F.h"
#include "TH1I.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TTree.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TString.h"
#include "TMath.h"
#include "TLorentzVector.h"
#include "TLegend.h"

//Standard C++ classes
#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <utility>
#include <ostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <memory>
#include <iomanip>

using namespace std;

namespace tools {

    void ERR( edm::InputTag& IT );
    
    double pfRelIso(const pat::Muon *mu);
    double pfRelIso(const pat::Electron *el, double myRho);

    std::vector<const pat::Muon* > ssbMuonSelector(const std::vector<pat::Muon>  & thePatMuons,
                                                   double v_muon_pt,
                                                   reco::Vertex::Point PV,
                                                   double v_muon_d0);
    
    std::vector<const pat::Electron* > ssbElectronSelector(const std::vector<pat::Electron>  & thePatElectrons,
                                                           double v_electron_pt,
                                                           reco::Vertex::Point PV,
                                                           double v_electron_d0,
                                                           bool bool_electron_chargeConsistency,
                                                           edm::Handle< std::vector<reco::Conversion> > &theConversions,
                                                           reco::BeamSpot::Point BS);

    std::vector<const pat::Jet* > JetSelector(const std::vector<pat::Jet>  & thePatJets,
                                              double  value_jet_et,
                                              double  value_jet_eta);
    
    
    std::vector<const pat::Jet* > JetSelector(const std::vector<pat::Jet>  & thePatJets,
                                              double  value_jet_et,
                                              double  value_jet_eta,
                                              std::vector<const pat::Electron*> vElectrons,
                                              std::vector<const pat::Muon*> vMuons);
    
    double MT_calc(TLorentzVector Vect, double MET, double MET_Phi);
    double Mll_calc(TLorentzVector Vect1, TLorentzVector Vect2);

}

#endif
