//This file is imported from:
// http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/UserCode/Mangano/WWAnalysis/AnalysisStep/interface/PatElectronEnergyCalibrator.h?revision=1.1&view=markup
#ifndef ElectronEnergyCalibrator_H
#define ElectronEnergyCalibrator_H

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "FWCore/Framework/interface/ESHandle.h"

class ElectronEnergyCalibrator
{
 public:

  ElectronEnergyCalibrator(std::string dataset, bool isAOD, bool isMC, bool updateEnergyError, int applyCorrections, bool verbose, bool synchronization) : dataset_(dataset),
   isAOD_(isAOD), isMC_(isMC), updateEnergyError_(updateEnergyError), applyCorrections_(applyCorrections), verbose_(verbose), synchronization_(synchronization) {}

    //  void correct(pat::Electron &, const edm::Event&, const edm::EventSetup&);
  void correct(reco::GsfElectron &, double r9,  const edm::Event&, const edm::EventSetup&, double newRegEnergy = -9999. , double newRegEnergyError = -9999. );

 private:

  void computeNewEnergy( const reco::GsfElectron &, float r9, int run) ;
  void computeNewRegEnergy( const reco::GsfElectron &, float r9, int run ) ;
  void computeEpCombination( const reco::GsfElectron & electron )  ;

  float newEnergy_ ;
  float newEnergyError_ ;
  
  math::XYZTLorentzVector newMomentum_ ;
  float errorTrackMomentum_ ;
  float finalMomentumError_ ;

  unsigned long long cacheIDTopo ;
  edm::ESHandle<CaloTopology> caloTopo ;
  
  std::string dataset_;
  bool isAOD_;
  bool isMC_;
  bool updateEnergyError_;
  int applyCorrections_;
  bool verbose_;
  bool synchronization_;
   
};

#endif




