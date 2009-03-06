#ifndef SimTrackSimVertexDumper_H
#define SimTrackSimVertexDumper_H
// 
//
// system include files
#include <memory>

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "HepMC/GenEvent.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"

class TH1F;
class TH2F;
#include <vector>

class SimTrackSimVertexDumper : public edm::EDAnalyzer{
   public:
  explicit SimTrackSimVertexDumper( const edm::ParameterSet& );
  virtual ~SimTrackSimVertexDumper() {};
  
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
  virtual void beginJob( const edm::EventSetup& );
  virtual void endJob(){};

  void particleAssignDaughters( HepMC::GenParticle* vp,float,float);

 private:
  
  edm::Service<TFileService> fs;
  
  

  TH1F * genP;                    
  TH1F * genPt;                   
  TH1F * genEta;                  
  TH1F * genPID;                  
  TH1F * genPIDLossNew;          
  TH1F * genPIDLossOld;         
  TH1F * genZimpact;              
  TH1F * genPWithOldPtCut;
  TH1F * genPtWithNewPCut;        
  TH1F * genZWithOldEtaCut;       
  TH1F * genEtaWithNewZCut;       
  TH1F * genPLossWithOldPtCut;
  TH1F * genPtLossWithNewPCut;        
  TH1F * genZLossWithOldEtaCut;       
  TH1F * genEtaLossWithNewZCut;       
         
  TH2F * genPversusPt;
  TH2F * genZversusEta;
                        
  TH1F * genChainP;               
  TH1F * genChainPt;              
  TH1F * genChainEta;             
  TH1F * genChainPID;             
                                 
  TH1F * genChainPWithOldPtCut;
  TH1F * genChainPtWithNewPCut;   
  TH1F * genChainZWithOldEtaCut;  
  TH1F * genChainEtaWithNewZCut;  

  std::string HepMCLabel;
  std::string SimTkLabel;
  std::string SimVtxLabel;
  bool dumpHepMC;

};

#endif
