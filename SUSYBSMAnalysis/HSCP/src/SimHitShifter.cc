// -*- C++ -*-
//
// Package:    SimHitShifter
// Class:      SimHitShifter
// 
/**\class SimHitShifter SimHitShifter.cc simhitshifter/SimHitShifter/src/SimHitShifter.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Camilo Andres Carrillo Montoya,40 2-B15,+41227671625,
//         Created:  Mon Aug 30 18:35:05 CEST 2010
// $Id: SimHitShifter.cc,v 1.1 2011/11/18 03:52:58 jiechen Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/RPCDigi/interface/RPCDigi.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include <DataFormats/RPCRecHit/interface/RPCRecHit.h>
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
#include <DataFormats/GeometrySurface/interface/LocalError.h>
#include <DataFormats/GeometryVector/interface/LocalPoint.h>
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DataFormats/DetId/interface/DetId.h"
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include <DataFormats/GeometrySurface/interface/LocalError.h>
#include <DataFormats/GeometryVector/interface/LocalPoint.h>
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
//#include "FastSimulation/Tracking/test/FastTrackAnalyzer.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"

#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"

#include "DataFormats/MuonDetId/interface/CSCDetId.h"

#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include <Geometry/RPCGeometry/interface/RPCRoll.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <Geometry/RPCGeometry/interface/RPCGeomServ.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>

#include <cmath>

//Root
#include "TFile.h"
#include "TF1.h"
#include "TH1F.h"
#include "TH1.h"
#include "TH2F.h"
#include "TROOT.h"
#include "TMath.h"
#include "TCanvas.h"

//Track
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h" 
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include<fstream>


//
// class declaration
//

class SimHitShifter : public edm::EDProducer {
   public:
      explicit SimHitShifter(const edm::ParameterSet&);
      ~SimHitShifter();
  //edm::ESHandle <RPCGeometry> rpcGeo;
      virtual void beginRun(const edm::Run&, const edm::EventSetup&);
      std::map<int,float> shiftinfo;


   private:
      std::string ShiftFileName;
      virtual void beginJob(const edm::Run&, const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
    
};

SimHitShifter::SimHitShifter(const edm::ParameterSet& iConfig)
{
  std::cout<<"in the constructor"<<std::endl;
  
  ShiftFileName  = iConfig.getUntrackedParameter<std::string>("ShiftFileName","/afs/cern.ch/user/c/carrillo/simhits/CMSSW_3_5_8_patch2/src/simhitshifter/SimHitShifter/Merged_Muon_RawId_Shift.txt");
 
  //iSetup.get<MuonGeometryRecord>().get(rpcGeo);

  std::ifstream ifin(ShiftFileName.c_str());

  int rawId;
  float offset;

  std::cout<<"In the constructor, The name of the file is "<<ShiftFileName.c_str()<<std::endl;

  if(!ifin) std::cout<<"Problem reading the map rawId shift "<<ShiftFileName.c_str()<<std::endl;
  assert(ifin);

  while (ifin.good()){
    ifin >>rawId >>offset;
    shiftinfo[rawId]=offset;
    std::cout<<"rawId ="<<rawId<<" offset="<<offset<<std::endl;
  }
  
  produces<edm::PSimHitContainer>("MuonCSCHits");
  produces<edm::PSimHitContainer>("MuonDTHits");
  produces<edm::PSimHitContainer>("MuonRPCHits");
}


SimHitShifter::~SimHitShifter()
{
}

void SimHitShifter::produce(edm::Event& iEvent, const edm::EventSetup& iSetup){
   using namespace edm;

   //std::cout << " Getting the SimHits " <<std::endl;
   std::vector<edm::Handle<edm::PSimHitContainer> > theSimHitContainers;
   iEvent.getManyByType(theSimHitContainers);
   //std::cout << " The Number of sim Hits is  " << theSimHitContainers.size() <<std::endl;

   std::auto_ptr<edm::PSimHitContainer> pcsc(new edm::PSimHitContainer);
   std::auto_ptr<edm::PSimHitContainer> pdt(new edm::PSimHitContainer);
   std::auto_ptr<edm::PSimHitContainer> prpc(new edm::PSimHitContainer);

   std::vector<PSimHit> theSimHits;

   using std::oct;
   using std::dec;
   
   for (int i = 0; i < int(theSimHitContainers.size()); i++){
     theSimHits.insert(theSimHits.end(),theSimHitContainers.at(i)->begin(),theSimHitContainers.at(i)->end());
   } 

   for (std::vector<PSimHit>::const_iterator iHit = theSimHits.begin(); iHit != theSimHits.end(); iHit++){
     DetId theDetUnitId((*iHit).detUnitId());
     DetId simdetid= DetId((*iHit).detUnitId());

     if(simdetid.det()!=DetId::Muon) continue;

     float newtof = 0;
    
     if(simdetid.det()==DetId::Muon &&  simdetid.subdetId()== MuonSubdetId::RPC){//Only RPCs
       //std::cout<<"\t\t We have an RPC Sim Hit! in t="<<(*iHit).timeOfFlight()<<" DetId="<<(*iHit).detUnitId()<<std::endl;
       if(shiftinfo.find(simdetid.rawId())==shiftinfo.end()){
	 std::cout<<"RPC Warning the RawId = "<<simdetid.det()<<" | "<<simdetid.rawId()<<"is not in the map"<<std::endl;
	 newtof = (*iHit).timeOfFlight();
       }else{
	 newtof = (*iHit).timeOfFlight()+shiftinfo[simdetid.rawId()];
       }
       
       PSimHit hit((*iHit).entryPoint(),(*iHit).exitPoint(),(*iHit).pabs(),
		   newtof,
		   (*iHit).energyLoss(),(*iHit).particleType(),simdetid,(*iHit). trackId(),(*iHit).thetaAtEntry(),(*iHit).phiAtEntry(),(*iHit).processType());
       prpc->push_back(hit);
     }
     else if(simdetid.det()==DetId::Muon &&  simdetid.subdetId()== MuonSubdetId::DT){//Only DTs
       int RawId = simdetid.rawId(); 
       std::cout<<"We found a DT simhit the RawId in Dec is";
       std::cout<<dec<<RawId<<std::endl;
       std::cout<<"and in oct"<<std::endl;
       std::cout<<oct<<RawId<< std::endl;
       std::cout<<"once masked in oct "<<std::endl;
       int compressedRawId = RawId/8/8/8/8/8;
       std::cout<<compressedRawId<<std::endl;
       std::cout<<"extendedRawId"<<std::endl;
       int extendedRawId = compressedRawId*8*8*8*8*8;
       std::cout<<extendedRawId<<std::endl;
       std::cout<<"converted again in decimal"<<std::endl;
       std::cout<<dec<<extendedRawId<<std::endl;
       
       if(shiftinfo.find(extendedRawId)==shiftinfo.end()){
	 //std::cout<<"DT Warning the RawId = "<<extendedRawId<<"is not in the map"<<std::endl;
	 newtof = (*iHit).timeOfFlight();
       }else{
	 newtof = (*iHit).timeOfFlight()+shiftinfo[extendedRawId];
	 std::cout<<"RawId = "<<extendedRawId<<"is in the map "<<(*iHit).timeOfFlight()<<" "<<newtof<<std::endl;
       }
       
       std::cout<<"\t\t We have an DT Sim Hit! in t="<<(*iHit).timeOfFlight()<<" DetId="<<(*iHit).detUnitId()<<std::endl;
       PSimHit hit((*iHit).entryPoint(),(*iHit).exitPoint(),(*iHit).pabs(),
		   newtof,
		   (*iHit).energyLoss(),(*iHit).particleType(),simdetid,(*iHit). trackId(),(*iHit).thetaAtEntry(),(*iHit).phiAtEntry(),(*iHit).processType());
       pdt->push_back(hit);
     }
     else if(simdetid.det()==DetId::Muon &&  simdetid.subdetId()== MuonSubdetId::CSC){//Only CSCs
       //std::cout<<"\t\t We have an CSC Sim Hit! in t="<<(*iHit).timeOfFlight()<<" DetId="<<(*iHit).detUnitId()<<std::endl;
       
       CSCDetId TheCSCDetId = CSCDetId(simdetid);
       CSCDetId TheChamberDetId = TheCSCDetId.chamberId();
       
       if(shiftinfo.find(TheChamberDetId.rawId())==shiftinfo.end()){
	 std::cout<<"The RawId is not in the map,perhaps it is on the CSCs station 1 ring 4"<<std::endl;
	 if(TheChamberDetId.station()==1 && TheChamberDetId.ring()==4){
	   CSCDetId TheChamberDetIdNoring4= CSCDetId(TheChamberDetId.endcap(),TheChamberDetId.station(),1 //1 instead of 4
						     ,TheChamberDetId.chamber(),TheChamberDetId.layer());
	   
	   if(shiftinfo.find(TheChamberDetIdNoring4.rawId())==shiftinfo.end()){
	     std::cout<<"CSC Warning the RawId = "<<TheChamberDetIdNoring4<<" "<<TheChamberDetIdNoring4.rawId()<<"is not in the map"<<std::endl;
	     newtof = (*iHit).timeOfFlight();
	   }else{
	     newtof = (*iHit).timeOfFlight()+shiftinfo[TheChamberDetIdNoring4.rawId()];
	   }
	 }
       }else{
	 newtof = (*iHit).timeOfFlight()+shiftinfo[TheChamberDetId.rawId()];
       }
       
       PSimHit hit((*iHit).entryPoint(),(*iHit).exitPoint(),(*iHit).pabs(),
		   newtof,
		   (*iHit).energyLoss(),(*iHit).particleType(),simdetid,(*iHit). trackId(),(*iHit).thetaAtEntry(),(*iHit).phiAtEntry(),(*iHit).processType());
       
       std::cout<<"CSC check newtof"<<newtof<<" "<<(*iHit).timeOfFlight()<<std::endl;
       if(newtof==(*iHit).timeOfFlight())std::cout<<"Warning!!!"<<std::endl;
       pcsc->push_back(hit);
     }     
   }

   std::cout<<"Putting collections in the event"<<std::endl;

   iEvent.put(pcsc,"MuonCSCHits");
   iEvent.put(pdt,"MuonDTHits");
   iEvent.put(prpc,"MuonRPCHits");
   
}

void 
SimHitShifter::beginRun(const edm::Run& run, const edm::EventSetup& iSetup)
{

}

// ------------ method called once each job just before starting event loop  ------------
void 
SimHitShifter::beginJob(const edm::Run& run, const edm::EventSetup& iSetup)
{

}

// ------------ method called once each job just after ending the event loop  ------------
void 
SimHitShifter::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(SimHitShifter);
