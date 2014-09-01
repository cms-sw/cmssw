
//FIXME THIS FILE NEED A BIG CLEANING... (Loic)


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
// $Id: SimHitShifter.cc,v 1.3 2012/08/15 14:41:07 querten Exp $
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
#include "FastSimulation/Tracking/test/FastTrackAnalyzer.h"

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
#include "TRandom3.h"  
#include "TStopwatch.h"  

//Track
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h" 
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include<fstream>

using std::cout;
using std::endl;

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
  bool Debug;
  bool KillMuonHits;
  bool ShiftAmpMuon;
  bool ShiftAmpTrack;
  bool RemoveNonSignalTrkHits;
  double VaryMuonThresh;  
  double AmpMuonShiftSize;
  double AmpTrackShiftSize;  
  bool ShiftTiming;  
  TRandom3 randGlobal;  

  int nCSCHits; 
  int nRPCHits; 
  int nDTHits;  
  TH1F* _hProbDT;  // probability for a DT hit to be recorded 

  bool keepDTMuonHit(double amplitude);  
  double getProbRecordHit(double dE, bool print=false);  
  void fillProbKeepDTMuonHit();  

  std::vector<int> particleIds_;
  std::string ShiftFileName; 
  virtual void beginJob(const edm::Run&, const edm::EventSetup&) ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
};

double SimHitShifter::getProbRecordHit(double dE, bool print) {
  // Calculate probability that a muon DT hit with energy loss dE (in keV) is recorded. 
  // Do calculation based on email from Anna Meneguzzo, 2012-06-03: 
  // https://hypernews.cern.ch/HyperNews/CMS/get/dt-performance/118/1/1/1/1/1/1/1.html
  // First get expected number of primary ionization electrons.  This is done based on a Poisson distribution. 

  double _probHitRecord = 0.50;  // probability to produce a primary election in a region such that it reaches the wire                                                  
  // Value from Anna Meneguzzo email, 2012-06-03:
  // If you look at fig 3 for example of http://cmsdoc.cern.ch/documents/02/note02_002.pdf  the electric
  // field lines which arrive on the wire do not cover the entire cell drift region, as can be seen in fig. 3. Approximately
  // only one half of the charge reaches the wire and it is there multiplied. In the note Enrico assumes 60% " From fig. 3 we evaluate that the average
  // collection factor C is C = 60 %" . I reported 50% indeed it is something between  50% and 60%.
  // The 50% is the probability that a primary electron is produced in the region such that it reach the wire.

  int _ntrials = 100000;   // number of trials to use to get the probability
  double _primElePerKev = 10;  // from Anna
  double meanPrimEleExp = dE * _primElePerKev;

  TF1 *fPoisson = new TF1("fPoisson", "TMath::Poisson(x,[0])", 0, 20);
  fPoisson->SetParameter(0, meanPrimEleExp);

  int nPass = 0;
  for (int itrial=0; itrial<_ntrials; itrial++) {
    bool passTrial = false;
    double numPrimEle = fPoisson->GetRandom();
    int numPrimEleInt = floor(numPrimEle + 0.5);
    for (int iEle=0; iEle<numPrimEleInt; iEle++) {
      if (randGlobal.Rndm()<_probHitRecord) { passTrial = true; }
    }
    if (passTrial) nPass++;    
  }

  double probRecord = double(nPass) / _ntrials;
  if (print) cout << "For energy loss of " << dE 
		  << " keV, # expected primary electrons = " << meanPrimEleExp
                  << "; probability at least one is recorded = " << probRecord
                  << endl;  
  return probRecord;
}

void SimHitShifter::fillProbKeepDTMuonHit() { 
  _hProbDT = new TH1F("_hProbDT", ";DT amplitude (GeV);probability to retain DT hit", 200, 0, 2e-6);
  for (int ibin=1; ibin<=_hProbDT->GetNbinsX(); ibin++) {
    double ampGeV = _hProbDT->GetBinCenter(ibin);  
    double ampKeV = ampGeV * 1.e6;  
    double prob = getProbRecordHit(ampKeV);
    _hProbDT->SetBinContent(ibin, prob);
  }
}

bool SimHitShifter::keepDTMuonHit(double amplitude){ 
  int ibin        = _hProbDT->FindBin(amplitude);  
  double probKeep = _hProbDT->GetBinContent(ibin);  
  if (ibin<1)                     probKeep = 0;
  if (ibin>_hProbDT->GetNbinsX()) probKeep = 1.0;  

  double randNum = randGlobal.Rndm(); 
  bool keep = randNum<probKeep;  
  return keep;
}  

SimHitShifter::SimHitShifter(const edm::ParameterSet& iConfig)
{
  ShiftFileName  = iConfig.getUntrackedParameter<std::string>("ShiftFileName","/afs/cern.ch/user/c/carrillo/simhits/CMSSW_3_5_8_patch2/src/simhitshifter/SimHitShifter/Merged_Muon_RawId_Shift.txt");
  std::ifstream ifin(ShiftFileName.c_str());

  int rawId;
  float offset;

  if(!ifin) std::cout<<"Problem reading the map rawId shift "<<ShiftFileName.c_str()<<std::endl;
  assert(ifin);

  while (ifin.good()){
    ifin >>rawId >>offset;
    shiftinfo[rawId]=offset;
    //std::cout<<"rawId ="<<rawId<<" offset="<<offset<<std::endl;
  }

  ShiftTiming            = iConfig.getUntrackedParameter<bool>("ShiftTiming");
  ShiftAmpMuon           = iConfig.getUntrackedParameter<bool>("ShiftAmpMuon");
  Debug                  = iConfig.getUntrackedParameter<bool>("Debug");  
  KillMuonHits           = iConfig.getUntrackedParameter<bool>("KillMuonHits");
  RemoveNonSignalTrkHits = iConfig.getUntrackedParameter<bool>("RemoveNonSignalTrkHits");
  ShiftAmpTrack          = iConfig.getUntrackedParameter<bool>("ShiftAmpTrack");
  VaryMuonThresh         = iConfig.getUntrackedParameter<double>("VaryMuonThresh"); 
  //  VaryMuonThresh = 0.99;  
  AmpMuonShiftSize   = iConfig.getUntrackedParameter<double>("AmpMuonShiftSize");
  AmpTrackShiftSize  = iConfig.getUntrackedParameter<double>("AmpTrackShiftSize");
  particleIds_       = iConfig.getParameter< std::vector<int> >("particleIds");  

  cout << "Running with parameters:  " << endl
       << "  Debug                  = " << Debug               << endl  
       << "  ShiftTiming            = " << ShiftTiming         << endl  
       << "  ShiftAmpMuon           = " << ShiftAmpMuon        << endl  
       << "  ShiftAmpTrack          = " << ShiftAmpTrack       << endl  
       << "  KillMuonHits           = " << KillMuonHits        << endl  
       << "  RemoveNonSignalTrkHits = " << RemoveNonSignalTrkHits << endl  
       << "  VaryMuonThresh         = " << VaryMuonThresh        << endl  
       << "  AmpMuonShiftSize       = " << AmpMuonShiftSize    << endl  
       << "  AmpTrackShiftSize      = " << AmpTrackShiftSize   << endl  
       << endl;  



  if (ShiftTiming  || 
      ShiftAmpMuon ||
      KillMuonHits) { 
    produces<edm::PSimHitContainer>("MuonCSCHits");
    produces<edm::PSimHitContainer>("MuonDTHits");
    produces<edm::PSimHitContainer>("MuonRPCHits");
  }  

  if (ShiftAmpTrack) {
    produces<edm::PSimHitContainer>("TrackerHitsPixelBarrelHighTof");
    produces<edm::PSimHitContainer>("TrackerHitsPixelBarrelLowTof");
    produces<edm::PSimHitContainer>("TrackerHitsPixelEndcapHighTof");
    produces<edm::PSimHitContainer>("TrackerHitsPixelEndcapLowTof");
    produces<edm::PSimHitContainer>("TrackerHitsTECHighTof");
    produces<edm::PSimHitContainer>("TrackerHitsTECLowTof");
    produces<edm::PSimHitContainer>("TrackerHitsTIBHighTof");
    produces<edm::PSimHitContainer>("TrackerHitsTIBLowTof");
    produces<edm::PSimHitContainer>("TrackerHitsTIDHighTof");
    produces<edm::PSimHitContainer>("TrackerHitsTIDLowTof");
    produces<edm::PSimHitContainer>("TrackerHitsTOBHighTof");
    produces<edm::PSimHitContainer>("TrackerHitsTOBLowTof");
  }

  // make plot of probability to keep DT muon hits
  fillProbKeepDTMuonHit();  
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
  std::auto_ptr<edm::PSimHitContainer> pdt (new edm::PSimHitContainer);
  std::auto_ptr<edm::PSimHitContainer> prpc(new edm::PSimHitContainer);
  std::auto_ptr<edm::PSimHitContainer> ptrk(new edm::PSimHitContainer);
  std::auto_ptr<edm::PSimHitContainer> ptrkPXBH(new edm::PSimHitContainer);
  std::auto_ptr<edm::PSimHitContainer> ptrkPXBL(new edm::PSimHitContainer);
  std::auto_ptr<edm::PSimHitContainer> ptrkPXEH(new edm::PSimHitContainer);
  std::auto_ptr<edm::PSimHitContainer> ptrkPXEL(new edm::PSimHitContainer);
  std::auto_ptr<edm::PSimHitContainer> ptrkTECH(new edm::PSimHitContainer);
  std::auto_ptr<edm::PSimHitContainer> ptrkTECL(new edm::PSimHitContainer);
  std::auto_ptr<edm::PSimHitContainer> ptrkTIBH(new edm::PSimHitContainer);
  std::auto_ptr<edm::PSimHitContainer> ptrkTIBL(new edm::PSimHitContainer);
  std::auto_ptr<edm::PSimHitContainer> ptrkTIDH(new edm::PSimHitContainer);
  std::auto_ptr<edm::PSimHitContainer> ptrkTIDL(new edm::PSimHitContainer);
  std::auto_ptr<edm::PSimHitContainer> ptrkTOBH(new edm::PSimHitContainer);
  std::auto_ptr<edm::PSimHitContainer> ptrkTOBL(new edm::PSimHitContainer);
  std::vector<PSimHit> theSimHits;

  using std::oct;
  using std::dec;

  std::vector<unsigned int> simTrackIds;  

  Handle<edm::SimTrackContainer> simTracksHandle;  
  iEvent.getByLabel("g4SimHits",simTracksHandle);   
  const SimTrackContainer simTracks = *(simTracksHandle.product());  

  SimTrackContainer::const_iterator simTrack;  
   
  for (int i = 0; i < int(theSimHitContainers.size()); i++){
    theSimHits.insert(theSimHits.end(),theSimHitContainers.at(i)->begin(),theSimHitContainers.at(i)->end());
  } 


  for (simTrack = simTracks.begin(); simTrack != simTracks.end(); ++simTrack) {
    // Check if the particleId is in our list
    std::vector<int>::const_iterator partIdItr = find(particleIds_.begin(),particleIds_.end(),simTrack->type());
    if(partIdItr==particleIds_.end()) continue;
    if (Debug) cout << "MC Signal:  pt = " << (*simTrack).momentum().pt() << endl;  
    simTrackIds.push_back(simTrack->trackId());  
  }  
   
  TRandom3 rand;  

  nCSCHits = 0;
  nRPCHits = 0;
  nDTHits  = 0;  

  // threshold values from running: 
  //  ~/workdirNewer]$ root -l -b -q 'studyHitAmplitudes.C+' | & tee studyHitAmplitudes.log  
  // old values, obtained with q=1, m=100 GeV signal MC
   // double thresholdCSC =  2.69356e-07;  
   // double thresholdRPC =  8.32867e-08; 
   double thresholdDT  =  2.70418e-08;  

   // new values, for simulated muons:  
   double thresholdCSC = 3.58981e-07; 
   double thresholdRPC = 7.58294e-08;  
   // double thresholdDT  = 3.25665e-08;  
   
  thresholdCSC *= 1. + VaryMuonThresh;  
  thresholdRPC *= 1. + VaryMuonThresh;  
  thresholdDT  *= 1. + VaryMuonThresh;  



  if (ShiftTiming  || 
      ShiftAmpMuon || 
      KillMuonHits) { 

    for (std::vector<PSimHit>::const_iterator iHit = theSimHits.begin(); iHit != theSimHits.end(); iHit++){
      DetId theDetUnitId((*iHit).detUnitId());
      DetId simdetid= DetId((*iHit).detUnitId());

      if(simdetid.det()!=DetId::Muon) continue;  

      float newtof = 0;
      float newamplitude = 0;
    
      if(simdetid.det()==DetId::Muon &&  simdetid.subdetId()== MuonSubdetId::RPC){//Only RPCs
	//std::cout<<"\t\t We have an RPC Sim Hit! in t="<<(*iHit).timeOfFlight()<<" DetId="<<(*iHit).detUnitId()<<std::endl;
	if(shiftinfo.find(simdetid.rawId())==shiftinfo.end()){
	  // std::cout<<"RPC Warning the RawId = "<<simdetid.det()<<" | "<<simdetid.rawId()<<"is not in the map"<<std::endl;
	  newtof = (*iHit).timeOfFlight();
	}else{
	  newtof = (*iHit).timeOfFlight();
	  newamplitude = (*iHit).energyLoss();
	  if (Debug) cout << "RPC:   old amplitude = " << newamplitude << "  new amplitude = ";
	  if(ShiftTiming) newtof = (*iHit).timeOfFlight()+shiftinfo[simdetid.rawId()];
	  if(ShiftAmpMuon) newamplitude = (*iHit).energyLoss()*(1+AmpMuonShiftSize);
	  if (Debug) cout << newamplitude << endl;
	}
       
	nRPCHits++;  
	PSimHit hit((*iHit).entryPoint(),(*iHit).exitPoint(),(*iHit).pabs(),
		    newtof,
		    newamplitude,(*iHit).particleType(),simdetid,(*iHit). trackId(),(*iHit).thetaAtEntry(),(*iHit).phiAtEntry(),(*iHit).processType());

	// Note that TRandom3::Rndm() returns uniformly-distributed floating points in ]0,1]  
	//	if (!KillMuonHits || (rand.Rndm()>KillMuonRate)) prpc->push_back(hit);

	if (!KillMuonHits || (newamplitude>thresholdRPC)) prpc->push_back(hit); // original
	//	if (!KillMuonHits || (newamplitude>0)) prpc->push_back(hit);  // testing   
      } else if(simdetid.det()==DetId::Muon &&  simdetid.subdetId()== MuonSubdetId::DT) { //Only DTs
	int RawId = simdetid.rawId(); 
	//       std::cout<<"We found a DT simhit the RawId in Dec is";
	//       std::cout<<dec<<RawId<<std::endl;
	//       std::cout<<"and in oct"<<std::endl;
	//       std::cout<<oct<<RawId<< std::endl;
	//       std::cout<<"once masked in oct "<<std::endl;
	int compressedRawId = RawId/8/8/8/8/8;
	//       std::cout<<compressedRawId<<std::endl;
	//       std::cout<<"extendedRawId"<<std::endl;
	int extendedRawId = compressedRawId*8*8*8*8*8;
	//       std::cout<<extendedRawId<<std::endl;
	//       std::cout<<"converted again in decimal"<<std::endl;
	//       std::cout<<dec<<extendedRawId<<std::endl;
       
	if(shiftinfo.find(extendedRawId)==shiftinfo.end()){
	  //std::cout<<"DT Warning the RawId = "<<extendedRawId<<"is not in the map"<<std::endl;
	  newtof = (*iHit).timeOfFlight();  
	}else{
	  newtof = (*iHit).timeOfFlight();
	  newamplitude = (*iHit).energyLoss();
	  if (Debug) cout << "DT:   old amplitude = " << newamplitude << "  new amplitude = ";
	  if(ShiftTiming) newtof = (*iHit).timeOfFlight()+shiftinfo[simdetid.rawId()];
	  // 	  if(ShiftAmpMuon) newamplitude = (*iHit).energyLoss()*(1+AmpMuonShiftSize); // original
	  newamplitude = (*iHit).energyLoss()*(1-VaryMuonThresh);  // testing  
	  if (Debug) cout << newamplitude << endl;
	  //newtof = (*iHit).timeOfFlight()+shiftinfo[extendedRawId];
	  //          std::cout<<"RawId = "<<extendedRawId<<"is in the map "<<(*iHit).timeOfFlight()<<" "<<newtof<<std::endl;
	}
       
	//       std::cout<<"\t\t We have an DT Sim Hit! in t="<<(*iHit).timeOfFlight()<<" DetId="<<(*iHit).detUnitId()<<std::endl;
	nDTHits++;  
	PSimHit hit((*iHit).entryPoint(),(*iHit).exitPoint(),(*iHit).pabs(),
		    newtof,
		    newamplitude,(*iHit).particleType(),simdetid,(*iHit). trackId(),(*iHit).thetaAtEntry(),(*iHit).phiAtEntry(),(*iHit).processType());
	//	if (!KillMuonHits || (rand.Rndm()>KillMuonRate)) pdt->push_back(hit);

	//	if (!KillMuonHits || (newamplitude>thresholdDT)) pdt->push_back(hit);  // original
	if (!KillMuonHits || (keepDTMuonHit(newamplitude))) pdt->push_back(hit);  // testing

      } else if(simdetid.det()==DetId::Muon &&  simdetid.subdetId()== MuonSubdetId::CSC) { //Only CSCs
	//       std::cout<<"\t\t We have an CSC Sim Hit! in t="<<(*iHit).timeOfFlight()<<" DetId="<<(*iHit).detUnitId()<<std::endl;
       
	CSCDetId TheCSCDetId = CSCDetId(simdetid);
	CSCDetId TheChamberDetId = TheCSCDetId.chamberId();
       
	if(shiftinfo.find(TheChamberDetId.rawId())==shiftinfo.end()){
	  //	 std::cout<<"The RawId is not in the map,perhaps it is on the CSCs station 1 ring 4"<<std::endl;
	  if(TheChamberDetId.station()==1 && TheChamberDetId.ring()==4){
	    CSCDetId TheChamberDetIdNoring4= CSCDetId(TheChamberDetId.endcap(),TheChamberDetId.station(),1 //1 instead of 4
						      ,TheChamberDetId.chamber(),TheChamberDetId.layer());
	   
	    if(shiftinfo.find(TheChamberDetIdNoring4.rawId())==shiftinfo.end()){
	      //	     std::cout<<"CSC Warning the RawId = "<<TheChamberDetIdNoring4<<" "<<TheChamberDetIdNoring4.rawId()<<"is not in the map"<<std::endl;
	      newtof = (*iHit).timeOfFlight();
	    }else{
	      newtof = (*iHit).timeOfFlight();
	      newamplitude = (*iHit).energyLoss();
	      if (Debug) cout << "CSC ring4: old amplitude = " << newamplitude << "  new amplitude = ";
	      if(ShiftTiming) newtof = (*iHit).timeOfFlight()+shiftinfo[TheChamberDetIdNoring4.rawId()];
	      if(ShiftAmpMuon) newamplitude = (*iHit).energyLoss()*(1+AmpMuonShiftSize);  
	      if (Debug) cout << newamplitude << endl;
	    }
	  }
	} else {
	  newtof = (*iHit).timeOfFlight();
	  newamplitude = (*iHit).energyLoss();
	  if (Debug) cout << "CSC:   old amplitude = " << newamplitude << "  new amplitude = ";
	  if(ShiftTiming) 	 newtof = (*iHit).timeOfFlight()+shiftinfo[TheChamberDetId.rawId()];
	  if(ShiftAmpMuon) newamplitude = (*iHit).energyLoss()*(1+AmpMuonShiftSize);  
	  if (Debug) cout << newamplitude << endl;
	}
       
	nCSCHits++;  
	PSimHit hit((*iHit).entryPoint(),(*iHit).exitPoint(),(*iHit).pabs(),
		    newtof,
		    newamplitude,(*iHit).particleType(),simdetid,(*iHit). trackId(),(*iHit).thetaAtEntry(),(*iHit).phiAtEntry(),(*iHit).processType());
       
	//       std::cout<<"CSC check newtof"<<newtof<<" "<<(*iHit).timeOfFlight()<<std::endl;
	//       if(newtof==(*iHit).timeOfFlight())std::cout<<"Warning!!!"<<std::endl;
	//	if (!KillMuonHits || (rand.Rndm()>KillMuonRate)) pcsc->push_back(hit);

	if (!KillMuonHits || (newamplitude>thresholdCSC)) pcsc->push_back(hit); // original
	//	if (!KillMuonHits || (newamplitude>0)) pcsc->push_back(hit);  // testing
      } else {  // end CSC's  
	if (Debug) cout << "Warning:  Found hit with simdetid.det()==DetId::Muon but not matched to CSC, RPC, or DT." << endl;  
      }      

    }  //    for (std::vector<PSimHit>::const_iterator iHit = theSimHits.begin(); iHit != theSimHits.end(); iHit++){

  } // end if (ShiftTiming || ShiftAmpMuon || KillMuonHits) { 


  if (Debug) cout << "nCSCHits = " << nCSCHits << endl;  
  if (Debug) cout << "nRPCHits = " << nRPCHits << endl;  
  if (Debug) cout << "nDTHits  = " << nDTHits  << endl;  


  /////////////////////////////////
  // Now do all the tracker sim hit variations.
  // Code copied from:
  // http://cmslxr.fnal.gov/lxr/source/Validation/TrackerHits/src/TrackerHitAnalyzer.cc#346
  ////////////////////////////////

  if (ShiftAmpTrack) { // only do this section if ShiftAmpTrack is true.  
   
    // iterator to access containers
    //  edm::PSimHitContainer::const_iterator itHit;
   
    double scaleEnergyLoss = 1.;
    if (ShiftAmpTrack) {
      scaleEnergyLoss = 1. + AmpTrackShiftSize;  
    }
    /////////////////////////////////
    // get Pixel Barrel information
    ////////////////////////////////
    // extract low container
    edm::Handle<edm::PSimHitContainer> PxlBrlLowContainer;
    iEvent.getByLabel("g4SimHits","TrackerHitsPixelBarrelLowTof", PxlBrlLowContainer);  
    if (!PxlBrlLowContainer.isValid()) {
      edm::LogError("TrackerHitAnalyzer::analyze")
	<< "Unable to find TrackerHitsPixelBarrelLowTof in event!";
      return;
    }  
 

    // extract high container
    edm::Handle<edm::PSimHitContainer> PxlBrlHighContainer;
    iEvent.getByLabel("g4SimHits","TrackerHitsPixelBarrelHighTof",PxlBrlHighContainer);
    if (!PxlBrlHighContainer.isValid()) {
      edm::LogError("TrackerHitAnalyzer::analyze")
	<< "Unable to find TrackerHitsPixelBarrelHighTof in event!";
      return;
    }


    /////////////////////////////////
    // get Pixel Forward information
    ////////////////////////////////
    // extract low container
    edm::Handle<edm::PSimHitContainer> PxlFwdLowContainer;
    iEvent.getByLabel("g4SimHits","TrackerHitsPixelEndcapLowTof",PxlFwdLowContainer);
    if (!PxlFwdLowContainer.isValid()) {
      edm::LogError("TrackerHitAnalyzer::analyze")
	<< "Unable to find TrackerHitsPixelEndcapLowTof in event!";
      return;
    }

    // extract high container
    edm::Handle<edm::PSimHitContainer> PxlFwdHighContainer;
    iEvent.getByLabel("g4SimHits","TrackerHitsPixelEndcapHighTof",PxlFwdHighContainer);
    if (!PxlFwdHighContainer.isValid()) {
      edm::LogError("TrackerHitAnalyzer::analyze")
	<< "Unable to find TrackerHitsPixelEndcapHighTof in event!";
      return;
    }

    ///////////////////////////////////
    // get Silicon TIB information
    //////////////////////////////////
    // extract TIB low container
    edm::Handle<edm::PSimHitContainer> SiTIBLowContainer;
    iEvent.getByLabel("g4SimHits","TrackerHitsTIBLowTof",SiTIBLowContainer);
    if (!SiTIBLowContainer.isValid()) {
      edm::LogError("TrackerHitProducer::analyze")
	<< "Unable to find TrackerHitsTIBLowTof in event!";
      return;
    }
    //////////////////////////////////
    // extract TIB low container
    edm::Handle<edm::PSimHitContainer> SiTIBHighContainer;
    iEvent.getByLabel("g4SimHits","TrackerHitsTIBHighTof",SiTIBHighContainer);
    if (!SiTIBHighContainer.isValid()) {
      edm::LogError("TrackerHitProducer::analyze")
	<< "Unable to find TrackerHitsTIBHighTof in event!";
      return;
    }
    ///////////////////////////////////
    // get Silicon TOB information
    //////////////////////////////////
    // extract TOB low container
    edm::Handle<edm::PSimHitContainer> SiTOBLowContainer;
    iEvent.getByLabel("g4SimHits","TrackerHitsTOBLowTof",SiTOBLowContainer);
    if (!SiTOBLowContainer.isValid()) {
      edm::LogError("TrackerHitProducer::analyze")
	<< "Unable to find TrackerHitsTOBLowTof in event!";
      return;
    }
    //////////////////////////////////
    // extract TOB low container
    edm::Handle<edm::PSimHitContainer> SiTOBHighContainer;
    iEvent.getByLabel("g4SimHits","TrackerHitsTOBHighTof",SiTOBHighContainer);
    if (!SiTOBHighContainer.isValid()) {
      edm::LogError("TrackerHitProducer::analyze")
	<< "Unable to find TrackerHitsTOBHighTof in event!";
      return;
    }

    ///////////////////////////////////
    // get Silicon TID information
    //////////////////////////////////
    // extract TID low container
    edm::Handle<edm::PSimHitContainer> SiTIDLowContainer;
    iEvent.getByLabel("g4SimHits","TrackerHitsTIDLowTof",SiTIDLowContainer);
    if (!SiTIDLowContainer.isValid()) {
      edm::LogError("TrackerHitProducer::analyze")
	<< "Unable to find TrackerHitsTIDLowTof in event!";
      return;
    }
    //////////////////////////////////
    // extract TID low container
    edm::Handle<edm::PSimHitContainer> SiTIDHighContainer;
    iEvent.getByLabel("g4SimHits","TrackerHitsTIDHighTof",SiTIDHighContainer);
    if (!SiTIDHighContainer.isValid()) {
      edm::LogError("TrackerHitProducer::analyze")
	<< "Unable to find TrackerHitsTIDHighTof in event!";
      return;
    }
    ///////////////////////////////////
    // get Silicon TEC information
    //////////////////////////////////
    // extract TEC low container
    edm::Handle<edm::PSimHitContainer> SiTECLowContainer;
    iEvent.getByLabel("g4SimHits","TrackerHitsTECLowTof",SiTECLowContainer);
    if (!SiTECLowContainer.isValid()) {
      edm::LogError("TrackerHitProducer::analyze")
	<< "Unable to find TrackerHitsTECLowTof in event!";
      return;
    }
    //////////////////////////////////
    // extract TEC low container
    edm ::Handle<edm::PSimHitContainer> SiTECHighContainer;
    iEvent.getByLabel("g4SimHits","TrackerHitsTECHighTof",SiTECHighContainer);
    if (!SiTECHighContainer.isValid()) {
      edm::LogError("TrackerHitProducer::analyze")
	<< "Unable to find TrackerHitsTECHighTof in event!";
      return;
    }

    // For each hit collection, do the amplitude variation only for hits from signal tracks.  
    // Hits from all non-signal tracks also need to filled, but should not be varied in amplitude.  
    double newEnergyLoss;  
    for (std::vector<PSimHit>::const_iterator itHit = PxlBrlHighContainer->begin(); itHit != PxlBrlHighContainer->end(); itHit++){
      newEnergyLoss = (*itHit).energyLoss();  
      if (std::find(simTrackIds.begin(), simTrackIds.end(), itHit->trackId()) != simTrackIds.end()) { newEnergyLoss = scaleEnergyLoss * (*itHit).energyLoss(); }    
      else { if (RemoveNonSignalTrkHits) continue; }  
      if (Debug) cout << "Trk PXBH: old amplitude = " << (*itHit).energyLoss() << "; newamplitude = " << scaleEnergyLoss * (*itHit).energyLoss() << "; process type = " << (*itHit).processType() << "; particle type = " << itHit->particleType() << endl;     
      PSimHit hit((*itHit).entryPoint(), (*itHit).exitPoint(), (*itHit).pabs(), (*itHit).timeOfFlight(), 
		  newEnergyLoss, 
		  (*itHit).particleType(), DetId((*itHit).detUnitId()), (*itHit).trackId(), (*itHit).thetaAtEntry(), (*itHit).phiAtEntry(), (*itHit).processType());
      ptrkPXBH->push_back(hit);  
    }  
    
    for (std::vector<PSimHit>::const_iterator itHit = PxlBrlLowContainer->begin(); itHit != PxlBrlLowContainer->end(); itHit++){
      newEnergyLoss = (*itHit).energyLoss();  
      if (std::find(simTrackIds.begin(), simTrackIds.end(), itHit->trackId()) != simTrackIds.end()) { newEnergyLoss = scaleEnergyLoss * (*itHit).energyLoss(); }    
      else { if (RemoveNonSignalTrkHits) continue; }  
      if (Debug) cout << "Trk PXBL: old amplitude = " << (*itHit).energyLoss() << "; newamplitude = " << scaleEnergyLoss * (*itHit).energyLoss() << "; process type = " << (*itHit).processType() << "; particle type = " << itHit->particleType() << endl;  
      PSimHit hit((*itHit).entryPoint(), (*itHit).exitPoint(), (*itHit).pabs(), (*itHit).timeOfFlight(), 
		  newEnergyLoss, 
		  (*itHit).particleType(), DetId((*itHit).detUnitId()), (*itHit).trackId(), (*itHit).thetaAtEntry(), (*itHit).phiAtEntry(), (*itHit).processType());
      ptrkPXBL->push_back(hit);  
    }  

    for (std::vector<PSimHit>::const_iterator itHit = PxlFwdHighContainer->begin(); itHit != PxlFwdHighContainer->end(); itHit++){
      newEnergyLoss = (*itHit).energyLoss();  
      if (std::find(simTrackIds.begin(), simTrackIds.end(), itHit->trackId()) != simTrackIds.end()) { newEnergyLoss = scaleEnergyLoss * (*itHit).energyLoss(); }    
      else { if (RemoveNonSignalTrkHits) continue; }  
      if (Debug) cout << "Trk PXEH: old amplitude = " << (*itHit).energyLoss() << "; newamplitude = " << scaleEnergyLoss * (*itHit).energyLoss() << "; process type = " << (*itHit).processType() << "; particle type = " << itHit->particleType() << endl;  
      PSimHit hit((*itHit).entryPoint(), (*itHit).exitPoint(), (*itHit).pabs(), (*itHit).timeOfFlight(), 
		  newEnergyLoss, 
		  (*itHit).particleType(), DetId((*itHit).detUnitId()), (*itHit).trackId(), (*itHit).thetaAtEntry(), (*itHit).phiAtEntry(), (*itHit).processType());
      ptrkPXEH->push_back(hit);  
    }  

    for (std::vector<PSimHit>::const_iterator itHit = PxlFwdLowContainer->begin(); itHit != PxlFwdLowContainer->end(); itHit++){
      newEnergyLoss = (*itHit).energyLoss();  
      if (std::find(simTrackIds.begin(), simTrackIds.end(), itHit->trackId()) != simTrackIds.end()) { newEnergyLoss = scaleEnergyLoss * (*itHit).energyLoss(); }    
      else { if (RemoveNonSignalTrkHits) continue; }  
      if (Debug) cout << "Trk PXEL: old amplitude = " << (*itHit).energyLoss() << "; newamplitude = " << scaleEnergyLoss * (*itHit).energyLoss() << "; process type = " << (*itHit).processType() << "; particle type = " << itHit->particleType() << endl;  
      PSimHit hit((*itHit).entryPoint(), (*itHit).exitPoint(), (*itHit).pabs(), (*itHit).timeOfFlight(), 
		  newEnergyLoss, 
		  (*itHit).particleType(), DetId((*itHit).detUnitId()), (*itHit).trackId(), (*itHit).thetaAtEntry(), (*itHit).phiAtEntry(), (*itHit).processType());
      ptrkPXEL->push_back(hit);  
    }  

    for (std::vector<PSimHit>::const_iterator itHit = SiTECHighContainer->begin(); itHit != SiTECHighContainer->end(); itHit++){
      newEnergyLoss = (*itHit).energyLoss();  
      if (std::find(simTrackIds.begin(), simTrackIds.end(), itHit->trackId()) != simTrackIds.end()) { newEnergyLoss = scaleEnergyLoss * (*itHit).energyLoss(); }    
      else { if (RemoveNonSignalTrkHits) continue; }  
      if (Debug) cout << "Trk TECH: old amplitude = " << (*itHit).energyLoss() << "; newamplitude = " << scaleEnergyLoss * (*itHit).energyLoss() << "; process type = " << (*itHit).processType() << "; particle type = " << itHit->particleType() << endl;  
      PSimHit hit((*itHit).entryPoint(), (*itHit).exitPoint(), (*itHit).pabs(), (*itHit).timeOfFlight(), 
		  newEnergyLoss, 
		  (*itHit).particleType(), DetId((*itHit).detUnitId()), (*itHit).trackId(), (*itHit).thetaAtEntry(), (*itHit).phiAtEntry(), (*itHit).processType());
      ptrkTECH->push_back(hit);  
    }  

    for (std::vector<PSimHit>::const_iterator itHit = SiTECLowContainer->begin(); itHit != SiTECLowContainer->end(); itHit++){
      newEnergyLoss = (*itHit).energyLoss();  
      if (std::find(simTrackIds.begin(), simTrackIds.end(), itHit->trackId()) != simTrackIds.end()) { newEnergyLoss = scaleEnergyLoss * (*itHit).energyLoss(); }    
      else { if (RemoveNonSignalTrkHits) continue; }  
      if (Debug) cout << "Trk TECL: old amplitude = " << (*itHit).energyLoss() << "; newamplitude = " << scaleEnergyLoss * (*itHit).energyLoss() << "; process type = " << (*itHit).processType() << "; particle type = " << itHit->particleType() << endl;  
      PSimHit hit((*itHit).entryPoint(), (*itHit).exitPoint(), (*itHit).pabs(), (*itHit).timeOfFlight(), 
		  newEnergyLoss, 
		  (*itHit).particleType(), DetId((*itHit).detUnitId()), (*itHit).trackId(), (*itHit).thetaAtEntry(), (*itHit).phiAtEntry(), (*itHit).processType());
      ptrkTECL->push_back(hit);  
    }  

    for (std::vector<PSimHit>::const_iterator itHit = SiTIBHighContainer->begin(); itHit != SiTIBHighContainer->end(); itHit++){
      newEnergyLoss = (*itHit).energyLoss();  
      if (std::find(simTrackIds.begin(), simTrackIds.end(), itHit->trackId()) != simTrackIds.end()) { newEnergyLoss = scaleEnergyLoss * (*itHit).energyLoss(); }    
      else { if (RemoveNonSignalTrkHits) continue; }  
      if (Debug) cout << "Trk TIBH: old amplitude = " << (*itHit).energyLoss() << "; newamplitude = " << scaleEnergyLoss * (*itHit).energyLoss() << "; process type = " << (*itHit).processType() << "; particle type = " << itHit->particleType() << endl;  
      PSimHit hit((*itHit).entryPoint(), (*itHit).exitPoint(), (*itHit).pabs(), (*itHit).timeOfFlight(), 
		  newEnergyLoss, 
		  (*itHit).particleType(), DetId((*itHit).detUnitId()), (*itHit).trackId(), (*itHit).thetaAtEntry(), (*itHit).phiAtEntry(), (*itHit).processType());
      ptrkTIBH->push_back(hit);  
    }  

    for (std::vector<PSimHit>::const_iterator itHit = SiTIBLowContainer->begin(); itHit != SiTIBLowContainer->end(); itHit++){
      newEnergyLoss = (*itHit).energyLoss();  
      if (std::find(simTrackIds.begin(), simTrackIds.end(), itHit->trackId()) != simTrackIds.end()) { newEnergyLoss = scaleEnergyLoss * (*itHit).energyLoss(); }    
      else { if (RemoveNonSignalTrkHits) continue; }  
      if (Debug) cout << "Trk TIBL: old amplitude = " << (*itHit).energyLoss() << "; newamplitude = " << scaleEnergyLoss * (*itHit).energyLoss() << "; process type = " << (*itHit).processType() << "; particle type = " << itHit->particleType() << endl;  
      PSimHit hit((*itHit).entryPoint(), (*itHit).exitPoint(), (*itHit).pabs(), (*itHit).timeOfFlight(), 
		  newEnergyLoss, 
		  (*itHit).particleType(), DetId((*itHit).detUnitId()), (*itHit).trackId(), (*itHit).thetaAtEntry(), (*itHit).phiAtEntry(), (*itHit).processType());
      ptrkTIBL->push_back(hit);  
    }  

    for (std::vector<PSimHit>::const_iterator itHit = SiTIDHighContainer->begin(); itHit != SiTIDHighContainer->end(); itHit++){
      newEnergyLoss = (*itHit).energyLoss();  
      if (std::find(simTrackIds.begin(), simTrackIds.end(), itHit->trackId()) != simTrackIds.end()) { newEnergyLoss = scaleEnergyLoss * (*itHit).energyLoss(); }    
      else { if (RemoveNonSignalTrkHits) continue; }  
      if (Debug) cout << "Trk TIDH: old amplitude = " << (*itHit).energyLoss() << "; newamplitude = " << scaleEnergyLoss * (*itHit).energyLoss() << "; process type = " << (*itHit).processType() << "; particle type = " << itHit->particleType() << endl;  
      PSimHit hit((*itHit).entryPoint(), (*itHit).exitPoint(), (*itHit).pabs(), (*itHit).timeOfFlight(), 
		  newEnergyLoss, 
		  (*itHit).particleType(), DetId((*itHit).detUnitId()), (*itHit).trackId(), (*itHit).thetaAtEntry(), (*itHit).phiAtEntry(), (*itHit).processType());
      ptrkTIDH->push_back(hit);  
    }  

    for (std::vector<PSimHit>::const_iterator itHit = SiTIDLowContainer->begin(); itHit != SiTIDLowContainer->end(); itHit++){
      newEnergyLoss = (*itHit).energyLoss();  
      if (std::find(simTrackIds.begin(), simTrackIds.end(), itHit->trackId()) != simTrackIds.end()) { newEnergyLoss = scaleEnergyLoss * (*itHit).energyLoss(); }    
      else { if (RemoveNonSignalTrkHits) continue; }  
      if (Debug) cout << "Trk TIDL: old amplitude = " << (*itHit).energyLoss() << "; newamplitude = " << scaleEnergyLoss * (*itHit).energyLoss() << "; process type = " << (*itHit).processType() << "; particle type = " << itHit->particleType() << endl;  
      PSimHit hit((*itHit).entryPoint(), (*itHit).exitPoint(), (*itHit).pabs(), (*itHit).timeOfFlight(), 
		  newEnergyLoss, 
		  (*itHit).particleType(), DetId((*itHit).detUnitId()), (*itHit).trackId(), (*itHit).thetaAtEntry(), (*itHit).phiAtEntry(), (*itHit).processType());
      ptrkTIDL->push_back(hit);  
    }  

    for (std::vector<PSimHit>::const_iterator itHit = SiTOBHighContainer->begin(); itHit != SiTOBHighContainer->end(); itHit++){
      newEnergyLoss = (*itHit).energyLoss();  
      if (std::find(simTrackIds.begin(), simTrackIds.end(), itHit->trackId()) != simTrackIds.end()) { newEnergyLoss = scaleEnergyLoss * (*itHit).energyLoss(); }    
      else { if (RemoveNonSignalTrkHits) continue; }  
      if (Debug) cout << "Trk TOBH: old amplitude = " << (*itHit).energyLoss() << "; newamplitude = " << scaleEnergyLoss * (*itHit).energyLoss() << "; process type = " << (*itHit).processType() << "; particle type = " << itHit->particleType() << endl;  
      PSimHit hit((*itHit).entryPoint(), (*itHit).exitPoint(), (*itHit).pabs(), (*itHit).timeOfFlight(), 
		  newEnergyLoss, 
		  (*itHit).particleType(), DetId((*itHit).detUnitId()), (*itHit).trackId(), (*itHit).thetaAtEntry(), (*itHit).phiAtEntry(), (*itHit).processType());
      ptrkTOBH->push_back(hit);  
    }  

    for (std::vector<PSimHit>::const_iterator itHit = SiTOBLowContainer->begin(); itHit != SiTOBLowContainer->end(); itHit++){
      newEnergyLoss = (*itHit).energyLoss();  
      if (std::find(simTrackIds.begin(), simTrackIds.end(), itHit->trackId()) != simTrackIds.end()) { newEnergyLoss = scaleEnergyLoss * (*itHit).energyLoss(); }    
      else { if (RemoveNonSignalTrkHits) continue; }  
      if (Debug) cout << "Trk TOBL: old amplitude = " << (*itHit).energyLoss() << "; newamplitude = " << scaleEnergyLoss * (*itHit).energyLoss() << "; process type = " << (*itHit).processType() << "; particle type = " << itHit->particleType() << endl;  
      PSimHit hit((*itHit).entryPoint(), (*itHit).exitPoint(), (*itHit).pabs(), (*itHit).timeOfFlight(), 
		  newEnergyLoss, 
		  (*itHit).particleType(), DetId((*itHit).detUnitId()), (*itHit).trackId(), (*itHit).thetaAtEntry(), (*itHit).phiAtEntry(), (*itHit).processType());
      ptrkTOBL->push_back(hit);  
    }  

  } // end    if (ShiftAmpTrack) {

  if (Debug) cout << "Number of hits added of each type: " << endl 
		  << "ptrkPXBH:  " << ptrkPXBH->size() << endl 
		  << "ptrkPXBL:  " << ptrkPXBL->size() << endl 
		  << "ptrkPXEH:  " << ptrkPXEH->size() << endl 
		  << "ptrkPXEL:  " << ptrkPXEL->size() << endl 
		  << "ptrkTECH:  " << ptrkTECH->size() << endl 
		  << "ptrkTECL:  " << ptrkTECL->size() << endl 
		  << "ptrkTIBH:  " << ptrkTIBH->size() << endl 
		  << "ptrkTIBL:  " << ptrkTIBL->size() << endl 
		  << "ptrkTIDH:  " << ptrkTIDH->size() << endl 
		  << "ptrkTIDL:  " << ptrkTIDL->size() << endl 
		  << "ptrkTOBH:  " << ptrkTOBH->size() << endl 
		  << "ptrkTOBL:  " << ptrkTOBL->size() << endl 
		  << "pmuonCSC:  " << pcsc    ->size() << endl 
		  << "pmuonRPC:  " << prpc    ->size() << endl 
		  << "pmuonDT:   " << pdt     ->size() << endl 
		  << endl;  

  if (ShiftTiming  || 
      ShiftAmpMuon || 
      KillMuonHits) { 
    iEvent.put(pcsc,"MuonCSCHits");
    iEvent.put(pdt, "MuonDTHits");
    iEvent.put(prpc,"MuonRPCHits");
  }

  if (ShiftAmpTrack) {
    iEvent.put(ptrkPXBH,"TrackerHitsPixelBarrelHighTof");  
    iEvent.put(ptrkPXBL,"TrackerHitsPixelBarrelLowTof");   
    iEvent.put(ptrkPXEH,"TrackerHitsPixelEndcapHighTof");  
    iEvent.put(ptrkPXEL,"TrackerHitsPixelEndcapLowTof");   
    iEvent.put(ptrkTECH,"TrackerHitsTECHighTof");	  
    iEvent.put(ptrkTECL,"TrackerHitsTECLowTof");	   
    iEvent.put(ptrkTIBH,"TrackerHitsTIBHighTof");	  
    iEvent.put(ptrkTIBL,"TrackerHitsTIBLowTof");	   
    iEvent.put(ptrkTIDH,"TrackerHitsTIDHighTof");	  
    iEvent.put(ptrkTIDL,"TrackerHitsTIDLowTof");	   
    iEvent.put(ptrkTOBH,"TrackerHitsTOBHighTof");	  
    iEvent.put(ptrkTOBL,"TrackerHitsTOBLowTof");    
  }

}   // end void SimHitShifter::produce(edm::Event& iEvent, const edm::EventSetup& iSetup){


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
SimHitShifter::endJob(){}

//define this as a plug-in
DEFINE_FWK_MODULE(SimHitShifter);
