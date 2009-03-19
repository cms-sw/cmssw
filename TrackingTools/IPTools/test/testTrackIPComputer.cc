#include <iostream>
#include <vector>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include "TrackingTools/IPTools/interface/ImpactParameterComputer.h"

class testTrackIPComputer : public edm::EDAnalyzer {

public:
  testTrackIPComputer(const edm::ParameterSet &params);
  ~testTrackIPComputer();
  
private:
  virtual void analyze(const edm::Event &event, const edm::EventSetup &es);

};

testTrackIPComputer::testTrackIPComputer(const edm::ParameterSet &params) { }

testTrackIPComputer::~testTrackIPComputer() { }

void testTrackIPComputer::analyze(const edm::Event &evt, const edm::EventSetup &es) {
  
    edm::Handle<edm::View<reco::Muon> > muons;
    evt.getByLabel("muons", muons);
  
    for(edm::View<reco::Muon>::const_iterator itMuon = muons->begin(); itMuon != muons->end(); ++itMuon){ // loop over MuonCollection
      if(!itMuon->isGlobalMuon()) continue; // use only global muons
  
      edm::Handle<reco::VertexCollection> primaryVertex;
      evt.getByLabel(edm::InputTag("offlinePrimaryVertices"), primaryVertex); // use offline PVs for IP calculation, BeamSpot would be an alternative choice
      const reco::VertexCollection& vtxs = *(primaryVertex.product());

      edm::Handle<reco::BeamSpot> beamSpot;
      evt.getByLabel(edm::InputTag("offlineBeamSpot"), beamSpot); // use offline PVs for IP calculation, BeamSpot would be an alternative choice
      const reco::BeamSpot bsp = *(beamSpot.product());
  
      if(vtxs.size()>0){
        reco::TrackRef trRef = itMuon->globalTrack(); // alternatively itMuon->track(), etc could be used
        if(!trRef.isNull()){
	  IPTools::ImpactParameterComputer IPComp(vtxs[0]); // use the vtx with the highest pt sum in this example 

	  Measurement1D mess1D = IPComp.computeIP(es, *trRef);
	  
	  // to calculate the ImpactParameter, its Error and the Significance use
	  double IP             =  mess1D.value();
	  double IPError        =  mess1D.error();
	  double IPSignificance =  mess1D.significance();
	  std::cout << "PrimaryVertex IP2D : IP2DErr :  " << IP << ": \t" << IPError << ": \t" << IPSignificance << "\t" << std::endl;

	  Measurement1D mess1D_3D = IPComp.computeIP(es, *trRef, true);
	  
	  // to calculate the ImpactParameter, its Error and the Significance use
	  IP             =  mess1D_3D.value();
	  IPError        =  mess1D_3D.error();
	  IPSignificance =  mess1D_3D.significance();
	  std::cout << "IP3D : IP3DErr : " << IP << ": \t" << IPError << ": \t" << IPSignificance << std::endl;
        }
      }

      reco::TrackRef trRef = itMuon->globalTrack(); // alternatively itMuon->track(), etc could be used
      if(!trRef.isNull()){
	IPTools::ImpactParameterComputer IPComp(bsp); // use the vtx with the highest pt sum in this example 

	Measurement1D mess1D = IPComp.computeIP(es, *trRef);
	  
	// to calculate the ImpactParameter, its Error and the Significance use
	double IP             =  mess1D.value();
	double IPError        =  mess1D.error();
	double IPSignificance =  mess1D.significance();
	std::cout << "BeamSpot IP2D : IP2DErr :  " << IP << ": \t" << IPError << ": \t" << IPSignificance << std::endl;
	
	Measurement1D mess1D_3D = IPComp.computeIP(es, *trRef, true);
	
	// to calculate the ImpactParameter, its Error and the Significance use
	IP             =  mess1D_3D.value();
	IPError        =  mess1D_3D.error();
	IPSignificance =  mess1D_3D.significance();
	std::cout << "IP3D : IP3DErr : " << IP << ": \t" << IPError << ": \t" << IPSignificance << std::endl;

	
	mess1D = IPComp.computeIPdz(es, *trRef);
	  
	// to calculate the ImpactParameter, its Error and the Significance use
	double dZ             =  mess1D.value();
	double dZErr          =  mess1D.error();
	std::cout << "BeamSpot dZ : dZError :  " << dZ << ": \t" << dZErr << std::endl;
      }


    }

    
}

DEFINE_FWK_MODULE(testTrackIPComputer);
