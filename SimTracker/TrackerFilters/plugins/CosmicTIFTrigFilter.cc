// livio.fano@cern.ch

#include "SimTracker/TrackerFilters/interface/CosmicTIFTrigFilter.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "FastSimulation/BaseParticlePropagator/interface/BaseParticlePropagator.h"
#include "FastSimulation/ParticlePropagator/interface/MagneticFieldMap.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "HepMC/GenVertex.h"
#include "CLHEP/Vector/LorentzVector.h"
#include <map>
#include <vector>

using namespace std;
namespace cms

{
  CosmicTIFTrigFilter::CosmicTIFTrigFilter(const edm::ParameterSet& conf):    conf_(conf)
  {
    trigconf  = conf_.getParameter<int>("trig_conf");
  }
  
  bool CosmicTIFTrigFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
  {
    
    
    edm::Handle<edm::HepMCProduct>HepMCEvt;
    iEvent.getByLabel("source","",HepMCEvt);
    const HepMC::GenEvent* MCEvt = HepMCEvt->GetEvent();
    
    bool hit1=false;
    bool hit2=false;
    bool hit3=false;
    bool hit4=false;
    
    
    for(HepMC::GenEvent::particle_const_iterator i=MCEvt->particles_begin(); i != MCEvt->particles_end();++i)
      {
	//old
	//	int myId = (*i)->ParticleID();
	int myId = (*i)->pdg_id();
	if (abs(myId)==13)
	  {
	    
	    // Get the muon position and momentum
	    //old
//	    HepLorentzVector vertex=(*i)->CreationVertex();
	    HepMC::GenVertex* pv = (*i)->production_vertex();
	    CLHEP::Hep3Vector vertex = (pv->position().x(), pv->position().y(), pv->position().z());
	    //old	 
	    //	    CLHEP::HepLorentzVector momentum=(*i)->Momentum();
	    CLHEP::Hep3Vector momentum( (*i)->momentum().px(),(*i)->momentum().py(),(*i)->momentum().pz());	  
	    
	    if(trigconf==1){

	      CLHEP::Hep3Vector S1(350.,1600.,500.);
	      CLHEP::Hep3Vector S2(350.,-1600.,400.);
	      CLHEP::Hep3Vector S3(350.,1600.,1600.);
	      
	      hit1=Sci_trig(vertex, momentum, S1);
	      hit2=Sci_trig(vertex, momentum, S2);
	      hit3=Sci_trig(vertex, momentum, S3);
	      
	      // trigger conditions
	      
	      if((hit1&&hit2) || (hit3&&hit2))
		{
		  /*
		  cout << "\tGot a trigger in configuration A " << endl; 
		  if(hit1)cout << "hit1 " << endl;
		  if(hit2)cout << "hit2 " << endl;
		  if(hit3)cout << "hit3 " << endl;
		  */
		  trig1++;
		  return true;
		}
	    }else if(trigconf ==2) {

	      CLHEP::Hep3Vector S1(350.,1600.,850.);
	      CLHEP::Hep3Vector S2(0.,-1550.,-1650.);
	      CLHEP::Hep3Vector S3(350.,1600.,2300.);
	      
	      hit1=Sci_trig(vertex, momentum, S1);
	      hit2=Sci_trig(vertex, momentum, S2);
	      hit3=Sci_trig(vertex, momentum, S3);
	      
	      // trigger conditions
	      
	      if((hit1&&hit2) || (hit3&&hit2))
		{
		  /*
		  cout << "\tGot a trigger in configuration B " << endl; 
		  if(hit1)cout << "hit1 " << endl;
		  if(hit2)cout << "hit2 " << endl;
		  if(hit3)cout << "hit3 " << endl;
		  */
		  trig2++;
		  return true;
		}

	    }else if(trigconf ==3) {

	      CLHEP::Hep3Vector S1(350.,1600.,850.);
	      CLHEP::Hep3Vector S3(350.,1600.,2300.);
	      CLHEP::Hep3Vector S2(350.,-1600.,400.);
	      CLHEP::Hep3Vector S4(0.,-1600.,-2000.);
	      
	      hit1=Sci_trig(vertex, momentum, S1);
	      hit2=Sci_trig(vertex, momentum, S2);
	      hit3=Sci_trig(vertex, momentum, S3);
	      hit4=Sci_trig(vertex, momentum, S4);
	      
	      // trigger conditions
	      if((hit1&&hit2) || (hit3&&hit2) || (hit1&&hit4) || (hit3&&hit4))
		{
		  /*
		  cout << "\tGot a trigger in configuration C " << endl; 
		  if(hit1)cout << "hit1 " << endl;
		  if(hit2)cout << "hit2 " << endl;
		  if(hit3)cout << "hit3 " << endl;
		  if(hit4)cout << "hit4 " << endl;
		  */
		  trig3++;
		  return true;
		}
	    }
	  }
      }
    
    return false;
  }
  
  
  bool CosmicTIFTrigFilter::Sci_trig(CLHEP::Hep3Vector vertex,  CLHEP::Hep3Vector momentum, CLHEP::Hep3Vector S)
  {
    float x0= vertex.x();
    float y0= vertex.y();
    float z0= vertex.z();
    float px0=momentum.x();
    float py0=momentum.y();
    float pz0=momentum.z();
    float Sx=S.x();
    float Sy=S.y();
    float Sz=S.z();
    
    //float ys=Sy;
    float zs=(Sy-y0)*(pz0/py0)+z0;
    //	  float xs=((Sy-y0)*(pz0/py0)-z0)*(px0/pz0)+x0;
    float xs=(Sy-y0)*(px0/py0)+x0;
    
    //	  cout << Sx << " " << Sz << " " << xs << " " << zs << endl;
    //	  cout << x0 << " " << z0 << " " << px0 << " " << py0 << " " << pz0 << endl;
    
    if((xs<Sx+500 && xs>Sx-500)&&(zs<Sz+500 && zs>Sz-500))
      {
	return true;
      }
    else
      {
	return false;
      }
    
  }
  
}

