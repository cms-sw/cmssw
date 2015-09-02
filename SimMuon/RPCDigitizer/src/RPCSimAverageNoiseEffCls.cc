#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/RPCGeometry/interface/RPCRollSpecs.h"
#include "SimMuon/RPCDigitizer/src/RPCSimAverageNoiseEffCls.h"
#include "SimMuon/RPCDigitizer/src/RPCSimSetUp.h"

#include "SimMuon/RPCDigitizer/src/RPCSynchronizer.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"

#include <cmath>

#include <FWCore/Framework/interface/Frameworkfwd.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Framework/interface/ESHandle.h>

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "SimMuon/RPCDigitizer/src/RPCSimSetUp.h"

#include<cstring>
#include<iostream>
#include<fstream>
#include<string>
#include<vector>
#include<stdlib.h>
#include <utility>
#include <map>

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandPoissonQ.h"

using namespace std;

RPCSimAverageNoiseEffCls::RPCSimAverageNoiseEffCls(const edm::ParameterSet& config) : 
  RPCSim(config)
{

  aveEff = config.getParameter<double>("averageEfficiency");
  aveCls = config.getParameter<double>("averageClusterSize");
  resRPC = config.getParameter<double>("timeResolution");
  timOff = config.getParameter<double>("timingRPCOffset");
  dtimCs = config.getParameter<double>("deltatimeAdjacentStrip");
  resEle = config.getParameter<double>("timeJitter");
  sspeed = config.getParameter<double>("signalPropagationSpeed");
  lbGate = config.getParameter<double>("linkGateWidth");
  rpcdigiprint = config.getParameter<bool>("printOutDigitizer");

  rate=config.getParameter<double>("Rate");
  nbxing=config.getParameter<int>("Nbxing");
  gate=config.getParameter<double>("Gate");
  frate=config.getParameter<double>("Frate");

  if (rpcdigiprint) {
    std::cout <<"Average Efficiency        = "<<aveEff<<std::endl;
    std::cout <<"Average Cluster Size      = "<<aveCls<<" strips"<<std::endl;
    std::cout <<"RPC Time Resolution       = "<<resRPC<<" ns"<<std::endl;
    std::cout <<"RPC Signal formation time = "<<timOff<<" ns"<<std::endl;
    std::cout <<"RPC adjacent strip delay  = "<<dtimCs<<" ns"<<std::endl;
    std::cout <<"Electronic Jitter         = "<<resEle<<" ns"<<std::endl;
    std::cout <<"Signal propagation time   = "<<sspeed<<" x c"<<std::endl;
    std::cout <<"Link Board Gate Width     = "<<lbGate<<" ns"<<std::endl;
  }

  _rpcSync = new RPCSynchronizer(config);

}

RPCSimAverageNoiseEffCls::~RPCSimAverageNoiseEffCls()
{
  delete _rpcSync;
}


int RPCSimAverageNoiseEffCls::getClSize(uint32_t id,float posX, CLHEP::HepRandomEngine* engine)
{
  std::vector<double> clsForDetId = getRPCSimSetUp()->getCls(id);

  int cnt = 1;
  int min = 1;
  double func=0.0;
  std::vector<double> sum_clsize;

  sum_clsize.clear();
  sum_clsize = clsForDetId;
  int vectOffset(0);

  double rr_cl = CLHEP::RandFlat::shoot(engine);

  if(0.0 <= posX && posX < 0.2)  {
    func = clsForDetId[19]*(rr_cl);
    vectOffset = 0;
  }
  if(0.2 <= posX && posX < 0.4) {
    func = clsForDetId[39]*(rr_cl);
    vectOffset = 20;
  }
  if(0.4 <= posX && posX < 0.6) {
    func = clsForDetId[59]*(rr_cl);
    vectOffset = 40;
  }
  if(0.6 <= posX && posX < 0.8) {
    func = clsForDetId[79]*(rr_cl);
    vectOffset = 60;
  }  
  if(0.8 <= posX && posX < 1.0)  {
    func = clsForDetId[89]*(rr_cl);
    vectOffset = 80;
  }
  

  for(int i = vectOffset; i<(vectOffset+20); i++){
    cnt++;
    if(func > clsForDetId[i]){
      min = cnt;
    }
    else if(func < clsForDetId[i]){
      break;
    }
  }
  return min;
}

int RPCSimAverageNoiseEffCls::getClSize(float posX, CLHEP::HepRandomEngine* engine)
{

  std::map< int, std::vector<double> > clsMap = getRPCSimSetUp()->getClsMap();

  int cnt = 1;
  int min = 1;
  double func=0.0;
  std::vector<double> sum_clsize;

  double rr_cl = CLHEP::RandFlat::shoot(engine);
  if(0.0 <= posX && posX < 0.2)  {
    func = (clsMap[1])[(clsMap[1]).size()-1]*(rr_cl);
    sum_clsize = clsMap[1];
  }
  if(0.2 <= posX && posX < 0.4) {
    func = (clsMap[2])[(clsMap[2]).size()-1]*(rr_cl);
    sum_clsize = clsMap[2];
  }
  if(0.4 <= posX && posX < 0.6) {
    func = (clsMap[3])[(clsMap[3]).size()-1]*(rr_cl);
    sum_clsize = clsMap[3];
  }
  if(0.6 <= posX && posX < 0.8) {
    func = (clsMap[4])[(clsMap[4]).size()-1]*(rr_cl);
    sum_clsize = clsMap[4];
  }
  if(0.8 <= posX && posX < 1.0)  {
    func = (clsMap[5])[(clsMap[5]).size()-1]*(rr_cl);
    sum_clsize = clsMap[5];
  }

  for(vector<double>::iterator iter = sum_clsize.begin();
      iter != sum_clsize.end(); ++iter){
    cnt++;
    if(func > (*iter)){
      min = cnt;
    }
    else if(func < (*iter)){
      break;
    }
  }
  return min;
}

void
RPCSimAverageNoiseEffCls::simulate(const RPCRoll* roll,
                                   const edm::PSimHitContainer& rpcHits,
                                   CLHEP::HepRandomEngine* engine)
{

  _rpcSync->setRPCSimSetUp(getRPCSimSetUp());
  theRpcDigiSimLinks.clear();
  theDetectorHitMap.clear();
  theRpcDigiSimLinks = RPCDigiSimLinks(roll->id().rawId());

  RPCDetId rpcId = roll->id();
  RPCGeomServ RPCname(rpcId);
  //std::string nameRoll = RPCname.name();

  const Topology& topology=roll->specs()->topology();

  for (edm::PSimHitContainer::const_iterator _hit = rpcHits.begin();
       _hit != rpcHits.end(); ++_hit){

    if(_hit-> particleType() == 11) continue;

    // Here I hould check if the RPC are up side down;
    const LocalPoint& entr=_hit->entryPoint();

    int time_hit = _rpcSync->getSimHitBx(&(*_hit), engine);
    float posX = roll->strip(_hit->localPosition()) - static_cast<int>(roll->strip(_hit->localPosition()));

    std::vector<float> veff = (getRPCSimSetUp())->getEff(rpcId.rawId());

    // Effinciecy
    int centralStrip = topology.channel(entr)+1;;
    float fire = CLHEP::RandFlat::shoot(engine);

    if (fire < veff[centralStrip-1]) {

      int fstrip=centralStrip;
      int lstrip=centralStrip;

      // Compute the cluster size
      // double w = CLHEP::RandFlat::shoot(engine);
      //if (w < 1.e-10) w=1.e-10;
//       int clsize = this->getClSize(posX, engine); // This is for one and the same cls for all the chambers
      int clsize = this->getClSize(rpcId.rawId(),posX, engine); // This is for cluster size chamber by chamber
      std::vector<int> cls;
      cls.push_back(centralStrip);
      if (clsize > 1){
	for (int cl = 0; cl < (clsize-1)/2; cl++){
	  if (centralStrip - cl -1 >= 1  ){
	    fstrip = centralStrip-cl-1;
	    cls.push_back(fstrip);
	  }
	  if (centralStrip + cl + 1 <= roll->nstrips() ){
	    lstrip = centralStrip+cl+1;
	    cls.push_back(lstrip);
	  }
	}
	if (clsize%2 == 0 ){
	  // insert the last strip according to the 
	  // simhit position in the central strip 
	  double deltaw=roll->centreOfStrip(centralStrip).x()-entr.x();
	  if (deltaw<0.) {
	    if (lstrip < roll->nstrips() ){
	      lstrip++;
	      cls.push_back(lstrip);
	    }
	  }else{
	    if (fstrip > 1 ){
	      fstrip--;
	      cls.push_back(fstrip);
	    }
	  }
	}
      }

      for (std::vector<int>::iterator i=cls.begin(); i!=cls.end();i++){
	// Check the timing of the adjacent strip
	if(*i != centralStrip){
	  if(CLHEP::RandFlat::shoot(engine) < veff[*i-1]){
	    std::pair<int, int> digi(*i,time_hit);
	    strips.insert(digi);

	    theDetectorHitMap.insert(DetectorHitMap::value_type(digi,&(*_hit)));
	  }
	} 
	else {
	  std::pair<int, int> digi(*i,time_hit);
	  theDetectorHitMap.insert(DetectorHitMap::value_type(digi,&(*_hit)));

	  strips.insert(digi);
	}
      }
    }
  }
}

void RPCSimAverageNoiseEffCls::simulateNoise(const RPCRoll* roll,
                                             CLHEP::HepRandomEngine* engine)
{

  RPCDetId rpcId = roll->id();

  RPCGeomServ RPCname(rpcId);
  //std::string nameRoll = RPCname.name();

  std::vector<float> vnoise = (getRPCSimSetUp())->getNoise(rpcId.rawId());
  std::vector<float> veff = (getRPCSimSetUp())->getEff(rpcId.rawId());

  unsigned int nstrips = roll->nstrips();
  double area = 0.0;
  
  if ( rpcId.region() == 0 )
    {
      const RectangularStripTopology* top_ = dynamic_cast<const
	RectangularStripTopology*>(&(roll->topology()));
      float xmin = (top_->localPosition(0.)).x();
      float xmax = (top_->localPosition((float)roll->nstrips())).x();
      float striplength = (top_->stripLength());
      area = striplength*(xmax-xmin);
    }
  else
    {
      const TrapezoidalStripTopology* top_=dynamic_cast<const TrapezoidalStripTopology*>(&(roll->topology()));
      float xmin = (top_->localPosition(0.)).x();
      float xmax = (top_->localPosition((float)roll->nstrips())).x();
      float striplength = (top_->stripLength());
      area = striplength*(xmax-xmin);
    }

  for(unsigned int j = 0; j < vnoise.size(); ++j){
    
    if(j >= nstrips) break; 

    // The efficiency of 0% does not imply on the noise rate.
    // If the strip is masked the noise rate should be 0 Hz/cm^2
    //    if(veff[j] == 0) continue;
    
    //    double ave = vnoise[j]*nbxing*gate*area*1.0e-9*frate;
    // The vnoise is the noise rate per strip, so we shout multiply not
    // by the chamber area,
    // but the strip area which is area/((float)roll->nstrips()));
    double ave =
      vnoise[j]*nbxing*gate*area*1.0e-9*frate/((float)roll->nstrips());

    CLHEP::RandPoissonQ randPoissonQ(*engine, ave);
    N_hits = randPoissonQ.fire();

    for (int i = 0; i < N_hits; i++ ){
      
      int time_hit = (static_cast<int>(CLHEP::RandFlat::shoot(engine, (nbxing*gate)/gate))) - nbxing/2;
      std::pair<int, int> digi(j+1,time_hit);
      strips.insert(digi);
    }
  }
}

