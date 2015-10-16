#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/RPCGeometry/interface/RPCRollSpecs.h"
#include "SimMuon/RPCDigitizer/src/RPCSimAsymmetricCls.h"
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
#include <FWCore/ParameterSet/interface/ParameterSet.h>
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

RPCSimAsymmetricCls::RPCSimAsymmetricCls(const edm::ParameterSet& config) : 
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

RPCSimAsymmetricCls::~RPCSimAsymmetricCls()
{
  delete _rpcSync;
}

int RPCSimAsymmetricCls::getClSize(uint32_t id,float posX, CLHEP::HepRandomEngine* engine)
{
  std::vector<double>  clsForDetId = getRPCSimSetUp()->getAsymmetricClsDistribution(id,slice(posX));

  int cnt = 1;
  int min = 1;

  double rr_cl = CLHEP::RandFlat::shoot(engine);
  LogDebug ("RPCSimAsymmetricCls")<<"[RPCSimAsymmetricCls::getClSize] Fired RandFlat :: "<<rr_cl;
  for(unsigned int i = 0 ; i < clsForDetId.size(); i++){
    cnt++;
    if(rr_cl > clsForDetId[i]){
      min = cnt;
    }
    else if(rr_cl < clsForDetId[i]){
      break;
    }
  }
  return min;
}

int RPCSimAsymmetricCls::getClSize(float posX, CLHEP::HepRandomEngine* engine)
{
  std::map< int, std::vector<double> > clsMap = getRPCSimSetUp()->getClsMap();

  int cnt = 1;
  int min = 1;
  double func=0.0;
  std::vector<double> sum_clsize;

  double rr_cl = CLHEP::RandFlat::shoot(engine);
  LogDebug ("RPCSimAsymmetricCls")<<"[RPCSimAsymmetricCls::getClSize] Fired RandFlat :: "<<rr_cl;

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
RPCSimAsymmetricCls::simulate(const RPCRoll* roll,
			      const edm::PSimHitContainer& rpcHits,
			      CLHEP::HepRandomEngine* engine)
{
  _rpcSync->setRPCSimSetUp(getRPCSimSetUp());
  theRpcDigiSimLinks.clear();
  theDetectorHitMap.clear();
  theRpcDigiSimLinks = RPCDigiSimLinks(roll->id().rawId());
  
  RPCDetId rpcId = roll->id();
  RPCGeomServ RPCname(rpcId);
  std::string nameRoll = RPCname.name();
  
  const Topology& topology=roll->specs()->topology();
  for (edm::PSimHitContainer::const_iterator _hit = rpcHits.begin();
       _hit != rpcHits.end(); ++_hit){
    if(_hit-> particleType() == 11) continue;
    // Here I hould check if the RPC are up side down;
    const LocalPoint& entr=_hit->entryPoint();
    
    int time_hit = _rpcSync->getSimHitBx(&(*_hit),engine);
    float posX = roll->strip(_hit->localPosition()) 
      - static_cast<int>(roll->strip(_hit->localPosition()));
    
    std::vector<float> veff = (getRPCSimSetUp())->getEff(rpcId.rawId());
    
    std::stringstream veffstream; veffstream<<"[";
    for(std::vector<float>::iterator veffIt = veff.begin(); veffIt != veff.end(); ++veffIt) { veffstream<<(*veffIt)<<","; }
    veffstream<<"]";
    std::string veffstr = veffstream.str();
    LogDebug("RPCSimAsymmetricCls")<<"Get Eff from RPCSimSetup for detId = "<<rpcId.rawId()<<" :: "<<veffstr;  


    // Efficiency
    int centralStrip = topology.channel(entr)+1;
    float fire = CLHEP::RandFlat::shoot(engine);
    LogDebug ("RPCSimAsymmetricCls")<<"[RPCSimAsymmetricCls::simulate] Fired RandFlat :: "<<fire<<" --> < "<<veff[centralStrip-1]<<" ? --> "<<((fire < veff[centralStrip-1])?1:0);

    if (fire < veff[centralStrip-1]) {
      LogDebug ("RPCSimAsymmetricCls")<<"Detector is Efficient for this simhit";

      int fstrip=centralStrip;
      int lstrip=centralStrip;
      
      // Compute the cluster size

      //double w = CLHEP::RandFlat::shoot(engine);
      //LogDebug ("RPCSimAsymmetricCls")<<"[RPCSimAsymmetricCls::simulate] Fired RandFlat :: "<<w<<" (w is not used)";
      //if (w < 1.e-10) w=1.e-10;

      int clsize = this->getClSize(rpcId.rawId(),posX, engine); // This is for cluster size chamber by chamber
      LogDebug ("RPCSimAsymmetricCls")<<"Clustersize = "<<clsize;

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
 	if (clsize%2 == 0){ //even cluster size is a special case
	  if(clsize>5){
	    // insert the last strip according to the 
	    // simhit position in the central strip
	    // needed for cls > 5, because higher cluster size has no asymmetry
	    // and thus is treated like in the old parametrization
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
	  else {
	    // needed for correct initial position for even cluster size
	    // in case of asymmetric cluster size
	    if (lstrip < roll->nstrips() ){
	      lstrip++;
	      cls.push_back(lstrip);
	    }
	  }	
	}
      }
      
      //Now calculate the shift according to the distribution
      float fire1 = CLHEP::RandFlat::shoot(engine);
      LogDebug ("RPCSimAsymmetricCls")<<"[RPCSimAsymmetricCls::simulate] Fired RandFlat :: "<<fire1<<" (fire1 is used for a shift of the cluster)";

      int strip_shift=0;
      
      int offset;
      
      if(clsize%2==0){
	offset = 2;
      }
      else {
	offset = 1;
      }
      
      //No shift (asymmetry) for higher cluster size.
      if(clsize>5){ 
	strip_shift = 0;
      }
      else {
	std::vector<double>  TMPclsAsymmForDetId 
	  = getRPCSimSetUp()->getAsymmetryForCls(rpcId,slice(posX),clsize); 
	
	for(unsigned int i = 0; i < TMPclsAsymmForDetId.size(); i ++){
	  if(fire1 < TMPclsAsymmForDetId[i]){
	    strip_shift = i - offset;
	    break;
	  }
	}
      }

      vector<int> shifted_cls; // vector to hold shifted strips
      shifted_cls.clear();
      
      int min_strip=100;
      int max_strip=0;
      
      //correction for the edges
      for (std::vector<int>::iterator i=cls.begin(); i!=cls.end();i++){
	if(*i+strip_shift < min_strip){
	  min_strip = *i+strip_shift;
	}
	if(*i+strip_shift > max_strip){
	  max_strip = *i+strip_shift;
	}
      }
      
      if(min_strip<1 || max_strip-roll->nstrips()>0){
	strip_shift = 0;
      }

      //Now shift the cluster
      for (std::vector<int>::iterator i=cls.begin(); i!=cls.end();i++){
	shifted_cls.push_back(*i+strip_shift);
      }
      for (std::vector<int>::iterator i=shifted_cls.begin(); 
	   i!=shifted_cls.end();i++){
	// Check the timing of the adjacent strip
	if(*i != centralStrip){
	  double fire2 = CLHEP::RandFlat::shoot(engine);
          LogDebug ("RPCSimAsymmetricCls")<<"[RPCSimAsymmetricCls::simulate] Fired RandFlat :: "<<fire2<<" (check whether adjacent strips are efficient)";
	  if(fire2 < veff[*i-1]){
	    std::pair<int, int> digi(*i,time_hit);
	    strips.insert(digi);
	    LogDebug ("RPCSimAsymmetricCls")<<"RPC Digi inserted :: Signl :: DetId :: "<<rpcId<<" = "<<rpcId.rawId()<<" ==> digi <"<<digi.first<<","<<digi.second<<">";  
	    
	    theDetectorHitMap.insert(DetectorHitMap::value_type(digi,&(*_hit)));
	  }
	} 
	else {
	  std::pair<int, int> digi(*i,time_hit);
	  theDetectorHitMap.insert(DetectorHitMap::value_type(digi,&(*_hit)));

	  strips.insert(digi);
	  LogDebug ("RPCSimAsymmetricCls")<<"RPC Digi inserted :: Signl :: DetId :: "<<rpcId<<" = "<<rpcId.rawId()<<" ==> digi <"<<digi.first<<","<<digi.second<<">";  
	}
      }
    }
  } 
}

void RPCSimAsymmetricCls::simulateNoise(const RPCRoll* roll,
					CLHEP::HepRandomEngine* engine)
{

  RPCDetId rpcId = roll->id();

  RPCGeomServ RPCname(rpcId);
  std::string nameRoll = RPCname.name();

  std::vector<float> vnoise = (getRPCSimSetUp())->getNoise(rpcId.rawId());
  std::vector<float> veff = (getRPCSimSetUp())->getEff(rpcId.rawId());

  LogDebug ("RPCSimAsymmetricCls")<<"[RPCSimAsymmetricCls::simulateNoise] Treating DetId :: "<<rpcId<<" = "<<rpcId.rawId()<<" which has "<<roll->nstrips()<<" strips";

  std::stringstream vnoisestream; vnoisestream<<"[";
  for(std::vector<float>::iterator vnoiseIt = vnoise.begin(); vnoiseIt != vnoise.end(); ++vnoiseIt) { vnoisestream<<(*vnoiseIt)<<","; }
  vnoisestream<<"]";
  std::string vnoisestr = vnoisestream.str();
  LogDebug("RPCSimAsymmetricCls")<<"Get Noise from RPCSimSetup for detId = "<<rpcId.rawId()<<" :: vector with "<<vnoise.size()<<"entries :: "<<vnoisestr;  
 

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

  LogDebug ("RPCSimAsymmetricCls")<<"Noise :: vnoise.size() = "<<vnoise.size();

  for(unsigned int j = 0; j < vnoise.size(); ++j){
    
    if(j >= nstrips) break; 

    // The efficiency of 0% does not imply on the noise rate.
    // If the strip is masked the noise rate should be 0 Hz/cm^2
    // if(veff[j] == 0) continue;
    
    // double ave = vnoise[j]*nbxing*gate*area*1.0e-9*frate;
    // The vnoise is the noise rate per strip, so we shout multiply not
    // by the chamber area,
    // but the strip area which is area/((float)roll->nstrips()));
    double ave = vnoise[j]*nbxing*gate*area*1.0e-9*frate/((float)roll->nstrips());
    LogDebug ("RPCSimAsymmetricCls")<<"Noise :: strip "<<j<<" Average = "<<ave<<" = vnoise[j]*nbxing*gate*area*1.0e-9*frate/((float)roll->nstrips()) = "<<vnoise[j]<<"*"<<nbxing<<"*"<<gate<<"*"<<area<<"*"<<1.0e-9<<"*"<<frate<<"/"<<((float)roll->nstrips());

    CLHEP::RandPoissonQ randPoissonQ(*engine, ave);
    N_hits = randPoissonQ.fire();
    LogDebug ("RPCSimAsymmetricCls")<<"[RPCSimAsymmetricCls::simulateNoise] Fired RandPoissonQ :: "<<N_hits;
    LogDebug ("RPCSimAsymmetricCls")<<"Noise :: Amount of Noise Hits for DetId :: "<<rpcId<<" = "<<rpcId.rawId()<<" = N_hits = randPoissonQ.fire() = "<<N_hits;

    for (int i = 0; i < N_hits; i++ ){
      double time2 = CLHEP::RandFlat::shoot((nbxing*gate)/gate);
      LogDebug ("RPCSimAsymmetricCls")<<"[RPCSimAsymmetricCls::simulateNoise] Fired RandFlat :: "<<time2;
      int time_hit = (static_cast<int>(time2) - nbxing/2);
      std::pair<int, int> digi(j+1,time_hit);
      strips.insert(digi);
      LogDebug ("RPCSimAsymmetricCls")<<"RPC Digi inserted :: Noise :: DetId :: "<<rpcId<<" = "<<rpcId.rawId()<<" ==> digi <"<<digi.first<<","<<digi.second<<">";
    }
  }
}

unsigned int RPCSimAsymmetricCls::slice(float posX){

  if(0.0 <= posX && posX < 0.2)  {
    return 0;
  }
  else  if(0.2 <= posX && posX < 0.4) {
    return 1;
  }
  else if(0.4 <= posX && posX < 0.6) {
    return 2;
  }
  else  if(0.6 <= posX && posX < 0.8) {
    return 3; 
  }  
  else  if(0.8 <= posX && posX < 1.0)  {
    return 4;
  }
  else  return 2;
}
