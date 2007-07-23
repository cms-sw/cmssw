#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/RPCGeometry/interface/RPCRollSpecs.h"
#include "SimMuon/RPCDigitizer/src/RPCSimAverage.h"

#include "SimMuon/RPCDigitizer/src/RPCSynchronizer.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"

#include <cmath>
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CLHEP/Random/RandomEngine.h"
#include "CLHEP/Random/RandFlat.h"
#include <CLHEP/Random/RandGaussQ.h>
#include <CLHEP/Random/RandFlat.h>

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

#include<cstring>
#include<iostream>
#include<fstream>
#include<string>
#include<vector>
#include<stdlib.h>
#include <utility>
#include <map>

using namespace std;

RPCSimAverage::RPCSimAverage(const edm::ParameterSet& config) : 
  RPCSim(config)
{

  _rpcSync = new RPCSynchronizer(config);

  aveEff = config.getParameter<double>("averageEfficiency");
  aveCls = config.getParameter<double>("averageClusterSize");
  resRPC = config.getParameter<double>("timeResolution");
  timOff = config.getParameter<double>("timingRPCOffset");
  dtimCs = config.getParameter<double>("deltatimeAdjacentStrip");
  resEle = config.getParameter<double>("timeJitter");
  sspeed = config.getParameter<double>("signalPropagationSpeed");
  lbGate = config.getParameter<double>("linkGateWidth");
  rpcdigiprint = config.getParameter<bool>("printOutDigitizer");

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

  string ifile="../data/ClSizeTot.dat";

  infile = new ifstream(ifile.c_str(), ios::in);
  if(! *infile) {
    cerr << "error: unable to open input file: "
         <<  ifile  << endl;
    //    return -1;
  }

  string buffer;
  double sum = 0;
  unsigned int counter = 1;
  unsigned int row = 1;
  std::vector<double> sum_clsize;

  while ( *infile >> buffer ) {
    const char *buffer1 = buffer.c_str();
    double dato = atof(buffer1);
    sum += dato;
    sum_clsize.push_back(sum);

    if(counter == row*20) {
      clsMap[row] = sum_clsize;
      row++;
      sum = 0;
      sum_clsize.clear();
    }
    counter++;
  }

  edm::Service<edm::RandomNumberGenerator> rng;
  if ( ! rng.isAvailable()) {
    throw cms::Exception("Configuration")
      << "RPCDigitizer requires the RandomNumberGeneratorService\n"
      "which is not present in the configuration file.  You must add the service\n"
      "in the configuration file or remove the modules that require it.";
  }
  
  rndEngine = &(rng->getEngine());
  flatDistribution = new CLHEP::RandFlat(rndEngine);
}

int RPCSimAverage::getClSize(float posX)
{

  int cnt = 1;
  int min = 1;
  int max = 1;
  double func=0.0;
  std::vector<double> sum_clsize;

  double rr_cl = RandFlat::shoot();
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
      max = cnt;
      break;
    }
  }
  return min;
}


void
RPCSimAverage::simulate(const RPCRoll* roll,
			const edm::PSimHitContainer& rpcHits,const RPCGeometry* geo)
{

  _rpcSync->setGeometry(geo);
  _rpcSync->setReadOutTime(geo);

  //  if(_rpcSync->getReadOutTime(roll->id()) < 1){
//     std::cout<<"Region = "<<roll->id().region()<<"  Ring = "<<roll->id().ring()<<"  Station = "<<roll->id().station()<<"  Sector = "<<roll->id().sector()<<"  Layer = "<<roll->id().layer()<<"  Subsector = "<<roll->id().subsector()<<"  Roll = "<<roll->id().roll()<<std::endl;
//     std::cout<<"IL TIME DELLA ROLL E': "<<_rpcSync->getReadOutTime(roll->id())<<std::endl;
    //  }
  //  _rpcSync->getReadOutTime(roll->id());
 
  const Topology& topology=roll->specs()->topology();

  for (edm::PSimHitContainer::const_iterator _hit = rpcHits.begin();
       _hit != rpcHits.end(); ++_hit){
 

    // Here I hould check if the RPC are up side down;
    const LocalPoint& entr=_hit->entryPoint();

    //    const LocalPoint& exit=_hit->exitPoint();

    float posX = roll->strip(_hit->localPosition()) - static_cast<int>(roll->strip(_hit->localPosition()));

    // Effinciecy

    if (flatDistribution->fire() < aveEff) {

      int centralStrip = topology.channel(entr)+1;  
      int fstrip=centralStrip;
      int lstrip=centralStrip;
      // Compute the cluster size
      double w = flatDistribution->fire();
      if (w < 1.e-10) w=1.e-10;
      int clsize = this->getClSize(posX);

      std::vector<int> cls;
      cls.push_back(centralStrip);
      if (clsize > 1){
	for (int cl = 0; cl < (clsize-1)/2; cl++)
	  if (centralStrip - cl -1 >= 1  ){
	    fstrip = centralStrip-cl-1;
	    cls.push_back(fstrip);
	  }
	for (int cl = 0; cl < (clsize-1)/2; cl++)
	  if (centralStrip + cl + 1 <= roll->nstrips() ){
	    lstrip = centralStrip+cl+1;
	    cls.push_back(lstrip);
	  }
	if (clsize%2 == 0 ){
	  // insert the last strip according to the 
	  // simhit position in the central strip 
	  double deltaw=roll->centreOfStrip(centralStrip).x()-entr.x();
	  if (deltaw>0.) {
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

//       std::cout<<"CLUSTER SIZE: "<<cls.size()<<std::endl;

      for (std::vector<int>::iterator i=cls.begin(); i!=cls.end();i++){
	// Check the timing of the adjacent strip
	std::pair<int, int> digi(*i,_rpcSync->getDigiBx(&(*_hit), centralStrip, *i));
	//	std::cout<<"STRIP: "<<*i<<"  "<<"BX: "<<bx<<std::endl;
	strips.insert(digi);
      }
    }
  }
}


