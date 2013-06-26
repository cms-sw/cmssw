#ifndef ECALFETODIGI_H
#define ECALFETODIGI_H

/* Class: EcalFEtoDigi
 * Description:
 *  produces Ecal trigger primitive digis from TCC input flat files
 * Original Author: Nuno Leonardo, P. Paganini, E. Perez
 * Created:  Thu Feb 22 11:32:53 CET 2007
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
//#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h" 

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>

#include "SimCalorimetry/EcalElectronicsEmulation/interface/TCCinput.h"

struct TCCinput; 
typedef std::vector<TCCinput> TCCInputData;
const int N_SM = 36; //number of ecal barrel supermodules

class EcalFEtoDigi : public edm::EDProducer {

public:

  explicit EcalFEtoDigi(const edm::ParameterSet&);
  ~EcalFEtoDigi(){}
  
private:

  virtual void beginJob() ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob();

  void readInput();
  EcalTrigTowerDetId         create_TTDetId (TCCinput);
  EcalTriggerPrimitiveSample create_TPSample(TCCinput, const edm::EventSetup&);
  EcalTriggerPrimitiveSample create_TPSampleTcp(TCCinput, const edm::EventSetup&);
  int SMidToTCCid(const int) const;
  void getLUT(unsigned int * lut, const int towerId,  const edm::EventSetup&) const ;

  TCCInputData inputdata_[N_SM];

  std::string basename_;
  bool useIdentityLUT_;
  int sm_;
  bool singlefile;

  int fileEventOffset_;
  bool debug_;
  std::ofstream outfile;

};

#endif
