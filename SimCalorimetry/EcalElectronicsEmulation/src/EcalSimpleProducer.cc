#include "SimCalorimetry/EcalElectronicsEmulation/interface/EcalSimpleProducer.h"
#include "TFormula.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include <iostream>
#include <string>

using namespace std;
using namespace edm;

void EcalSimpleProducer::produce(edm::Event& evt, const edm::EventSetup&){
  const int ievt = evt.id().event();
  if(formula_.get()!=0){
    auto_ptr<EBDigiCollection> digis(new EBDigiCollection);
    
    digis->reserve(170*360);
    
    const int nSamples = digis->stride();
    for(int iEta0=0; iEta0<170; ++iEta0){
      for(int iPhi0=0; iPhi0<360; ++iPhi0){
	int iEta1 = cIndex2iEta(iEta0);
	int iPhi = cIndex2iPhi(iPhi0);
	if(verbose_) cout << "(" << iEta0 << "," << iPhi0 << "): ";
	digis->push_back(EBDetId(iEta1,iPhi));
	DataFrame dframe(digis->back());
	
	for(int t = 0; t < nSamples; ++t){
	  uint16_t encodedAdc = (uint16_t)formula_->Eval(iEta0, iPhi0, ievt-1, t);
	  if(verbose_) cout << encodedAdc << ((t<(nSamples-1))?"\t":"\n");
	  dframe[t]=encodedAdc;
	}
      }
    }
    evt.put(digis);
    //puts an empty digi collecion for endcap:
    evt.put(auto_ptr<EEDigiCollection>(new EEDigiCollection()));
  }
  if(tpFormula_.get()!=0){
    auto_ptr<EcalTrigPrimDigiCollection> tps
      = auto_ptr<EcalTrigPrimDigiCollection>(new EcalTrigPrimDigiCollection);
    tps->reserve(56*72);
    const int nSamples = 5;
    for(int iTtEta0=0; iTtEta0<56; ++iTtEta0){
      for(int iTtPhi0=0; iTtPhi0<72; ++iTtPhi0){
	int iTtEta1 = cIndex2iTtEta(iTtEta0);
	int iTtPhi = cIndex2iTtPhi(iTtPhi0);
	
	if(verbose_) cout << "(" << iTtEta0 << "," << iTtPhi0 << "): ";
	int zside = iTtEta1<0?-1:1;
	EcalTriggerPrimitiveDigi
	  tpframe(EcalTrigTowerDetId(zside, EcalTriggerTower,
				     abs(iTtEta1), iTtPhi));
	
	tpframe.setSize(nSamples);
	
	if(verbose_) cout << "TP: ";
	for(int t = 0; t < nSamples; ++t){
	  uint16_t encodedTp = (uint16_t)tpFormula_->Eval(iTtEta0, iTtPhi0, ievt-1, t);
	  
	  if(verbose_) cout << "TP(" << iTtEta0 << "," << iTtPhi0 << ") = "
			    << encodedTp << ((t<(nSamples-1))?"\t":"\n");
	  tpframe.setSample(t, EcalTriggerPrimitiveSample(encodedTp));
	}
	tps->push_back(tpframe);
      }
    }
    evt.put(tps);
  }
  if(simHitFormula_.get()!=0){//generation of barrel sim hits
    auto_ptr<PCaloHitContainer> hits
      = auto_ptr<PCaloHitContainer>(new PCaloHitContainer);
    for(int iEta0=0; iEta0<170; ++iEta0){
      for(int iPhi0=0; iPhi0<360; ++iPhi0){
	int iEta1 = cIndex2iEta(iEta0);
	int iPhi = cIndex2iPhi(iPhi0);
	if(verbose_) cout << "(" << iEta0 << "," << iPhi0 << "): ";
	
	double em = simHitFormula_->Eval(iEta0, iPhi0, ievt-1);
        double eh = 0.;
	double t = 0.;
	const PCaloHit hit(EBDetId(iEta1,iPhi).rawId(), em, eh, t, 0);
	hits->push_back(hit);
      }
    }
    evt.put(hits, "EcalHitsEB");
    //puts an empty digi collecion for endcap:
    evt.put(auto_ptr<PCaloHitContainer>(new PCaloHitContainer()),
	    "EcalHitsEE");
  }
}

EcalSimpleProducer::EcalSimpleProducer(const edm::ParameterSet& pset):
  EDProducer(){
  string formula = pset.getParameter<string>("formula");
  string tpFormula = pset.getParameter<string>("tpFormula");
  string simHitFormula = pset.getParameter<string>("simHitFormula");
  
  verbose_ = pset.getUntrackedParameter<bool>("verbose", false);
  //  replaceAll(formula, "itt0", "((((ieta0<85)*(84-ieta0)+(ieta0>=85)*(ieta0-85))/5-18)*4+((iphi0/5+2)%4))");
  replaceAll(formula, "ebm", "(ieta0<85)");
  replaceAll(formula, "ebp", "(ieta0>84)");
  replaceAll(formula, "ieta0", "x");
  replaceAll(formula, "iphi0", "y");
  replaceAll(formula, "ievt0", "z");
  replaceAll(formula, "isample0", "t");
  //  cout << "----------> " << formula << endl;
  
  replaceAll(tpFormula, "itt0", "((ieta0<28)*(27-ieta0)+(ieta0>=28)*(ieta0-28))*4+(iphi0+2)%4");
  replaceAll(tpFormula, "eb", "(ieta0>10 && ieta0<45)");
  replaceAll(tpFormula, "ebm", "(ieta0>10 && ieta0<28)");
  replaceAll(tpFormula, "ebp", "(ieta0>27 && ieta0<45)");
  replaceAll(tpFormula, "ee", "(ieta0<11 || ieta0>44)");
  replaceAll(tpFormula, "eem", "(ieta0<11)");
  replaceAll(tpFormula, "eep", "(ieta0>44)");
  replaceAll(tpFormula, "ieta0", "x");
  replaceAll(tpFormula, "iphi0", "y");
  replaceAll(tpFormula, "ievt0", "z");
  replaceAll(tpFormula, "isample0", "t");
  //cout << "----------> " << tpFormula << endl;
  

  //replaceAll(simHitormula, "itt0", "((((ieta0<85)*(84-ieta0)+(ieta0>=85)*(ieta0-85))/5-18)*4+((iphi0/5+2)%4))");
  replaceAll(simHitFormula, "ebm", "(ieta0<85)");
  replaceAll(simHitFormula, "ebp", "(ieta0>84)");
  replaceAll(simHitFormula, "ieta0", "x");
  replaceAll(simHitFormula, "iphi0", "y");
  replaceAll(simHitFormula, "ievt0", "z");
  
  if(formula.size()!=0){
    formula_ = auto_ptr<TFormula>(new TFormula("f", formula.c_str()));
    Int_t err = formula_->Compile();
    if(err!=0){
      throw cms::Exception("Error in EcalSimpleProducer 'formula' config.");
    }
    produces<EBDigiCollection>();
    produces<EEDigiCollection>();
  }
  if(tpFormula.size()!=0){
    tpFormula_ = auto_ptr<TFormula>(new TFormula("f", tpFormula.c_str()));
    Int_t err = tpFormula_->Compile();
    if(err!=0){
      throw cms::Exception("Error in EcalSimpleProducer 'tpFormula' config.");
    }
    produces<EcalTrigPrimDigiCollection>();
  }
  if(simHitFormula.size()!=0){
    simHitFormula_
      = auto_ptr<TFormula>(new TFormula("f", simHitFormula.c_str()));
    Int_t err = simHitFormula_->Compile();
    if(err!=0){
      throw cms::Exception("Error in EcalSimpleProducer "
			   "'simHitFormula' config.");
    }
    produces<edm::PCaloHitContainer>("EcalHitsEB");
    produces<edm::PCaloHitContainer>("EcalHitsEE");
  }
}

void EcalSimpleProducer::replaceAll(std::string& s, const std::string& from,
				  const std::string& to) const{
  string::size_type pos = 0;
  //  cout << "replaceAll(" << s << "," << from << "," << to << ")\n";
  while((pos=s.find(from, pos))!=string::npos){
    // cout << "replace(" << pos << "," << from.size() << "," << to << ")\n";
    s.replace(pos, from.size(), to);
    //   cout << " -> " << s << "\n"; 
  }
}
