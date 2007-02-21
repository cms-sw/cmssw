#include "Validation/GlobalDigis/interface/GlobalDigisProducer.h"

GlobalDigisProducer::GlobalDigisProducer(const edm::ParameterSet& iPSet) :
  fName(""), verbosity(0), frequency(0), label(""), getAllProvenances(false),
  printProvenanceInfo(false), count(0)
{
  std::string MsgLoggerCat = "GlobalDigisProducer_GlobalDigisProducer";

  // get information from parameter set
  fName = iPSet.getUntrackedParameter<std::string>("Name");
  verbosity = iPSet.getUntrackedParameter<int>("Verbosity");
  frequency = iPSet.getUntrackedParameter<int>("Frequency");
  label = iPSet.getParameter<std::string>("Label");
  edm::ParameterSet m_Prov =
    iPSet.getParameter<edm::ParameterSet>("ProvenanceLookup");
  getAllProvenances = 
    m_Prov.getUntrackedParameter<bool>("GetAllProvenances");
  printProvenanceInfo = 
    m_Prov.getUntrackedParameter<bool>("PrintProvenanceInfo");

  //get Labels to use to extract information
  ECalEBSrc_ = iPSet.getParameter<edm::InputTag>("ECalEBSrc");
  ECalEESrc_ = iPSet.getParameter<edm::InputTag>("ECalEESrc");
  ECalESSrc_ = iPSet.getParameter<edm::InputTag>("ECalESSrc");
  HCalSrc_ = iPSet.getParameter<edm::InputTag>("HCalSrc");

  // use value of first digit to determine default output level (inclusive)
  // 0 is none, 1 is basic, 2 is fill output, 3 is gather output
  verbosity %= 10;

  // create persistent object
  produces<PGlobalDigi>(label);

  // print out Parameter Set information being used
  if (verbosity >= 0) {
    edm::LogInfo(MsgLoggerCat) 
      << "\n===============================\n"
      << "Initialized as EDProducer with parameter values:\n"
      << "    Name          = " << fName << "\n"
      << "    Verbosity     = " << verbosity << "\n"
      << "    Frequency     = " << frequency << "\n"
      << "    Label         = " << label << "\n"
      << "    GetProv       = " << getAllProvenances << "\n"
      << "    PrintProv     = " << printProvenanceInfo << "\n"
      << "    ECalEBSrc     = " << ECalEBSrc_.label() 
      << ":" << ECalEBSrc_.instance() << "\n"
      << "    ECalEESrc     = " << ECalEESrc_.label() 
      << ":" << ECalEESrc_.instance() << "\n"
      << "    ECalESSrc     = " << ECalESSrc_.label() 
      << ":" << ECalESSrc_.instance() << "\n"
      << "    HCalSrc       = " << HCalSrc_.label() 
      << ":" << HCalSrc_.instance() << "\n"
      << "===============================\n";
  }

  // set default constants
  // ECal
  ECalgainConv_[0] = 0.;
  ECalgainConv_[1] = 1.;
  ECalgainConv_[2] = 2.;
  ECalgainConv_[3] = 12.;  
  ECalbarrelADCtoGeV_ = 0.035;
  ECalendcapADCtoGeV_ = 0.06;

}

GlobalDigisProducer::~GlobalDigisProducer() 
{
}

void GlobalDigisProducer::beginJob(const edm::EventSetup& iSetup)
{
  std::string MsgLoggerCat = "GlobalDigisProducer_beginJob";

  // setup calorimeter constants from service
  edm::ESHandle<EcalADCToGeVConstant> pAgc;
  iSetup.get<EcalADCToGeVConstantRcd>().get(pAgc);
  const EcalADCToGeVConstant* agc = pAgc.product();
  
  EcalMGPAGainRatio * defaultRatios = new EcalMGPAGainRatio();

  ECalgainConv_[0] = 0.;
  ECalgainConv_[1] = 1.;
  ECalgainConv_[2] = defaultRatios->gain12Over6() ;
  ECalgainConv_[3] = ECalgainConv_[2]*(defaultRatios->gain6Over1()) ;

  delete defaultRatios;

  ECalbarrelADCtoGeV_ = agc->getEBValue();
  ECalendcapADCtoGeV_ = agc->getEEValue();

  if (verbosity >= 0) {
    edm::LogInfo(MsgLoggerCat) 
      << "Modified Calorimeter gain constants: g0 = " << ECalgainConv_[0]
      << ", g1 = " << ECalgainConv_[1] << ", g2 = " << ECalgainConv_[2]
      << ", g3 = " << ECalgainConv_[3];
    edm::LogInfo(MsgLoggerCat)
      << "Modified Calorimeter ADCtoGeV constants: barrel = " 
      << ECalbarrelADCtoGeV_ << ", endcap = " << ECalendcapADCtoGeV_;
  }

  // clear storage vectors
  clear();
  return;
}

void GlobalDigisProducer::endJob()
{
  std::string MsgLoggerCat = "GlobalDigisProducer_endJob";
  if (verbosity >= 0)
    edm::LogInfo(MsgLoggerCat) 
      << "Terminating having processed " << count << " events.";
  return;
}

void GlobalDigisProducer::produce(edm::Event& iEvent, 
				  const edm::EventSetup& iSetup)
{
  std::string MsgLoggerCat = "GlobalDigisProducer_produce";

  // keep track of number of events processed
  ++count;

  // get event id information
  int nrun = iEvent.id().run();
  int nevt = iEvent.id().event();

  if (verbosity > 0) {
    edm::LogInfo(MsgLoggerCat)
      << "Processing run " << nrun << ", event " << nevt
      << " (" << count << " events total)";
  } else if (verbosity == 0) {
    if (nevt%frequency == 0 || nevt == 1) {
      edm::LogInfo(MsgLoggerCat)
	<< "Processing run " << nrun << ", event " << nevt
	<< " (" << count << " events total)";
    }
  }

  // clear event holders
  clear();

  // look at information available in the event
  if (getAllProvenances) {

    std::vector<const edm::Provenance*> AllProv;
    iEvent.getAllProvenance(AllProv);

    if (verbosity >= 0)
      edm::LogInfo(MsgLoggerCat)
	<< "Number of Provenances = " << AllProv.size();

    if (printProvenanceInfo && (verbosity >= 0)) {
      TString eventout("\nProvenance info:\n");      

      for (unsigned int i = 0; i < AllProv.size(); ++i) {
	eventout += "\n       ******************************";
	eventout += "\n       Module       : ";
	eventout += (AllProv[i]->product).moduleLabel();
	eventout += "\n       ProductID    : ";
	eventout += (AllProv[i]->product).productID_.id_;
	eventout += "\n       ClassName    : ";
	eventout += (AllProv[i]->product).fullClassName_;
	eventout += "\n       InstanceName : ";
	eventout += (AllProv[i]->product).productInstanceName_;
	eventout += "\n       BranchName   : ";
	eventout += (AllProv[i]->product).branchName_;
      }
      eventout += "\n       ******************************\n";
      edm::LogInfo(MsgLoggerCat) << eventout << "\n";
      printProvenanceInfo = false;
    }
    getAllProvenances = false;
  }

  // call fill functions
  // gather Ecal information from event
  fillECal(iEvent, iSetup);
  // gather Hcal information from event
  fillHCal(iEvent, iSetup);

  if (verbosity > 0)
    edm::LogInfo (MsgLoggerCat)
      << "Done gathering data from event.";

  // produce object to put into event
  std::auto_ptr<PGlobalDigi> pOut(new PGlobalDigi);

  if (verbosity > 2)
    edm::LogInfo (MsgLoggerCat)
      << "Saving event contents:";

  // call store functions
  // store ECal information in produce
  storeECal(*pOut);
  // store HCal information in produce
  storeHCal(*pOut);

  // store information in event
  iEvent.put(pOut,label);

  return;
}

void GlobalDigisProducer::fillECal(edm::Event& iEvent, 
				   const edm::EventSetup& iSetup)
{
  std::string MsgLoggerCat = "GlobalDigisProducer_fillECal";

  TString eventout;
  if (verbosity > 0)
    eventout = "\nGathering info:";  

  // extract crossing frame from event
  edm::Handle<CrossingFrame> crossingFrame;
  iEvent.getByType(crossingFrame);
  if (!crossingFrame.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find crossingFrame in event!";
    return;
  }

  ////////////////////////
  //extract EB information
  ////////////////////////
  bool isBarrel = true;
  edm::Handle<EBDigiCollection> EcalDigiEB;  
  const EBDigiCollection *EBdigis = 0;
  iEvent.getByLabel(ECalEBSrc_, EcalDigiEB);
  if (!EcalDigiEB.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find EcalDigiEB in event!";
    return;
  }  
  EBdigis = EcalDigiEB.product();
  if (EBdigis->size() == 0) isBarrel = false;

  if (isBarrel) {

    // loop over simhits
    const std::string barrelHitsName("EcalHitsEB");
    std::auto_ptr<MixCollection<PCaloHit> >
      barrelHits(new MixCollection<PCaloHit>
		 (crossingFrame.product(), barrelHitsName));

    // keep track of sum of simhit energy in each crystal
    MapType ebSimMap;
    for (MixCollection<PCaloHit>::MixItr hitItr 
	   = barrelHits->begin();
	 hitItr != barrelHits->end();
	 ++hitItr) {

      EBDetId ebid = EBDetId(hitItr->id());

      uint32_t crystid = ebid.rawId();
      ebSimMap[crystid] += hitItr->energy();
    }

    // loop over digis
    const EBDigiCollection *barrelDigi = EcalDigiEB.product();

    std::vector<double> ebAnalogSignal;
    std::vector<double> ebADCCounts;
    std::vector<double> ebADCGains;
    ebAnalogSignal.reserve(EBDataFrame::MAXSAMPLES);
    ebADCCounts.reserve(EBDataFrame::MAXSAMPLES);
    ebADCGains.reserve(EBDataFrame::MAXSAMPLES);

    int i = 0;
    for (std::vector<EBDataFrame>::const_iterator digis =
	   barrelDigi->begin();
	 digis != barrelDigi->end();
	 ++digis) {

      ++i;

      EBDetId ebid = digis->id();

      double Emax = 0;
      int Pmax = 0;
      double pedestalPreSample = 0.;
      double pedestalPreSampleAnalog = 0.;
        
      for (int sample = 0; sample < digis->size(); ++sample) {
	ebAnalogSignal[sample] = 0.;
	ebADCCounts[sample] = 0.;
	ebADCGains[sample] = -1.;
      }
  
      // calculate maximum energy and pedestal
      for (int sample = 0; sample < digis->size(); ++sample) {
	ebADCCounts[sample] = (digis->sample(sample).adc());
	ebADCGains[sample] = (digis->sample(sample).gainId());
	ebAnalogSignal[sample] = 
	  (ebADCCounts[sample] * ECalgainConv_[(int)ebADCGains[sample]]
	   * ECalbarrelADCtoGeV_);
	if (Emax < ebAnalogSignal[sample]) {
	  Emax = ebAnalogSignal[sample];
	  Pmax = sample;
	}
	if ( sample < 3 ) {
	  pedestalPreSample += ebADCCounts[sample] ;
	  pedestalPreSampleAnalog += 
	    ebADCCounts[sample] * ECalgainConv_[(int)ebADCGains[sample]]
	    * ECalbarrelADCtoGeV_ ;
	}
	
      }
      pedestalPreSample /= 3. ; 
      pedestalPreSampleAnalog /= 3. ; 

      // calculate pedestal subtracted digi energy in the crystal
      double Erec = Emax - pedestalPreSampleAnalog
	* ECalgainConv_[(int)ebADCGains[Pmax]];

      // gather necessary information
      EBCalAEE.push_back(Erec);
      EBCalSHE.push_back(ebSimMap[ebid.rawId()]);
      EBCalmaxPos.push_back(Pmax);
    }
    
    if (verbosity > 1) {
      eventout += "\n          Number of EBDigis collected:.............. ";
      eventout += i;
    }
  }

  /////////////////////////
  //extract EE information
  ////////////////////////
  bool isEndCap = true;
  edm::Handle<EEDigiCollection> EcalDigiEE;  
  const EEDigiCollection *EEdigis = 0;
  iEvent.getByLabel(ECalEESrc_, EcalDigiEE);
  if (!EcalDigiEE.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find EcalDigiEE in event!";
    return;
  }  
  EEdigis = EcalDigiEE.product();
  if (EEdigis->size() == 0) isEndCap = false;

  if (isEndCap) {

    // loop over simhits
    const std::string endcapHitsName("EcalHitsEE");
    std::auto_ptr<MixCollection<PCaloHit> >
      endcapHits(new MixCollection<PCaloHit>
		 (crossingFrame.product(), endcapHitsName));

    // keep track of sum of simhit energy in each crystal
    MapType eeSimMap;
    for (MixCollection<PCaloHit>::MixItr hitItr 
	   = endcapHits->begin();
	 hitItr != endcapHits->end();
	 ++hitItr) {

      EEDetId eeid = EEDetId(hitItr->id());

      uint32_t crystid = eeid.rawId();
      eeSimMap[crystid] += hitItr->energy();
    }

    // loop over digis
    const EEDigiCollection *endcapDigi = EcalDigiEE.product();

    std::vector<double> eeAnalogSignal;
    std::vector<double> eeADCCounts;
    std::vector<double> eeADCGains;
    eeAnalogSignal.reserve(EEDataFrame::MAXSAMPLES);
    eeADCCounts.reserve(EEDataFrame::MAXSAMPLES);
    eeADCGains.reserve(EEDataFrame::MAXSAMPLES);

    int i = 0;
    for (std::vector<EEDataFrame>::const_iterator digis =
	   endcapDigi->begin();
	 digis != endcapDigi->end();
	 ++digis) {

      ++i;

      EEDetId eeid = digis->id();

      double Emax = 0;
      int Pmax = 0;
      double pedestalPreSample = 0.;
      double pedestalPreSampleAnalog = 0.;
        
      for (int sample = 0; sample < digis->size(); ++sample) {
	eeAnalogSignal[sample] = 0.;
	eeADCCounts[sample] = 0.;
	eeADCGains[sample] = -1.;
      }
  
      // calculate maximum enery and pedestal
      for (int sample = 0; sample < digis->size(); ++sample) {
	eeADCCounts[sample] = (digis->sample(sample).adc());
	eeADCGains[sample] = (digis->sample(sample).gainId());
	eeAnalogSignal[sample] = 
	  (eeADCCounts[sample] * ECalgainConv_[(int)eeADCGains[sample]]
	   * ECalbarrelADCtoGeV_);
	if (Emax < eeAnalogSignal[sample]) {
	  Emax = eeAnalogSignal[sample];
	  Pmax = sample;
	}
	if ( sample < 3 ) {
	  pedestalPreSample += eeADCCounts[sample] ;
	  pedestalPreSampleAnalog += 
	    eeADCCounts[sample] * ECalgainConv_[(int)eeADCGains[sample]]
	    * ECalbarrelADCtoGeV_ ;
	}
	
      }
      pedestalPreSample /= 3. ; 
      pedestalPreSampleAnalog /= 3. ; 

      // calculate pedestal subtracted digi energy in the crystal
      double Erec = Emax - pedestalPreSampleAnalog
	* ECalgainConv_[(int)eeADCGains[Pmax]];

      // gather necessary information
      EECalAEE.push_back(Erec);
      EECalSHE.push_back(eeSimMap[eeid.rawId()]);
      EECalmaxPos.push_back(Pmax);
    }
    
    if (verbosity > 1) {
      eventout += "\n          Number of EEDigis collected:.............. ";
      eventout += i;
    }
  }

  /////////////////////////
  //extract ES information
  ////////////////////////
  bool isPreshower = true;
  edm::Handle<ESDigiCollection> EcalDigiES;  
  const ESDigiCollection *ESdigis = 0;
  iEvent.getByLabel(ECalESSrc_, EcalDigiES);
  if (!EcalDigiES.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find EcalDigiES in event!";
    return;
  }  
  ESdigis = EcalDigiES.product();
  if (ESdigis->size() == 0) isPreshower = false;

  if (isPreshower) {

    // loop over simhits
    const std::string preshowerHitsName("EcalHitsES");
    std::auto_ptr<MixCollection<PCaloHit> >
      preshowerHits(new MixCollection<PCaloHit>
		 (crossingFrame.product(), preshowerHitsName));

    // keep track of sum of simhit energy in each crystal
    MapType esSimMap;
    for (MixCollection<PCaloHit>::MixItr hitItr 
	   = preshowerHits->begin();
	 hitItr != preshowerHits->end();
	 ++hitItr) {

      ESDetId esid = ESDetId(hitItr->id());

      uint32_t crystid = esid.rawId();
      esSimMap[crystid] += hitItr->energy();
    }

    // loop over digis
    const ESDigiCollection *preshowerDigi = EcalDigiES.product();

    std::vector<double> esADCCounts;
    esADCCounts.reserve(ESDataFrame::MAXSAMPLES);

    int i = 0;
    for (std::vector<ESDataFrame>::const_iterator digis =
	   preshowerDigi->begin();
	 digis != preshowerDigi->end();
	 ++digis) {

      ++i;

      ESDetId esid = digis->id();
        
      for (int sample = 0; sample < digis->size(); ++sample) {
	esADCCounts[sample] = 0.;
      }
  
      // gether ADC counts
      for (int sample = 0; sample < digis->size(); ++sample) {
	esADCCounts[sample] = (digis->sample(sample).adc());
      }

      ESCalADC0.push_back(esADCCounts[0]);
      ESCalADC1.push_back(esADCCounts[1]);
      ESCalADC2.push_back(esADCCounts[2]);
      ESCalSHE.push_back(esSimMap[esid.rawId()]);
    }
    
    if (verbosity > 1) {
      eventout += "\n          Number of ESDigis collected:.............. ";
      eventout += i;
    }
  }

  if (verbosity > 0)
    edm::LogInfo(MsgLoggerCat) << eventout << "\n";

  return;
}

void GlobalDigisProducer::storeECal(PGlobalDigi& product)
{
  std::string MsgLoggerCat = "GlobalDigisProducer_storeECal";

  if (verbosity > 2) {
    TString eventout("\n         nEBDigis     = ");
    eventout += EBCalmaxPos.size();
    for (unsigned int i = 0; i < EBCalmaxPos.size(); ++i) {
      eventout += "\n      (maxPos, AEE, SHE) = (";
      eventout += EBCalmaxPos[i];
      eventout += ", ";
      eventout += EBCalAEE[i];
      eventout += ", ";
      eventout += EBCalSHE[i];
      eventout += ")";
    }
    eventout += "\n         nEEDigis     = ";
    eventout += EECalmaxPos.size();
    for (unsigned int i = 0; i < EECalmaxPos.size(); ++i) {
      eventout += "\n      (maxPos, AEE, SHE) = (";
      eventout += EECalmaxPos[i];
      eventout += ", ";
      eventout += EECalAEE[i];
      eventout += ", ";
      eventout += EECalSHE[i];
      eventout += ")";
    }
    eventout += "\n         nESDigis          = ";
    eventout += ESCalADC0.size();
    for (unsigned int i = 0; i < ESCalADC0.size(); ++i) {
      eventout += "\n      (ADC0, ADC1, ADC2, SHE) = (";
      eventout += ESCalADC0[i];
      eventout += ", ";
      eventout += ESCalADC1[i];
      eventout += ", ";
      eventout += ESCalADC2[i];
      eventout += ", ";
      eventout += ESCalSHE[i];
      eventout += ")";
    }
    edm::LogInfo(MsgLoggerCat) << eventout << "\n";
  }

  product.putEBCalDigis(EBCalmaxPos,EBCalAEE,EBCalSHE);
  product.putEECalDigis(EECalmaxPos,EECalAEE,EECalSHE);
  product.putESCalDigis(ESCalADC0,ESCalADC1,ESCalADC2,ESCalSHE);

  return;
}

void GlobalDigisProducer::fillHCal(edm::Event& iEvent, 
				   const edm::EventSetup& iSetup)
{
  std::string MsgLoggerCat = "GlobalDigisProducer_fillHCal";

  TString eventout;
  if (verbosity > 0)
    eventout = "\nGathering info:";  

  // get calibration info
  edm::ESHandle<HcalDbService> HCalconditions;
  iSetup.get<HcalDbRecord>().get(HCalconditions);
  if (!HCalconditions.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find HCalconditions in event!";
    return;
  } 
  const HcalQIEShape *shape = HCalconditions->getHcalShape();
  HcalCalibrations calibrations;
  CaloSamples tool;

  ///////////////////////
  // extract simhit info
  //////////////////////
  edm::Handle<edm::PCaloHitContainer> hcalHits;
  iEvent.getByLabel(HCalSrc_,hcalHits);
  if (!hcalHits.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find hcalHits in event!";
    return;
  }  
  const edm::PCaloHitContainer *simhitResult = hcalHits.product();
  
  MapType fHBEnergySimHits;
  MapType fHEEnergySimHits;
  MapType fHOEnergySimHits;
  MapType fHFEnergySimHits;
  for (std::vector<PCaloHit>::const_iterator simhits = simhitResult->begin();
       simhits != simhitResult->end();
       ++simhits) {
    
    HcalDetId detId(simhits->id());
    uint32_t cellid = detId.rawId();

    if (detId.subdet() == sdHcalBrl){  
      fHBEnergySimHits[cellid] += simhits->energy(); 
    }
    if (detId.subdet() == sdHcalEC){  
      fHEEnergySimHits[cellid] += simhits->energy(); 
    }    
    if (detId.subdet() == sdHcalOut){  
      fHOEnergySimHits[cellid] += simhits->energy(); 
    }    
    if (detId.subdet() == sdHcalFwd){  
      fHFEnergySimHits[cellid] += simhits->energy(); 
    }    
  }

  ////////////////////////
  // get HBHE information
  ///////////////////////
  edm::Handle<edm::SortedCollection<HBHEDataFrame> > hbhe;
  iEvent.getByType(hbhe);
  if (!hbhe.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find HBHEDataFrame in event!";
    return;
  }    
  edm::SortedCollection<HBHEDataFrame>::const_iterator ihbhe;
  
  int iHB = 0;
  int iHE = 0; 
  for (ihbhe = hbhe->begin(); ihbhe != hbhe->end(); ++ihbhe) {
    HcalDetId cell(ihbhe->id()); 

    if ((cell.subdet() == sdHcalBrl) || (cell.subdet() == sdHcalEC)) {
      
      HCalconditions->makeHcalCalibration(cell, &calibrations);
      const HcalQIECoder *channelCoder = HCalconditions->getHcalCoder(cell);
      HcalCoderDb coder(*channelCoder, *shape);
      coder.adc2fC(*ihbhe, tool);
      
      // get HB info
      if (cell.subdet() == sdHcalBrl) {

	++iHB;
	float fDigiSum = 0;
	for  (int ii = 0; ii < tool.size(); ++ii) {
	  // default ped is 4.5
	  int capid = (*ihbhe)[ii].capid();
	  fDigiSum += (tool[ii] - calibrations.pedestal(capid));
	}
	
	HBCalAEE.push_back(fDigiSum);
	HBCalSHE.push_back(fHBEnergySimHits[cell.rawId()]);
      }
	
      // get HE info
      if (cell.subdet() == sdHcalEC) {
	
	++iHE;
	float fDigiSum = 0;
	for  (int ii = 0; ii < tool.size(); ++ii) {
	  int capid = (*ihbhe)[ii].capid();
	  fDigiSum += (tool[ii]-calibrations.pedestal(capid));
	}
	
	HECalAEE.push_back(fDigiSum);
	HECalSHE.push_back(fHEEnergySimHits[cell.rawId()]);
      }
    }
  }

  if (verbosity > 1) {
    eventout += "\n          Number of HBDigis collected:.............. ";
    eventout += iHB;
  }
  
  if (verbosity > 1) {
    eventout += "\n          Number of HEDigis collected:.............. ";
    eventout += iHE;
  }

  ////////////////////////
  // get HO information
  ///////////////////////
  edm::Handle<edm::SortedCollection<HODataFrame> > ho;
  iEvent.getByType(ho);
  if (!ho.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find HODataFrame in event!";
    return;
  }    
  edm::SortedCollection<HODataFrame>::const_iterator iho;
  
  int iHO = 0; 
  for (iho = ho->begin(); iho != ho->end(); ++iho) {
    HcalDetId cell(iho->id()); 

    if (cell.subdet() == sdHcalOut) {
      
      HCalconditions->makeHcalCalibration(cell, &calibrations);
      const HcalQIECoder *channelCoder = HCalconditions->getHcalCoder(cell);
      HcalCoderDb coder (*channelCoder, *shape);
      coder.adc2fC(*iho, tool);

      ++iHO;
      float fDigiSum = 0;
      for  (int ii = 0; ii < tool.size(); ++ii) {
	// default ped is 4.5
	int capid = (*iho)[ii].capid();
	fDigiSum += (tool[ii] - calibrations.pedestal(capid));
      }
	
      HOCalAEE.push_back(fDigiSum);
      HOCalSHE.push_back(fHOEnergySimHits[cell.rawId()]);
    }
  }

  if (verbosity > 1) {
    eventout += "\n          Number of HODigis collected:.............. ";
    eventout += iHO;
  }

  ////////////////////////
  // get HF information
  ///////////////////////
  edm::Handle<edm::SortedCollection<HFDataFrame> > hf;
  iEvent.getByType(hf);
  if (!hf.isValid()) {
    edm::LogWarning(MsgLoggerCat)
      << "Unable to find HFDataFrame in event!";
    return;
  }    
  edm::SortedCollection<HFDataFrame>::const_iterator ihf;
  
  int iHF = 0; 
  for (ihf = hf->begin(); ihf != hf->end(); ++ihf) {
    HcalDetId cell(ihf->id()); 

    if (cell.subdet() == sdHcalFwd) {
      
      HCalconditions->makeHcalCalibration(cell, &calibrations);
      const HcalQIECoder *channelCoder = HCalconditions->getHcalCoder(cell);
      HcalCoderDb coder (*channelCoder, *shape);
      coder.adc2fC(*ihf, tool);

      ++iHF;
      float fDigiSum = 0;
      for  (int ii = 0; ii < tool.size(); ++ii) {
	// default ped is 1.73077
	int capid = (*ihf)[ii].capid();
	fDigiSum += (tool[ii] - calibrations.pedestal(capid));
      }
	
      HFCalAEE.push_back(fDigiSum);
      HFCalSHE.push_back(fHOEnergySimHits[cell.rawId()]);
    }
  }

  if (verbosity > 1) {
    eventout += "\n          Number of HFDigis collected:.............. ";
    eventout += iHF;
  }

  if (verbosity > 0)
    edm::LogInfo(MsgLoggerCat) << eventout << "\n";

  return;
}

void GlobalDigisProducer::storeHCal(PGlobalDigi& product)
{
  std::string MsgLoggerCat = "GlobalDigisProducer_storeHCal";

  if (verbosity > 2) {
    TString eventout("\n         nHBDigis     = ");
    eventout += HBCalAEE.size();
    for (unsigned int i = 0; i < HBCalAEE.size(); ++i) {
      eventout += "\n      (AEE, SHE) = (";
      eventout += HBCalAEE[i];
      eventout += ", ";
      eventout += HBCalSHE[i];
      eventout += ")";
    }
    eventout += "\n         nHEDigis     = ";
    eventout += HECalAEE.size();
    for (unsigned int i = 0; i < HECalAEE.size(); ++i) {
      eventout += "\n      (AEE, SHE) = (";
      eventout += HECalAEE[i];
      eventout += ", ";
      eventout += HECalSHE[i];
      eventout += ")";
    }
    eventout += "\n         nHODigis     = ";
    eventout += HOCalAEE.size();
    for (unsigned int i = 0; i < HOCalAEE.size(); ++i) {
      eventout += "\n      (AEE, SHE) = (";
      eventout += HOCalAEE[i];
      eventout += ", ";
      eventout += HOCalSHE[i];
      eventout += ")";
    }
    eventout += "\n         nHFDigis     = ";
    eventout += HFCalAEE.size();
    for (unsigned int i = 0; i < HFCalAEE.size(); ++i) {
      eventout += "\n      (AEE, SHE) = (";
      eventout += HFCalAEE[i];
      eventout += ", ";
      eventout += HFCalSHE[i];
      eventout += ")";
    }

    edm::LogInfo(MsgLoggerCat) << eventout << "\n";
  }

  product.putHBCalDigis(HBCalAEE,HBCalSHE);
  product.putHECalDigis(HECalAEE,HECalSHE);
  product.putHOCalDigis(HOCalAEE,HOCalSHE);
  product.putHFCalDigis(HFCalAEE,HFCalSHE);

  return;
}

void GlobalDigisProducer::clear()
{
  std::string MsgLoggerCat = "GlobalDigisProducer_clear";

  if (verbosity > 0)
    edm::LogInfo(MsgLoggerCat)
      << "Clearing event holders"; 

  // reset electromagnetic info
  // EE info
  EECalmaxPos.clear(); 
  EECalAEE.clear(); 
  EECalSHE.clear();
  // EB info
  EBCalmaxPos.clear(); 
  EBCalAEE.clear(); 
  EBCalSHE.clear();
  // ES info
  ESCalADC0.clear();
  ESCalADC1.clear();
  ESCalADC2.clear();
  ESCalSHE.clear();

  return;
}

//define this as a plug-in
DEFINE_FWK_MODULE(GlobalDigisProducer);

/*
void GlobalDigisProducer::fillHCal(edm::Event& iEvent, 
				   const edm::EventSetup& iSetup)
{
  std::string MsgLoggerCat = "GlobalDigisProducer_fillHCal";

  TString eventout;
  if (verbosity > 0)
    eventout = "\nGathering info:";  

  if (verbosity > 0)
    edm::LogInfo(MsgLoggerCat) << eventout << "\n";

  return;
}

void GlobalDigisProducer::storeHCal(PGlobalDigiHit& product)
{
  std::string MsgLoggerCat = "GlobalDigisProducer_storeHCal";

  return;
}
*/
