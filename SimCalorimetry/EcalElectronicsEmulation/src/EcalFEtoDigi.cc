
#include "SimCalorimetry/EcalElectronicsEmulation/interface/EcalFEtoDigi.h"
//#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h" 

EcalFEtoDigi::EcalFEtoDigi(const edm::ParameterSet& iConfig) {
  db_ = NULL;
  databaseFileNameEB_ = iConfig.getParameter<std::string>("DatabaseFileEB");;
  databaseFileNameEE_ = iConfig.getParameter<std::string>("DatabaseFileEE");;
  basename_           = iConfig.getUntrackedParameter<string>("FlatBaseName");
  sm_                 = iConfig.getUntrackedParameter<int>("SuperModuleId");
  skipEvents_         = iConfig.getUntrackedParameter<int>("SkipEvents");
  doCompressEt_       = iConfig.getUntrackedParameter<bool>("doCompressEt");
  debug               = iConfig.getUntrackedParameter<bool>("debugPrintFlag");

  singlefile = (sm_==-1)?false:true;

  produces<EcalTrigPrimDigiCollection>();
}

EcalFEtoDigi::~EcalFEtoDigi() {
  delete db_ ;
}


/// method called to produce the data
void
EcalFEtoDigi::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  
  /// event counter
  static int current_bx = -1;
  current_bx++;

  if(debug)
    cout << "[EcalFEtoDigi::produce] producing event " << current_bx+1 << endl;
  
  std::auto_ptr<EcalTrigPrimDigiCollection>  
    e_tpdigis (new EcalTrigPrimDigiCollection);


  vector<TCCinput>::const_iterator it;

  for(int i=0; i<N_SM; i++) {

    if(!singlefile)
      sm_=i+1;

    for(it = inputdata_[i].begin(); it != inputdata_[i].end(); it++) {

      if(!(*it).is_current(current_bx)) 
	continue;
      else
      	if(debug && (*it).input!=0 )
	  cout << "[EcalFEtoDigi] " 
	       << "\tsupermodule:" << sm_ 
	       << "\tevent: "      << current_bx 
	       << "\tbx: "         << (*it).bunchCrossing
	       << "\tvalue:0x"     << setfill('0') << setw(4) 
	       << hex << (*it).input << setfill(' ') << dec 
	       << endl;

      
      /// create EcalTrigTowerDetId
      const EcalTrigTowerDetId  e_id = create_TTDetId(*it);
      
      //EcalElectronicsMapping theMapping;
      //const EcalTrigTowerDetId  e_id 
      //= theMapping.getTrigTowerDetId(SMidToTCCid(sm_),(*it).tower);
      //EcalElectronicsMapping::getTrigTowerDetId(int TCCid, int iTT)
      
      /// create EcalTriggerPrimitiveDigi
      EcalTriggerPrimitiveDigi *e_digi = new EcalTriggerPrimitiveDigi(e_id);
      
      /// create EcalTriggerPrimitiveSample
      EcalTriggerPrimitiveSample e_sample = create_TPSample(*it);
      
      /// set sample
      e_digi->setSize(1); //set sampleOfInterest to 0
      e_digi->setSample(0,e_sample);
      
      /// add to EcalTrigPrimDigiCollection 
      e_tpdigis->push_back(*e_digi);
    
      if(debug) 
	outfile << (*it).tower << '\t' 
		<< (*it).bunchCrossing << '\t'<< setfill('0') << hex
		<< "0x" << setw(4) << (*it).input << '\t'
		<< "0"	<< dec << setfill(' ') 
		<< endl;

      /// print & debug 
      if(debug && (*it).input!=0 )
	cout << "[EcalFEtoDigi] debug id: " << e_digi->id() << "\n\t" 
	     << dec 
	     << "\tieta: "     << e_digi->id().ieta()
	     << "\tiphi: "     << e_digi->id().iphi()
	     << "\tsize: "     << e_digi->size()
	     << "\tfg: "       <<(e_digi->fineGrain()?1:0)    
	     << hex 	 
	     << "\tEt: 0x"     << e_digi->compressedEt() 
	     << " (0x"         << (*it).get_energy() << ")" 
	     << "\tttflag: 0x" << e_digi->ttFlag()
	     << dec
	     << endl;

      delete e_digi;

    }
    
    if(singlefile)
      break;
  }

  ///in case no info was found for the event:need to create something
  if(e_tpdigis->size()==0) {
    cout << "[EcalFEtoDigi] creating empty collection for the event!\n";
    EcalTriggerPrimitiveDigi *e_digi = new EcalTriggerPrimitiveDigi();
    e_tpdigis->push_back(*e_digi);
  }


  iEvent.put(e_tpdigis);

}


/// open and read in input (flat) data file
void 
EcalFEtoDigi::readInput() {

  if(debug)
    cout << "\n[EcalFEtoDigi::readInput] Reading input data\n";
  
  stringstream s;
  int tcc;

  for (int i=0; i<N_SM; i++) {

    tcc = (sm_==-1)?SMidToTCCid(i+1):SMidToTCCid(sm_);

    s.str("");
    s << basename_ << tcc << ".txt"; 

    ifstream f(s.str().c_str());
    
    if(debug) {
      cout << "  opening " << s.str().c_str() << "..." << endl;
      if(!f.good())
	cout << " skipped!"; 
      cout << endl;       
    }
    //if (!f.good() || f.eof()) 
    //  throw cms::Exception("BadInputFile") 
    //	<< "EcalFEtoDigi: cannot open file " << s.str().c_str() << endl; 
    
    int n_bx=0;
    int tt; int bx; unsigned val; int dummy;

    while(f.good()) {
      if(f.eof()) break;
      tt=0; bx=-1; val=0x0; dummy=0;
      f >> tt >> bx >> hex >> val >> dec >> dummy;
      if(bx==-1 || bx < skipEvents_ ) continue;
      if( !n_bx || (bx!=(inputdata_[i].back()).bunchCrossing) )
	n_bx++;
      TCCinput ttdata(tt,bx,val);
      inputdata_[i].push_back(ttdata);
      
      if(debug&&val!=0)
	printf("\treading tower:%d  bx:%d input:0x%x dummy:%2d\n", 
	       tt, bx, val, dummy);
    } 
    
    f.close();
    
    if(sm_!=-1)
      break;    
  }

  if(debug)
    cout << "[EcalFEtoDigi::readInput] Done reading." << endl;

  return; 
}

/// create EcalTrigTowerDetId from input data (line)
EcalTrigTowerDetId 
EcalFEtoDigi::create_TTDetId(TCCinput data) {

  // (EcalBarrel only)
  static const int kTowersInPhi = 4;
  
  int iTT   = data.tower;
  int zside = (sm_>18)?-1:+1;
  int SMid  = sm_;

  int jtower = iTT-1;
  int etaTT  = jtower / kTowersInPhi +1;
  int phiTT;
  if (zside < 0) 
    phiTT = (SMid-19) * kTowersInPhi + jtower % kTowersInPhi;
  else 
    phiTT = (SMid- 1) * kTowersInPhi + kTowersInPhi-(jtower % kTowersInPhi)-1;

  phiTT ++;
  //needed as phi=0 (iphi=1) is at middle of lower SMs (1 and 19), need shift by 2
  phiTT = phiTT -2; 
  if (phiTT <= 0) phiTT = 72+phiTT;

  /// construct the EcalTrigTowerDetId object
  if(debug&&data.get_energy()!=0)
    printf("[EcalFEtoDigi] Creating EcalTrigTowerDetId (SMid,itt)=(%d,%d)->(eta,phi)=(%d,%d) \n", SMid, iTT, etaTT, phiTT);
  
  EcalTrigTowerDetId 
    e_id( zside , EcalBarrel, etaTT, phiTT, 0);
  
  return e_id;
}

/// create EcalTriggerPrimitiveSample from input data (line)
EcalTriggerPrimitiveSample 
EcalFEtoDigi::create_TPSample(TCCinput data) {

  int tower      = data.tower;
  int  Et        = data.get_energy();
  bool tt_fg     = data.get_fg();
  //unsigned input = data.input;
  //int  Et    = input & 0x3ff; //get bits 0-9
  //bool tt_fg = input & 0x400; //get bit number 10

  /// setup look up table
  std::vector<unsigned int> lut_ ;
  if(doCompressEt_)
    lut_ = db_->getTowerParameters(sm_, tower) ;
  else
    for(int i=0; i<500; i++) lut_.push_back(i); //identity lut!
  
  /// compress energy 10 -> 8  bit
  int lut_out = lut_[Et] ;
  int ttFlag  = (lut_out & 0x700) >> 8 ;
  int cEt     = (lut_out & 0xff );

  ///create sample
  if(debug&&data.get_energy()!=0)
    printf("[EcalFEtoDigi] Creating sample; input:0x%X (Et:0x%x) cEt:0x%x fg:%d ttflag:0x%x \n",
	   data.input, Et, cEt, tt_fg, ttFlag);
  
  EcalTriggerPrimitiveSample e_sample(cEt, tt_fg, ttFlag);
  
  return e_sample;
}


/// method called once each job just before starting event loop
void 
EcalFEtoDigi::beginJob(const edm::EventSetup&){

  ///check SM numbering convetion:: here assume 1-38 
  /// [or -1 flag to indicate all sm's will be read in]
  if(sm_!=-1 && sm_<1 || sm_>36) 
    throw cms::Exception("InvalidDetId") 
      << "EcalFEtoDigi: Adapt SM numbering convention.";

  ///debug: open file for recreating input copy
  outfile.open("inputcopy.txt");

  db_ = new DBInterface(databaseFileNameEB_,databaseFileNameEE_);

  readInput();
  
}  


/// method called once each job just after ending the event loop
void 
EcalFEtoDigi::endJob() {

  ///debug: close file with recreated input copy
  outfile.close();
}

/// translate input supermodule id into TCC id
int 
EcalFEtoDigi::SMidToTCCid( const int smid ) const {

  return (smid<=18) ? smid+55-1 : smid+37-19;

}
