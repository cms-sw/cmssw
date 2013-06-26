#include "SimG4Core/GFlash/TB/TreeProducerCalibSimul.h"

using namespace std;


// -------------------------------------------------
// contructor
TreeProducerCalibSimul::TreeProducerCalibSimul( const edm::ParameterSet& iConfig )
{
  // now do what ever initialization is needed
  xtalInBeam                 = iConfig.getUntrackedParameter<int>("xtalInBeam",-1000);
  rootfile_                  = iConfig.getUntrackedParameter<std::string>("rootfile","mySimMatrixTree.root");
  txtfile_                   = iConfig.getUntrackedParameter<std::string>("txtfile", "mySimMatrixTree.txt");
  EBRecHitCollection_        = iConfig.getParameter<std::string>("EBRecHitCollection");
  RecHitProducer_            = iConfig.getParameter<std::string>("RecHitProducer");
  hodoRecInfoCollection_     = iConfig.getParameter<std::string>("hodoRecInfoCollection");
  hodoRecInfoProducer_       = iConfig.getParameter<std::string>("hodoRecInfoProducer");
  tdcRecInfoCollection_      = iConfig.getParameter<std::string>("tdcRecInfoCollection");
  tdcRecInfoProducer_        = iConfig.getParameter<std::string>("tdcRecInfoProducer");
  eventHeaderCollection_     = iConfig.getParameter<std::string>("eventHeaderCollection");
  eventHeaderProducer_       = iConfig.getParameter<std::string>("eventHeaderProducer");

  // summary
  cout << endl;
  cout <<"Constructor" << endl;
  cout << endl;
  cout << "TreeProducerCalibSimul"    << endl;
  cout << "xtal in beam = " << xtalInBeam << endl;
  cout <<"Fetching hitCollection: "   << EBRecHitCollection_.c_str()     << " prod by " << RecHitProducer_.c_str()      <<endl; 
  cout <<"Fetching hodoCollection: "  << hodoRecInfoCollection_.c_str()  << " prod by " << hodoRecInfoProducer_.c_str() <<endl;
  cout <<"Fetching tdcCollection: "   << tdcRecInfoCollection_.c_str()   << " prod by " << tdcRecInfoProducer_.c_str()  <<endl;
  cout <<"Fetching evHeaCollection: " << eventHeaderCollection_.c_str()  << " prod by " << eventHeaderProducer_.c_str() <<endl;
  cout << endl;
}


// -------------------------------------------------
// destructor
TreeProducerCalibSimul::~TreeProducerCalibSimul()
{
  cout << endl;
  cout << "Deleting" << endl;
  cout << endl;

  delete myTree;
}



// ------------------------------------------------------
// initializations
void TreeProducerCalibSimul::beginJob()
{
  cout << endl;
  cout << "BeginJob" << endl;
  cout << endl;

  // tree
  myTree = new TreeMatrixCalib(rootfile_.c_str());

  // counters
  tot_events      = 0;
  tot_events_ok   = 0;
  noHits          = 0;
  noHodo          = 0;
  noTdc           = 0;
  noHeader        = 0;
} 



// -------------------------------------------
// finalizing
void TreeProducerCalibSimul::endJob() 
{
  cout << endl;
  cout << "EndJob" << endl;
  cout << endl;

  ofstream *MyOut = new ofstream(txtfile_.c_str());   
  *MyOut << "total events: "                                   << tot_events      << endl;
  *MyOut << "events skipped because of no hits: "              << noHits          << endl;
  *MyOut << "events skipped because of no hodos: "             << noHodo          << endl;
  *MyOut << "events skipped because of no tdc: "               << noTdc           << endl;
  *MyOut << "events skipped because of no header: "            << noHeader        << endl;
  *MyOut << "total OK events (passing the basic selection): "  << tot_events_ok   << endl;
  MyOut->close();
  delete MyOut;
}



// -----------------------------------------------
// my analysis
void TreeProducerCalibSimul::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace cms;

  // counting events
  tot_events++;

  if ( tot_events%5000 == 0){ cout << "event " << tot_events << endl;}
  

  // ---------------------------------------------------------------------
  // taking what I need: hits
  Handle< EBRecHitCollection > pEBRecHits ;
  const EBRecHitCollection*  EBRecHits = 0 ;
  //try {
  iEvent.getByLabel (RecHitProducer_, EBRecHitCollection_, pEBRecHits) ;
  EBRecHits = pEBRecHits.product(); 
  //} catch ( std::exception& ex ) {
  //std::cout<<"Error! can't get the product " << EBRecHitCollection_.c_str () << std::endl ;
  //std::cerr<<"Error! can't get the product " << EBRecHitCollection_.c_str () << std::endl ;
  //}

  // taking what I need: hodoscopes
  Handle<EcalTBHodoscopeRecInfo> pHodo;
  const EcalTBHodoscopeRecInfo* recHodo=0;
  //try {
  iEvent.getByLabel( hodoRecInfoProducer_, hodoRecInfoCollection_, pHodo);
  recHodo = pHodo.product();
  //} catch ( std::exception& ex ) {
  //std::cout<<"Error! can't get the product "<<hodoRecInfoCollection_.c_str() << std::endl;
  //std::cerr<<"Error! can't get the product "<< hodoRecInfoCollection_.c_str() << std::endl;
  //}
  
  // taking what I need: tdc
  Handle<EcalTBTDCRecInfo> pTDC;
  const EcalTBTDCRecInfo* recTDC=0;
  //try {
  iEvent.getByLabel( tdcRecInfoProducer_, tdcRecInfoCollection_, pTDC);
  recTDC = pTDC.product(); 
  //} catch ( std::exception& ex ) {
  //std::cout<<"Error! can't get the product " << tdcRecInfoCollection_.c_str() << std::endl;
  //std::cerr<<"Error! can't get the product " << tdcRecInfoCollection_.c_str() << std::endl;
  //}

  // taking what I need: event header
  Handle<EcalTBEventHeader> pEventHeader;
  const EcalTBEventHeader* evtHeader=0;
  //try {
  iEvent.getByLabel( eventHeaderProducer_ , pEventHeader );
  evtHeader = pEventHeader.product(); 
  //} catch ( std::exception& ex ) {
  //std::cout << "Error! can't get the event header " <<std::endl;
  //std::cerr << "Error! can't get the event header " <<std::endl;
  //}
   
  // checking everything is there and fine
  if ( (!EBRecHits) || (EBRecHits->size() == 0)){ noHits++;  return; }
  if (!recTDC)    { noTdc++;     return; }        
  if (!recHodo)   { noHodo++;    return; }                  
  if (!evtHeader) { noHeader++;  return; }                
  tot_events_ok++;



  // ---------------------------------------------------------------------
  // info on the event
  int run   = -999;
  int tbm   = -999;
  int event = evtHeader->eventNumber();

  // ---------------------------------------------------------------------
  // xtal-in-beam  
  int nomXtalInBeam  = -999;
  int nextXtalInBeam = -999;

  EBDetId xtalInBeamId(1,xtalInBeam, EBDetId::SMCRYSTALMODE); 
  if (xtalInBeamId==EBDetId(0)){ return; }
  int mySupCry = xtalInBeamId.ic();
  int mySupEta = xtalInBeamId.ieta();
  int mySupPhi = xtalInBeamId.iphi();

  
  // ---------------------------------------------------------------------
  // hodoscope information
  double x  = recHodo->posX();
  double y  = recHodo->posY();
  double sx = recHodo->slopeX();
  double sy = recHodo->slopeY();
  double qx = recHodo->qualX();
  double qy = recHodo->qualY();
    

  // ---------------------------------------------------------------------
  // tdc information
  double tdcOffset = recTDC->offset();
  

  // ---------------------------------------------------------------------
  // Find EBDetId in a 7x7 Matrix
  EBDetId Xtals7x7[49];
  double energy[49];
  int crystal[49];
  int allMatrix = 1; 
  for (unsigned int icry=0; icry<49; icry++){
    unsigned int row    = icry/7;
    unsigned int column = icry%7;
    //try
    //  {
    Xtals7x7[icry]=EBDetId(xtalInBeamId.ieta()+column-3, xtalInBeamId.iphi()+row-3, EBDetId::ETAPHIMODE);
	
    if ( Xtals7x7[icry].ism() == 1){ 
      energy[icry] = EBRecHits->find(Xtals7x7[icry])->energy(); 
      crystal[icry] = Xtals7x7[icry].ic();
    } else {
      energy[icry] = -100.;
      crystal[icry] = -100;
      allMatrix = 0; 
    }
    /* 
    catch (...)
      {
	// can not construct 7x7 matrix 
	energy[icry]  = -100.; 
	crystal[icry] = -100;
	allMatrix = 0;  
      }
    */
  }


  // ---------------------------------------------------------------------
  // Looking for the max energy crystal
  double maxEne      = -999.;
  int maxEneCry      = 9999;
  int maxEneInMatrix = -999;
  for (int ii=0; ii<49; ii++){ if (energy[ii] > maxEne){ 
    maxEne         = energy[ii]; 
    maxEneCry      = crystal[ii]; 
    maxEneInMatrix = ii;} 
  }  
  


  // Position reconstruction - skipped here
  double Xcal   = -999.;
  double Ycal   = -999.;

  // filling the tree
  myTree->fillInfo(run, event, mySupCry, maxEneCry, nomXtalInBeam, nextXtalInBeam, mySupEta, mySupPhi, tbm, x, y, Xcal, Ycal, sx, sy, qx, qy, tdcOffset, allMatrix, energy, crystal);
  myTree->store();
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//define this as a plug-in

DEFINE_FWK_MODULE(TreeProducerCalibSimul);
