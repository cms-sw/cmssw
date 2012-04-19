#include "SimCalorimetry/HcalSimProducers/src/HcalHitRelabeller.h"
#include "SimDataFormats/CaloTest/interface/HcalTestNumbering.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

HcalHitRelabeller::HcalHitRelabeller(const edm::ParameterSet& ps) : m_crossFrame(0) {
  // try to make sure the memory gets pinned in place
  m_pileupRelabelled.reserve(5000);
  m_segmentation.resize(29);
  for (int i=0; i<29; i++) {
    char name[10];
    snprintf(name,10,"Eta%d",i+1);
    if (i>0) {
      m_segmentation[i]=ps.getUntrackedParameter<std::vector<int> >(name,m_segmentation[i-1]);
    } else {
      m_segmentation[i]=ps.getUntrackedParameter<std::vector<int> >(name);
    }
  }
}

void HcalHitRelabeller::process(const CrossingFrame<PCaloHit>& cf) {
  if (m_crossFrame) {
    delete m_crossFrame;
    m_crossFrame = 0;
  }
  m_signalRelabelled.clear();
  m_pileupRelabelled.clear();
 
  m_crossFrame=new CrossingFrame<PCaloHit>(cf.getBunchRange().first,cf.getBunchRange().second,cf.getBunchSpace(),"RELABEL_HCAL",400);

  // cf.print(1);
  // m_crossFrame->print(1);
  
/*std::vector<PCaloHit>::const_iterator ibegin;
  std::vector<PCaloHit>::const_iterator iend;
  std::vector<PCaloHit>::const_iterator i;
  // signal
  cf.getSignal(ibegin,iend);
  
  for (i=ibegin; i!=iend; i++) {
  DetId newid=relabel(i->id());
  m_signalRelabelled.push_back(PCaloHit(newid.rawId(),i->energy(),i->time(),i->geantTrackId(),i->energyEM()/i->energy(),i->depth()));
  }  

  for (int dp=1; dp<=7; ++dp) {
    for (int eta=1; eta<30; ++eta) {
      for (int i=0; i<72; ++i) {
	int iz=1, phi;
	if (i%2 != 0) iz = -1;
	phi = i/2;
	phi = 2*phi + 1;
	HcalSubdetector sd=HcalBarrel;
	if (eta > 16 || (eta == 16 && dp > 2)) sd=HcalEndcap;
	HcalDetId cell = HcalDetId(sd,iz*eta,phi,dp);
	uint32_t id = cell.denseIndex();
	HcalDetId newCell = HcalDetId::detIdFromDenseIndex(id);
//	if (cell() != newCell()) std::cout << cell << " Id " << id << " Convert " << newCell << std::endl;
      }
    }
  }
*/
  // Begin Change by Wetzel - Commented lines caused errors
  std::vector<const PCaloHit*>::const_iterator ibegin;// = cf.getSignal().begin();
  std::vector<const PCaloHit*>::const_iterator iend;// = cf.getSignal().end();
  // signal
  cf.getSignal(ibegin,iend);
  
  // std::cout << "Sig hit size: " << cf.getSignal().size() << '\n';
  // std::cout << "pu hit size: " << cf.getPileups().size() << '\n';
  
  for (std::vector<const PCaloHit*>::const_iterator i = ibegin; 
       i != iend; ++i) {

    DetId testId = (*i)->id(); //std::vector<const PCaloHit*>::const_iterator i
    int det, z, depth, eta, phi, layer;
    HcalTestNumbering::unpackHcalIndex(testId,det,z,depth,eta,phi,layer);
    /*
    std::cout << std::endl 
	      << "TestNumId ->  det: " << det << " "
	      << "depth: " << depth << " "
	      << "ieta: "  << eta << " "
	      << "iphi: "  << phi << " "
	      << "layer: " << layer 
	      << std::endl;
    
    HcalDetId cell((*i)->id()); 

    int iphi, ieta, sub;
    depth = cell.depth();
    iphi  = cell.iphi()-1;
    ieta  = cell.ieta();
    sub   = cell.subdet();
    std::cout << "  HcalDetId ->  sub: " << sub << " "
	      << "depth: " << depth << " "
	      << "ieta: "  << ieta << " "
	      << "iphi: "  << iphi << " " 
	      << std::endl;


    const CaloCellGeometry *cellGeometry =
      theGeometry->getSubdetectorGeometry(cell)->getGeometry(cell);
    const GlobalPoint& globalposition = cellGeometry->getPosition();
    
    std::cout << "PCaloHit position: " << globalposition << std::endl;

    */
      

    // std::cout << std::hex << (*i)->id() << std::dec << '\n';
    DetId newid = relabel((*i)->id());
    // std::cout << std::hex << newid.rawId() << std::dec << '\n';

    HcalDetId newcell(newid);


    const CaloCellGeometry *cellGeometry =
      theGeometry->getSubdetectorGeometry(newcell)->getGeometry(newcell);
    GlobalPoint globalposition = (GlobalPoint)(cellGeometry->getPosition());
    
    //    std::cout << "PCaloHit " << newcell << " position: " << globalposition << std::endl;


    PCaloHit newHit(newid.rawId(), (*i)->energy(), (*i)->time(), 
		    (*i)->geantTrackId(), (*i)->energyEM()/(*i)->energy(), 
		    newcell.depth());

    // std::cout << newHit.energy() << '\n';
    newHit.setEventId((*i)->eventId());
    // std::cout << newHit.eventId().event() << '\n';
    m_signalRelabelled.push_back(newHit);
    // std::cout << "Hit added." << std::endl;
  }
  //End Change by Wetzel
  
  
  m_crossFrame->addSignals(&m_signalRelabelled,cf.getEventID());
  
  // pileup
  cf.getPileups(ibegin, iend);
  // const unsigned int base=cf.getNrSignals();
  // const unsigned int np=cf.getNrPileups();
  // for (unsigned int j=0; j<np; j++) {
  EncodedEventId lastId;
  int ievent = 0;
  for (std::vector<const PCaloHit*>::const_iterator i = ibegin;
       i != iend; ++i) {
    if ((!m_pileupRelabelled[ievent].empty()) && (lastId != (*i)->eventId())) {
      m_crossFrame->addPileups(lastId.bunchCrossing(), &m_pileupRelabelled[ievent], 
			       lastId.event());
      ++ievent;
    }
    // const PCaloHit& pch=cf.getObject(base+j);
    DetId newid=relabel((*i)->id());
    PCaloHit newHit(newid.rawId(), (*i)->energy(), (*i)->time(), 
		    (*i)->geantTrackId(), (*i)->energyEM()/(*i)->energy(), 
		    (*i)->depth());
    newHit.setEventId((*i)->eventId());
    m_pileupRelabelled[ievent].push_back(newHit);
    lastId = (*i)->eventId();
  }
  if(!m_pileupRelabelled[ievent].empty()){
    m_crossFrame->addPileups(lastId.bunchCrossing(), &m_pileupRelabelled[ievent],
                             lastId.event());
  }
}

void HcalHitRelabeller::clear() {
  m_signalRelabelled.clear();
  m_pileupRelabelled.clear();
}


void HcalHitRelabeller::setGeometry(const CaloGeometry*& geom) {

  theGeometry = geom;
}

DetId HcalHitRelabeller::relabel(const uint32_t testId) const {
  HcalDetId hid;

  int det, z, depth, eta, phi, layer, sign;

  //  std::cout << " HcalHitRelabeller::relabel " << std::endl;

  HcalTestNumbering::unpackHcalIndex(testId,det,z,depth,eta,phi,layer);

  layer-=1; // one is added in the simulation, here used for indexing  

  sign=(z==0)?(-1):(1);

  /*  
  std::cout << "det: " << det << " "
  	    << "z: " << z << " "
   	    << "depth: " << depth << " "
   	    << "ieta: " << eta << " "
   	    << "iphi: " << phi << " "
   	    << "layer: " << layer << " ";
  std::cout.flush();
  */

  int newDepth = 0; // moved out of if's just for printing purposes...

  if (det==int(HcalBarrel)) {
    newDepth=m_segmentation[eta-1][layer];
    if(eta==16 && newDepth > 2) newDepth=2;// tower 16 HACK to be watched out..
    hid=HcalDetId(HcalBarrel,eta*sign,phi,newDepth);        
  }
  if (det==int(HcalEndcap)) {
    newDepth=m_segmentation[eta-1][layer];
    if (eta>=21 && (phi%2)==0) phi--; // combine double-width towers in HE
    if (eta==16 && newDepth<3) newDepth=3; // tower 16 HACK to be watched out..
    hid=HcalDetId(HcalEndcap,eta*sign,phi,newDepth);    
  }
  if (det==int(HcalOuter)) {
    hid=HcalDetId(HcalOuter,eta*sign,phi,4);    
    newDepth = 4;
  }
  if (det==int(HcalForward)) {
    if ((phi%2)==0) phi--;
    if (eta>=40 && ((phi-1)%4)==0) {
      phi-=2;
      if (phi<0) phi+=72;
    }

    hid=HcalDetId(HcalForward,eta*sign,phi,depth);
    newDepth = depth; 

  }

  /*
  std::cout << " new HcalDetId -> hex.RawID = "
	    << std::hex << hid.rawId() << std::dec;
  std::cout.flush();
  std::cout << " det, z, depth, eta, phi = "
	    << det << " "
	    << z << " "
	    << newDepth << " "
	    << eta << " "
	    << phi << " "
	    <<  " ---> " << hid << std::endl;  
  */

  return hid;
}
