#include "FWCore/Framework/interface/EventSetup.h"
#include "SimGeneral/HepPDTESSource/test/HepPDTAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include <iostream>
#include <iomanip>

using namespace edm;
using namespace std;

HepPDTAnalyzer::HepPDTAnalyzer(const edm::ParameterSet& iConfig) :
  particleName_( iConfig.getParameter<std::string>( "particleName" ) ) {
}

HepPDTAnalyzer::~HepPDTAnalyzer() {
}

void
HepPDTAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  
  ESHandle <ParticleDataTable> pdt;
  iSetup.getData( pdt );

  if ( particleName_ == "all" ) {
    std::cout << " Number of particles in table = " << pdt->size() << std::endl;
    printBanner();
    for ( ParticleDataTable::const_iterator iter = pdt->begin() ; iter != pdt->end() ; iter++ ) {
      const ParticleData * part = pdt->particle ( (*iter).first );
       printInfo(part); 	
    }
  } else if ( particleName_ == "names" ) {
    std::cout << " Number of particles in table = " << pdt->size() << std::endl;
    printBanner();
    for ( ParticleDataTable::const_iterator iter = pdt->begin() ; iter != pdt->end() ; iter++ ) {
      const ParticleData * part = pdt->particle ( (*iter).first );
       std::cout << " " << part->name() << std::endl; 	
    }
  } else if ( particleName_ == "checkAllowedIDs") {
    int maxID = 2000000000;
    int minID = -1*maxID;
    int legalcount = 0;
    std::cout << " Scanning IDs from " << minID << " to " << maxID << std::endl;
    printBanner();
    for (int ii=minID; ii<=maxID; ii++) {
      const ParticleData *part = pdt->particle(ii);
      if (part) {
        legalcount++;
        printInfo(part);
      }
      if (abs(ii)>=1000000000) ii=ii+9;
    }
    std::cout << "  Found " << legalcount << " legal IDs in range." << std::endl;
  }
  else {
    printBanner();
    const ParticleData * part = pdt->particle( particleName_ );
    printInfo(part);
  }
   
}

void 
HepPDTAnalyzer::printInfo(const ParticleData* & part) {

  cout << setfill(' ') << setw(14); 
  cout << part->name();
  cout << setfill(' ') << setw(12); 
  cout << part->pid(); 
  cout << setfill(' ') << setw(8) << setprecision(3) ; 
  cout << part->charge();
  cout << setfill(' ') << setw(12) << setprecision(6) ; 
  cout << part->mass();
  cout << setfill(' ') << setw(14) << setprecision(5) ; 
  cout << part->totalWidth();
  cout << setfill(' ') << setw(14) << setprecision(5) ;
  cout << part->lifetime() << endl;

}

void
HepPDTAnalyzer::printBanner() {
    cout << " Particle name    ID        Charge     Mass       Tot. Width      Lifetime " << endl; 
    cout << " ------------------------------------------------------------------------- " << endl;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(HepPDTAnalyzer);
