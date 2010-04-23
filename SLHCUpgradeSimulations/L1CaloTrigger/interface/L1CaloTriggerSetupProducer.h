#ifndef L1CALOTRIGGERSETUPPRODUCER_H
#define L1CALOTRIGGERSETUPPRODUCER_H

#include <memory>
#include "boost/shared_ptr.hpp"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducts.h"
#include "SimDataFormats/SLHC/interface/L1CaloTriggerSetup.h"
#include "SimDataFormats/SLHC/interface/L1CaloTriggerSetupRcd.h"

//Include XERCES-C XML Parser
#include <xercesc/dom/DOM.hpp>
#include <xercesc/dom/DOMDocument.hpp>
#include <xercesc/dom/DOMDocumentType.hpp>
#include <xercesc/dom/DOMElement.hpp>
#include <xercesc/dom/DOMImplementation.hpp>
#include <xercesc/dom/DOMImplementationLS.hpp>
#include <xercesc/dom/DOMNodeIterator.hpp>
#include <xercesc/dom/DOMNodeList.hpp>
#include <xercesc/dom/DOMText.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/util/XMLUni.hpp>
#include <string>
#include <stdexcept>


// Error codes
enum {
  ERROR_ARGS = 1,
  ERROR_XERCES_INIT,
  ERROR_PARSE,
   ERROR_EMPTY_DOCUMENT
};


class L1CaloTriggerSetupProducer : public edm::ESProducer {
   public:
      L1CaloTriggerSetupProducer(const edm::ParameterSet&);
      ~L1CaloTriggerSetupProducer();

      typedef boost::shared_ptr<L1CaloTriggerSetup> ReturnType;

      ReturnType produce(const L1CaloTriggerSetupRcd&);
   private:


  void openFile(std::string&) throw(std::runtime_error);
  void config(L1CaloTriggerSetup&); //Configure


  edm::FileInPath inputFile_;
  xercesc::XercesDOMParser *m_ConfigFileParser; //Main Parser object


  //DECLARE THE XML TAGS AND ATTRIBUTES


     XMLCh* TAG_GEO; //Tag For Geometry and Dimension
     XMLCh* ATT_GEO_eta; //Eta0 of the lattice
     XMLCh* ATT_GEO_phi; //Phi0 of the lattice
     XMLCh* ATT_GEO_etam; //EtaMax of the lattice
     XMLCh* ATT_GEO_phim; //PhiMax of the lattice
     XMLCh* ATT_GEO_dim; //Dimension of the lattice (Square
     XMLCh* TAG_SETTINGS;//Tag for Card general settings

     //Thresholds/Settings
      XMLCh* ATT_SETTINGS_ECALTower;
      XMLCh* ATT_SETTINGS_HCALTower;
      XMLCh* ATT_SETTINGS_ElectronCutA;
      XMLCh* ATT_SETTINGS_ElectronCutB;
      XMLCh* ATT_SETTINGS_ElectronCutC;

      XMLCh* ATT_SETTINGS_TauSeedTower;

      XMLCh* ATT_SETTINGS_ClusterCut;
      XMLCh* ATT_SETTINGS_IsolationEA;
      XMLCh* ATT_SETTINGS_IsolationEB;
      XMLCh* ATT_SETTINGS_IsolationTA;
      XMLCh* ATT_SETTINGS_IsolationTB;
      XMLCh* ATT_SETTINGS_IsolationZone;
      XMLCh* ATT_SETTINGS_IsolationPedestalEG;
      XMLCh* ATT_SETTINGS_IsolationPedestalTau;


      XMLCh* ATT_SETTINGS_JetCenter;
      XMLCh* ATT_SETTINGS_JetET;

      //Wires
      XMLCh* TAG_WIRE;
      XMLCh* ATT_WIRE_bin;
      XMLCh* ATT_WIRE_eta;
      XMLCh* ATT_WIRE_phi;


    XMLCh* TAG_CARD; //Tag for  Card  ---General

};
#endif

