#ifndef L1CALOTRIGGERSETUPPRODUCER_H
#define L1CALOTRIGGERSETUPPRODUCER_H

#include <string>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <list>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>

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

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


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


  edm::FileInPath mInputfile;
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


  //      XMLCh* ATT_SETTINGS_JetCenter;
      XMLCh* ATT_SETTINGS_JetET;
      XMLCh* ATT_SETTINGS_FineGrainPass;
      //Wires
      XMLCh* TAG_WIRE;
      XMLCh* ATT_WIRE_bin;
      XMLCh* ATT_WIRE_eta;
      XMLCh* ATT_WIRE_phi;


    XMLCh* TAG_CARD; //Tag for  Card  ---General

};
#endif







using namespace xercesc;
using namespace std;



L1CaloTriggerSetupProducer::L1CaloTriggerSetupProducer(const edm::ParameterSet& iConfig)
{
  mInputfile = iConfig.getParameter<edm::FileInPath>("InputXMLFile");

   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced(this);

}


L1CaloTriggerSetupProducer::~L1CaloTriggerSetupProducer()
{



}


//
// member functions
//

// ------------ method called to produce the data  ------------
L1CaloTriggerSetupProducer::ReturnType
L1CaloTriggerSetupProducer::produce(const L1CaloTriggerSetupRcd& iRecord)
{
   using namespace edm::es;

   L1CaloTriggerSetup *setup = new L1CaloTriggerSetup();
   std::string filePath = mInputfile.fullPath();

   edm::LogInfo ("INFO") << "Configuration File:"<<filePath << endl;



   edm::LogInfo ("INFO") << "Initializing XERCES" << endl;
   XMLPlatformUtils::Initialize();  // Initialize Xerces infrastructure


   edm::LogInfo ("INFO") << "Creating XML Tags" << endl;

   //Mother Card : ClusteringCard
    TAG_CARD =XMLString::transcode("CARD");

    //Geometry Setup
    TAG_GEO =XMLString::transcode("GEOMETRY");

    ATT_GEO_eta=XMLString::transcode("eta0");
    ATT_GEO_phi=XMLString::transcode("phi0");
    ATT_GEO_dim=XMLString::transcode("dim");
    ATT_GEO_etam=XMLString::transcode("etam");
    ATT_GEO_phim=XMLString::transcode("phim");

    //Algorithm Settings
    TAG_SETTINGS=XMLString::transcode("SETTINGS");

    ATT_SETTINGS_ECALTower=XMLString::transcode("ECALTower");
    ATT_SETTINGS_HCALTower=XMLString::transcode("HCALTower");
    ATT_SETTINGS_ClusterCut=XMLString::transcode("ClusterThr");

    ATT_SETTINGS_ElectronCutA=XMLString::transcode("ElectronConstant");
    ATT_SETTINGS_ElectronCutB=XMLString::transcode("ElectronThreshold");
    ATT_SETTINGS_ElectronCutC=XMLString::transcode("ElectronSlope");

    ATT_SETTINGS_TauSeedTower=XMLString::transcode("TauSeedTower");

    ATT_SETTINGS_IsolationEA=XMLString::transcode("IsolationElectronA");
    ATT_SETTINGS_IsolationEB=XMLString::transcode("IsolationElectronB");
    ATT_SETTINGS_IsolationTA=XMLString::transcode("IsolationTauA");
    ATT_SETTINGS_IsolationTB=XMLString::transcode("IsolationTauB");
    ATT_SETTINGS_IsolationZone=XMLString::transcode("IsolationZone");
    ATT_SETTINGS_IsolationPedestalEG=XMLString::transcode("IsolationThresholdElectron");
    ATT_SETTINGS_IsolationPedestalTau=XMLString::transcode("IsolationThresholdTau");
    //    ATT_SETTINGS_JetCenter=XMLString::transcode("JetCenter");
    ATT_SETTINGS_JetET=XMLString::transcode("JetEt");
    ATT_SETTINGS_FineGrainPass=XMLString::transcode("FineGrainPass");

    //Wire Information
    TAG_WIRE=XMLString::transcode("WIRE");
    ATT_WIRE_bin=XMLString::transcode("no");
    ATT_WIRE_eta=XMLString::transcode("eta");
    ATT_WIRE_phi=XMLString::transcode("phi");

    TAG_CARD = XMLString::transcode("CARD");


    //Initialize the parser

   edm::LogInfo ("INFO") << "Initializing XERCES Parser" << endl;
   m_ConfigFileParser = new XercesDOMParser;


   edm::LogInfo ("INFO") << "Opening File..." << endl;
   openFile(filePath);
   edm::LogInfo ("INFO") << "Parsing Configuration"<<endl;
   config(*setup);

   edm::LogInfo ("INFO") << "terminating XERCES"<<endl;
   XMLPlatformUtils::Terminate();  // Terminate Xerces

   XMLString::release(&TAG_CARD);
   XMLString::release(&TAG_GEO);
   XMLString::release(&ATT_GEO_eta);
   XMLString::release(&ATT_GEO_phi);
   XMLString::release(&ATT_GEO_dim);
   XMLString::release(&TAG_SETTINGS);
   XMLString::release(&ATT_SETTINGS_ECALTower);
   XMLString::release(&ATT_SETTINGS_HCALTower);
   XMLString::release(&ATT_SETTINGS_TauSeedTower);
   XMLString::release(&ATT_SETTINGS_ElectronCutA);
   XMLString::release(&ATT_SETTINGS_ElectronCutB);
   XMLString::release(&ATT_SETTINGS_ElectronCutC);

   XMLString::release(&ATT_SETTINGS_ClusterCut);
   XMLString::release(&ATT_SETTINGS_IsolationEA);
   XMLString::release(&ATT_SETTINGS_IsolationEB);
   XMLString::release(&ATT_SETTINGS_IsolationTA);
   XMLString::release(&ATT_SETTINGS_IsolationTB);
   XMLString::release(&ATT_SETTINGS_IsolationZone);
   //   XMLString::release(&ATT_SETTINGS_JetCenter);
   XMLString::release(&ATT_SETTINGS_JetET);
   XMLString::release(&ATT_SETTINGS_FineGrainPass);
   XMLString::release(&TAG_WIRE);
   XMLString::release(&ATT_WIRE_bin);
   XMLString::release(&ATT_WIRE_eta);
   XMLString::release(&ATT_WIRE_phi);
   XMLString::release(&ATT_SETTINGS_IsolationPedestalEG);
   XMLString::release(&ATT_SETTINGS_IsolationPedestalTau);


   edm::LogInfo ("INFO") << "Creating Setup Module"<<endl;
   boost::shared_ptr<L1CaloTriggerSetup> rcd =(boost::shared_ptr<L1CaloTriggerSetup>) setup ;
   edm::LogInfo ("INFO") << "Event Setup Successfull"<<endl;
   return rcd ;

}



void
L1CaloTriggerSetupProducer::openFile(string& configFile)
  throw( std::runtime_error )
{
  // Test to see if the file is ok.

  struct stat fileStatus;

  int iretStat = stat(configFile.c_str(), &fileStatus);
  if( iretStat == ENOENT )
    printf("Path file_name does not exist, or path is an empty string.");

  else if( iretStat == ENOTDIR )
    printf("A component of the path is not a directory.");
  else if( iretStat == ELOOP )
    printf("Too many symbolic links encountered while traversing the path.");
  else if( iretStat == EACCES )
    printf("Permission denied.");
  else if( iretStat == ENAMETOOLONG )
    printf("File can not be read\n");

  // Configure DOM parser.

  m_ConfigFileParser->setValidationScheme( XercesDOMParser::Val_Never );
  m_ConfigFileParser->setDoNamespaces( false );
  m_ConfigFileParser->setDoSchema( false );
  m_ConfigFileParser->setLoadExternalDTD( false );
  m_ConfigFileParser->parse( configFile.c_str() );

}



void
L1CaloTriggerSetupProducer::config(L1CaloTriggerSetup& rcd)
{


       //Read Document
       DOMDocument* xmlDoc = m_ConfigFileParser->getDocument();
        //get Root XML element <root>
       DOMElement* elementRoot = xmlDoc->getDocumentElement();
        if( !elementRoot ) throw(std::runtime_error( "empty XML document" ));
       DOMNodeList*  children = elementRoot->getChildNodes();
       const  XMLSize_t nodeCount = children->getLength();

       //loop through XML entries
      for( XMLSize_t xx = 0; xx < nodeCount; ++xx )
  {
    DOMNode* currentNode = children->item(xx);
    if( currentNode->getNodeType() && currentNode->getNodeType() == DOMNode::ELEMENT_NODE )
      {
            DOMElement* currentElement = dynamic_cast< xercesc::DOMElement* >( currentNode );

      //Search for Clustering Card
            if( XMLString::equals(currentElement->getTagName(), TAG_CARD))
        {
    edm::LogInfo ("INFO") << "FOUND XML Setup information"<<endl;

    //Loop on the Children of the Clustering Card
    DOMNodeList*  card_c = currentElement->getChildNodes();
    XMLSize_t cardc_count = card_c->getLength();
    for( XMLSize_t i = 0; i < cardc_count; ++i)
       {
                 DOMNode* Nodei = card_c->item(i);
               if( Nodei->getNodeType() && Nodei->getNodeType() == DOMNode::ELEMENT_NODE )
           {

             DOMElement* cc_el = dynamic_cast< xercesc::DOMElement* >( Nodei );
             //Look for geo
             if( XMLString::equals(cc_el->getTagName(), TAG_GEO))
         {
           int geo_eta =atoi(XMLString::transcode(cc_el->getAttribute(ATT_GEO_eta)));
           int geo_phi =atoi(XMLString::transcode(cc_el->getAttribute(ATT_GEO_phi)));
           int geo_etam =atoi(XMLString::transcode(cc_el->getAttribute(ATT_GEO_etam)));
           int geo_phim =atoi(XMLString::transcode(cc_el->getAttribute(ATT_GEO_phim)));
           int geo_dim =atoi(XMLString::transcode(cc_el->getAttribute(ATT_GEO_dim)));
           //Store geo on the card
           rcd.setGeometry(geo_eta,geo_phi,geo_etam,geo_phim,geo_dim);
           edm::LogInfo ("INFO") << "Geometry Set "<<endl;

         }

             //Look for SETTINGS
             if( XMLString::equals(cc_el->getTagName(), TAG_SETTINGS))
         {
           int att_ECALTower     =atoi(XMLString::transcode(cc_el->getAttribute(ATT_SETTINGS_ECALTower)));
           int att_HCALTower     =atoi(XMLString::transcode(cc_el->getAttribute(ATT_SETTINGS_HCALTower)));

           int att_TauSeedTower     =atoi(XMLString::transcode(cc_el->getAttribute(ATT_SETTINGS_TauSeedTower)));

           int att_ElectronCutA   =atoi(XMLString::transcode(cc_el->getAttribute(ATT_SETTINGS_ElectronCutA)));
           int att_ElectronCutB   =atoi(XMLString::transcode(cc_el->getAttribute(ATT_SETTINGS_ElectronCutB)));
           int att_ElectronCutC   =atoi(XMLString::transcode(cc_el->getAttribute(ATT_SETTINGS_ElectronCutC)));

           int att_ClusterCut    =atoi(XMLString::transcode(cc_el->getAttribute(ATT_SETTINGS_ClusterCut)));
           int att_isoEA         =atoi(XMLString::transcode(cc_el->getAttribute(ATT_SETTINGS_IsolationEA)));
           int att_isoEB         =atoi(XMLString::transcode(cc_el->getAttribute(ATT_SETTINGS_IsolationEB)));
           int att_isoTA         =atoi(XMLString::transcode(cc_el->getAttribute(ATT_SETTINGS_IsolationTA)));
           int att_isoTB         =atoi(XMLString::transcode(cc_el->getAttribute(ATT_SETTINGS_IsolationTB)));
           int att_isoZone       =atoi(XMLString::transcode(cc_el->getAttribute(ATT_SETTINGS_IsolationZone)));
           int att_isoPedEG      =atoi(XMLString::transcode(cc_el->getAttribute(ATT_SETTINGS_IsolationPedestalEG)));
           int att_isoPedTau     =atoi(XMLString::transcode(cc_el->getAttribute(ATT_SETTINGS_IsolationPedestalTau)));

	   //           int att_jetC          =atoi(XMLString::transcode(cc_el->getAttribute(ATT_SETTINGS_JetCenter)));
           int att_jetET         =atoi(XMLString::transcode(cc_el->getAttribute(ATT_SETTINGS_JetET)));
	   int att_fineGrainPass = atoi(XMLString::transcode(cc_el->getAttribute(ATT_SETTINGS_FineGrainPass)));

           //Store Activity Cuts on the Card
           rcd.setThresholds(att_ECALTower,
                 att_HCALTower,
                 att_ElectronCutA,
                 att_ElectronCutB,
                 att_ElectronCutC,
                 att_TauSeedTower,
                 att_ClusterCut,
                 att_isoEA,
                 att_isoEB,
                 att_isoTA,
                 att_isoTB,
                 att_isoZone,
                 att_isoPedEG,
                 att_isoPedTau,
			     //                 att_jetC,
		 att_jetET,
		 att_fineGrainPass     );
           edm::LogInfo ("INFO") << "Thresholds Set"<<endl;

         }

             //Look for Wires
             if( XMLString::equals(cc_el->getTagName(), TAG_WIRE))
         {
           int wire_bin  =atoi(XMLString::transcode(cc_el->getAttribute(ATT_WIRE_bin)));
           int wire_eta  =atoi(XMLString::transcode(cc_el->getAttribute(ATT_WIRE_eta)));
           int wire_phi  =atoi(XMLString::transcode(cc_el->getAttribute(ATT_WIRE_phi)));
           //store wires
           rcd.addWire(wire_bin,wire_eta,wire_phi);

         }
           }
       }

        }
      }
  }

}



DEFINE_FWK_EVENTSETUP_MODULE(L1CaloTriggerSetupProducer);



