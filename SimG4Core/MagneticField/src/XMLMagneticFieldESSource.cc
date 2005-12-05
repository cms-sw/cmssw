#include "SimG4Core/MagneticField/interface/XMLMagneticFieldESSource.h"
#include "GeometryReaders/XMLIdealGeometryESSource/interface/GeometryConfiguration.h"

#include "DetectorDescription/Base/interface/DDException.h"
#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDRoot.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"

#include <memory>

XMLMagneticFieldESSource::XMLMagneticFieldESSource(const edm::ParameterSet & p): rootNodeName_(p.getParameter<std::string>("rootNodeName"))
{
    DDLParser * parser = DDLParser::instance();
    GeometryConfiguration cf;
    edm::FileInPath fp = p.getParameter<edm::FileInPath>("GeometryConfiguration");
    DCOUT ('X', "FileInPath is looking for " + fp.fullPath());
    int result1 = cf.readConfig(fp.fullPath());
    if (result1 !=0) throw DDException("DDLConfiguration: readConfig failed !");
    int result2 = parser->parse(cf);
    if (result2 != 0) throw DDException("DDD-Parser: parsing failed!");
    if(rootNodeName_ == "" || rootNodeName_ == "\""){
       rootNodeName_ = DDRootDef::instance().root().ddname();
       //std::cout <<"default name \""<<rootNodeName_<<"\""<<std::endl;
       setWhatProduced(this);
    }else {
       setWhatProduced(this,rootNodeName_);
    }
    findingRecord<IdealMagneticFieldRecord>();
}

XMLMagneticFieldESSource::~XMLMagneticFieldESSource() {}

std::auto_ptr<DDCompactView>
XMLMagneticFieldESSource::produce(const IdealMagneticFieldRecord &)
{ 
   //std::cout <<"got in produce"<<std::endl;
   DDName ddName(rootNodeName_);
   //std::cout <<"ddName \""<<ddName<<"\""<<std::endl;
   DDLogicalPart rootNode(ddName);
   //std::cout <<"made the DDLogicalPart"<<std::endl;
   if(! rootNode.isValid()){
      throw cms::Exception("Geometry")<<"There is no valid node named \""
				      <<rootNodeName_<<"\"";
   }
   std::auto_ptr<DDCompactView> returnValue(new DDCompactView(rootNode));
   //copy the graph from the global one
   DDCompactView globalOne;
   returnValue->writeableGraph() = globalOne.graph();
   //std::cout <<"made the view"<<std::endl;
   return returnValue;
}

void XMLMagneticFieldESSource::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &,
					      const edm::IOVSyncValue & iosv, 
					      edm::ValidityInterval & oValidity)
{
   edm::ValidityInterval infinity(iosv.beginOfTime(), iosv.endOfTime());
   oValidity = infinity;
}


#include "FWCore/Framework/interface/SourceFactory.h"


DEFINE_FWK_EVENTSETUP_SOURCE(XMLMagneticFieldESSource)


