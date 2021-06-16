// -*- C++ -*-
//
// Package:    Validation/HGCalValidation
// Class:      HGCalWaferValidation
//
/**\class HGCalWaferValidation HGCalWaferValidation.cc Validation/HGCalValidation/plugins/HGCalWaferValidation.cc

 Description: Validates HGCal wafer data inside DD against specifications given in a flat text file.

 Implementation:
     * Uses GraphWalker to follow DD hierarchy to find HGCal EE module and the HE modules.
     * Search of wafer layers and iterates each wafer found.
     * Extract x, y coordinate position from wafer positioning; thickness, u & v coords from copyNo.
       Wafer shape and rotation are extracted from given names of wafer logical volumes.
     * All extracted wafer info saved into a map indexed by (layer#, u, v).
     * Each line in flat text file are compared against wafer information in the map.
       Any errors are reported, counted and summarized at the end.
     * Unaccounted wafers, which are in DD but not in the flat text file, are also reported and counted.
*/
//
// Original Author:  Imran Yusuff
//         Created:  Thu, 27 May 2021 19:47:08 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalCommonData/interface/HGCalTypes.h"

#include <fstream>
#include <regex>

//
// class declaration
//

// If the analyzer does not use TFileService, please remove
// the template argument to the base class so the class inherits
// from  edm::one::EDAnalyzer<>
// This will improve performance in multithreaded jobs.

class HGCalWaferValidation : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalWaferValidation(const edm::ParameterSet&);
  ~HGCalWaferValidation() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // This is MessageLogger logging category
  const std::string logcat = "HGCalWaferValidation";

  // wafer coordinate is (layer, u, v), used as index for map
  using WaferCoord = std::tuple<int, int, int>;

  std::string strWaferCoord(const WaferCoord& coord);

  bool DDFindHGCal(DDCompactView::GraphWalker& walker, std::string targetName);
  void DDFindWafers(DDCompactView::GraphWalker& walker);
  void ProcessWaferLayer(DDCompactView::GraphWalker& walker);

  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------
  // module parameters
  edm::FileInPath geometryFileName_;

  // struct to hold wafer information from DD in map
  struct WaferInfo {
    int thickClass;
    double x;
    double y;
    std::string shapeCode;
    int rotCode;
  };

  // EDM token to access DD
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> viewToken_;

  // map holding all wafer properties from DD
  std::map<WaferCoord, struct WaferInfo> waferData;

  // boolean map to keep track of unaccounted DD wafers (not in flat file)
  std::map<WaferCoord, bool> waferValidated;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
HGCalWaferValidation::HGCalWaferValidation(const edm::ParameterSet& iConfig)
    : geometryFileName_(iConfig.getParameter<edm::FileInPath>("GeometryFileName")) {
  viewToken_ = esConsumes<DDCompactView, IdealGeometryRecord>();
  //now do what ever initialization is needed
}

HGCalWaferValidation::~HGCalWaferValidation() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  //
  // please remove this method altogether if it would be left empty
}

//
// member functions
//

// convert WaferCoord tuple to string representation (i.e. for printing)
std::string HGCalWaferValidation::strWaferCoord(const WaferCoord& coord) {
  std::stringstream ss;
  ss << "(" << std::get<0>(coord) << "," << std::get<1>(coord) << "," << std::get<2>(coord) << ")";
  return ss.str();
}

// ----- find HGCal entry among the DD -----
bool HGCalWaferValidation::DDFindHGCal(DDCompactView::GraphWalker& walker, std::string targetName) {
  if (walker.current().first.name().name() == targetName) {
    // target found
    return true;
  }
  if (walker.firstChild()) {
    do {
      if (DDFindHGCal(walker, targetName))
        // target inside child
        return true;
    } while (walker.nextSibling());
    walker.parent();
  }
  return false;
}

// ----- find the next wafer, then process the wafer layer -----
void HGCalWaferValidation::DDFindWafers(DDCompactView::GraphWalker& walker) {
  if (walker.current().first.name().fullname().rfind("hgcalwafer:", 0) == 0) {
    // first wafer found. Now process the entire layer of wafers.
    ProcessWaferLayer(walker);
    return;
  }
  if (walker.firstChild()) {
    do {
      DDFindWafers(walker);
    } while (walker.nextSibling());
    walker.parent();
  }
}

// ----- process the layer of wafers -----
void HGCalWaferValidation::ProcessWaferLayer(DDCompactView::GraphWalker& walker) {
  static int waferLayer = 0;  // layer numbers in DD are assumed to be sequential from 1
  waferLayer++;
  edm::LogVerbatim(logcat) << "ProcessWaferLayer: Processing layer " << waferLayer;
  do {
    if (walker.current().first.name().fullname().rfind("hgcalwafer:", 0) == 0) {
      auto wafer = walker.current();
      const std::string waferName(walker.current().first.name().fullname());
      const int copyNo = wafer.second->copyno();
      // extract DD layer properties
      const int waferType = HGCalTypes::getUnpackedType(copyNo);
      const int waferU = HGCalTypes::getUnpackedU(copyNo);
      const int waferV = HGCalTypes::getUnpackedV(copyNo);
      const WaferCoord waferCoord(waferLayer, waferU, waferV);  // map index
      // build struct of DD wafer properties
      struct WaferInfo waferInfo;
      waferInfo.thickClass = waferType;
      waferInfo.x = wafer.second->translation().x();
      waferInfo.y = wafer.second->translation().y();
      const std::string waferNameData =
          std::regex_replace(waferName,
                             std::regex("(HGCal[EH]E)(Wafer[01])(Fine|Coarse[12])([a-z]*)([0-9]*)"),
                             "$1 $2-$3 $4 $5",
                             std::regex_constants::format_no_copy);
      std::stringstream ss(waferNameData);
      std::string EEorHE;
      std::string typeStr;
      std::string shapeStr;
      std::string rotStr;
      ss >> EEorHE >> typeStr >> shapeStr >> rotStr;
      // assume rotational symmetry of full-sized wafers
      if (shapeStr.empty())
        shapeStr = "F";
      if (rotStr.empty())
        rotStr = "0";
      const int rotCode(std::stoi(rotStr));
      //edm::LogVerbatim(logcat) << "rotStr " << rotStr << " rotCode " << rotCode;
      waferInfo.shapeCode = shapeStr;
      waferInfo.rotCode = rotCode;
      // populate the map
      waferData[waferCoord] = waferInfo;
      waferValidated[waferCoord] = false;
    }
  } while (walker.nextSibling());
}

// ------------ method called for each event  ------------
void HGCalWaferValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  // Get the CMS DD
  auto viewH = iSetup.getHandle(viewToken_);

  if (!viewH.isValid()) {
    edm::LogPrint(logcat) << "Error obtaining geometry handle!";
    return;
  }

  edm::LogVerbatim(logcat) << "Root is : " << viewH->root();
  edm::LogVerbatim(logcat) << std::endl;

  // find HGCalEE
  auto eeWalker = viewH->walker();
  const bool eeFound = DDFindHGCal(eeWalker, "HGCalEE");
  if (eeFound) {
    edm::LogVerbatim(logcat) << "HGCalEE found!";
    edm::LogVerbatim(logcat) << "name     = " << eeWalker.current().first.name().name();
    edm::LogVerbatim(logcat) << "fullname = " << eeWalker.current().first.name().fullname();
  } else {
    edm::LogPrint(logcat) << "HGCalEE not found!";
  }
  edm::LogVerbatim(logcat) << std::endl;

  // find HGCalHEsil
  auto hesilWalker = viewH->walker();
  const bool hesilFound = DDFindHGCal(hesilWalker, "HGCalHEsil");
  if (hesilFound) {
    edm::LogVerbatim(logcat) << "HGCalHEsil found!";
    edm::LogVerbatim(logcat) << "name     = " << hesilWalker.current().first.name().name();
    edm::LogVerbatim(logcat) << "fullname = " << hesilWalker.current().first.name().fullname();
  } else {
    edm::LogPrint(logcat) << "HGCalHEsil not found!";
  }
  edm::LogVerbatim(logcat) << std::endl;

  // find HGCalHEmix
  auto hemixWalker = viewH->walker();
  const bool hemixFound = DDFindHGCal(hemixWalker, "HGCalHEmix");
  if (hemixFound) {
    edm::LogVerbatim(logcat) << "HGCalHEmix found!";
    edm::LogVerbatim(logcat) << "name     = " << hemixWalker.current().first.name().name();
    edm::LogVerbatim(logcat) << "fullname = " << hemixWalker.current().first.name().fullname();
  } else {
    edm::LogPrint(logcat) << "HGCalHEmix not found!";
  }
  edm::LogVerbatim(logcat) << std::endl;

  // give up if no HGCal found at all
  if (!(eeFound || hesilFound || hemixFound)) {
    edm::LogPrint(logcat) << "Nothing found. Giving up.";
    return;
  }

  // Now walk the HGCalEE walker to find the first wafer on each layer and process them
  edm::LogVerbatim(logcat) << "Calling DDFindWafers(eeWalker);";
  DDFindWafers(eeWalker);

  // Walk the HGCalHEsilwalker to find the first wafer on each layer and process them
  edm::LogVerbatim(logcat) << "Calling DDFindWafers(hesilWalker);";
  DDFindWafers(hesilWalker);

  // Walk the HGCalHEmix walker to find the first wafer on each layer and process them
  edm::LogVerbatim(logcat) << "Calling DDFindWafers(hemixWalker);";
  DDFindWafers(hemixWalker);

  // Confirm all the DD wafers have been read
  edm::LogVerbatim(logcat) << "Number of wafers read from DD: " << waferData.size();

  // Now open the geometry text file
  std::string fileName = geometryFileName_.fullPath();
  edm::LogVerbatim(logcat) << "Opening geometry text file: " << fileName;
  std::ifstream geoTxtFile(fileName);

  if (!geoTxtFile) {
    edm::LogPrint(logcat) << "Cannot open geometry text file.";
    return;
  }

  // total processed counter
  int nTotalProcessed = 0;

  // geometry error counters
  int nMissing = 0;
  int nThicknessError = 0;
  int nPosXError = 0;
  int nPosYError = 0;
  int nShapeError = 0;
  int nRotError = 0;
  int nUnaccounted = 0;

  std::string buf;

  // process each line on the text file
  while (std::getline(geoTxtFile, buf)) {
    std::stringstream ss(buf);
    std::vector<std::string> tokens;
    while (ss >> buf)
      if (!buf.empty())
        tokens.push_back(buf);
    if (tokens.size() != 8)
      continue;

    nTotalProcessed++;

    // extract wafer info from a textfile line
    const int waferLayer(std::stoi(tokens[0]));
    const std::string waferShapeCode(tokens[1]);
    const int waferThickness(std::stoi(tokens[2]));
    const double waferX(std::stod(tokens[3]));
    const double waferY(std::stod(tokens[4]));
    const int waferRotCode(std::stoi(tokens[5]));
    const int waferU(std::stoi(tokens[6]));
    const int waferV(std::stoi(tokens[7]));

    // map index for crosschecking with DD
    const WaferCoord waferCoord(waferLayer, waferU, waferV);

    // now check for (and report) wafer data disagreements

    if (waferData.find(waferCoord) == waferData.end()) {
      nMissing++;
      edm::LogVerbatim(logcat) << "MISSING: " << strWaferCoord(waferCoord);
      continue;
    }

    const struct WaferInfo waferInfo = waferData[waferCoord];
    waferValidated[waferCoord] = true;

    if ((waferInfo.thickClass == 0 && waferThickness != 120) || (waferInfo.thickClass == 1 && waferThickness != 200) ||
        (waferInfo.thickClass == 2 && waferThickness != 300)) {
      nThicknessError++;
      edm::LogVerbatim(logcat) << "THICKNESS ERROR: " << strWaferCoord(waferCoord);
    }

    // it seems that wafer x-coords relative to their layer plane are mirrored...
    if (fabs(-waferInfo.x - waferX) > 0.015) {  // assuming this much tolerance
      nPosXError++;
      edm::LogVerbatim(logcat) << "POSITION x ERROR: " << strWaferCoord(waferCoord);
    }

    if (fabs(waferInfo.y - waferY) > 0.015) {  // assuming this much tolerance
      nPosYError++;
      edm::LogVerbatim(logcat) << "POSITION y ERROR: " << strWaferCoord(waferCoord);
    }

    if (waferInfo.shapeCode != waferShapeCode) {
      nShapeError++;
      edm::LogVerbatim(logcat) << "SHAPE ERROR: " << strWaferCoord(waferCoord);
    }

    if ((waferShapeCode != 'F' && waferInfo.rotCode != waferRotCode) ||
        (waferShapeCode == 'F' && (waferInfo.rotCode % 2 != waferRotCode % 2))) {
      nRotError++;
      edm::LogVerbatim(logcat) << "ROTATION ERROR: " << strWaferCoord(waferCoord) << "  ( " << waferInfo.rotCode
                               << " != " << waferRotCode << " )";
    }
  }

  geoTxtFile.close();

  // Find unaccounted DD wafers
  for (auto const& accounted : waferValidated) {
    if (!accounted.second) {
      nUnaccounted++;
      edm::LogVerbatim(logcat) << "UNACCOUNTED: " << strWaferCoord(accounted.first);
    }
  }

  // Print out error counts
  edm::LogVerbatim(logcat) << std::endl;
  edm::LogVerbatim(logcat) << "*** ERROR COUNTS ***";
  edm::LogVerbatim(logcat) << "Missing         :  " << nMissing;
  edm::LogVerbatim(logcat) << "Thickness error :  " << nThicknessError;
  edm::LogVerbatim(logcat) << "Pos-x error     :  " << nPosXError;
  edm::LogVerbatim(logcat) << "Pos-y error     :  " << nPosYError;
  edm::LogVerbatim(logcat) << "Shape error     :  " << nShapeError;
  edm::LogVerbatim(logcat) << "Rotation error  :  " << nRotError;
  edm::LogVerbatim(logcat) << "Unaccounted     :  " << nUnaccounted;
  edm::LogVerbatim(logcat) << std::endl;
  edm::LogVerbatim(logcat) << "Total wafers processed from geotxtfile = " << nTotalProcessed;
  edm::LogVerbatim(logcat) << std::endl;

  // Issue a LogPrint (warning) if there is at least one wafer errors
  if (nMissing > 0 || nThicknessError > 0 || nPosXError > 0 || nPosYError > 0 || nShapeError > 0 || nRotError > 0 ||
      nUnaccounted > 0) {
    edm::LogPrint(logcat) << "There are at least one wafer error.";
  }
}

// ------------ method called once each job just before starting event loop  ------------
void HGCalWaferValidation::beginJob() {
  // please remove this method if not needed
}

// ------------ method called once each job just after ending the event loop  ------------
void HGCalWaferValidation::endJob() {
  // please remove this method if not needed
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HGCalWaferValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.add<edm::FileInPath>("GeometryFileName",
                            edm::FileInPath("Validation/HGCalValidation/data/geomnew_corrected_360.txt"));
  descriptions.add("hgcalWaferValidation", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalWaferValidation);
