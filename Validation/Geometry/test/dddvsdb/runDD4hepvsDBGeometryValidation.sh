#! /bin/tcsh

cmsenv

echo " START Geometry Validation"

# $1 is the Global Tag
# $2 is the scenario, like "ExtendedGeometry2021". Omit "DD4hep".
# $3 is "round" to round values in comparisons  to 0 if < |1.e7|.
# Omit this option to show differences down to |1.e-23|.

# Note this script only currently supports Run 3.
# In future, it should be enhanced to support Runs 1-2 and Phase 2.

set roundFlag = ''
if ($#argv == 0) then
    set gtag="auto:upgrade2021"
    set geometry="ExtendedGeometry2021"
else if($#argv == 1) then
    set gtag=`echo ${1}`
    set geometry="ExtendedGeometry2021"
else if ($#argv == 2) then
    set gtag=`echo ${1}`
    set geometry=`echo ${2}`
else if ($#argv == 3) then 
    set gtag=`echo ${1}`
    set geometry=`echo ${2}`
    set roundFlag = `echo ${3}`
endif
echo GlobalTag = ${gtag}
echo geometry = ${geometry}
echo roundFlag = ${roundFlag}

set tolerance = '1.0e-7'
# If rounding enabled, tolerance for numerical comparisons. Absolute values less than this are set to 0.

#global tag gtag is assumed to be of the form GeometryWORD such as GeometryExtended or GeometryIdeal
#as of 3.4.X loaded objects in the DB, these correspond to condlabels Extended, Ideal, etc...
# Run 2 Extended condlabel corresponds to GeometryExtended2015 scenario. 
set condlabel = `(echo $geometry | sed -e '{s/Geometry//g}' -e '{s/Plan//g}' -e '{s/[0-9]*//g}')`
echo ${condlabel} " geometry label from db"

set workArea = `(echo $geometry)`
mkdir ${workArea}
cd ${workArea}
set myDir=`pwd`
echo $myDir

cp $CMSSW_RELEASE_BASE/src/CondTools/Geometry/test/writehelpers/geometryExtended2021DD4hep_xmlwriter.py geometryExtendedDD4hep_xmlwriter.py
echo $geometry
sed -i "{s/ExtendedGeometry2021/${geometry}/}" geometryExtendedDD4hep_xmlwriter.py >  GeometryValidation.log
cmsRun geometryExtendedDD4hep_xmlwriter.py >>  GeometryValidation.log

cp $CMSSW_RELEASE_BASE/src/CondTools/Geometry/test/writehelpers/geometryExtended2021DD4hep_writer.py .
# cp $CMSSW_BASE/src/CondTools/Geometry/test/writehelpers/geometryExtended2021DD4hep_writer.py .
# When more Reco writer configs are available, there should be a way to choose the correct version.
# sed -i "{s/GeometryExtended/${geometry}/}" geometrywriter.py >>  GeometryValidation.log
cmsRun geometryExtended2021DD4hep_writer.py >>  GeometryValidation.log
if ( -e myfile.db ) then
    echo "The local DB file is present" | tee -a GeometryValidation.log
else
    echo "ERROR the local DB file is not present" | tee -a GeometryValidation.log
    exit
endif

echo "Start compare the content of GT and the local DB" | tee -a GeometryValidation.log

# (MEC:1) The following two tests with the diff below them actually make
# sure that the Global Tag (GT) and Local DB XML file blobs are fine... 
# meaning that the full simulation geometry source is fine (XML blob)
# as well as the reco geometries.
# cp $CMSSW_BASE/src/CondTools/Geometry/test/geometrytest_local.py .
cp $CMSSW_RELEASE_BASE/src/CondTools/Geometry/test/geometrytest_local.py .
sed -i "{/process.GlobalTag.globaltag/d}" geometrytest_local.py >> GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\from Configuration.AlCa.GlobalTag import GlobalTag\nprocess.GlobalTag = GlobalTag(process.GlobalTag, '${gtag}', '')" geometrytest_local.py >> GeometryValidation.log
set geomabbrev = `(echo $geometry | sed -e '{s/Geometry//g}')`
sed -i "{s/Extended_TagXX/TagXX_${geomabbrev}_mc/}" geometrytest_local.py >>  GeometryValidation.log
if ( "${roundFlag}" == round ) then                                                               
  sed -i "/roundValues/s/False/True/" geometrytest_local.py >> GeometryValidation.log
endif

cmsRun geometrytest_local.py > outLocalDB.log
if ( -s outLocalDB.log ) then
    echo "Local DB access run ok" | tee -a GeometryValidation.log
else
    echo "ERROR the output of Local DB access test is empty" | tee -a GeometryValidation.log
    exit
endif

cp $CMSSW_RELEASE_BASE/src/CondTools/Geometry/test/geometrytestDD4hep_db.py .
# cp $CMSSW_BASE/src/CondTools/Geometry/test/geometrytestDD4hep_db.py .
sed -i "{/process.GlobalTag.globaltag/d}" geometrytestDD4hep_db.py >> GeometryValidation.log 
sed -i "/FrontierConditions_GlobalTag_cff/ a\from Configuration.AlCa.GlobalTag import GlobalTag\nprocess.GlobalTag = GlobalTag(process.GlobalTag, '${gtag}', '')" geometrytestDD4hep_db.py >> GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\process.DDDetectorESProducerFromDB.label = cms.string('${condlabel}')" geometrytestDD4hep_db.py >> GeometryValidation.log 
if ( "${roundFlag}" == round ) then                                                               
  sed -i "/roundValues/s/False/True/" geometrytestDD4hep_db.py >> GeometryValidation.log
endif
cmsRun geometrytestDD4hep_db.py > outGTDB.log
if ( -s outGTDB.log ) then
    echo "GT DB access run ok" | tee -a GeometryValidation.log
else
    echo "ERROR the output of GT DB access test is empty" | tee -a GeometryValidation.log
    exit
endif

diff outLocalDB.log outGTDB.log > logDiffLocalvsGT.log
if ( -s logDiffLocalvsGT.log ) then
    echo "WARNING THE CONTENT OF GLOBAL TAG MAY BE DIFFERENT WITH RESPECT TO THE LOCAL DB FILE" | tee -a GeometryValidation.log
    cp $CMSSW_BASE/src/Validation/Geometry/test/dddvsdb/sortXML.sh .
    cp $CMSSW_BASE/src/Validation/Geometry/test/dddvsdb/sortCompositeMaterials.py .
    ./sortXML.sh outLocalDB.log localdb.xml
    ./sortXML.sh outGTDB.log gtdb.xml
    diff localdb.xml gtdb.xml > logDiffLocXMLvsGTXML.log
    sort  localdb.xml > localdb.sort
    sort gtdb.xml > gtdb.sort
    diff localdb.sort gtdb.sort > logDiffLocvsGTSort.log
    echo Examine logDiffLocXMLvsGTXML.log to see the differences in the local and GT XML files. | tee -a GeometryValidation.log
    echo Examine logDiffLocvsGTSort.log to see the differences in sorted content of the local and GT XML files. | tee -a GeometryValidation.log
    echo The two XML files may have real differences, or they may have identical content that is simply re-arranged. | tee -a GeometryValidation.log
    echo Examining these log files can help you determine whether the XML files have significant differences. | tee -a GeometryValidation.log
endif

echo "End compare the content of GT and the local DB" | tee -a GeometryValidation.log

echo "Start Tracker RECO geometry validation" | tee -a GeometryValidation.log

mkdir tkdb
mkdir tkdblocal
mkdir tkxml

cp myfile.db tkdblocal

cd tkdb
# cp $CMSSW_BASE/src/Geometry/TrackerGeometryBuilder/test/python/testTrackerModuleInfoDBDD4hep_cfg.py .
cp $CMSSW_RELEASE_BASE/src/Geometry/TrackerGeometryBuilder/test/python/testTrackerModuleInfoDBDD4hep_cfg.py .
sed -i "{/process.GlobalTag.globaltag/d}" testTrackerModuleInfoDBDD4hep_cfg.py >> ../GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\from Configuration.AlCa.GlobalTag import GlobalTag\nprocess.GlobalTag = GlobalTag(process.GlobalTag, '${gtag}', '')" testTrackerModuleInfoDBDD4hep_cfg.py >> ../GeometryValidation.log 
if ( "${roundFlag}" == round ) then                                                               
  sed -i "/tolerance/s/1.0e-23/${tolerance}/" testTrackerModuleInfoDBDD4hep_cfg.py >> GeometryValidation.log
endif
cmsRun testTrackerModuleInfoDBDD4hep_cfg.py >> ../GeometryValidation.log
mv testTrackerModuleInfoDBDD4hep_cfg.py ../
if ( -s ModuleInfo.log ) then
    echo "TK test from DB run ok" | tee -a ../GeometryValidation.log
else
    echo "ERROR the output of TK test from DB is empty" | tee -a ../GeometryValidation.log
    exit
endif

cd ../tkdblocal
cp $CMSSW_RELEASE_BASE/src/Geometry/TrackerGeometryBuilder/test/python/trackerModuleInfoLocalDBDD4hep_cfg.py .
# cp $CMSSW_BASE/src/Geometry/TrackerGeometryBuilder/test/python/trackerModuleInfoLocalDBDD4hep_cfg.py .
sed -i "{/process.GlobalTag.globaltag/d}" trackerModuleInfoLocalDBDD4hep_cfg.py >> ../GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\from Configuration.AlCa.GlobalTag import GlobalTag\nprocess.GlobalTag = GlobalTag(process.GlobalTag, '${gtag}', '')" trackerModuleInfoLocalDBDD4hep_cfg.py >> ../GeometryValidation.log 
if ( "${roundFlag}" == round ) then                                                               
  sed -i "/tolerance/s/1.0e-23/${tolerance}/" trackerModuleInfoLocalDBDD4hep_cfg.py >> GeometryValidation.log
endif
sed -i "{s/Extended2021/${geomabbrev}/}" trackerModuleInfoLocalDBDD4hep_cfg.py >>  GeometryValidation.log
cmsRun trackerModuleInfoLocalDBDD4hep_cfg.py >> ../GeometryValidation.log
mv trackerModuleInfoLocalDBDD4hep_cfg.py ../
if ( -s ModuleInfo.log ) then
    echo "TK test from Local DB run ok" | tee -a ../GeometryValidation.log
else
    echo "ERROR the output of TK test from Local DB is empty" | tee -a ../GeometryValidation.log
    exit
endif

cd ../tkxml
# cp $CMSSW_BASE/src/Geometry/TrackerGeometryBuilder/test/python/testTrackerModuleInfoDD4hep_cfg.py .
cp $CMSSW_RELEASE_BASE/src/Geometry/TrackerGeometryBuilder/test/python/testTrackerModuleInfoDD4hep_cfg.py .
sed -i "{s/Extended2021/${geomabbrev}/}" testTrackerModuleInfoDD4hep_cfg.py >>  GeometryValidation.log
sed -i "{/process.GlobalTag.globaltag/d}" testTrackerModuleInfoDD4hep_cfg.py >> ../GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\from Configuration.AlCa.GlobalTag import GlobalTag\nprocess.GlobalTag = GlobalTag(process.GlobalTag, '${gtag}', '')" testTrackerModuleInfoDD4hep_cfg.py >> ../GeometryValidation.log 
if ( "${roundFlag}" == round ) then                                                               
  sed -i "/tolerance/s/1.0e-23/${tolerance}/" testTrackerModuleInfoDD4hep_cfg.py >> GeometryValidation.log
endif
cmsRun testTrackerModuleInfoDD4hep_cfg.py >> ../GeometryValidation.log
mv testTrackerModuleInfoDD4hep_cfg.py ../
if ( -s ModuleInfo.log ) then
    echo "TK test from DD4hep XML run ok" | tee -a ../GeometryValidation.log
else
    echo "ERROR the output of TK test from DD4hep XML is empty" | tee -a ../GeometryValidation.log
    exit
endif

cd ../
rm -f tkdblocal/myfile.db
diff -r tkdb/ tkxml/ > logTkDiffGTvsXML.log
if ( -s logTkDiffGTvsXML.log ) then
    echo "WARNING THE TRACKER RECO GEOMETRY IS DIFFERENT BETWEEN XML AND GT DB" | tee -a GeometryValidation.log
endif

diff -r tkdblocal/ tkxml/ > logTkDiffLocalvsXML.log
if ( -s logTkDiffLocalvsXML.log ) then
    echo "WARNING THE TRACKER RECO GEOMETRY IS DIFFERENT BETWEEN XML AND LOCAL DB" | tee -a GeometryValidation.log
endif

diff -r tkdb/ tkdblocal/ > logTkDiffGTvsLocal.log
if ( -s logTkDiffGTvsLocal.log ) then
    echo "WARNING THE TRACKER RECO GEOMETRY IS DIFFERENT BETWEEN GT DB AND LOCAL DB" | tee -a GeometryValidation.log
endif

echo "End Tracker RECO geometry validation" | tee -a GeometryValidation.log

echo "Start DT RECO geometry validation" | tee -a GeometryValidation.log

# cp $CMSSW_BASE/src/Geometry/DTGeometry/test/testDTGeometryFromDBDD4hep_cfg.py .
cp $CMSSW_RELEASE_BASE/src/Geometry/DTGeometry/test/testDTGeometryFromDBDD4hep_cfg.py .
sed -i "{/process.GlobalTag.globaltag/d}" testDTGeometryFromDBDD4hep_cfg.py >> GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\from Configuration.AlCa.GlobalTag import GlobalTag\nprocess.GlobalTag = GlobalTag(process.GlobalTag, '${gtag}', '')" testDTGeometryFromDBDD4hep_cfg.py >> GeometryValidation.log 
if ( "${roundFlag}" == round ) then                                                               
  sed -i "/tolerance/s/1.0e-23/${tolerance}/" testDTGeometryFromDBDD4hep_cfg.py >> GeometryValidation.log
endif
cmsRun testDTGeometryFromDBDD4hep_cfg.py > outDB_DT.log
if ( -s outDB_DT.log ) then
    echo "DT test from DB run ok" | tee -a GeometryValidation.log
else
    echo "ERROR the output of DT test from DB is empty" | tee -a GeometryValidation.log
    exit
endif

# cp $CMSSW_BASE/src/Geometry/DTGeometry/test/testDTGeometryFromLocalDBDD4hep_cfg.py .
cp $CMSSW_RELEASE_BASE/src/Geometry/DTGeometry/test/testDTGeometryFromLocalDBDD4hep_cfg.py .
sed -i "{s/Extended2021/${geomabbrev}/}" testDTGeometryFromLocalDBDD4hep_cfg.py >>  GeometryValidation.log
sed -i "{/process.GlobalTag.globaltag/d}" testDTGeometryFromLocalDBDD4hep_cfg.py >> GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\from Configuration.AlCa.GlobalTag import GlobalTag\nprocess.GlobalTag = GlobalTag(process.GlobalTag, '${gtag}', '')" testDTGeometryFromLocalDBDD4hep_cfg.py >> GeometryValidation.log 
if ( "${roundFlag}" == round ) then                                                               
  sed -i "/tolerance/s/1.0e-23/${tolerance}/" testDTGeometryFromLocalDBDD4hep_cfg.py >> GeometryValidation.log
endif
cmsRun testDTGeometryFromLocalDBDD4hep_cfg.py > outLocalDB_DT.log
if ( -s outDB_DT.log ) then
    echo "DT test from Local DB run ok" | tee -a GeometryValidation.log
else
    echo "ERROR the output of DT test from Local DB is empty" | tee -a GeometryValidation.log
    exit
endif

# cp $CMSSW_BASE/src/Geometry/DTGeometry/test/testDTGeometryDD4hep_cfg.py .
cp $CMSSW_RELEASE_BASE/src/Geometry/DTGeometry/test/testDTGeometryDD4hep_cfg.py .
sed -i "{s/Extended2021/${geomabbrev}/}" testDTGeometryDD4hep_cfg.py >>  GeometryValidation.log
sed -i "{/process.GlobalTag.globaltag/d}" testDTGeometryDD4hep_cfg.py >> GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\from Configuration.AlCa.GlobalTag import GlobalTag\nprocess.GlobalTag = GlobalTag(process.GlobalTag, '${gtag}', '')" testDTGeometryDD4hep_cfg.py >> GeometryValidation.log 
if ( "${roundFlag}" == round ) then                                                               
  sed -i "/tolerance/s/1.0e-23/${tolerance}/" testDTGeometryDD4hep_cfg.py >> GeometryValidation.log
endif
cmsRun testDTGeometryDD4hep_cfg.py > outXML_DT.log
if ( -s outXML_DT.log ) then
    echo "DT test from XML run ok" | tee -a GeometryValidation.log
else
    echo "ERROR the output of DT test from XML is empty" | tee -a GeometryValidation.log
    exit
endif

diff --ignore-matching-lines='Geometry node for DTGeom' outDB_DT.log outXML_DT.log > logDTDiffGTvsXML.log
if ( -s logDTDiffGTvsXML.log ) then
    echo "WARNING THE DT RECO GEOMETRY IS DIFFERENT BETWEEN XML AND GT DB" | tee -a GeometryValidation.log
endif

diff --ignore-matching-lines='Geometry node for DTGeom' outLocalDB_DT.log outXML_DT.log > logDTDiffLocalvsXML.log
if ( -s logDTDiffLocalvsXML.log ) then
    echo "WARNING THE DT RECO GEOMETRY IS DIFFERENT BETWEEN XML AND LOCAL DB" | tee -a GeometryValidation.log
endif

diff --ignore-matching-lines='Geometry node for DTGeom' outDB_DT.log outLocalDB_DT.log > logDTDiffGTvsLocal.log
if ( -s logDTDiffGTvsLocal.log ) then
    echo "WARNING THE DT RECO GEOMETRY IS DIFFERENT BETWEEN GT DB AND  LOCAL DB" | tee -a GeometryValidation.log
endif

echo "End DT RECO geometry validation" | tee -a GeometryValidation.log

echo "Start CSC RECO geometry validation" | tee -a GeometryValidation.log

cp $CMSSW_RELEASE_BASE/src/Geometry/CSCGeometry/test/testCSCGeometryFromDBDD4hep_cfg.py .
# cp $CMSSW_BASE/src/Geometry/CSCGeometry/test/testCSCGeometryFromDBDD4hep_cfg.py .
sed -i "{/process.GlobalTag.globaltag/d}" testCSCGeometryFromDBDD4hep_cfg.py >> GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\from Configuration.AlCa.GlobalTag import GlobalTag\nprocess.GlobalTag = GlobalTag(process.GlobalTag, '${gtag}', '')" testCSCGeometryFromDBDD4hep_cfg.py >> GeometryValidation.log 
cmsRun testCSCGeometryFromDBDD4hep_cfg.py > outDB_CSC.log
if ( -s outDB_CSC.log ) then
    echo "CSC test from GT DB run ok" | tee -a GeometryValidation.log
else
    echo "ERROR the output of CSC test from GT DB is empty" | tee -a GeometryValidation.log
    exit
endif

cp $CMSSW_RELEASE_BASE/src/Geometry/CSCGeometry/test/testCSCGeometryFromLocalDBDD4hep_cfg.py .
# cp $CMSSW_BASE/src/Geometry/CSCGeometry/test/testCSCGeometryFromLocalDBDD4hep_cfg.py .
sed -i "{/process.GlobalTag.globaltag/d}" testCSCGeometryFromLocalDBDD4hep_cfg.py >> GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\from Configuration.AlCa.GlobalTag import GlobalTag\nprocess.GlobalTag = GlobalTag(process.GlobalTag, '${gtag}', '')" testCSCGeometryFromLocalDBDD4hep_cfg.py >> GeometryValidation.log 
cmsRun testCSCGeometryFromLocalDBDD4hep_cfg.py > outLocalDB_CSC.log
if ( -s outLocalDB_CSC.log ) then
    echo "CSC test from Local DB run ok" | tee -a GeometryValidation.log
else
    echo "ERROR the output of CSC test from Local DB is empty" | tee -a GeometryValidation.log
    exit
endif

cp $CMSSW_RELEASE_BASE/src/Geometry/CSCGeometry/test/testCSCGeometryDD4hep_cfg.py .
# cp $CMSSW_BASE/src/Geometry/CSCGeometry/test/testCSCGeometryDD4hep_cfg.py .
sed -i "{s/GeometryExtended/${geometry}/}" testCSCGeometryDD4hep_cfg.py >>  GeometryValidation.log
sed -i "{/process.GlobalTag.globaltag/d}" testCSCGeometryDD4hep_cfg.py >> GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\from Configuration.AlCa.GlobalTag import GlobalTag\nprocess.GlobalTag = GlobalTag(process.GlobalTag, '${gtag}', '')" testCSCGeometryDD4hep_cfg.py >> GeometryValidation.log 
cmsRun testCSCGeometryDD4hep_cfg.py > outXML_CSC.log
if ( -s outXML_CSC.log ) then
    echo "CSC test from XML run ok" | tee -a GeometryValidation.log
else
    echo "ERROR the output of CSC test from XML is empty" | tee -a GeometryValidation.log
    exit
endif

diff --ignore-matching-lines='Geometry node for CSCGeom' outDB_CSC.log outXML_CSC.log > logCSCDiffGTvsXML.log
if ( -s logCSCDiffGTvsXML.log ) then
    echo "WARNING THE CSC RECO GEOMETRY IS DIFFERENT BETWEEN XML AND GT DB" | tee -a GeometryValidation.log
endif

diff --ignore-matching-lines='Geometry node for CSCGeom' outLocalDB_CSC.log outXML_CSC.log > logCSCDiffLocalvsXML.log
if ( -s logCSCDiffLocalvsXML.log ) then
    echo "WARNING THE CSC RECO GEOMETRY IS DIFFERENT BETWEEN XML AND LOCAL DB" | tee -a GeometryValidation.log
endif

diff --ignore-matching-lines='Geometry node for CSCGeom' outLocalDB_CSC.log outDB_CSC.log > logCSCDiffLocalvsGT.log
if ( -s logCSCDiffLocalvsGT.log ) then
    echo "WARNING THE CSC RECO GEOMETRY IS DIFFERENT BETWEEN GT DB AND LOCAL DB" | tee -a GeometryValidation.log
endif

echo "End CSC RECO geometry validation" | tee -a GeometryValidation.log

echo "Start RPC RECO geometry validation" | tee -a GeometryValidation.log

# cp $CMSSW_BASE/src/Geometry/RPCGeometry/test/testRPCGeometryFromDBDD4hep_cfg.py .
cp $CMSSW_RELEASE_BASE/src/Geometry/RPCGeometry/test/testRPCGeometryFromDBDD4hep_cfg.py .
sed -i "{/process.GlobalTag.globaltag/d}" testRPCGeometryFromDBDD4hep_cfg.py >> GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\from Configuration.AlCa.GlobalTag import GlobalTag\nprocess.GlobalTag = GlobalTag(process.GlobalTag, '${gtag}', '')" testRPCGeometryFromDBDD4hep_cfg.py >> GeometryValidation.log 
cmsRun testRPCGeometryFromDBDD4hep_cfg.py > outDB_RPC.log
if ( -s outDB_RPC.log ) then
    echo "RPC test from GT DB run ok" | tee -a GeometryValidation.log
else
    echo "ERROR the output of RPC test from GT DB is empty" | tee -a GeometryValidation.log
    exit
endif

# cp $CMSSW_BASE/src/Geometry/RPCGeometry/test/testRPCGeometryFromLocalDBDD4hep_cfg.py .
cp $CMSSW_RELEASE_BASE/src/Geometry/RPCGeometry/test/testRPCGeometryFromLocalDBDD4hep_cfg.py .
sed -i "{/process.GlobalTag.globaltag/d}" testRPCGeometryFromLocalDBDD4hep_cfg.py >> GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\from Configuration.AlCa.GlobalTag import GlobalTag\nprocess.GlobalTag = GlobalTag(process.GlobalTag, '${gtag}', '')" testRPCGeometryFromLocalDBDD4hep_cfg.py >> GeometryValidation.log 
cmsRun testRPCGeometryFromLocalDBDD4hep_cfg.py > outLocalDB_RPC.log
if ( -s outLocalDB_RPC.log ) then
    echo "RPC test from Local DB run ok" | tee -a GeometryValidation.log
else
    echo "ERROR the output of RPC test from Local DB is empty" | tee -a GeometryValidation.log
    exit
endif

# cp $CMSSW_BASE/src/Geometry/RPCGeometry/test/testRPCGeometryDD4hep_cfg.py .
cp $CMSSW_RELEASE_BASE/src/Geometry/RPCGeometry/test/testRPCGeometryDD4hep_cfg.py .
sed -i "{s/GeometryExtended/${geometry}/}" testRPCGeometryDD4hep_cfg.py >>  GeometryValidation.log
sed -i "{/process.GlobalTag.globaltag/d}" testRPCGeometryDD4hep_cfg.py >> GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\from Configuration.AlCa.GlobalTag import GlobalTag\nprocess.GlobalTag = GlobalTag(process.GlobalTag, '${gtag}', '')" testRPCGeometryDD4hep_cfg.py >> GeometryValidation.log 
cmsRun testRPCGeometryDD4hep_cfg.py > outXML_RPC.log
if ( -s outXML_RPC.log ) then
    echo "RPC test from XML run ok" | tee -a GeometryValidation.log
else
    echo "ERROR the output of RPC test from XML is empty" | tee -a GeometryValidation.log
    exit
endif

diff --ignore-matching-lines='Geometry node for RPCGeom' outDB_RPC.log outXML_RPC.log > logRPCDiffGTvsXML.log
if ( -s logRPCDiffGTvsXML.log ) then
    echo "WARNING THE RPC RECO GEOMETRY IS DIFFERENT BETWEEN XML AND GT DB" | tee -a GeometryValidation.log
endif

diff --ignore-matching-lines='Geometry node for RPCGeom' outLocalDB_RPC.log outXML_RPC.log > logRPCDiffLocalvsXML.log
if ( -s logRPCDiffLocalvsXML.log ) then
    echo "WARNING THE RPC RECO GEOMETRY IS DIFFERENT BETWEEN XML AND LOCAL DB" | tee -a GeometryValidation.log
endif

diff --ignore-matching-lines='Geometry node for RPCGeom' outLocalDB_RPC.log outDB_RPC.log > logRPCDiffLocalvsDB.log
if ( -s logRPCDiffLocalvsDB.log ) then
    echo "WARNING THE RPC RECO GEOMETRY IS DIFFERENT BETWEEN GT DB AND LOCAL DB" | tee -a GeometryValidation.log
endif

echo "End RPC RECO geometry validation" | tee -a GeometryValidation.log

echo "Start GEM RECO geometry validation" | tee -a GeometryValidation.log

# cp $CMSSW_BASE/src/Geometry/GEMGeometry/test/testGEMGeometryFromDBDD4hep_cfg.py .
cp $CMSSW_RELEASE_BASE/src/Geometry/GEMGeometry/test/testGEMGeometryFromDBDD4hep_cfg.py .
sed -i "{/process.GlobalTag.globaltag/d}" testGEMGeometryFromDBDD4hep_cfg.py >> GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\from Configuration.AlCa.GlobalTag import GlobalTag\nprocess.GlobalTag = GlobalTag(process.GlobalTag, '${gtag}', '')" testGEMGeometryFromDBDD4hep_cfg.py >> GeometryValidation.log 
cmsRun testGEMGeometryFromDBDD4hep_cfg.py
mv GEMtestOutput.out outDB_GEM.log
if ( -s outDB_GEM.log ) then
    echo "GEM test from GT DB run ok" | tee -a GeometryValidation.log
else
    echo "ERROR the output of GEM test from GT DB is empty" | tee -a GeometryValidation.log
    exit
endif

cp $CMSSW_RELEASE_BASE/src/Geometry/GEMGeometry/test/testGEMGeometryFromLocalDBDD4hep_cfg.py .
# cp $CMSSW_BASE/src/Geometry/GEMGeometry/test/testGEMGeometryFromLocalDBDD4hep_cfg.py .
sed -i "{/process.GlobalTag.globaltag/d}" testGEMGeometryFromLocalDBDD4hep_cfg.py >> GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\from Configuration.AlCa.GlobalTag import GlobalTag\nprocess.GlobalTag = GlobalTag(process.GlobalTag, '${gtag}', '')" testGEMGeometryFromLocalDBDD4hep_cfg.py >> GeometryValidation.log 
cmsRun testGEMGeometryFromLocalDBDD4hep_cfg.py
mv GEMtestOutput.out outLocalDB_GEM.log
if ( -s outLocalDB_GEM.log ) then
    echo "GEM test from Local DB run ok" | tee -a GeometryValidation.log
else
    echo "ERROR the output of GEM test from Local DB is empty" | tee -a GeometryValidation.log
    exit
endif

cp $CMSSW_RELEASE_BASE/src/Geometry/GEMGeometry/test/testGEMGeometryDD4hep_cfg.py .
# cp $CMSSW_BASE/src/Geometry/GEMGeometry/test/testGEMGeometryDD4hep_cfg.py .
sed -i "{s/GeometryExtended/${geometry}/}" testGEMGeometryDD4hep_cfg.py >>  GeometryValidation.log
sed -i "{/process.GlobalTag/d}" testGEMGeometryDD4hep_cfg.py >> GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\from Configuration.AlCa.GlobalTag import GlobalTag\nprocess.GlobalTag = GlobalTag(process.GlobalTag, '${gtag}', '')" testGEMGeometryDD4hep_cfg.py >> GeometryValidation.log 
cmsRun testGEMGeometryDD4hep_cfg.py
mv GEMtestOutput.out outXML_GEM.log
if ( -s outXML_GEM.log ) then
    echo "GEM test from XML run ok" | tee -a GeometryValidation.log
else
    echo "ERROR the output of GEM test from XML is empty" | tee -a GeometryValidation.log
    exit
endif

diff --ignore-matching-lines='Geometry node for GEMGeom' outDB_GEM.log outXML_GEM.log > logGEMDiffGTvsXML.log
if ( -s logGEMDiffGTvsXML.log ) then
    echo "WARNING THE GEM RECO GEOMETRY IS DIFFERENT BETWEEN XML AND GT DB" | tee -a GeometryValidation.log
endif

diff --ignore-matching-lines='Geometry node for GEMGeom' outLocalDB_GEM.log outXML_GEM.log > logGEMDiffLocalvsXML.log
if ( -s logGEMDiffLocalvsXML.log ) then
    echo "WARNING THE GEM RECO GEOMETRY IS DIFFERENT BETWEEN XML AND LOCAL DB" | tee -a GeometryValidation.log
endif

diff --ignore-matching-lines='Geometry node for GEMGeom' outLocalDB_GEM.log outDB_GEM.log > logGEMDiffLocalvsDB.log
if ( -s logGEMDiffLocalvsDB.log ) then
    echo "WARNING THE GEM RECO GEOMETRY IS DIFFERENT BETWEEN GT DB AND LOCAL DB" | tee -a GeometryValidation.log
endif

echo "End GEM RECO geometry validation" | tee -a GeometryValidation.log

echo "Start CALO RECO geometry validation" | tee -a GeometryValidation.log

cp myfile.db $CMSSW_BASE/src/Geometry/CaloEventSetup/test/
cd $CMSSW_BASE/src/Geometry/CaloEventSetup/
cd data
# wget -i download.url
# wget commented out -- use files in "data" directory instead
cd ../test
source setup.scr >> ${myDir}/GeometryValidation.log
cp runTestCaloGeometryDD4hep_cfg.py ${myDir}/runTestCaloGeometryDD4hep_cfg.py
sed -i "{s/Extended2021/${geomabbrev}/}" ${myDir}/runTestCaloGeometryDD4hep_cfg.py >> ${myDir}/GeometryValidation.log
cmsRun ${myDir}/runTestCaloGeometryDD4hep_cfg.py > GeometryCaloValidationXML.log
if ( -s GeometryCaloValidationXML.log ) then
    echo "CALO test from XML run ok" | tee -a ${myDir}/GeometryValidation.log
else
    echo "ERROR the output of CALO test from XML is empty" | tee -a ${myDir}/GeometryValidation.log
    exit
endif

cp runTestCaloGeometryDBDD4hep_cfg.py ${myDir}/runTestCaloGeometryDBDD4hep_cfg.py
sed -i "s/auto:upgrade2021/${gtag}/" ${myDir}/runTestCaloGeometryDBDD4hep_cfg.py >> ${myDir}/GeometryValidation.log 
cmsRun ${myDir}/runTestCaloGeometryDBDD4hep_cfg.py > GeometryCaloValidationDB.log
if ( -s GeometryCaloValidationDB.log ) then
    echo "CALO test from GT DB run ok" | tee -a ${myDir}/GeometryValidation.log
else
    echo "ERROR the output of CALO test from GT DB is empty" | tee -a ${myDir}/GeometryValidation.log
    exit
endif

cp runTestCaloGeometryLocalDBDD4hep_cfg.py ${myDir}/runTestCaloGeometryLocalDBDD4hep_cfg.py
sed -i "s/auto:upgrade2021/${gtag}/" ${myDir}/runTestCaloGeometryLocalDBDD4hep_cfg.py >> ${myDir}/GeometryValidation.log 
cmsRun ${myDir}/runTestCaloGeometryLocalDBDD4hep_cfg.py > GeometryCaloValidationLocal.log
if ( -s GeometryCaloValidationLocal.log ) then
    echo "CALO Local test from Local DB run ok" | tee -a ${myDir}/GeometryValidation.log
else
    echo "ERROR the output of CALO test from Local DB is empty" | tee -a ${myDir}/GeometryValidation.log
    exit
endif
source clean.scr >> ${myDir}/GeometryValidation.log
rm myfile.db
cd ${myDir}

grep SUCCEED $CMSSW_BASE/src/Geometry/CaloEventSetup/test/GeometryCaloValidationXML.log >> GeometryValidation.log
grep SUCCEED $CMSSW_BASE/src/Geometry/CaloEventSetup/test/GeometryCaloValidationDB.log >> GeometryValidation.log
grep SUCCEED $CMSSW_BASE/src/Geometry/CaloEventSetup/test/GeometryCaloValidationLocal.log >> GeometryValidation.log
cp $CMSSW_BASE/src/Geometry/CaloEventSetup/test/GeometryCaloValidationXML.log .
cp $CMSSW_BASE/src/Geometry/CaloEventSetup/test/GeometryCaloValidationDB.log .
cp $CMSSW_BASE/src/Geometry/CaloEventSetup/test/GeometryCaloValidationLocal.log .

grep 'BIG DISAGREEMENT FOUND' $CMSSW_BASE/src/Geometry/CaloEventSetup/test/GeometryCaloValidationXML.log > CALOXMLError.log 
grep 'BIG DISAGREEMENT FOUND' $CMSSW_BASE/src/Geometry/CaloEventSetup/test/GeometryCaloValidationDB.log > CALODBError.log 
grep 'BIG DISAGREEMENT FOUND' $CMSSW_BASE/src/Geometry/CaloEventSetup/test/GeometryCaloValidationLocal.log > CALOLocalError.log 

rm -f $CMSSW_BASE/src/Geometry/CaloEventSetup/test/GeometryCaloValidationXML.log
rm -f $CMSSW_BASE/src/Geometry/CaloEventSetup/test/GeometryCaloValidationDB.log
rm -f $CMSSW_BASE/src/Geometry/CaloEventSetup/test/GeometryCaloValidationLocal.log

if ( -s CALOXMLError.log ) then                                                               
    echo "WARNING THE CALO GEOMETRY IS DIFFERENT BETWEEN XML AND REF" | tee -a GeometryValidation.log                                                                                  
endif                                                                                                      

if ( -s CALODBError.log ) then                                                               
    echo "WARNING THE CALO GEOMETRY IS DIFFERENT BETWEEN GT DB AND REF" | tee -a GeometryValidation.log                                                                                  
endif                                                                                                      

if ( -s CALOLocalError.log ) then                                                               
    echo "WARNING THE CALO GEOMETRY IS DIFFERENT BETWEEN LOCAL DB AND REF" | tee -a GeometryValidation.log                                                                                  
endif                                                                                                      
                                                                                              
echo "End CALO RECO geometry validation" | tee -a GeometryValidation.log

echo "Start Simulation geometry validation" | tee -a GeometryValidation.log

cp $CMSSW_RELEASE_BASE/src/SimG4Core/PrintGeomInfo/test/python/runDD4hepXML_cfg.py .
# cp $CMSSW_BASE/src/SimG4Core/PrintGeomInfo/test/python/runDD4hepXML_cfg.py .
sed -i "{s/Extended2021/${geomabbrev}/}" runDD4hepXML_cfg.py >>  GeometryValidation.log
sed -i "{s/DumpSummary      = cms.untracked.bool(True/DumpSummary      = cms.untracked.bool(False/}" runDD4hepXML_cfg.py >>  GeometryValidation.log
sed -i "{s/DumpSense      = cms.untracked.bool(False/DumpSense      = cms.untracked.bool(True/}" runDD4hepXML_cfg.py >>  GeometryValidation.log
sed -i "{s/DumpParams      = cms.untracked.bool(False/DumpParams      = cms.untracked.bool(True/}" runDD4hepXML_cfg.py >>  GeometryValidation.log
sed -i "{/MaterialFileName/d}" runDD4hepXML_cfg.py >> GeometryValidation.log
sed -i "{/SolidFileName/d}" runDD4hepXML_cfg.py >> GeometryValidation.log
sed -i "{/LVFileName/d}" runDD4hepXML_cfg.py >> GeometryValidation.log
sed -i "{/PVFileName/d}" runDD4hepXML_cfg.py >> GeometryValidation.log
sed -i "{/TouchFileName/d}" runDD4hepXML_cfg.py >> GeometryValidation.log
( cmsRun runDD4hepXML_cfg.py > readXMLAndDump.log ) >>& GeometryValidation.log

cp $CMSSW_RELEASE_BASE/src/SimG4Core/PrintGeomInfo/test/python/runDD4hepDB_cfg.py .
# cp $CMSSW_BASE/src/SimG4Core/PrintGeomInfo/test/python/runDD4hepDB_cfg.py .
sed -i "{s/Extended2021/${geomabbrev}/}" runDD4hepDB_cfg.py >>  GeometryValidation.log
sed -i "{s/DumpSummary      = cms.untracked.bool(True/DumpSummary      = cms.untracked.bool(False/}" runDD4hepDB_cfg.py >>  GeometryValidation.log
sed -i "{s/DumpSense      = cms.untracked.bool(False/DumpSense      = cms.untracked.bool(True/}" runDD4hepDB_cfg.py >>  GeometryValidation.log
sed -i "{s/DumpParams      = cms.untracked.bool(False/DumpParams      = cms.untracked.bool(True/}" runDD4hepDB_cfg.py >>  GeometryValidation.log
sed -i "{/MaterialFileName/d}" runDD4hepDB_cfg.py >> GeometryValidation.log
sed -i "{/SolidFileName/d}" runDD4hepDB_cfg.py >> GeometryValidation.log
sed -i "{/LVFileName/d}" runDD4hepDB_cfg.py >> GeometryValidation.log
sed -i "{/PVFileName/d}" runDD4hepDB_cfg.py >> GeometryValidation.log
sed -i "{/TouchFileName/d}" runDD4hepDB_cfg.py >> GeometryValidation.log
sed -i "{/process.GlobalTag.globaltag/d}" runDD4hepDB_cfg.py >> GeometryValidation.log
sed -i "/from Configuration.AlCa.GlobalTag/ a\process.GlobalTag = GlobalTag(process.GlobalTag, '${gtag}', '')" runDD4hepDB_cfg.py >> GeometryValidation.log
( cmsRun runDD4hepDB_cfg.py > readXMLfromGTDB.log) >& /dev/null

cp $CMSSW_RELEASE_BASE/src/SimG4Core/PrintGeomInfo/test/python/runDD4hepLocalDB_cfg.py .
# cp $CMSSW_BASE/src/SimG4Core/PrintGeomInfo/test/python/runDD4hepLocalDB_cfg.py .
( cmsRun runDD4hepLocalDB_cfg.py > readXMLfromLocDB.log ) >>& GeometryValidation.log

if ( ! -s readXMLAndDump.log ) then
    echo "ERROR THE MULTI-XML FILE GEOMETRY WAS NOT DUMPED PROPERLY." | tee -a GeometryValidation.log
    exit 1
else
    echo "Geometry dump from multiple XML files done."
endif

if ( ! -s readXMLfromGTDB.log ) then
    echo "ERROR THE GLOBAL TAG DATABASE GEOMETRY WAS NOT DUMPED PROPERLY." | tee -a GeometryValidation.log
    exit 1
else
    echo "Geometry dump from global tag database done."
endif

if ( ! -s readXMLfromLocDB.log ) then
    echo "ERROR THE LOCAL DATABASE GEOMETRY WAS NOT DUMPED PROPERLY." | tee -a GeometryValidation.log
    exit 1
else
    echo "Geometry dump from local database done."
endif

diff readXMLAndDump.log readXMLfromGTDB.log > tcdfXMLvsDB.log
diff readXMLAndDump.log readXMLfromLocDB.log > tcdfXMLvsLocDB.log
diff readXMLfromLocDB.log readXMLfromGTDB.log > tcdfLocDbvsDB.log


if ( -s tcdfXMLvsDB.log ) then
    echo "WARNING THE GEOMETRYFILE IS DIFFERENT BETWEEN XML FILES AND DB." | tee -a GeometryValidation.log
    echo See tcdfXMLvsDB.log  for differences | tee -a GeometryValidation.log
endif

if ( -s tcdfXMLvsLocDB.log ) then
    echo "WARNING THE GEOMETRYFILE IS DIFFERENT BETWEEN XML FILES AND LOCAL DATABASE BLOB." | tee -a GeometryValidation.log
    echo See tcdfXMLvsLocDB.log  for differences | tee -a GeometryValidation.log
endif

if ( -s tcdfLocDBvsDB.log ) then
    echo "WARNING THE GEOMETRYFILE IS DIFFERENT BETWEEN LOCAL AND GLOBALTAG DATABASE BLOBS."  | tee -a GeometryValidation.log
    echo See tcdfLocDBvsDB.log  for differences | tee -a GeometryValidation.log
endif

cp $CMSSW_RELEASE_BASE/src/DetectorDescription/DDCMS/test/python/testTGeoIterator.py .
# cp $CMSSW_BASE/src/DetectorDescription/DDCMS/test/python/testTGeoIterator.py .
sed -i "{s/ExtendedGeometry2021/${geometry}/}" testTGeoIterator.py >> GeometryValidation.log
cmsRun testTGeoIterator.py
if ( -s navGeometry.log ) then
  mv navGeometry.log navGeoXML.log
else
  echo Failed to dump paths from XML files | tee -a GeometryValidation.log
endif

cp $CMSSW_RELEASE_BASE/src/DetectorDescription/DDCMS/test/python/testTGeoIteratorDB.py .
# cp $CMSSW_BASE/src/DetectorDescription/DDCMS/test/python/testTGeoIteratorDB.py .
sed -i "{/process.GlobalTag.globaltag/d}" testTGeoIteratorDB.py >> GeometryValidation.log
sed -i "{/from Configuration.AlCa.autoCond/d}" testTGeoIteratorDB.py >> GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\from Configuration.AlCa.GlobalTag import GlobalTag\nprocess.GlobalTag = GlobalTag(process.GlobalTag, '${gtag}', '')" testTGeoIteratorDB.py >> GeometryValidation.log 
cmsRun testTGeoIteratorDB.py
if ( -s navGeometry.log ) then
  mv navGeometry.log navGeoDB.log
else
  echo Failed to dump paths from DB | tee -a GeometryValidation.log
endif

cp $CMSSW_RELEASE_BASE/src/DetectorDescription/DDCMS/test/python/testTGeoIteratorLocalDB.py .
# cp $CMSSW_BASE/src/DetectorDescription/DDCMS/test/python/testTGeoIteratorLocalDB.py .
sed -i "{s/Extended2021/${geomabbrev}/g}" testTGeoIteratorLocalDB.py >> GeometryValidation.log
cmsRun testTGeoIteratorLocalDB.py
if ( -s navGeometry.log ) then
  mv navGeometry.log navGeoLocDB.log
else
  echo Failed to dump paths from local DB | tee -a GeometryValidation.log
endif

diff --ignore-matching-lines='Begin processing' navGeoXML.log navGeoDB.log > pathsXMLvsDB.log
diff --ignore-matching-lines='Begin processing' navGeoXML.log navGeoLocDB.log > pathsXMLvsLocDB.log
diff --ignore-matching-lines='Begin processing' navGeoLocDB.log navGeoDB.log > pathsLocDBvsDB.log

if ( -s pathsXMLvsDB.log ) then
    echo "WARNING PATHS ARE DIFFERENT BETWEEN XML FILES AND DB." | tee -a GeometryValidation.log
    echo See pathsXMLvsDB.log for differences | tee -a GeometryValidation.log
endif
if ( -s pathsXMLvsLocDB.log ) then
    echo "WARNING PATHS ARE DIFFERENT BETWEEN XML FILES AND LOCAL DATABASE BLOB." | tee -a GeometryValidation.log
    echo See pathsXMLvsLocDB.log  for differences | tee -a GeometryValidation.log
endif
if ( -s pathsLocDBvsDB.log ) then
    echo "WARNING PATHS ARE DIFFERENT BETWEEN LOCAL AND GLOBALTAG DATABASE BLOBS."  | tee -a GeometryValidation.log
    echo See pathsLocDBvsDB.log for differences | tee -a GeometryValidation.log
endif

echo "End Simulation geometry validation" | tee -a GeometryValidation.log
