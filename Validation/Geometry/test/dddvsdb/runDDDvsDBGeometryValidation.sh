#! /bin/tcsh

echo " START Geometry Validation"

if ($#argv == 0) then
    set gtag="MC_31X_V8::All"
    set geometry="GeometryIdeal"
else if($#argv == 1) then
    set gtag=`echo ${1}`
    set geometry="GeometryIdeal"
else
    set gtag=`echo ${1}`
    set geometry=`echo ${2}`
endif

cmsenv
mkdir workArea
cd workArea
source $CMSSW_BASE/src/CondTools/Geometry/test/blob_preparation.txt > GeometryValidation.log
cp $CMSSW_BASE/src/CondTools/Geometry/test/geometryxmlwriter.py .
sed -i "{s/GeometryExtended/${geometry}/}" geometryxmlwriter.py >>  GeometryValidation.log
cmsRun geometryxmlwriter.py >>  GeometryValidation.log

cp $CMSSW_BASE/src/CondTools/Geometry/test/geometrywriter.py .
sed -i "{s/GeometryExtended/${geometry}/}" geometrywriter.py >>  GeometryValidation.log
cmsRun geometrywriter.py >>  GeometryValidation.log
if ( -e myfile.db ) then
    echo "The local DB file is present" | tee -a GeometryValidation.log
else
    echo "ERROR the local DB file is not present" | tee -a GeometryValidation.log
    exit
endif

echo "Start Tracker RECO geometry validation" | tee -a GeometryValidation.log
mkdir tkdb
mkdir tkddd
cp myfile.db tkdb
cd tkdb
cp $CMSSW_BASE/src/Geometry/TrackerGeometryBuilder/test/trackerModuleInfoDB_cfg.py .
sed -i "{s/MC_31X_V8::All/${gtag}/}" trackerModuleInfoDB_cfg.py >> ../GeometryValidation.log
cmsRun trackerModuleInfoDB_cfg.py >> ../GeometryValidation.log
cd ../tkddd
cp $CMSSW_BASE/src/Geometry/TrackerGeometryBuilder/test/trackerModuleInfoDDD_cfg.py .
sed -i "{s/MC_31X_V8::All/${gtag}/}" trackerModuleInfoDDD_cfg.py >> ../GeometryValidation.log
cmsRun trackerModuleInfoDDD_cfg.py >> ../GeometryValidation.log
cd ../
rm -f tkdb/myfile.db
diff -r tkdb/ tkddd/ > logTkDiff.log
if ( -s logTkDiff.log ) then
    echo "WARNING THE TRACKER RECO GEOMETRY IS DIFFERENT BETWEEN DDD AND DB" | tee -a GeometryValidation.log
endif
echo "End Tracker RECO geometry validation" | tee -a GeometryValidation.log

echo "Start DT RECO geometry validation" | tee -a GeometryValidation.log

cp $CMSSW_BASE/src/Geometry/DTGeometry/test/testDTGeometryFromDB_cfg.py .  
sed -i "{s/MC_31X_V8::All/${gtag}/}" testDTGeometryFromDB_cfg.py >> GeometryValidation.log
cmsRun testDTGeometryFromDB_cfg.py > outDB_DT.log
if ( -s outDB_DT.log ) then
    echo "DT test from DB run ok" | tee -a GeometryValidation.log
else
    echo "ERROR the output of DT test from DB is empty" | tee -a GeometryValidation.log
    exit
endif

cp $CMSSW_BASE/src/Geometry/DTGeometry/test/testDTGeometry_cfg.py .
sed -i "{s/GeometryExtended/${geometry}/}" testDTGeometry_cfg.py >>  GeometryValidation.log
sed -i "{s/MC_31X_V8::All/${gtag}/}" testDTGeometry_cfg.py >>  GeometryValidation.log
cmsRun testDTGeometry_cfg.py > outDDD_DT.log

if ( -s outDDD_DT.log ) then
    echo "DT test from DDD run ok" | tee -a GeometryValidation.log
else
    echo "ERROR the output of DT test from DDD is empty" | tee -a GeometryValidation.log
    exit
endif

diff --ignore-matching-lines='Geometry node for DTGeom' outDDD_DT.log outDB_DT.log > logDTDiff.log
if ( -s logDTDiff.log ) then
    echo "WARNING THE DT RECO GEOMETRY IS DIFFERENT BETWEEN DDD AND DB" | tee -a GeometryValidation.log
endif

echo "End DT RECO geometry validation" | tee -a GeometryValidation.log

echo "Start CSC RECO geometry validation" | tee -a GeometryValidation.log

cp $CMSSW_BASE/src/Geometry/CSCGeometry/test/testCSCGeometryFromDB_cfg.py .  
sed -i "{s/MC_31X_V8::All/${gtag}/}" testCSCGeometryFromDB_cfg.py >> GeometryValidation.log
cmsRun testCSCGeometryFromDB_cfg.py > outDB_CSC.log
if ( -s outDB_CSC.log ) then
    echo "CSC test from DB run ok" | tee -a GeometryValidation.log
else
    echo "ERROR the output of CSC test from DB is empty" | tee -a GeometryValidation.log
    exit
endif

cp $CMSSW_BASE/src/Geometry/CSCGeometry/test/testCSCGeometry_cfg.py .
sed -i "{s/GeometryExtended/${geometry}/}" testCSCGeometry_cfg.py >>  GeometryValidation.log
sed -i "{s/MC_31X_V8::All/${gtag}/}" testCSCGeometry_cfg.py >> GeometryValidation.log
cmsRun testCSCGeometry_cfg.py > outDDD_CSC.log

if ( -s outDDD_CSC.log ) then
    echo "CSC test from DDD run ok" | tee -a GeometryValidation.log
else
    echo "ERROR the output of CSC test from DDD is empty" | tee -a GeometryValidation.log
    exit
endif

diff --ignore-matching-lines='Geometry node for CSCGeom' outDDD_CSC.log outDB_CSC.log > logCSCDiff.log
if ( -s logCSCDiff.log ) then
    echo "WARNING THE CSC RECO GEOMETRY IS DIFFERENT BETWEEN DDD AND DB" | tee -a GeometryValidation.log
endif

echo "End CSC RECO geometry validation" | tee -a GeometryValidation.log

echo "Start RPC RECO geometry validation" | tee -a GeometryValidation.log

cp $CMSSW_BASE/src/Geometry/RPCGeometry/test/testRPCGeometryFromDB_cfg.py .  
sed -i "{s/MC_31X_V8::All/${gtag}/}" testRPCGeometryFromDB_cfg.py >> GeometryValidation.log
cmsRun testRPCGeometryFromDB_cfg.py > outDB_RPC.log
if ( -s outDB_RPC.log ) then
    echo "RPC test from DB run ok" | tee -a GeometryValidation.log
else
    echo "ERROR the output of RPC test from DB is empty" | tee -a GeometryValidation.log
    exit
endif

cp $CMSSW_BASE/src/Geometry/RPCGeometry/test/testRPCGeometry_cfg.py .
sed -i "{s/GeometryExtended/${geometry}/}" testRPCGeometry_cfg.py >>  GeometryValidation.log
sed -i "{s/MC_31X_V8::All/${gtag}/}" testRPCGeometry_cfg.py >> GeometryValidation.log
cmsRun testRPCGeometry_cfg.py > outDDD_RPC.log

if ( -s outDDD_RPC.log ) then
    echo "RPC test from DDD run ok" | tee -a GeometryValidation.log
else
    echo "ERROR the output of RPC test from DDD is empty" | tee -a GeometryValidation.log
    exit
endif

diff --ignore-matching-lines='Geometry node for RPCGeom' outDDD_RPC.log outDB_RPC.log > logRPCDiff.log
if ( -s logRPCDiff.log ) then
    echo "WARNING THE RPC RECO GEOMETRY IS DIFFERENT BETWEEN DDD AND DB" | tee -a GeometryValidation.log
endif

echo "End RPC RECO geometry validation" | tee -a GeometryValidation.log

echo "Start CALO RECO geometry validation" | tee -a GeometryValidation.log

addpkg Geometry/CaloEventSetup
cd $CMSSW_BASE/src/Geometry/CaloEventSetup/test/
source setup.scr > GeometryCaloValidation.log
sed -i "{s/MC_31X_V8::All/${gtag}/}" runTestCaloGeometryXMLDB_cfg.py >> GeometryCaloValidation.log
cmsRun runTestCaloGeometryXMLDB_cfg.py >> GeometryCaloValidation.log
cd -
less $CMSSW_BASE/src/Geometry/CaloEventSetup/test/GeometryCaloValidation.log | tee -a GeometryValidation.log

echo "End CALO RECO geometry validation" | tee -a GeometryValidation.log


echo "Start Simulation geometry validation" | tee -a GeometryValidation.log
addpkg GeometryReaders/XMLIdealGeometryESSource
cd $CMSSW_BASE/src/GeometryReaders/XMLIdealGeometryESSource/test/
source runXMLBigFileToDBAndBackValidation.sh ${geometry} > GeometryXMLValidation.log
cd -
less $CMSSW_BASE/src/GeometryReaders/XMLIdealGeometryESSource/test/GeometryXMLValidation.log | tee -a GeometryValidation.log

echo "End Simulation geometry validation" | tee -a GeometryValidation.log



