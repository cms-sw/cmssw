#!/bin/bash -ex

VGEO_DIR=${CMSSW_BASE}/src/Validation/Geometry/test
TEST_DIR=.

cmsRun ${VGEO_DIR}/single_neutrino_cfg.py nEvents=1000 >$TEST_DIR/single_neutrino_cfg.log 2>&1

python_cmd="python2"
python3 -c "from FWCore.PythonFramework.CmsRun import CmsRun" 2>/dev/null && python_cmd="python3"

for geom in {'Extended2015','Extended2017Plan1'}; do
  ${python_cmd} ${VGEO_DIR}/runP_Tracker.py geom=$geom label=Tracker >$TEST_DIR/runP_Tracker_cfg.log 2>&1
done

${python_cmd} ${VGEO_DIR}/MaterialBudget.py -s -d Tracker -g 'Extended2017Plan1'
${python_cmd} ${VGEO_DIR}/MaterialBudget.py -s -d Tracker -g 'Extended2015' -gc 'Extended2017Plan1'

echo '<html> <head><title>Reference Tracker Material Budget</title></head> <style>img.Reference{margin: 20px auto 20px auto; border: 10px solid green; border-radius: 10px;}img.PullRequest{margin: 20px auto 20px auto; border: 10px solid red; border-radius: 10px;}</style> <body> <h1>Reference Plots</h1> <p>Please check any diferences in the plots compared with the given reference. <en>Reference plots</en> have a green border followed by the output of the PR which features a red border</p><h2>Geometry: Extended2017Plan1</h2> <hr/> <img class="Reference" src="https://twiki.cern.ch/twiki/pub/CUAHEP/ValidationGeometryReferencePlots/Tracker_l_vs_z_vs_R_bw.png" width="1000"/><br/> <img class="PullRequest" src="Tracker_l_vs_z_vs_R_Extended2017Plan1_bw.png" width="1000"/><br/> <hr/> <img class="Reference" src="https://twiki.cern.ch/twiki/pub/CUAHEP/ValidationGeometryReferencePlots/Tracker_x_vs_eta.png" width="375"/> <img class="PullRequest" src="Tracker_x_vs_eta_Extended2017Plan1.png" width="375"/><br/> <hr/> <img class="Reference" src="https://twiki.cern.ch/twiki/pub/CUAHEP/ValidationGeometryReferencePlots/Tracker_x_vs_phi.png" width="375"/> <img class="PullRequest" src="Tracker_x_vs_phi_Extended2017Plan1.png" width="375"/><br/> <hr/> <h2>Geometry Comparison: Extended2017Plan1 vs Extended2015</h2> <h3>Comparison Ratio</h3> <img class="Reference" src="https://twiki.cern.ch/twiki/pub/CUAHEP/ValidationGeometryReferencePlots/Tracker_ComparisonRatio_l_vs_R.png" width="1000"/><br/> <img class="PullRequest" src="Tracker_ComparisonRatio_l_vs_R_Extended2015_vs_Extended2017Plan1.png" width="1000"/><br/> <hr/> <img class="Reference" src="https://twiki.cern.ch/twiki/pub/CUAHEP/ValidationGeometryReferencePlots/Tracker_ComparisonRatio_l_vs_z_vs_R.png" width="1000"/><br/> <img class="PullRequest" src="Tracker_Comparison_l_vs_z_vs_R_geocomp_Extended2015_vs_Extended2017Plan1.png" width="1000"/><br/> <hr/> <img class="Reference" src="https://twiki.cern.ch/twiki/pub/CUAHEP/ValidationGeometryReferencePlots/Tracker_Comparison_x_vs_eta.png" width="1000"/><br/> <img class="PullRequest" src="Tracker_ComparisonRatio_x_vs_eta_Extended2015_vs_Extended2017Plan1.png" width="1000"/><br/> <hr/> </body></html>' > ${TEST_DIR}/index.html

if [[ ! -z ${JENKINS_UPLOAD_DIR} ]] ; then
   mkdir ${JENKINS_UPLOAD_DIR}/materialBudgetTrackerPlots
   cp  ${TEST_DIR}/Images/*.png ${JENKINS_UPLOAD_DIR}/materialBudgetTrackerPlots/
   cp  ${TEST_DIR}/index.html ${JENKINS_UPLOAD_DIR}/materialBudgetTrackerPlots/
   cp  ${TEST_DIR}/*.log ${JENKINS_UPLOAD_DIR}/materialBudgetTrackerPlots/ 
fi
