##!/bin/csh -x
#!/bin/csh 

#This script can be used to generate a web page to compare histograms from 
#two input root files produced using the EDAnalyzers in RecoEgamma/Examples,
#by running one of:
#
#  
#  #  "Validation/RecoEgamma/test/PhotonValidator_cfg.py
#
# The default list of histograms (configurable) is based on version VXX-XX-XX
# of Validation/RecoEgamma
#
#Two files are created by this script: validation.C and validation.html.
#validation.C should be run inside root to greate a set of gif images
#which can then be viewed in a web browser using validation.html.

#=============BEGIN CONFIGURATION=================

setenv RUNTYPE Central
#setenv RUNTYPE Local
setenv STARTUP True
setenv FASTSIM False
setenv UPGRADE True
## TYPE options: Photons, GEDPhotons
## ANALYZERNAME options: PhotonValidator, oldpfPhotonValidator, pfPhotonValidator
#setenv TYPE Photons
#setenv ANALYZERNAME PhotonValidator
setenv TYPE GEDPhotons
setenv ANALYZERNAME pfPhotonValidator


setenv CMSSWver1 7_1_0
setenv CMSSWver2 7_1_0
setenv OLDRELEASE 7_1_0
setenv NEWRELEASE 7_1_0
setenv OLDPRERELEASE pre3
setenv NEWPRERELEASE pre4
setenv UPGRADEVER  UPG2017
setenv LHCENERGY   13
setenv PU True
setenv PUlevel 50ns

if ( $STARTUP == True &&  $FASTSIM == False && $PU == False && $UPGRADE == True ) then
setenv OLDGLOBALTAG POSTLS171_V1-v1
setenv NEWGLOBALTAG POSTLS171_V1-v2
else if ( $UPGRADE == True && $PU == True &&  $FASTSIM == False ) then
setenv OLDGLOBALTAG PU${PUlevel}_POSTLS171_V2-v6
setenv NEWGLOBALTAG PU${PUlevel}_POSTLS171_V2-v2
else if (  $STARTUP == True  && $FASTSIM == True) then
setenv OLDGLOBALTAG POSTLS171_V1_FastSim-v1
setenv NEWGLOBALTAG POSTLS171_V1_FastSim-v2
 endif




setenv OLDRELEASE ${OLDRELEASE}_${OLDPRERELEASE}
#setenv OLDRELEASE ${OLDRELEASE}
setenv NEWRELEASE ${NEWRELEASE}_${NEWPRERELEASE}
#setenv NEWRELEASE ${NEWRELEASE}

#setenv WorkDir1   /afs/cern.ch/user/n/nancy/scratch0/CMSSW/test/CMSSW_${CMSSWver1}/src/Validation/RecoEgamma/test
#setenv WorkDir2   /afs/cern.ch/user/n/nancy/scratch0/CMSSW/test/CMSSW_${CMSSWver2}_${NEWPRERELEASE}/src/Validation/RecoEgamma/test

setenv WorkDir1   /afs/cern.ch/user/n/nancy/scratch0/CMSSW/test/CMSSW_${CMSSWver1}_${OLDPRERELEASE}/src/Validation/RecoEgamma/test
setenv WorkDir2   /afs/cern.ch/user/n/nancy/scratch0/CMSSW/test/CMSSW_${CMSSWver2}_${NEWPRERELEASE}/src/Validation/RecoEgamma/test

#setenv WorkDir1   /afs/cern.ch/user/n/nancy/scratch0/CMSSW/test/CMSSW_${CMSSWver1}_${OLDPRERELEASE}/src/Validation/RecoEgamma/test
#setenv WorkDir2   /afs/cern.ch/user/n/nancy/scratch0/CMSSW/test/CMSSW_${CMSSWver2}/src/Validation/RecoEgamma/test

#setenv WorkDir1   /afs/cern.ch/user/n/nancy/scratch0/CMSSW/test/CMSSW_${CMSSWver1}/src/Validation/RecoEgamma/test
#setenv WorkDir2   /afs/cern.ch/user/n/nancy/scratch0/CMSSW/test/CMSSW_${CMSSWver2}/src/Validation/RecoEgamma/test

#Name of sample (affects output directory name and htmldescription only) 


#setenv SAMPLE SingleGammaPt10
#setenv SAMPLE SingleGammaPt35
##setenv SAMPLE SingleGammaFlatPt10_100
setenv SAMPLE H130GGgluonfusion
#setenv SAMPLE PhotonJets_Pt_10
#setenv SAMPLE GammaJets_Pt_80_120
#setenv SAMPLE QCD_Pt_80_120


if ( $RUNTYPE == Central ) then
setenv HISTOPATHNAME_Efficiencies DQMData/Run\ 1/EgammaV/Run\ summary/${ANALYZERNAME}/Efficiencies
setenv HISTOPATHNAME_Photons DQMData/Run\ 1/EgammaV/Run\ summary/${ANALYZERNAME}/Photons
setenv HISTOPATHNAME_Conversions DQMData/Run\ 1/EgammaV/Run\ summary/${ANALYZERNAME}/ConversionInfo
endif

if ( $RUNTYPE == Local ) then
setenv HISTOPATHNAME_Efficiencies DQMData/EgammaV/${ANALYZERNAME}/Efficiencies
setenv HISTOPATHNAME_Photons DQMData/EgammaV/${ANALYZERNAME}/Photons
setenv HISTOPATHNAME_Conversions DQMData/EgammaV/${ANALYZERNAME}/ConversionInfo
endif




#==============END BASIC CONFIGURATION==================


#Input root trees for the two cases to be compared 

if ($SAMPLE == SingleGammaPt10) then

if ( $RUNTYPE == Local ) then
setenv OLDFILE ${WorkDir1}/PhotonValidationRelVal${OLDRELEASE}_SingleGammaPt10.root
setenv NEWFILE ${WorkDir2}/PhotonValidationRelVal${NEWRELEASE}_SingleGammaPt10.root
else if ( $RUNTYPE == Central ) then
setenv OLDFILE ${WorkDir1}/DQM_V0001_R000000001__RelValSingleGammaPt10_UP15__CMSSW_${OLDRELEASE}-${OLDGLOBALTAG}__DQM.root
if ( $UPGRADE == True ) then
setenv NEWFILE ${WorkDir2}/DQM_V0001_R000000001__RelValSingleGammaPt10_UP15__CMSSW_${NEWRELEASE}-${NEWGLOBALTAG}__DQM.root
#setenv NEWFILE ${WorkDir2}/DQM_V0001_R000000001__RelValSingleGammaPt10_${UPGRADEVER}__CMSSW_${NEWRELEASE}-${NEWGLOBALTAG}__DQM.root
else 
setenv NEWFILE ${WorkDir2}/DQM_V0001_R000000001__RelValSingleGammaPt10_UP15__CMSSW_${NEWRELEASE}-${NEWGLOBALTAG}__DQM.root
endif
endif


else if ($SAMPLE == SingleGammaPt35) then 

if ( $RUNTYPE == Local ) then
setenv OLDFILE ${WorkDir1}/PhotonValidationRelVal${OLDRELEASE}_SingleGammaPt35.root
setenv NEWFILE ${WorkDir2}/PhotonValidationRelVal${NEWRELEASE}_SingleGammaPt35.root

else if ( $RUNTYPE == Central ) then
setenv OLDFILE ${WorkDir1}/DQM_V0001_R000000001__RelValSingleGammaPt35__CMSSW_${OLDRELEASE}-${OLDGLOBALTAG}__DQM.root
if ( $UPGRADE == True ) then
setenv NEWFILE ${WorkDir2}/DQM_V0001_R000000001__RelValSingleGammaPt35_${UPGRADEVER}__CMSSW_${NEWRELEASE}-${NEWGLOBALTAG}__DQM.root
else 
setenv NEWFILE ${WorkDir2}/DQM_V0001_R000000001__RelValSingleGammaPt35__CMSSW_${NEWRELEASE}-${NEWGLOBALTAG}__DQM.root
endif 
endif


else if ($SAMPLE == SingleGammaFlatPt10_100) then 


if ( $RUNTYPE == Local ) then
setenv OLDFILE ${WorkDir1}/PhotonValidationRelVal${OLDRELEASE}_SingleGammaFlatPt10To100.root
setenv NEWFILE ${WorkDir2}/PhotonValidationRelVal${NEWRELEASE}_SingleGammaFlatPt10To100.root
endif 


else if ($SAMPLE == H130GGgluonfusion) then 


if ( $RUNTYPE == Local ) then
setenv OLDFILE ${WorkDir1}/PhotonValidationRelVal${OLDRELEASE}_H130GGgluonfusion.root
setenv NEWFILE ${WorkDir2}/PhotonValidationRelVal${NEWRELEASE}_H130GGgluonfusion.root
else if ( $RUNTYPE == Central ) then

#setenv OLDFILE ${WorkDir1}/DQM_V0001_R000000001__RelValH130GGgluonfusion__CMSSW_${OLDRELEASE}-${OLDGLOBALTAG}__GEN-SIM-DIGI-RECO.root
#setenv OLDFILE ${WorkDir1}/DQM_V0001_R000000001__RelValH130GGgluonfusion__CMSSW_${OLDRELEASE}-${OLDGLOBALTAG}__DQM.root
setenv OLDFILE ${WorkDir1}/DQM_V0001_R000000001__RelValH130GGgluonfusion_${UPGRADEVER}_${LHCENERGY}__CMSSW_${OLDRELEASE}-${OLDGLOBALTAG}__DQM.root

if ( $UPGRADE == True ) then

setenv OLDFILE ${WorkDir1}/DQM_V0001_R000000001__RelValH130GGgluonfusion_${LHCENERGY}__CMSSW_${OLDRELEASE}-${OLDGLOBALTAG}__DQM.root
setenv NEWFILE ${WorkDir2}/DQM_V0001_R000000001__RelValH130GGgluonfusion_${LHCENERGY}__CMSSW_${NEWRELEASE}-${NEWGLOBALTAG}__DQM.root
else
setenv NEWFILE ${WorkDir2}/DQM_V0001_R000000001__RelValH130GGgluonfusion__CMSSW_${NEWRELEASE}-${NEWGLOBALTAG}__DQM.root
endif

endif



else if ($SAMPLE == PhotonJets_Pt_10) then

if ( $RUNTYPE == Local ) then
setenv OLDFILE ${WorkDir1}/PhotonValidationRelVal${OLDRELEASE}_PhotonJets_Pt_10.root
setenv NEWFILE ${WorkDir2}/PhotonValidationRelVal${NEWRELEASE}_PhotonJets_Pt_10.root
else if ( $RUNTYPE == Central ) then

setenv OLDFILE ${WorkDir1}/DQM_V0001_R000000001__RelValPhotonJets_Pt_10__CMSSW_${OLDRELEASE}-${OLDGLOBALTAG}__DQM.root
if ( $UPGRADE == True ) then
setenv NEWFILE ${WorkDir2}/DQM_V0001_R000000001__RelValPhotonJets_Pt_10_${UPGRADEVER}_${LHCENERGY}__CMSSW_${NEWRELEASE}-${NEWGLOBALTAG}__DQM.root
else
setenv NEWFILE ${WorkDir2}/DQM_V0001_R000000001__RelValPhotonJets_Pt_10__CMSSW_${NEWRELEASE}-${NEWGLOBALTAG}__DQM.root
endif 
endif

else if ($SAMPLE ==  GammaJets_Pt_80_120) then 

if ( $RUNTYPE == Local ) then
setenv OLDFILE ${WorkDir1}/PhotonValidationRelVal${OLDRELEASE}_GammaJets_Pt_80_120.root
setenv NEWFILE ${WorkDir2}/PhotonValidationRelVal${NEWRELEASE}_DQM.root
endif

else if ($SAMPLE == QCD_Pt_80_120) then 


endif



#Location of output.  The default will put your output in:
#http://cmsdoc.cern.ch/Physics/egamma/www/validation/

setenv CURRENTDIR $PWD
setenv OUTPATH /afs/cern.ch/cms/Physics/egamma/www/validation/Photons
cd $OUTPATH
setenv NEWRELEASE {$NEWRELEASE}
setenv OLDRELEASE {$OLDRELEASE}

if (! -d $NEWRELEASE) then
  mkdir $NEWRELEASE
endif
setenv OUTPATH $OUTPATH/$NEWRELEASE
cd $OUTPATH

if (! -d ${TYPE}) then
  mkdir ${TYPE}
endif
setenv OUTPATH $OUTPATH/${TYPE}
cd  $OUTPATH

if (! -d vs${OLDRELEASE}) then
  mkdir vs${OLDRELEASE}
endif
setenv OUTPATH $OUTPATH/vs${OLDRELEASE}


if ( $FASTSIM == True) then 
setenv OUTDIR $OUTPATH/${SAMPLE}FastSim
else if ( $FASTSIM == False && $UPGRADE == False && $PU == True ) then 
setenv OUTDIR $OUTPATH/${SAMPLE}PU
else if ( $FASTSIM == False && $PU == False && $UPGRADE == False ) then 
setenv OUTDIR $OUTPATH/${SAMPLE}
else if ( $SAMPLE == H130GGgluonfusion  && $UPGRADE == True && $PU == True && $FASTSIM == False) then
setenv OUTDIR $OUTPATH/${SAMPLE}_${LHCENERGY}TeV_PU${PUlevel}
else if ( $SAMPLE == H130GGgluonfusion  && $UPGRADE == True ) then
setenv OUTDIR $OUTPATH/${SAMPLE}_${LHCENERGY}TeV
else if ( $SAMPLE ==  PhotonJets_Pt_10  && $UPGRADE == True ) then
setenv OUTDIR $OUTPATH/${SAMPLE}_${LHCENERGY}TeV
else if ( $SAMPLE ==  SingleGammaPt10  && $UPGRADE == True ) then
setenv OUTDIR $OUTPATH/${SAMPLE}
else if ( $SAMPLE ==  SingleGammaPt35  && $UPGRADE == True ) then
setenv OUTDIR $OUTPATH/${SAMPLE}
endif


#else if ( $SAMPLE == H130GGgluonfusion ||  PhotonJets_Pt_10  && $UPGRADE == True ) then
#if ( $PU == True) then
#setenv OUTDIR $OUTPATH/${SAMPLE}PU
#else if ( $PU == False) then
#setenv OUTDIR $OUTPATH/${SAMPLE}
#endif



if (! -d $OUTDIR) then
  cd $OUTPATH
  mkdir $OUTDIR
  cd $OUTDIR
  mkdir gifs
endif
cd $OUTDIR


#The list of histograms to be compared for each TYPE can be configured below:


if ( $TYPE == Photons ||  $TYPE == GEDPhotons ) then


cat > efficiencyForPhotons <<EOF
  recoEffVsEta
  recoEffVsPhi
  recoEffVsEt
  deadChVsEta
  deadChVsPhi
  deadChVsEt
EOF


#  gamgamMassAll
#  gamgamMassBarrel
#  gamgamMassEndcap
#  gamgamMassNoConvAll
#  gamgamMassNoConvBarrel
#  gamgamMassNoConvEndcap
#  gamgamMassConvAll
#  gamgamMassConvBarrel
#  gamgamMassConvEndcap


cat > scaledhistosForPhotons <<EOF
  nOfflineVtx
  scEta
  scPhi
  scEAll
  scEtAll
  phoEta
  phoPhi
  phoDEta
  phoDPhi
  phoEAll
  phoEtAll
  eResAll
  eResBarrel
  eResEndcap
  eResunconvAll
  eResunconvBarrel
  eResunconvEndcap
  eResconvAll
  eResconvBarrel
  eResconvEndcap
  isoTrkSolidConeDR04All
  isoTrkSolidConeDR04Barrel
  isoTrkSolidConeDR04Endcap
  nTrkSolidConeDR04All
  nTrkSolidConeDR04Barrel
  nTrkSolidConeDR04Endcap
  r9Barrel
  r9Endcap
  r1Barrel
  r1Endcap
  r2Barrel
  r2Endcap
  sigmaIetaIetaBarrel
  sigmaIetaIetaEndcap
  hOverEAll
  hOverEBarrel
  hOverEEndcap
  newhOverEAll
  newhOverEBarrel
  newhOverEEndcap
  hcalTowerSumEtConeDR04Barrel
  hcalTowerSumEtConeDR04Endcap
  hcalTowerBcSumEtConeDR04Barrel
  hcalTowerBcSumEtConeDR04Endcap
  ecalRecHitSumEtConeDR04Barrel
  ecalRecHitSumEtConeDR04Endcap


EOF

cat > scaledhistosForPhotonsLogScale <<EOF
  hOverEAll
  hOverEBarrel
  hOverEEndcap
  newhOverEAll
  newhOverEBarrel
  newhOverEEndcap
  hcalTowerSumEtConeDR04Barrel
  hcalTowerSumEtConeDR04Endcap
  hcalTowerBcSumEtConeDR04Barrel
  hcalTowerBcSumEtConeDR04Endcap
  ecalRecHitSumEtConeDR04Barrel
  ecalRecHitSumEtConeDR04Endcap
  r9Barrel
  r9Endcap
  r1Barrel
  r1Endcap
  r2Barrel
  r2Endcap
  sigmaIetaIetaAll
  sigmaIetaIetaBarrel
  sigmaIetaIetaEndcap



EOF


cat > unscaledhistosForPhotons <<EOF
pEResVsEtAll
pEResVsEtBarrel
pEResVsEtEndcap
pEResVsEtaAll
pEResVsEtaUnconv
pEResVsEtaConv
pEcalRecHitSumEtConeDR04VsEtaAll
pEcalRecHitSumEtConeDR04VsEtBarrel
pEcalRecHitSumEtConeDR04VsEtEndcap
pHcalTowerSumEtConeDR04VsEtaAll
pHcalTowerSumEtConeDR04VsEtBarrel
pHcalTowerSumEtConeDR04VsEtEndcap
pHcalTowerBcSumEtConeDR04VsEtaAll
pHcalTowerBcSumEtConeDR04VsEtBarrel
pHcalTowerBcSumEtConeDR04VsEtEndcap
pHoverEVsEtaAll
pHoverEVsEtAll
pnewHoverEVsEtaAll
pnewHoverEVsEtAll


EOF



cat > scaledhistosGEDspecific <<EOF
  eResRegr1unconvAll
  eResRegr1unconvBarrel
  eResRegr1unconvEndcap
  eResRegr1convAll
  eResRegr1convBarrel
  eResRegr1convEndcap
  eResRegr2unconvAll
  eResRegr2unconvBarrel
  eResRegr2unconvEndcap
  eResRegr2convAll
  eResRegr2convBarrel
  eResRegr2convEndcap
  chargedHadIsoBarrel
  chargedHadIsoEndcap
  neutralHadIsoBarrel
  neutralHadIsoEndcap
  photonIsoBarrel
  photonIsoEndcap
  pfMVABarrel
  pfMVAEndcap
  nCluOutMustacheBarrel
  nCluOutMustacheEndcap
  dRPhoPFcand_Pho_unCleanedBarrel
  dRPhoPFcand_Pho_unCleanedEndcap
  dRPhoPFcand_Pho_CleanedBarrel
  dRPhoPFcand_Pho_CleanedEndcap
  dRPhoPFcand_ChHad_unCleanedBarrel
  dRPhoPFcand_ChHad_unCleanedEndcap
  dRPhoPFcand_ChHad_CleanedBarrel
  dRPhoPFcand_ChHad_CleanedEndcap
  dRPhoPFcand_NeuHad_unCleanedBarrel
  dRPhoPFcand_NeuHad_unCleanedEndcap
  dRPhoPFcand_NeuHad_CleanedBarrel
  dRPhoPFcand_NeuHad_CleanedEndcap
  SumPtOverPhoPt_Pho_unCleanedBarrel
  SumPtOverPhoPt_Pho_unCleanedEndcap
  SumPtOverPhoPt_Pho_CleanedBarrel
  SumPtOverPhoPt_Pho_CleanedEndcap
  SumPtOverPhoPt_ChHad_unCleanedBarrel
  SumPtOverPhoPt_ChHad_unCleanedEndcap
  SumPtOverPhoPt_ChHad_CleanedBarrel
  SumPtOverPhoPt_ChHad_CleanedEndcap
  SumPtOverPhoPt_NeuHad_unCleanedBarrel
  SumPtOverPhoPt_NeuHad_unCleanedEndcap
  SumPtOverPhoPt_NeuHad_CleanedBarrel
  SumPtOverPhoPt_NeuHad_CleanedEndcap

EOF


cat > scaledhistosGEDspecificLogScale <<EOF
  photonIsoBarrel
  photonIsoEndcap
  chargedHadIsoBarrel
  chargedHadIsoEndcap
  neutralHadIsoBarrel
  neutralHadIsoEndcap


EOF



cat > efficiencyForConvertedPhotons <<EOF

  convEffVsEtaTwoTracks
  convEffVsPhiTwoTracks
  convEffVsRTwoTracks
  convEffVsZTwoTracks
  convEffVsEtTwoTracks
  convEffVsEtaTwoTracksAndVtxProbGT0
  convEffVsRTwoTracksAndVtxProbGT0

EOF



cat > scaledhistosForConvertedPhotons <<EOF

  convEta2
  convPhi
  convEResAll
  convEResBarrel
  convEResEndcap
  PoverEtracksAll
  PoverEtracksBarrel 
  PoverEtracksEndcap
  convPtResAll
  convPtResBarrel
  convPtResEndcap
  convVtxdR
  convVtxdR_barrel
  convVtxdR_endcap
  convVtxdZ
  convVtxdZ_barrel
  convVtxdZ_endcap
  convVtxdX
  convVtxdX_barrel
  convVtxdX_endcap
  convVtxdY
  convVtxdY_barrel
  convVtxdY_endcap
  mvaOutAll
  mvaOutBarrel
  mvaOutEndcap

EOF


cat > scaledhistosForConvertedPhotonsLogScale <<EOF
  EoverPtracksAll
  EoverPtracksBarrel 
  EoverPtracksEndcap
  vtxChi2ProbAll
  vtxChi2ProbBarrel
  vtxChi2ProbEndcap


EOF



cat > unscaledhistosForConvertedPhotons <<EOF
pEoverEtrueVsEtaAll
pEoverPVsEtaAll
pEoverPVsRAll
pConvVtxdRVsR
pConvVtxdRVsEta
pConvVtxdXVsX
pConvVtxdYVsY
pConvVtxdZVsZ
EOF


cat > 2dhistosForConvertedPhotons <<EOF
  convVtxRvsZAll
EOF

cat > projectionsForConvertedPhotons <<EOF
   convVtxRvsZBarrel
   convVtxRvsZEndcap
EOF



cat > fakeRateForConvertedPhotons <<EOF

  convFakeRateVsEtaTwoTracks
  convFakeRateVsPhiTwoTracks
  convFakeRateVsRTwoTracks
  convFakeRateVsZTwoTracks
  convFakeRateVsEtTwoTracks

EOF



cat > scaledhistosForTracks <<EOF

tkChi2AllTracks
hTkPtPullAll
hTkPtPullBarrel
hTkPtPullEndcap
hDPhiTracksAtVtxAll
hDCotTracksAll
zPVFromTracksAll
zPVFromTracksBarrel
zPVFromTracksEndcap
dzPVFromTracksAll
dzPVFromTracksBarrel
dzPVFromTracksEndcap

EOF

cat > unscaledhistosForTracks <<EOF
h_nHitsVsEtaAllTracks
h_nHitsVsRAllTracks
pChi2VsEtaAll
pChi2VsRAll
pDCotTracksVsEtaAll
pDCotTracksVsRAll
pdzPVVsR


EOF



endif

#=================END CONFIGURATION=====================

if (-e validation.C) rm validation.C
touch validation.C
cat > begin.C <<EOF
{
TFile *file_old = TFile::Open("$OLDFILE");
TFile *file_new = TFile::Open("$NEWFILE");

EOF
cat begin.C >>& validation.C
rm begin.C

setenv N 1



foreach i (`cat efficiencyForPhotons`)
  cat > temp$N.C <<EOF

TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
c$i->Divide(1,2);
c$i->cd(1);
//file_old->cd("DQMData/EgammaV/${ANALYZERNAME}/Efficiencies");
file_old->cd("$HISTOPATHNAME_Efficiencies");
$i->SetStats(0);
int nBins = $i->GetNbinsX();
float xMin=$i->GetBinLowEdge(1);
float xMax=$i->GetBinLowEdge(nBins)+$i->GetBinWidth(nBins);
TH1F* hold=new  TH1F("hold"," ",nBins,xMin,xMax);
hold=$i;
if ( $i==deadChVsEta ||  $i==deadChVsPhi ||  $i==deadChVsEt ) {
$i->SetMinimum(0.);
$i->SetMaximum(0.2);
} else if (  $i==recoEffVsEt ) {
$i->GetXaxis()->SetRangeUser(0.,200.);
} else {
$i->SetMinimum(0.);
$i->SetMaximum(1.1);
}
$i->SetLineColor(kPink+8);
$i->SetMarkerColor(kPink+8);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(1);
$i->SetLineWidth(1);
$i->Draw();
//file_new->cd("DQMData/EgammaV/${ANALYZERNAME}/Efficiencies");
file_new->cd("$HISTOPATHNAME_Efficiencies");
$i->SetStats(0);
$i->SetMinimum(0.);
$i->SetMaximum(1.1);
TH1F* hnew=new  TH1F("hnew"," ",nBins,xMin,xMax);
hnew=$i;
$i->SetLineColor(kBlack);
$i->SetMarkerColor(kBlack);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(1);
$i->SetLineWidth(1);
$i->Draw("same");
c$i->cd(2);
TH1F* ratio=new  TH1F("ratio"," ",nBins,xMin,xMax);
ratio->Divide(hnew,hold);
ratio->SetStats(0);
for ( int i=1; i<=ratio->GetNbinsX(); i++ ) {
float num=hnew->GetBinContent(i);
float den=hold->GetBinContent(i);
float dNum=hnew->GetBinError(i);
float dDen=hold->GetBinError(i);
float erro=0;
if ( num!=0 && den!=0) {
erro= ((1./den)*(1./den)*dNum*dNum) + ((num*num)/(den*den*den*den) * (dDen*dDen));
erro=sqrt(erro);
}
ratio->SetBinError(i, erro);
}
ratio->SetLineColor(1);
ratio->SetLineWidth(2);
ratio->SetMinimum(0.);
ratio->SetMaximum(2.);
ratio->Draw("e");
TLine *l = new TLine(xMin,1.,xMax,1.);
l->Draw(); 
c$i->SaveAs("gifs/$i.gif");

EOF
  setenv N `expr $N + 1`
end

foreach i (`cat scaledhistosForPhotons`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
c$i->Divide(1,2);
c$i->cd(1);
//file_new->cd("DQMData/EgammaV/${ANALYZERNAME}/Photons");
file_new->cd("$HISTOPATHNAME_Photons");
int nBins = $i->GetNbinsX();
float xMin=$i->GetBinLowEdge(1);
float xMax=$i->GetBinLowEdge(nBins)+$i->GetBinWidth(nBins);
Double_t mnew=$i->GetMaximum();
Double_t nnew=$i->GetEntries();
//file_old->cd("DQMData/EgammaV/${ANALYZERNAME}/Photons");
file_old->cd("$HISTOPATHNAME_Photons");

TH1F* hold=new  TH1F("hold"," ",nBins,xMin,xMax);
hold=$i;
Double_t mold=$i->GetMaximum();
Double_t nold=$i->GetEntries();
if ( $i==scEAll || $i==phoEAll ) {  
$i->GetYaxis()->SetRangeUser(0.,2000.);
}
$i->SetStats(0);
$i->SetMinimum(0.);
if ( mnew > mold+sqrt(mold) )  { 
$i->SetMaximum(mnew+2*sqrt(mnew)); 
}  else { 
$i->SetMaximum(mold+2*sqrt(mold)); 
}
$i->SetLineColor(kPink+8);
$i->SetFillColor(kPink+8);
//$i->SetLineWidth(3);
$i->Draw();
//file_new->cd("DQMData/EgammaV/${ANALYZERNAME}/Photons");
file_new->cd("$HISTOPATHNAME_Photons");
Double_t nnew=$i->GetEntries();
$i->SetStats(0);
$i->SetLineColor(kBlack);
$i->SetMarkerColor(kBlack);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(0.8);
//$i->SetLineWidth(1);
$i->Scale(nold/nnew);
TH1F* hnew=new  TH1F("hnew"," ",nBins,xMin,xMax);
hnew=$i;
$i->Draw("e1same");
c$i->cd(2);
TH1F* ratio=new  TH1F("ratio"," ",nBins,xMin,xMax);
ratio->Divide(hnew,hold);
for ( int i=1; i<=ratio->GetNbinsX(); i++ ) {
float num=hnew->GetBinContent(i);
float den=hold->GetBinContent(i);
float dNum=hnew->GetBinError(i);
float dDen=hold->GetBinError(i);
float erro=0;
if ( num!=0 && den!=0) {
erro= ((1./den)*(1./den)*dNum*dNum) + ((num*num)/(den*den*den*den) * (dDen*dDen));
erro=sqrt(erro);
}
ratio->SetBinError(i, erro);
}
ratio->SetStats(0);
ratio->SetLineColor(1);
ratio->SetLineWidth(2);
ratio->SetMinimum(0.);
ratio->SetMaximum(4.);
ratio->Draw("e");
TLine *l = new TLine(xMin,1.,xMax,1.);
l->Draw(); 
c$i->SaveAs("gifs/$i.gif");
//TString gifName=TString("gifs/$i")+"_ratio.gif";
//c$i->SaveAs(gifName);
EOF
  setenv N `expr $N + 1`
end


foreach i (`cat scaledhistosGEDspecific`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
c$i->Divide(1,2);
c$i->cd(1);
//file_new->cd("DQMData/EgammaV/${ANALYZERNAME}/Photons");
file_new->cd("$HISTOPATHNAME_Photons");
int nBins = $i->GetNbinsX();
float xMin=$i->GetBinLowEdge(1);
float xMax=$i->GetBinLowEdge(nBins)+$i->GetBinWidth(nBins);
Double_t mnew=$i->GetMaximum();
Double_t nnew=$i->GetEntries();
//file_old->cd("DQMData/EgammaV/${ANALYZERNAME}/Photons");
file_old->cd("$HISTOPATHNAME_Photons");

TH1F* hold=new  TH1F("hold"," ",nBins,xMin,xMax);
hold=$i;
Double_t mold=$i->GetMaximum();
Double_t nold=$i->GetEntries();
if ( $i==scEAll || $i==phoEAll ) {  
$i->GetYaxis()->SetRangeUser(0.,2000.);
}
if ($i==chargedHadIsoBarrel || $i==chargedHadIsoBarrel ) {
$i->GetXaxis()->SetRangeUser(0.,12.);
}
$i->SetStats(0);
$i->SetMinimum(0.);
if ( mnew > mold+sqrt(mold) )  { 
$i->SetMaximum(mnew+2*sqrt(mnew)); 
}  else { 
$i->SetMaximum(mold+2*sqrt(mold)); 
}
$i->SetLineColor(kPink+8);
$i->SetFillColor(kPink+8);
//$i->SetLineWidth(3);
$i->Draw();
//file_new->cd("DQMData/EgammaV/${ANALYZERNAME}/Photons");
file_new->cd("$HISTOPATHNAME_Photons");
Double_t nnew=$i->GetEntries();
$i->SetStats(0);
$i->SetLineColor(kBlack);
$i->SetMarkerColor(kBlack);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(0.8);
//$i->SetLineWidth(1);
$i->Scale(nold/nnew);
TH1F* hnew=new  TH1F("hnew"," ",nBins,xMin,xMax);
hnew=$i;
$i->Draw("e1same");
c$i->cd(2);
TH1F* ratio=new  TH1F("ratio"," ",nBins,xMin,xMax);
ratio->Divide(hnew,hold);
for ( int i=1; i<=ratio->GetNbinsX(); i++ ) {
float num=hnew->GetBinContent(i);
float den=hold->GetBinContent(i);
float dNum=hnew->GetBinError(i);
float dDen=hold->GetBinError(i);
float erro=0;
if ( num!=0 && den!=0) {
erro= ((1./den)*(1./den)*dNum*dNum) + ((num*num)/(den*den*den*den) * (dDen*dDen));
erro=sqrt(erro);
}
ratio->SetBinError(i, erro);
}
ratio->SetStats(0);
ratio->SetLineColor(1);
ratio->SetLineWidth(2);
ratio->SetMinimum(0.);
ratio->SetMaximum(4.);
ratio->Draw("e");
TLine *l = new TLine(xMin,1.,xMax,1.);
l->Draw(); 
c$i->SaveAs("gifs/$i.gif");
//TString gifName=TString("gifs/$i")+"_ratio.gif";
//c$i->SaveAs(gifName);
EOF
  setenv N `expr $N + 1`
end



foreach i (`cat scaledhistosGEDspecificLogScale`)
  cat > temp$N.C <<EOF
TCanvas *cc$i = new TCanvas("cc$i");
cc$i->cd();
cc$i->SetFillColor(10);
cc$i->SetLogy();
//file_new->cd("DQMData/EgammaV/${ANALYZERNAME}/Photons");
file_new->cd("$HISTOPATHNAME_Photons");
Double_t nnew=$i->GetEntries();
//file_old->cd("DQMData/EgammaV/${ANALYZERNAME}/Photons");
file_old->cd("$HISTOPATHNAME_Photons");
if ( $i==hcalTowerSumEtConeDR04Barrel ||  $i==hcalTowerSumEtConeDR04Endcap  ) {  
$i->GetXaxis()->SetRangeUser(0.,10.);
}
Double_t nold=$i->GetEntries();
$i->SetStats(0);
$i->SetMinimum(1);
$i->SetLineColor(kPink+8);
$i->SetFillColor(kPink+8);
$i->Draw();
//file_new->cd("DQMData/EgammaV/${ANALYZERNAME}/Photons");
file_new->cd("$HISTOPATHNAME_Photons");
Double_t nnew=$i->GetEntries();
$i->SetStats(0);
$i->SetLineColor(kBlack);
$i->SetMarkerColor(kBlack);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(1);
$i->Draw("e1same");
cc$i->SaveAs("gifs/log$i.gif");

EOF
  setenv N `expr $N + 1`
end





foreach i (`cat scaledhistosForPhotonsLogScale`)
  cat > temp$N.C <<EOF
TCanvas *cc$i = new TCanvas("cc$i");
cc$i->cd();
cc$i->SetFillColor(10);
cc$i->SetLogy();
//file_new->cd("DQMData/EgammaV/${ANALYZERNAME}/Photons");
file_new->cd("$HISTOPATHNAME_Photons");
Double_t nnew=$i->GetEntries();
//file_old->cd("DQMData/EgammaV/${ANALYZERNAME}/Photons");
file_old->cd("$HISTOPATHNAME_Photons");
if ( $i==hcalTowerSumEtConeDR04Barrel ||  $i==hcalTowerSumEtConeDR04Endcap  ) {  
$i->GetXaxis()->SetRangeUser(0.,10.);
}
Double_t nold=$i->GetEntries();
$i->SetStats(0);
$i->SetMinimum(1);
$i->SetLineColor(kPink+8);
$i->SetFillColor(kPink+8);
$i->Draw();
//file_new->cd("DQMData/EgammaV/${ANALYZERNAME}/Photons");
file_new->cd("$HISTOPATHNAME_Photons");
Double_t nnew=$i->GetEntries();
$i->SetStats(0);
$i->SetLineColor(kBlack);
$i->SetMarkerColor(kBlack);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(1);
$i->Draw("e1same");
cc$i->SaveAs("gifs/log$i.gif");

EOF
  setenv N `expr $N + 1`
end





foreach i (`cat unscaledhistosForPhotons`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
//file_old->cd("DQMData/EgammaV/${ANALYZERNAME}/Photons");
file_old->cd("$HISTOPATHNAME_Photons");
$i->SetStats(0);
if ( $i==pEcalRecHitSumEtConeDR04VsEtaAll   ) {  
$i->GetYaxis()->SetRangeUser(0.,5.);
} else if ( $i==pEcalRecHitSumEtConeDR04VsEtBarrel ) 
{ $i->GetYaxis()->SetRangeUser(0.,20.); 
} else if ( $i==pEcalRecHitSumEtConeDR04VsEtEndcap ) 
{ $i->GetYaxis()->SetRangeUser(0.,20.);
} else if ( $i==pHcalTowerSumEtConeDR04VsEtaAll) 
{ $i->GetYaxis()->SetRangeUser(0.,0.5);
} else if ( $i==pHcalTowerBcSumEtConeDR04VsEtaAll   ) 
{ $i->GetYaxis()->SetRangeUser(0.,1.);
} else if ( $i==pHcalTowerSumEtConeDR04VsEtBarrel ||  $i==pHcalTowerBcSumEtConeDR04VsEtBarrel) 
{ $i->GetYaxis()->SetRangeUser(0.,5.);
} else if ( $i==pHcalTowerSumEtConeDR04VsEtEndcap  || $i==pHcalTowerBcSumEtConeDR04VsEtEndcap ) 
{ $i->GetYaxis()->SetRangeUser(0.,5.);
} else if ( $i==pHoverEVsEtaAll || $i==pnewHoverEVsEtaAll  ) 
{ $i->GetYaxis()->SetRangeUser(-0.05,0.05);
} else if ( $i==pHoverEVsEtAll ||  $i==pnewHoverEVsEtAll ) 
{ $i->GetYaxis()->SetRangeUser(-0.05,0.05);
} else  {
$i->SetMinimum(0.94);
$i->SetMaximum(1.04);
}
$i->SetLineColor(kPink+8);
$i->SetMarkerColor(kPink+8);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(1);
$i->SetLineWidth(1);
$i->Draw();
//file_new->cd("DQMData/EgammaV/${ANALYZERNAME}/Photons");
file_new->cd("$HISTOPATHNAME_Photons");
$i->SetStats(0);
$i->SetLineColor(kBlack);
$i->SetMarkerColor(kBlack);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(1);
$i->SetLineWidth(1);
$i->Draw("e1same");
c$i->SaveAs("gifs/$i.gif");

EOF
  setenv N `expr $N + 1`
end





foreach i (`cat efficiencyForConvertedPhotons`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
c$i->Divide(1,2);
c$i->cd(1);
//file_old->cd("DQMData/EgammaV/${ANALYZERNAME}/Efficiencies");
file_old->cd("$HISTOPATHNAME_Efficiencies");
$i->SetStats(0);
int nBins = $i->GetNbinsX();
float xMin=$i->GetBinLowEdge(1);
float xMax=$i->GetBinLowEdge(nBins)+$i->GetBinWidth(nBins);
TH1F* hold=new  TH1F("hold"," ",nBins,xMin,xMax);
hold=$i;
$i->SetMinimum(0.);
$i->SetMaximum(1.);
$i->SetLineColor(kPink+8);
$i->SetMarkerColor(kPink+8);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(1);
$i->SetLineWidth(1);
$i->Draw();
//file_new->cd("DQMData/EgammaV/${ANALYZERNAME}/Efficiencies");
file_new->cd("$HISTOPATHNAME_Efficiencies");
TH1F* hnew=new  TH1F("hnew"," ",nBins,xMin,xMax);
hnew=$i;
$i->SetStats(0);
$i->SetMinimum(0.);
$i->SetMaximum(1.);
$i->SetLineColor(kBlack);
$i->SetMarkerColor(kBlack);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(1);
$i->SetLineWidth(1);
$i->Draw("same");
TH1F* ratio=new  TH1F("ratio"," ",nBins,xMin,xMax);
ratio->Divide(hnew,hold);
for ( int i=1; i<=ratio->GetNbinsX(); i++ ) {
float num=hnew->GetBinContent(i);
float den=hold->GetBinContent(i);
float dNum=hnew->GetBinError(i);
float dDen=hold->GetBinError(i);
float erro=0;
if ( num!=0 && den!=0) {
erro= ((1./den)*(1./den)*dNum*dNum) + ((num*num)/(den*den*den*den) * (dDen*dDen));
erro=sqrt(erro);
}
ratio->SetBinError(i, erro);
}
ratio->SetStats(0);
ratio->SetLineColor(1);
ratio->SetLineWidth(2);
ratio->SetMinimum(0.);
ratio->SetMaximum(2.);
c$i->cd(2);
ratio->Draw("e");
TLine *l = new TLine(xMin,1.,xMax,1.);
l->Draw();
c$i->SaveAs("gifs/$i.gif");

EOF
  setenv N `expr $N + 1`
end




foreach i (`cat scaledhistosForConvertedPhotons`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
//file_new->cd("DQMData/EgammaV/${ANALYZERNAME}/ConversionInfo");
file_new->cd("$HISTOPATHNAME_Conversions");
Double_t mnew=$i->GetMaximum();
//file_old->cd("DQMData/EgammaV/${ANALYZERNAME}/ConversionInfo");
file_old->cd("$HISTOPATHNAME_Conversions");
Double_t mold=$i->GetMaximum();
$i->SetStats(0);
$i->SetMinimum(0.);
if ( mnew > mold) 
$i->SetMaximum(mnew+mnew*0.1);
else 
$i->SetMaximum(mold+mold*0.1);
$i->SetLineColor(kPink+8);
$i->SetFillColor(kPink+8);
$i->SetLineWidth(3);
$i->Draw();
Double_t nold=$i->GetEntries();
//file_new->cd("DQMData/EgammaV/${ANALYZERNAME}/ConversionInfo");
file_new->cd("$HISTOPATHNAME_Conversions");
Double_t nnew=$i->GetEntries();
$i->SetStats(0);
$i->SetLineColor(kBlack);
$i->SetMarkerColor(kBlack);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(1);
$i->SetLineWidth(1);
$i->Scale(nold/nnew);
$i->Draw("e1same");
c$i->SaveAs("gifs/$i.gif");

EOF
  setenv N `expr $N + 1`
end


foreach i (`cat scaledhistosForConvertedPhotonsLogScale`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
c$i->SetLogy(1);
//file_new->cd("DQMData/EgammaV/${ANALYZERNAME}/ConversionInfo");
file_new->cd("$HISTOPATHNAME_Conversions");
Double_t mnew=$i->GetMaximum();
//file_old->cd("DQMData/EgammaV/${ANALYZERNAME}/ConversionInfo");
file_old->cd("$HISTOPATHNAME_Conversions");
Double_t mold=$i->GetMaximum();
$i->SetStats(0);
$i->SetLineColor(kPink+8);
$i->SetFillColor(kPink+8);
$i->SetLineWidth(3);
$i->Draw();
Double_t nold=$i->GetEntries();
//file_new->cd("DQMData/EgammaV/${ANALYZERNAME}/ConversionInfo");
file_new->cd("$HISTOPATHNAME_Conversions");
Double_t nnew=$i->GetEntries();
$i->SetStats(0);
$i->SetLineColor(kBlack);
$i->SetMarkerColor(kBlack);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(1);
$i->SetLineWidth(1);
$i->Scale(nold/nnew);
$i->Draw("e1same");
c$i->SaveAs("gifs/$i.gif");

EOF
  setenv N `expr $N + 1`
end





foreach i (`cat unscaledhistosForConvertedPhotons`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
//file_old->cd("DQMData/EgammaV/${ANALYZERNAME}/ConversionInfo");
file_old->cd("$HISTOPATHNAME_Conversions");
$i->SetStats(0);
$i->GetYaxis()->SetRangeUser(0.6.,2);
if (  $i == pConvVtxdRVsR || $i == pConvVtxdRVsEta || $i == pConvVtxdXVsX ||  $i ==  pConvVtxdYVsY ) {
$i->GetYaxis()->SetRangeUser(-10.,10);
} else if ( $i == pConvVtxdZVsZ ) {
$i->GetYaxis()->SetRangeUser(-10.,10);
}
$i->SetLineColor(kPink+8);
$i->SetMarkerColor(kPink+8);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(1);
$i->SetLineWidth(1);
$i->Draw();
//file_new->cd("DQMData/EgammaV/${ANALYZERNAME}/ConversionInfo");
file_new->cd("$HISTOPATHNAME_Conversions");
$i->SetStats(0);
$i->SetLineColor(kBlack);
$i->SetMarkerColor(kBlack);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(1);
$i->SetLineWidth(1);
$i->Draw("e1same");
c$i->SaveAs("gifs/$i.gif");

EOF
  setenv N `expr $N + 1`
end



foreach i (`cat fakeRateForConvertedPhotons`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
//file_old->cd("DQMData/EgammaV/${ANALYZERNAME}/Efficiencies");
file_old->cd("$HISTOPATHNAME_Efficiencies");
$i->SetStats(0);
$i->SetMinimum(0.);
$i->SetMaximum(1.);
$i->SetLineColor(kPink+8);
$i->SetMarkerColor(kPink+8);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(1);
$i->SetLineWidth(1);
$i->Draw();
//file_new->cd("DQMData/EgammaV/${ANALYZERNAME}/Efficiencies");
file_new->cd("$HISTOPATHNAME_Efficiencies");
$i->SetStats(0);
$i->SetMinimum(0.);
$i->SetLineColor(kBlack);
$i->SetMarkerColor(kBlack);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(1);
$i->SetLineWidth(1);
$i->Draw("same");
c$i->SaveAs("gifs/$i.gif");

EOF
  setenv N `expr $N + 1`
end


foreach i (`cat 2dhistosForConvertedPhotons`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
//file_old->cd("DQMData/EgammaV/${ANALYZERNAME}/ConversionInfo");
file_old->cd("$HISTOPATHNAME_Conversions");
$i->SetStats(0);
$i->SetMinimum(0.);
$i->SetMarkerColor(kPink+8);
$i->Draw();
//file_new->cd("DQMData/EgammaV/${ANALYZERNAME}/ConversionInfo");
file_new->cd("$HISTOPATHNAME_Conversions");
$i->SetStats(0);
$i->SetMarkerColor(kBlack);
$i->Draw("same");
c$i->SaveAs("gifs/$i.gif");

EOF
  setenv N `expr $N + 1`
end

foreach i (`cat projectionsForConvertedPhotons`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
//file_old->cd("DQMData/EgammaV/${ANALYZERNAME}/ConversionInfo");
file_old->cd("$HISTOPATHNAME_Conversions");
if ($i==convVtxRvsZBarrel)
TH1D *tmp1$i= $i->ProjectionY();
else if ($i==convVtxRvsZEndcap)
TH1D *tmp1$i= $i->ProjectionX();
Double_t nold=tmp1$i->GetEntries();
Double_t mold=tmp1$i->GetMaximum();
//file_new->cd("DQMData/EgammaV/${ANALYZERNAME}/ConversionInfo");
file_new->cd("$HISTOPATHNAME_Conversions");
//TH1D *tmp2$i= $i->ProjectionY();
if ($i==convVtxRvsZBarrel)
TH1D *tmp2$i= $i->ProjectionY();
else if ($i==convVtxRvsZEndcap)
TH1D *tmp2$i= $i->ProjectionX();
Double_t nnew=tmp2$i->GetEntries();
Double_t mnew=tmp2$i->GetMaximum();
tmp1$i->SetStats(0);
tmp1$i->SetMinimum(0.);
if ( mnew > mold) 
tmp1$i->SetMaximum(mnew+mnew*0.2);
else 
tmp1$i->SetMaximum(mold+mold*0.2);
tmp1$i->SetLineColor(kPink+8);
tmp1$i->SetFillColor(kPink+8);
tmp1$i->SetLineWidth(3);
tmp1$i->Draw();
tmp2$i->SetStats(0);
tmp2$i->SetLineColor(kBlack);
tmp2$i->SetLineWidth(3);
tmp2$i->Scale(nold/nnew);
tmp2$i->Draw("same");
c$i->SaveAs("gifs/$i.gif");

EOF
  setenv N `expr $N + 1`
end



###if ($ANALYZERNAME == pfPhotonValidator)  goto skippingHistosForTracks
foreach i (`cat scaledhistosForTracks`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
//file_new->cd("DQMData/EgammaV/${ANALYZERNAME}/ConversionInfo");
file_new->cd("$HISTOPATHNAME_Conversions");
Double_t mnew=$i->GetMaximum();
//file_old->cd("DQMData/EgammaV/${ANALYZERNAME}/ConversionInfo");
file_old->cd("$HISTOPATHNAME_Conversions");
Double_t mold=$i->GetMaximum();
$i->SetStats(0);
$i->SetMinimum(0.);
if ( mnew > mold) 
$i->SetMaximum(mnew+mnew*0.1);
else 
$i->SetMaximum(mold+mold*0.1);
$i->SetLineColor(kPink+8);
$i->SetFillColor(kPink+8);
$i->SetLineWidth(3);
$i->Draw();
Double_t nold=$i->GetEntries();
//file_new->cd("DQMData/EgammaV/${ANALYZERNAME}/ConversionInfo");
file_new->cd("$HISTOPATHNAME_Conversions");
Double_t nnew=$i->GetEntries();
$i->SetStats(0);
$i->SetLineColor(kBlack);
$i->SetMarkerColor(kBlack);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(1);
$i->SetLineWidth(1);
$i->Scale(nold/nnew);
$i->Draw("e1same");
c$i->SaveAs("gifs/$i.gif");

EOF
  setenv N `expr $N + 1`
end



foreach i (`cat unscaledhistosForTracks`)
  cat > temp$N.C <<EOF
TCanvas *c$i = new TCanvas("c$i");
c$i->SetFillColor(10);
//file_new->cd("DQMData/EgammaV/${ANALYZERNAME}/ConversionInfo");
file_new->cd("$HISTOPATHNAME_Conversions");
Double_t mnew=$i->GetMaximum();
//file_old->cd("DQMData/EgammaV/${ANALYZERNAME}/ConversionInfo");
file_old->cd("$HISTOPATHNAME_Conversions");
Double_t mold=$i->GetMaximum();
$i->SetStats(0);
if ($i==pDCotTracksVsEtaAll ||  $i==pDCotTracksVsRAll ) {
$i->SetMinimum(-0.05);
$i->SetMaximum(0.05);
} else if ( $i==pdzPVVsR ) { 
$i->GetYaxis()->SetRangeUser(-3.,3.);
} else {
$i->SetMinimum(0.);
    if ( mnew > mold) 
    $i->SetMaximum(mnew+mnew*0.4);
    else 
    $i->SetMaximum(mold+mold*0.4);
}

$i->SetLineColor(kPink+8);
$i->SetMarkerColor(kPink+8);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(1);
$i->SetLineWidth(1);
$i->Draw();
Double_t nold=$i->GetEntries();
//file_new->cd("DQMData/EgammaV/${ANALYZERNAME}/ConversionInfo");
file_new->cd("$HISTOPATHNAME_Conversions");
Double_t nnew=$i->GetEntries();
$i->SetStats(0);
$i->SetLineColor(kBlack);
$i->SetMarkerColor(kBlack);
$i->SetMarkerStyle(20);
$i->SetMarkerSize(1);
$i->SetLineWidth(1);
//$i->Scale(nold/nnew);
$i->Draw("e1same");
c$i->SaveAs("gifs/$i.gif");

EOF
  setenv N `expr $N + 1`
end

#skippingHistosForTracks:
#  echo "Skipping histograms which are not defined for pfPhotons"



setenv NTOT `expr $N - 1`
setenv N 1
while ( $N <= $NTOT )
  cat temp$N.C >>& validation.C
  rm temp$N.C
  setenv N `expr $N + 1`
end

cat > end.C <<EOF
}
EOF
cat end.C >>& validation.C
rm end.C


if ( $TYPE == PixelMatchGsfElectron ) then
  setenv ANALYZER PixelMatchGsfElectronAnalyzer
  setenv CFG read_gsfElectrons
else if ( $TYPE == Photons  || $TYPE == GEDPhotons) then
  setenv ANALYZER ${ANALYZERNAME}
  setenv CFG PhotonValidator_cfg
endif

if  ( $PU == True &&  $FASTSIM == False ) then
setenv SAMPLE ${SAMPLE}PU
else if ( $PU == False && $FASTSIM == True) then
setenv SAMPLE ${SAMPLE}FastSim
else if ( $SAMPLE == H130GGgluonfusion && $UPGRADE == True ) then
setenv SAMPLE ${SAMPLE}_${LHCENERGY}TeV
else if ( $SAMPLE == H130GGgluonfusion && $UPGRADE == True &&  $PU == True ) then
setenv SAMPLE ${SAMPLE}_${LHCENERGY}TeV_PU${PUlevel}
endif



if (-e validation.html) rm validation.html
if (-e validationPlotsTemplate.html) rm validationPlotsTemplate.html
cp ${CURRENTDIR}/validationPlotsTemplate.html validationPlotsTemplate.html
touch validation.html
cat > begin.html <<EOF
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">
<html>
<head>
<meta http-equiv="content-type" content="text/html; charset=UTF-8" />
<title>$NEWRELEASE vs $OLDRELEASE $TYPE validation</title>
</head>

<h1>$NEWRELEASE vs $OLDRELEASE $TYPE validation
<br>
 $SAMPLE 
</h1>

<p>The following plots were made using <a href="http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/CMSSW/Validation/RecoEgamma/src/$ANALYZER.cc">Validation/RecoEgamma/src/$ANALYZER</a>, 
using <a href="http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/CMSSW/Validation/RecoEgamma/test/$CFG.py">Validation/RecoEgamma/test/$CFG.py</a>
<p>The script used to make the plots is <a href="validation.C">here</a>.
<br>
In all plots below, $OLDRELEASE is in purple , $NEWRELEASE in black. 
<br>
Click on the plots to see them enlarged.
<br>
Responsible: N. Marinelli
<br>
<br>



EOF
cat begin.html >>& validation.html
rm begin.html
cat  validationPlotsTemplate.html >>& validation.html
rm  validationPlotsTemplate.html 

rm scaledhistosForPhotons
rm scaledhistosGEDspecific
rm scaledhistosGEDspecificLogScale
rm unscaledhistosForPhotons
rm scaledhistosForConvertedPhotons
rm scaledhistosForConvertedPhotonsLogScale
rm unscaledhistosForConvertedPhotons
rm efficiencyForPhotons
rm efficiencyForConvertedPhotons
rm fakeRateForConvertedPhotons
rm 2dhistosForConvertedPhotons
rm projectionsForConvertedPhotons
rm scaledhistosForTracks
rm unscaledhistosForTracks
rm scaledhistosForPhotonsLogScale

#echo "Now paste the following into your terminal window:"
#echo ""
echo "cd $OUTDIR"
#echo " root -b"
#echo ".x validation.C"
#echo ".q"
#echo "cd $CURRENTDIR"
#echo ""


root -b -l -q validation.C
cd $CURRENTDIR
echo "Then you can view your validation plots here:"
echo "http://cmsdoc.cern.ch/Physics/egamma/www/$OUTPATH/validation.html"
