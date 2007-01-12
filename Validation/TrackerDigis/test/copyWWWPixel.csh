#! /bin/csh
setenv RELEASE $CMSSW_VERSION
setenv WWWDIR /afs/cern.ch/cms/cpt/Software/html/General/Validation/SVSuite/TrackerBTau/$RELEASE/Digi/Pixel

mkdir $WWWDIR/eps
mkdir $WWWDIR/eps/AdcPXB
mkdir $WWWDIR/eps/RowPXB
mkdir $WWWDIR/eps/ColPXB
mkdir $WWWDIR/eps/DigiMulPXB
mkdir $WWWDIR/eps/AdcPXF
mkdir $WWWDIR/eps/RowPXF
mkdir $WWWDIR/eps/ColPXF
mkdir $WWWDIR/eps/DigiMulPXF

mkdir $WWWDIR/gif
mkdir $WWWDIR/gif/AdcPXB
mkdir $WWWDIR/gif/RowPXB
mkdir $WWWDIR/gif/ColPXB
mkdir $WWWDIR/gif/DigiMulPXB
mkdir $WWWDIR/gif/AdcPXF
mkdir $WWWDIR/gif/RowPXF
mkdir $WWWDIR/gif/ColPXF
mkdir $WWWDIR/gif/DigiMulPXF


echo "...Copying..."

mv Adc*PXB*.eps $WWWDIR/eps/AdcPXB
mv Row*PXB*.eps $WWWDIR/eps/RowPXB
mv Col*PXB*.eps $WWWDIR/eps/ColPXB
mv DigiMul*PXB*.eps $WWWDIR/eps/DigiMulPXB
mv AdcZ*Disk*.eps $WWWDIR/eps/AdcPXF
mv RowZ*Disk*.eps $WWWDIR/eps/RowPXF
mv ColZ*Disk*.eps $WWWDIR/eps/ColPXF
mv DigiMul*Endcap*.eps $WWWDIR/eps/DigiMulPXF

mv Adc*PXB*.gif $WWWDIR/gif/AdcPXB
mv Row*PXB*.gif $WWWDIR/gif/RowPXB
mv Col*PXB*.gif $WWWDIR/gif/ColPXB
mv DigiMul*PXB*.gif $WWWDIR/gif/DigiMulPXB
mv AdcZ*Disk*.gif $WWWDIR/gif/AdcPXF
mv RowZ*Disk*.gif $WWWDIR/gif/RowPXF
mv ColZ*Disk*.gif $WWWDIR/gif/ColPXF
mv DigiMul*Endcap*.gif $WWWDIR/gif/DigiMulPXF

echo "...Done..."
