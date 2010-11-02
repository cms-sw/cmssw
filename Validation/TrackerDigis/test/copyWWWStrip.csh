#! /bin/csh
setenv RELEASE $CMSSW_VERSION

if ( ! -d /afs/cern.ch/cms/performance/tracker/activities/validation/$RELEASE/ ) mkdir /afs/cern.ch/cms/performance/tracker/activities/validation/$RELEASE/
if ( ! -d /afs/cern.ch/cms/performance/tracker/activities/validation/$RELEASE/Digi ) mkdir /afs/cern.ch/cms/performance/tracker/activities/validation/$RELEASE/Digi

if ($2 =="") then
setenv WWWDIRObj /afs/cern.ch/cms/performance/tracker/activities/validation/$RELEASE/Digi

else
setenv WWWDIRObj /afs/cern.ch/cms/performance/tracker/activities/validation/$RELEASE/Digi/$2
if (! -d $WWWDIRObj) mkdir $WWWDIRObj
endif

mkdir $WWWDIRObj/Strip

setenv WWWDIR $WWWDIRObj/Strip

mkdir $WWWDIR/eps
mkdir $WWWDIR/eps/NdigiTIB
mkdir $WWWDIR/eps/AdcTIB
mkdir $WWWDIR/eps/StripNumTIB
mkdir $WWWDIR/eps/NdigiTOB
mkdir $WWWDIR/eps/AdcTOB
mkdir $WWWDIR/eps/StripNumTOB
mkdir $WWWDIR/eps/NdigiTID
mkdir $WWWDIR/eps/AdcTID
mkdir $WWWDIR/eps/StripNumTID
mkdir $WWWDIR/eps/NdigiTEC
mkdir $WWWDIR/eps/AdcTEC
mkdir $WWWDIR/eps/StripNumTEC

mkdir $WWWDIR/gif
mkdir $WWWDIR/gif/NdigiTIB
mkdir $WWWDIR/gif/AdcTIB
mkdir $WWWDIR/gif/StripNumTIB
mkdir $WWWDIR/gif/NdigiTOB
mkdir $WWWDIR/gif/AdcTOB
mkdir $WWWDIR/gif/StripNumTOB
mkdir $WWWDIR/gif/NdigiTID
mkdir $WWWDIR/gif/AdcTID
mkdir $WWWDIR/gif/StripNumTID
mkdir $WWWDIR/gif/NdigiTEC
mkdir $WWWDIR/gif/AdcTEC
mkdir $WWWDIR/gif/StripNumTEC


echo "...Copying..."

mv NdigiTIB*.eps.gz $WWWDIR/eps/NdigiTIB
mv Adc*TIB*.eps.gz $WWWDIR/eps/AdcTIB
mv StripNu*TIB*.eps.gz $WWWDIR/eps/StripNumTIB
mv NdigiTOB*.eps.gz $WWWDIR/eps/NdigiTOB
mv Adc*TOB*.eps.gz $WWWDIR/eps/AdcTOB
mv StripNu*TOB*.eps.gz $WWWDIR/eps/StripNumTOB
mv NdigiTID*.eps.gz $WWWDIR/eps/NdigiTID
mv Adc*TID*.eps.gz $WWWDIR/eps/AdcTID
mv StripNu*TID*.eps.gz $WWWDIR/eps/StripNumTID
mv NdigiTEC*.eps.gz $WWWDIR/eps/NdigiTEC
mv Adc*TEC*.eps.gz $WWWDIR/eps/AdcTEC
mv StripNu*TEC*.eps.gz $WWWDIR/eps/StripNumTEC

mv NdigiTIB*.gif $WWWDIR/gif/NdigiTIB
mv Adc*TIB*.gif $WWWDIR/gif/AdcTIB
mv StripNu*TIB*.gif $WWWDIR/gif/StripNumTIB
mv NdigiTOB*.gif $WWWDIR/gif/NdigiTOB
mv Adc*TOB*.gif $WWWDIR/gif/AdcTOB
mv StripNu*TOB*.gif $WWWDIR/gif/StripNumTOB
mv NdigiTID*.gif $WWWDIR/gif/NdigiTID
mv Adc*TID*.gif $WWWDIR/gif/AdcTID
mv StripNu*TID*.gif $WWWDIR/gif/StripNumTID
mv NdigiTEC*.gif $WWWDIR/gif/NdigiTEC
mv Adc*TEC*.gif $WWWDIR/gif/AdcTEC
mv StripNu*TEC*.gif $WWWDIR/gif/StripNumTEC

echo "...Done..."
