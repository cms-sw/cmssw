#! /bin/csh
setenv RELEASE 100pre3
setenv WWWDIR /afs/fnal.gov/files/home/room2/elis/public_html/Validation/$RELEASE

mkdir $WWWDIR
mkdir $WWWDIR/TK-strips
mkdir $WWWDIR/TK-strips/eloss
mkdir $WWWDIR/TK-strips/Localx
mkdir $WWWDIR/TK-strips/Localy
mkdir $WWWDIR/TK-strips/Entryx-Exitx
mkdir $WWWDIR/TK-strips/Entryy-Exity
mkdir $WWWDIR/TK-strips/Entryz-Exitz
mkdir $WWWDIR/TK-pixels
mkdir $WWWDIR/TK-pixels/eloss
mkdir $WWWDIR/TK-pixels/Localx
mkdir $WWWDIR/TK-pixels/Localy
mkdir $WWWDIR/TK-pixels/Entryx-Exitx
mkdir $WWWDIR/TK-pixels/Entryy-Exity
mkdir $WWWDIR/TK-pixels/Entryz-Exitz
mkdir $WWWDIR/TK-summary

echo "...Copying..."
cp plots/muon/eloss_T*_KS* $WWWDIR/TK-strips/eloss
cp plots/muon/pos_Entryx-Exitx_T*_KS* $WWWDIR/TK-strips/Entryx-Exitx
cp plots/muon/pos_Entryy-Exity_T*_KS* $WWWDIR/TK-strips/Entryy-Exity
cp plots/muon/pos_Entryz-Exitz_T*_KS* $WWWDIR/TK-strips/Entryz-Exitz
cp plots/muon/pos_Localy_T*_KS* $WWWDIR/TK-strips/Localy
cp plots/muon/pos_Localx_T*_KS* $WWWDIR/TK-strips/Localx

cp plots/muon/eloss_*PIX_KS* $WWWDIR/TK-pixels/eloss
cp plots/muon/pos_Entryx-Exitx_*PIX_KS* $WWWDIR/TK-pixels/Entryx-Exitx
cp plots/muon/pos_Entryy-Exity_*PIX_KS* $WWWDIR/TK-pixels/Entryy-Exity
cp plots/muon/pos_Entryz-Exitz_*PIX_KS* $WWWDIR/TK-pixels/Entryz-Exitz
cp plots/muon/pos_Localy_*PIX_KS* $WWWDIR/TK-pixels/Localy
cp plots/muon/pos_Localx_*PIX_KS* $WWWDIR/TK-pixels/Localx

cp plots/muon/summary* $WWWDIR/TK-summary
cp LowKS* $WWWDIR/TK-summary
echo "...Done..."
