#! /bin/tcsh 

#setenv file1 stdgeom_btag_PU0.root
#setenv legend1 "stdgeom PU0"
#setenv file1 phase1_btag_PU0.root
#setenv legend1 "Phase 1 PU0"
#setenv file1 stdgeomfastsim_btag_PU0.root
#setenv legend1 "stdgeom fastsim PU0"
setenv file1 r39v423_btagvalid_step01_dlosspu50.root
setenv legend1 "Phase 1 fullsim PU50"

#setenv file2 phase1_btag_PU0.root
#setenv legend2 "Phase 1 PU0"
#setenv file2 stdgeomfastsim_btag_PU0.root
#setenv legend2 "stdgeom fastsim PU0"
setenv file2 phase1fastsim_btag_PU50.root
setenv legend2 "Phase 1 fastsim PU50"

cat plottino_mix.C | sed \
      -e s/FILE1LEGEND/"$legend1"/g \
      -e s/FILE2LEGEND/"$legend2"/g \
      -e s/plottino_mix/tmp_plottino_mix/g \
    > ! tmp_plottino_mix.C

setenv a423 CSV
setenv b423 CSVMVA
setenv c423 SSVHE
setenv d423 SSVHP
setenv e423 TCHE
setenv f423 TCHP

root -l -b -q tmp_plottino_mix.C\(\"$file1\",\"$file2\",\"$a423\",\"$a423\",0,0,0\)
root -l -b -q tmp_plottino_mix.C\(\"$file1\",\"$file2\",\"$b423\",\"$b423\",0,0,0\)
root -l -b -q tmp_plottino_mix.C\(\"$file1\",\"$file2\",\"$c423\",\"$c423\",0,0,0\)
root -l -b -q tmp_plottino_mix.C\(\"$file1\",\"$file2\",\"$d423\",\"$d423\",0,0,0\)
root -l -b -q tmp_plottino_mix.C\(\"$file1\",\"$file2\",\"$e423\",\"$e423\",0,0,0\)
root -l -b -q tmp_plottino_mix.C\(\"$file1\",\"$file2\",\"$f423\",\"$f423\",0,0,0\)

