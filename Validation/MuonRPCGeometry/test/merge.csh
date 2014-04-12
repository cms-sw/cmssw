#! /bin/csh


set out = /tmp/fruboes/muons.txt
rm $out
zcat out/mu*.gz > $out


ln -s $out muons.txt
echo Cands `cat muons.txt | wc -l`
echo Saved in $out
