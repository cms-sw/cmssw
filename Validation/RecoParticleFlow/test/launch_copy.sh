#!/bin/sh

cmsswDir=/afs/cern.ch/user/p/pjanot/scratch0/CMSSW_2_2_0/src
castorDir=/castor/cern.ch/user/p/pjanot/CMSSW220
castor="\/castor\/cern.ch\/user\/p\/pjanot\/CMSSW220"
simu=Full

for ((out=0; out<=2; out++));
    do
    case $out in
      0)
	  type=reco
	  ;;
      1)
	  type=aod
	  ;;
      2)
	  type=display
	  ;;
    esac
	
    for ((job=0;job<=8;job++));
	do
	case $job in
	0)
	    name=QCDDiJet_20_30
	    ;;
	1)
	    name=QCDDiJet_30_50
	    ;;
	2)
	    name=QCDDiJet_50_80
	    ;;
	3)
	    name=QCDDiJet_80_120
	    ;;
	4)
	    name=QCDDiJet_120_160
	    ;;
	5)
	    name=QCDDiJet_160_250
	    ;;
	6)
	    name=QCDDiJet_250_350
	    ;;
	7)
	    name=QCDDiJet_350_500
	    ;;
	8)
	    name=QCDDiJet_500_700
	    ;;
	esac
	
	filename=${type}"_"${name}"_"${simu}".root"
	castorname=${castorDir}"/"${type}"_"${name}"_"${simu}".root"
	echo $filename" -> "$castorname
	    
	sed -e "s/==CASTOR==/${castor}/" -e "s/==TYPE==/${type}/" -e "s/==NAME==/${name}/" -e "s/==SIMU==/${simu}/" copy_cfg.py > tmp_cfg
	
	sed -e "s/==CASTOR==/${castor}/" -e "s/==TYPE==/${type}/" -e "s/==NAME==/${name}/" -e "s/==SIMU==/${simu}/" drop.sh > drop_${type}_${name}.sh
	
#Start to write the script
	cat > copy_${type}_${name}.sh << EOF

#!/bin/sh
cd $cmsswDir
eval \`scramv1 runtime -sh\`
cd -
#commande pour decoder le .cfg
cat > TEST_cfg.py << "EOF"
EOF
    
#Ajoute le .cfg au script
cat  tmp_cfg>> copy_${type}_${name}.sh

# On poursuit le script
echo "EOF" >> copy_${type}_${name}.sh
cat >> copy_${type}_${name}.sh << EOF
cmsRun TEST_cfg.py

rfcp $filename $castorname
rm $filename

EOF
chmod 755 copy_${type}_${name}.sh
chmod 755 drop_${type}_${name}.sh

# uncomment if you wish to copy
#copy_${type}_${name}.sh
#rm copy_${type}_${name}.sh

# uncomment if you wish to drop
# It's unwise to automatically drop before checking the outcome of the copy stage !
#drop_${type}_${name}.sh
#rm drop_${type}_${name}.sh


  done
done

