#!/bin/sh

cmsswDir=/afs/cern.ch/user/p/pjanot/scratch0/CMSSW_3_1_0_pre2/src
castorDir=/castor/cern.ch/user/p/pjanot/CMSSW310pre2
castor="\/castor\/cern.ch\/user\/p\/pjanot\/CMSSW310pre2"

name=QCDForPF
simu=Full

for ((job=1; job<3; job++));
    do

jobin=${job}
  case $job in
      0)
	  jobin=""
	  ;;
  esac


  for ((out=1; out<=2; out++));
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
    
    filename=${type}"_"${name}"_"${simu}"_00"${job}".root"
    castorname=${castorDir}"/"${type}"_"${name}"_"${simu}"_00"${job}".root"
    echo $filename" -> "$castorname
    
    sed -e "s/==CASTOR==/${castor}/" -e "s/==TYPE==/${type}/" -e "s/==NAME==/${name}/" -e "s/==SIMU==/${simu}/" -e "s/==JOBIN==/${jobin}/" -e "s/==JOB==/${job}/" copy_cfg.py > tmp_cfg
	
    sed -e "s/==CASTOR==/${castor}/" -e "s/==TYPE==/${type}/" -e "s/==NAME==/${name}/" -e "s/==SIMU==/${simu}/" -e "s/==JOBIN==/${jobin}/" drop.sh > drop_${type}_${name}_${job}.sh
	
#Start to write the script
	cat > copy_${type}_${name}_${job}.sh << EOF

#!/bin/sh
cd $cmsswDir
eval \`scramv1 runtime -sh\`
cd -
#commande pour decoder le .cfg
cat > TEST_cfg.py << "EOF"
EOF
    
#Ajoute le .cfg au script
cat  tmp_cfg>> copy_${type}_${name}_${job}.sh

# On poursuit le script
echo "EOF" >> copy_${type}_${name}_${job}.sh
cat >> copy_${type}_${name}_${job}.sh << EOF
cmsRun TEST_cfg.py

rfcp $filename $castorname
rm $filename

EOF
chmod 755 copy_${type}_${name}_${job}.sh
chmod 755 drop_${type}_${name}_${job}.sh

# uncomment if you wish to copy
#copy_${type}_${name}_${job}.sh
#rm copy_${type}_${name}_${job}.sh

# uncomment if you wish to drop
# It's unwise to automatically drop before checking the outcome of the copy stage !
#drop_${type}_${name}_${job}.sh
#rm drop_${type}_${name}_${job}.sh


  done
done

