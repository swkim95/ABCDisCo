for smear in 75 100 150 200
do
   echo $smear
   /opt/anaconda3/bin/python -u ABCD_topjets_HLF_mD_smear.py --gpunum=0  --smear=$smear > log_topjets_HLF_mD_smear$smear
done

