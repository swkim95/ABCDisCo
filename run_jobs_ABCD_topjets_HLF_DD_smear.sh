for smear in 75 100 150 200
do
   echo $smear
   /opt/anaconda3/bin/python -u ABCD_topjets_HLF_DD_smear.py --gpunum=1  --smear=$smear > log_topjets_HLF_DD_smear$smear
done

