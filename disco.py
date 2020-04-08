import torch

def distance_corr(yqcd,mqcd,normedweight,power=1):
    xx = yqcd.view(-1, 1).repeat(1, len(yqcd)).view(len(yqcd),len(yqcd))
    yy = yqcd.repeat(len(yqcd),1).view(len(yqcd),len(yqcd))
    amat=(xx-yy).abs()

    xx = mqcd.view(-1, 1).repeat(1, len(mqcd)).view(len(mqcd),len(mqcd))
    yy = mqcd.repeat(len(mqcd),1).view(len(mqcd),len(mqcd))
    bmat=(xx-yy).abs()

    amatavg=torch.mean(amat*normedweight,dim=1)
    Amat=amat-amatavg.repeat(len(yqcd),1).view(len(yqcd),len(yqcd))\
        -amatavg.view(-1, 1).repeat(1, len(yqcd)).view(len(yqcd),len(yqcd))\
        +torch.mean(amatavg*normedweight)

    bmatavg=torch.mean(bmat*normedweight,dim=1)
    Bmat=bmat-bmatavg.repeat(len(mqcd),1).view(len(mqcd),len(mqcd))\
        -bmatavg.view(-1, 1).repeat(1, len(mqcd)).view(len(mqcd),len(mqcd))\
        +torch.mean(bmatavg*normedweight)

    ABavg=torch.mean(Amat*Bmat*normedweight,dim=1)
    AAavg=torch.mean(Amat*Amat*normedweight,dim=1)
    BBavg=torch.mean(Bmat*Bmat*normedweight,dim=1)

    if(power==1):
        dCorr=(torch.mean(ABavg*normedweight))/torch.sqrt((torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight)))
    elif(power==2):
        dCorr=(torch.mean(ABavg*normedweight))**2/(torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight))
#        dCorr=(torch.mean(ABavg*normedweight))/torch.sqrt((torch.mean(AAavg*normedweight)))
    else:
        dCorr=(torch.mean(ABavg*normedweight))/torch.sqrt((torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight)))**power
    return dCorr


def distance_corr_unbiased(yqcd,mqcd,normedweight,power=1):
    xx = yqcd.view(-1, 1).repeat(1, len(yqcd)).view(len(yqcd),len(yqcd))
    yy = yqcd.repeat(len(yqcd),1).view(len(yqcd),len(yqcd))
    amat=(xx-yy).abs()

    xx = mqcd.view(-1, 1).repeat(1, len(mqcd)).view(len(mqcd),len(mqcd))
    yy = mqcd.repeat(len(mqcd),1).view(len(mqcd),len(mqcd))
    bmat=(xx-yy).abs()

    amatavg=1/(len(yqcd)-2)*torch.sum(amat*normedweight,dim=1)
    Amat=amat-amatavg.repeat(len(yqcd),1).view(len(yqcd),len(yqcd))\
        -amatavg.view(-1, 1).repeat(1, len(yqcd)).view(len(yqcd),len(yqcd))\
        +1/(len(yqcd)-1)*torch.sum(amatavg*normedweight)

    bmatavg=1/(len(yqcd)-2)*torch.sum(bmat*normedweight,dim=1)
    Bmat=bmat-bmatavg.repeat(len(mqcd),1).view(len(mqcd),len(mqcd))\
        -bmatavg.view(-1, 1).repeat(1, len(mqcd)).view(len(mqcd),len(mqcd))\
        +1/(len(yqcd)-1)*torch.sum(bmatavg*normedweight)

    Amat.fill_diagonal_(0)
    Bmat.fill_diagonal_(0)
    
    ABavg=torch.mean(Amat*Bmat*normedweight,dim=1)
    AAavg=torch.mean(Amat*Amat*normedweight,dim=1)
    BBavg=torch.mean(Bmat*Bmat*normedweight,dim=1)

    if(power==1):
        denom=(torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight))
        dCorr=0.
        if(denom>0):
            dCorr=(torch.mean(ABavg*normedweight))/torch.sqrt(denom)
    elif(power==2):
        dCorr=(torch.mean(ABavg*normedweight))**2/(torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight))
#    elif(power==-1):
#        dCorr=(torch.mean(ABavg*normedweight)-1/len(yqcd)*torch.mean(amat)*torch.mean(bmat))/torch.sqrt((torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight)))
    else:
        dCorr=(torch.mean(ABavg*normedweight))/torch.sqrt((torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight)))**power
    return dCorr



#######
# These are alternative routines to calculate distance covariance designed to be more memory-economical, 
# so that they can handle very large inputs.  They are slow because they are not fully vectorized. 
def dcovsq_unbiased_slow(yqcd,mqcd,normedweight):
    
# this is formula (3.1) and (3.2) in 1310.2926
# I have checked analytically that it reduces to 
# sum_{i,j}Aij Bij - 2/(n-2) sum_i ai bi + a b/(n-2)/(n-1)
# where 
# Aij = |xi-xj|
# ai = sum_j Aij
# a = sum_i ai
# and similarly for B. 
    term1=0
    term2=0
    term3a=0
    term3b=0
    for i in range(len(yqcd)):
#        amat_vec=(yqcd-yqcd[i]).abs()
#        bmat_vec=(mqcd-mqcd[i]).abs()
#        term1+=torch.sum(amat_vec*bmat_vec)
#        term2+=torch.sum(amat_vec)*torch.sum(bmat_vec)
#        term3a+=torch.sum(amat_vec)
#        term3b+=torch.sum(bmat_vec)

# use numpy instead of torch, it's more accurate
        amat_vec=np.abs(yqcd-yqcd[i])
        bmat_vec=np.abs(mqcd-mqcd[i])
        term1+=np.mean(amat_vec*bmat_vec)
        term2+=np.mean(amat_vec)*np.mean(bmat_vec)
        term3a+=np.mean(amat_vec)
        term3b+=np.mean(bmat_vec)
    
    dCovsq=term1-2/(len(yqcd)-2)*len(yqcd)*term2+len(yqcd)*term3a*term3b/((len(yqcd)-2)*(len(yqcd)-1))
    
    return dCovsq

def dcorrsq_unbiased_slow(yqcd,mqcd,normedweight):
    
    numerator=dcovsq_unbiased_slow(yqcd,mqcd,normedweight)
    denominator=np.sqrt(dcovsq_unbiased_slow(yqcd,yqcd,normedweight)*dcovsq_unbiased_slow(mqcd,mqcd,normedweight))

    dCorrsq=numerator/denominator
    
    return dCorrsq