import numpy as np
import sys, os
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import subprocess
from scipy.ndimage import gaussian_filter1d
from scipy.special import erf
import warnings
import math

def read_fortran_fragments_out(inFile='fragments.out'):
    with open(inFile,'r') as fIn:
        lines = fIn.readlines()
    mass = []
    charge = []
    for l in lines:
        lSplit = l.replace('\n','').split()
        if 'mass' in lSplit:
            toAppend = mass
            continue
        if 'charge' in lSplit:
            toAppend = charge
            continue
        toAppend.append(np.array(lSplit).astype(float))
    
    mass = np.array(mass)
    charge = np.array(charge)
    
    m1 = mass[:mass.shape[0]//2]
    m2 = np.flip(mass[mass.shape[0]//2:],axis=0)
    
    c1 = charge[:charge.shape[0]//2]
    c2 = np.flip(charge[charge.shape[0]//2:],axis=0)
    
    m1Df = pd.DataFrame(data=m1,columns=['A','min'])
    m2Df = pd.DataFrame(data=m2,columns=['A','max'])
    
    massDf = pd.merge(m1Df,m2Df,how='outer',on='A')
    massDf[['min','max']] = massDf[['min','max']]/massDf[['min','max']].sum() * 100
    
    c1Df = pd.DataFrame(data=c1,columns=['Z','min'])
    c2Df = pd.DataFrame(data=c2,columns=['Z','max'])
    
    chargeDf = pd.merge(c1Df,c2Df,how='outer',on='Z')
    chargeDf[['min','max']] = chargeDf[['min','max']]/chargeDf[['min','max']].sum() * 100
    
    #Do expect this to throw problems later
    massDf = massDf.set_index('A')
    chargeDf = chargeDf.set_index('Z')
    
    return massDf, chargeDf

def write_fragments_out(massFName,chargeFName,dfByA,dfByZ):
    dfByA.to_csv(massFName,sep='\t',index=False)
    dfByZ.to_csv(chargeFName,sep='\t',index=False)
    return None

class FragmentYieldUtilities:
    """
    For things that could conceivably be used elsewhere
    """
    @staticmethod
    def deformed_liquid_drop_energy(A,Z,beta=None):
        """
        Computes binding energy (A,Z) using global liquid-drop parameters
        of Myers and Swiatecki, Nucl Phys 81 (1966) 1
        
        Default deformation is zero b/c microcanonical ensemble is (apparently)
        robust to deformation
        
        Agrees well with Fortran code

        Parameters
        ----------
        A : np.ndarray
            A for multiple nuclei
        Z : np.ndarray
            Z for multiple nuclei
        beta : np.ndarray, optional
            The nuclear deformation for each nucleus. The default is None.
        
        Raises
        ------
        NotImplementedError
            Deformation has not yet been benchmarked.
        
        Returns
        -------
        np.ndarray
            The energy of each nucleus.

        """
        if beta is not None and not np.array_equal(beta,np.zeros(A.shape)):
            raise NotImplementedError
        else:
            beta = np.zeros(A.shape)
            
        a1 = 15.677
        a2 = 18.56
        c3 = 0.717
        c4 = 1.21129
        ak = 1.79
        delta = 11.0
        spi = (A-2.0*Z)/A
        
        c1 = a1*(1.0-ak*spi**2)
        c2 = a2*(1.0-ak*spi**2)
        d1 = (1.0+(2.0/5.0)*beta**2)
        d2 = (1.0-(1.0/5.0)*beta**2)
        
        a13 = A**(1/3)
        
        #Checking for odd-even staggering
        delta = np.zeros(A.shape)
        for i in range(len(A)):
            if A[i] % 2 == 0:
                if Z[i] % 2  == 0:
                    delta[i] = -4.31/A[i]**(0.31)
                else:
                    delta[i] = 4.31/A[i]**(0.31)
        
        return -c1*A + c2*a13**2*d1 + c3*Z**2*d2/a13 - c4*Z**2/A + delta
    
    @staticmethod
    def coulomb_energy(A,Z,beta=np.zeros(2)):
        """
        Deformed Coulomb energy between two nuclei. From [somewhere]
        
        Agrees well with Fortran code

        Parameters
        ----------
        A : np.ndarray
            A for both nuclei
        Z : np.ndarray
            Z for both nuclei
        beta : np.ndarray, optional
            The nuclear deformation for both nucleus. The default is np.zeros(2).

        Raises
        ------
        NotImplementedError
            Deformation has not yet been benchmarked.
        
        Returns
        -------
        float
            The energy of the two-nucleus system.

        """
        if not np.array_equal(beta,np.zeros(2)):
            raise NotImplementedError
            
        for arr in [A,Z,beta]:
            assert isinstance(arr,np.ndarray)
            assert arr.shape == (2,)
        
        c1 = np.sqrt(9/(20*np.pi))
        c2 = 3/(7*np.pi)
        
        r = 1.12*A**(1/3)
        
        t1 = 1.44*Z[0]*Z[1]/(r[0]+r[1])
        
        #Deformed charge distribution - swaps beta array if fragments aren't ordered
        #as (large,small)
        if A[0] < A[1]:
            beta = np.flip(beta)
            
        v = c1*beta*r**2 + c2*(beta*r)**2
        t2 = Z[0]*Z[1]*(v[0] + v[1])*1.44/(r[0] + r[1])**3
        
        return t1 + t2
    
    @staticmethod
    def net_energy(A,Z,beta=np.zeros(2)):
        """
        Combined Coulomb + liquid drop energies
        
        Agrees well with Fortran code

        Parameters
        ----------
        A : np.ndarray
            A for both nuclei
        Z : np.ndarray
            Z for both nuclei
        beta : np.ndarray, optional
            The nuclear deformation for both nucleus. The default is np.zeros(2).
            
        Raises
        ------
        NotImplementedError
            Deformation has not yet been benchmarked.

        Returns
        -------
        float
            The energy of the two-nucleus system.

        """
        if not np.array_equal(beta,np.zeros(2)):
            raise NotImplementedError
        
        for arr in [A,Z,beta]:
            assert isinstance(arr,np.ndarray)
            assert arr.shape == (2,)
            
        ldEneg = FragmentYieldUtilities.deformed_liquid_drop_energy(A,Z,beta)
        
        coul = FragmentYieldUtilities.coulomb_energy(A,Z,beta)
        
        return coul + ldEneg.sum()
    
    @staticmethod
    def microcanonical_probability(A,eneg):
        """
        The microcanonical ensemble probability for a given configuration from
        [cite]

        Parameters
        ----------
        A : np.ndarray
            A for both nuclei
        eneg : float
            The energy of the configuration

        Returns
        -------
        float
            The probability of the configuration

        """
        assert eneg > 0
        
        a = A/10 #Level density parameter
        
        #Handling of square-root term to avoid overflow
        term1 = A[0]**4/(A[0]**(5/3) + A[1]**(5/3))**(3/2)
        term2 = A[1]**4/(A[0]+A[1])**(3/2)
        term3 = np.sqrt(a[0]*a[1]/(a[0]+a[1])**5)
        sqrtTerm = term1*term2*term3
            
        line2 = 1 - 1/(2*np.sqrt((a[0]+a[1])*eneg))
        line2 *= eneg**(9/4)
        
        line3 = np.exp(2*np.sqrt((a[0]+a[1])*eneg))
        
        return sqrtTerm * line2 * line3

class FragmentYields:
    def __init__(self,nFrags,zFrags,nNeck,zNeck,nErr=1,zErr=1,beta=None):
        self.nFrags = nFrags
        self.zFrags = zFrags
        self.nNeck = nNeck
        self.zNeck = zNeck
        self.nErr = nErr
        self.zErr = zErr
        
        assert self.nFrags[0] >= self.nFrags[1]
        assert self.zFrags[0] >= self.zFrags[1]
        
        if beta is not None:
            raise NotImplementedError
        else:
            self.beta = np.zeros(2)
            
    def get_partitions(self,N,Z,nNeck,zNeck):
        #Large and small fragments
        neutronPartitions = np.zeros((nNeck+1,2))
        protonPartitions = np.zeros((zNeck+1,2))
        
        for nIter in range(nNeck+1):
            neutronPartitions[nIter,0] = N[0] + nIter
            neutronPartitions[nIter,1] = N[1] + (nNeck-nIter)
            
        for zIter in range(zNeck+1):
            protonPartitions[zIter,0] = Z[0] + zIter
            protonPartitions[zIter,1] = Z[1] + (zNeck-zIter)
        
        return neutronPartitions, protonPartitions
    
    def get_raw_probs(self,N,Z,nNeck,zNeck):
        aven = 40
        
        neutronPartitions, protonPartitions = self.get_partitions(N,Z,nNeck,zNeck)
        
        nPossibleNeutrons = int(neutronPartitions.max()+1)
        nPossibleProtons = int(protonPartitions.max()+1)
        partitionShape = (nPossibleNeutrons,nPossibleProtons)
        
        enegArr = np.zeros(partitionShape)
        
        for nIter in range(neutronPartitions.shape[0]):
            for zIter in range(protonPartitions.shape[0]):
                #Yes, this order matters
                nLargeIdx, nSmallIdx = neutronPartitions[nIter].astype(int)
                zLargeIdx, zSmallIdx = protonPartitions[zIter].astype(int)
                
                #Note that the energy is symmetric if we swap *both* N and Z in
                #the partitions
                enegArr[nLargeIdx,zLargeIdx] = \
                    FragmentYieldUtilities.net_energy(neutronPartitions[nIter]+protonPartitions[zIter],
                                                      protonPartitions[zIter],
                                                      beta=self.beta)
                enegArr[nSmallIdx,zSmallIdx] = enegArr[nLargeIdx,zLargeIdx]
                enegArr[nLargeIdx,zSmallIdx] = \
                    FragmentYieldUtilities.net_energy(neutronPartitions[nIter]+np.flip(protonPartitions[zIter]),
                                                      np.flip(protonPartitions[zIter]),
                                                      beta=self.beta)
                enegArr[nSmallIdx,zLargeIdx] = enegArr[nLargeIdx,zSmallIdx]
                
        e0 = enegArr.min()
        enegArr[np.where(enegArr!=0)] = e0 - enegArr[np.where(enegArr!=0)] + aven
        
        #I trust that this array is created and arranged correctly
        probArr = np.zeros(partitionShape)
        for nIter in range(neutronPartitions.shape[0]):
            for zIter in range(protonPartitions.shape[0]):
                n1, n2 = neutronPartitions[nIter].astype(int)
                z1, z2 = protonPartitions[zIter].astype(int)
                
                if enegArr[n1,z1] > 0:
                    probArr[n1,z1] = \
                        FragmentYieldUtilities.microcanonical_probability(neutronPartitions[nIter]+protonPartitions[zIter],
                                                                          enegArr[n1,z1])
                    probArr[n2,z2] = probArr[n1,z1]
        
        probArr = probArr/probArr.max()
        
        #Trust the output of this
        return probArr
    
    def get_dist_by_A(self,N,Z,nNeck,zNeck,probArr,sigmaA):
        Atot = N.sum() + Z.sum() + nNeck + zNeck
        
        neutronPartitions, protonPartitions = self.get_partitions(N,Z,nNeck,zNeck)       
        
        n1Arr = np.zeros((nNeck+1,zNeck+1))
        z1Arr = np.zeros((nNeck+1,zNeck+1))
        
        n2Arr = np.zeros((nNeck+1,zNeck+1))
        z2Arr = np.zeros((nNeck+1,zNeck+1))
        
        for nIter in range(nNeck+1):
            n1Arr[nIter], n2Arr[nIter] = neutronPartitions[nIter]
        for zIter in range(zNeck+1):
            z1Arr[:,zIter], z2Arr[:,zIter] = protonPartitions[zIter]
        
        a1Arr = n1Arr + z1Arr
        a2Arr = n2Arr + z2Arr
        
        n1Arr = n1Arr.astype(int)
        z1Arr = z1Arr.astype(int)
        probArr = probArr[n1Arr[0,0]:n1Arr[-1,0]+1,
                          z1Arr[0,0]:z1Arr[0,-1]+1]
                
        #Trust this process for symm and asymm fission
        uniqueAVals = np.arange(a2Arr.min(),a1Arr.max()+1)
        dfByA = pd.DataFrame(index=range(len(uniqueAVals)),columns=['A','prob'],
                             data=np.zeros((len(uniqueAVals),2)),dtype=float)
        for (i,A) in enumerate(uniqueAVals):
            dfByA.iloc[i]['A'] = A
            if np.any(a1Arr==A):
                idx = np.where(a1Arr==A)
                dfByA.iloc[i]['prob'] += np.sum(probArr[idx])
            if np.any(a2Arr==A) and A != Atot/2:
                idx = np.where(a2Arr==A)
                dfByA.iloc[i]['prob'] += np.sum(probArr[idx])
            
        dfPad1 = pd.DataFrame(columns=['A','prob'],index=range(3*sigmaA),
                              dtype=(int,float))
        dfPad1['A'] = np.arange(dfByA['A'].min()-3*sigmaA,dfByA['A'].min())
        dfPad1['prob'] = 0
        
        dfPad2 = pd.DataFrame(columns=['A','prob'],index=range(3*sigmaA),
                              dtype=(int,float))
        dfPad2['A'] = np.arange(dfByA['A'].max()+1,dfByA['A'].max()+1+3*sigmaA)
        dfPad2['prob'] = 0
        
        dfByA = pd.concat([dfByA,dfPad1,dfPad2],ignore_index=True)
        dfByA = dfByA.sort_values('A',ignore_index=True)
        
        return dfByA
    
    def get_dist_by_Z(self,N,Z,nNeck,zNeck,probArr,sigmaZ):
        Ztot = Z.sum() + zNeck
        
        neutronPartitions, protonPartitions = self.get_partitions(N,Z,nNeck,zNeck)
        
        n1Arr = np.zeros((nNeck+1,zNeck+1))
        z1Arr = np.zeros((nNeck+1,zNeck+1))
        
        n2Arr = np.zeros((nNeck+1,zNeck+1))
        z2Arr = np.zeros((nNeck+1,zNeck+1))
        
        for nIter in range(nNeck+1):
            n1Arr[nIter], n2Arr[nIter] = neutronPartitions[nIter]
        for zIter in range(zNeck+1):
            z1Arr[:,zIter], z2Arr[:,zIter] = protonPartitions[zIter]
            
        n1Arr = n1Arr.astype(int)
        z1Arr = z1Arr.astype(int)
        probArr = probArr[n1Arr[0,0]:n1Arr[-1,0]+1,
                          z1Arr[0,0]:z1Arr[0,-1]+1]
        
        uniqueZVals = np.arange(z2Arr.min(),z1Arr.max()+1)
        dfByZ = pd.DataFrame(index=range(len(uniqueZVals)),columns=['Z','prob'],
                             data=np.zeros((len(uniqueZVals),2)),dtype=float)
        for (i,Z) in enumerate(uniqueZVals):
            dfByZ.iloc[i]['Z'] = Z
            if np.any(z1Arr==Z):
                idx = np.where(z1Arr==Z)
                dfByZ.iloc[i]['prob'] += np.sum(probArr[idx])
            if np.any(z2Arr==Z) and Z != Ztot/2:
                idx = np.where(z2Arr==Z)
                dfByZ.iloc[i]['prob'] += np.sum(probArr[idx])
                
        dfPad1 = pd.DataFrame(columns=['Z','prob'],index=range(4*sigmaZ),
                              dtype=(int,float))
        dfPad1['Z'] = np.arange(dfByZ['Z'].min()-4*sigmaZ,dfByZ['Z'].min())
        dfPad1['prob'] = 0
        
        dfPad2 = pd.DataFrame(columns=['Z','prob'],index=range(4*sigmaZ),
                              dtype=(int,float))
        dfPad2['Z'] = np.arange(dfByZ['Z'].max()+1,dfByZ['Z'].max()+1+4*sigmaZ)
        dfPad2['prob'] = 0
        
        dfByZ = pd.concat([dfByZ,dfPad1,dfPad2],ignore_index=True)
        dfByZ = dfByZ.sort_values('Z',ignore_index=True)
        return dfByZ
    
    def jhilam_convolution_A(self,dfByA,sigmaA):
        Atot = self.nFrags.sum() + self.zFrags.sum() + self.nNeck + self.zNeck
        
        for col in ['min','max']:
            dfByA['conv-'+col] = 0
            
            #Handles symmetric and asymmetric differently
            if self.nFrags[0] == self.nFrags[1] and self.zFrags[0] == self.zFrags[1]:
                sumInds = np.arange(1,Atot)
            else:
                sumInds = np.arange(Atot//2,Atot)
            
            for A in sumInds:
                if A not in dfByA.index:
                    continue
                for Atilde in sumInds:
                    if Atilde not in dfByA.index:
                        continue
                    x1 = (A - Atilde + 0.5)/(np.sqrt(2)*sigmaA)
                    x2 = (A - Atilde - 0.5)/(np.sqrt(2)*sigmaA)
                    v1 = erf(x1)/2
                    v2 = erf(x2)/2
                    ya1 = dfByA.loc[Atilde,col]*(v1-v2)
                    dfByA.loc[A,'conv-'+col] += ya1
                
            #Symmetrizing the asymmetric case
            if self.nFrags[0] != self.nFrags[1] or self.zFrags[0] != self.zFrags[1]:
                for i in np.arange(dfByA.index[0],Atot//2):
                    dfByA.loc[i,'conv-'+col] = dfByA.loc[Atot-i,'conv-'+col]
                    
        del dfByA['max']
        del dfByA['min']
        dfByA = dfByA.rename(columns={'conv-min':'min','conv-max':'max'})
                    
        return dfByA
    
    def jhilam_convolution_Z(self,dfByZ,sigmaZ,sigmaZFinal):
        warnings.warn('This method should not be trusted - Jhilam\'s code has a bug that hasn\'t been reproduced here')
        
        Ztot = self.zFrags.sum() + self.zNeck
        
        for col in ['min','max']:
            dfByZ['conv-intermediate'] = 0
            dfByZ['conv-'+col] = 0
            
            #Jhilam attempts to handle symmetric and asymmetric differently
            if self.nFrags[0] == self.nFrags[1] and self.zFrags[0] == self.zFrags[1]:
                sumInds = np.arange(1,Ztot,2)
                fullInds = np.arange(1,Ztot)
            else:
                sumInds = np.arange(Ztot//2,Ztot,2)
                fullInds = np.arange(Ztot//2,Ztot)
            
            #Odd loop
            for Z in sumInds:
                if Z not in dfByZ.index:
                    continue
                for Ztilde in sumInds:
                    if Ztilde not in dfByZ.index:
                        continue
                    x1 = (Z - Ztilde + 0.5)/(np.sqrt(2)*sigmaZ)
                    x2 = (Z - Ztilde - 0.5)/(np.sqrt(2)*sigmaZ)
                    v1 = erf(x1)/2
                    v2 = erf(x2)/2
                    ya1 = dfByZ.loc[Ztilde,col]*(v1-v2)
                    dfByZ.loc[Z,'conv-intermediate'] += ya1
            
            #Even loop
            for Z in sumInds+1:
                if Z not in dfByZ.index:
                    continue
                for Ztilde in sumInds+1:
                    if Ztilde not in dfByZ.index:
                        continue
                    x1 = (Z - Ztilde + 0.5)/(np.sqrt(2)*sigmaZ)
                    x2 = (Z - Ztilde - 0.5)/(np.sqrt(2)*sigmaZ)
                    v1 = erf(x1)/2
                    v2 = erf(x2)/2
                    ya1 = dfByZ.loc[Ztilde,col]*(v1-v2)
                    dfByZ.loc[Z,'conv-intermediate'] += ya1
                
            #Final convolution
            for Z in fullInds:
                if Z not in dfByZ.index:
                    continue
                for Ztilde in fullInds:
                    if Ztilde not in dfByZ.index:
                        continue
                    x1 = (Z - Ztilde + 0.5)/(np.sqrt(2)*sigmaZFinal)
                    x2 = (Z - Ztilde - 0.5)/(np.sqrt(2)*sigmaZFinal)
                    v1 = erf(x1)/2
                    v2 = erf(x2)/2
                    ya1 = dfByZ.loc[Ztilde,'conv-intermediate']*(v1-v2)
                    dfByZ.loc[Z,'conv-'+col] += ya1
                    
            del dfByZ['conv-intermediate']
            
            #Symmetrizing the asymmetric case
            if self.nFrags[0] != self.nFrags[1] or self.zFrags[0] != self.zFrags[1]:
                for i in np.arange(dfByZ.index[0],Ztot//2):
                    dfByZ.loc[i,'conv-'+col] = dfByZ.loc[Ztot-i,'conv-'+col]
            
        del dfByZ['max']
        del dfByZ['min']
        dfByZ = dfByZ.rename(columns={'conv-min':'min','conv-max':'max'})
        
        return dfByZ
        
    def convolution_python_new(self,dfByA,dfByZ,sigmaA,sigmaZ,sigmaZFinal):
        dfByA['conv-min'] = gaussian_filter1d(dfByA['min'].to_numpy(),sigmaA)
        dfByA['conv-max'] = gaussian_filter1d(dfByA['max'].to_numpy(),sigmaA)
        
        #Don't quite trust this yet
        evenInds = dfByZ.index % 2 == 0
        oddInds = dfByZ.index % 2 != 0
        
        for col in ['min','max']:
            dfByZ['even'] = 0
            dfByZ.loc[evenInds,'even'] = dfByZ.loc[evenInds,col]
            dfByZ['even-conv'] = gaussian_filter1d(dfByZ['even'],sigmaZ)
            
            dfByZ['odd'] = 0
            dfByZ.loc[oddInds,'odd'] = dfByZ.loc[oddInds,col]
            dfByZ['odd-conv'] = gaussian_filter1d(dfByZ['odd'],sigmaZ)
            
            dfByZ.loc[evenInds,'pre-conv-'+col] = dfByZ.loc[evenInds,'even-conv']
            dfByZ.loc[oddInds,'pre-conv-'+col] = dfByZ.loc[oddInds,'odd-conv']
            
            dfByZ['conv-'+col] = gaussian_filter1d(dfByZ['pre-conv-'+col],sigmaZFinal)
            
            del dfByZ['pre-conv-'+col]
            del dfByZ['odd-conv']
            del dfByZ['even-conv']
            
        del dfByA['max']
        del dfByA['min']
        dfByA = dfByA.rename(columns={'conv-min':'min','conv-max':'max'})
        
        del dfByZ['max']
        del dfByZ['min']
        dfByZ = dfByZ.rename(columns={'conv-min':'min','conv-max':'max'})
        
        return dfByA, dfByZ
    
    def jhilam_fortran(self):
        with open("frag-loc.in","w") as f:
            f.write("%d %d %d %d %d %d 0.0 0"%(self.nFrags.max(),self.zFrags.max(),
                                               self.nFrags.min(),self.zFrags.min(),
                                               self.nNeck,self.zNeck))
            
            f.write("\n")
            
        #Calling Jhilam's code. Technically is maybe a vulnerability b/c of 
        #'shell=True'
        # if not os.path.isfile('compute-distribution'):
        if os.path.isfile('fragments-dis.f'):
            subprocess.run('gfortran fragments-dis.f -o compute-distribution',
                           shell=True)
        else:
            raise ValueError("Fragment yield calculation unavailable - missing executable")
        
        subprocess.run('./compute-distribution')
        
        #May have been compiled on a different OS
        # try:
            
        # except OSError:
        #     os.remove('compute-distribution')
        #     subprocess.run('gfortran fragments-dis.f -o compute-distribution',
        #                    shell=True)
        #     subprocess.run('./compute-distribution',shell=True)
            
        dfByA, dfByZ = read_fortran_fragments_out()
            
        return dfByA, dfByZ
            
    def __call__(self,mode='python-new',sigmaA=3,sigmaZ=2,sigmaZFinal=0.5):
        availableModes = ['python-new','python-jhilam','fortran-jhilam']
        if mode not in availableModes:
            raise ValueError('Convolution mode must be one of ',availableModes)
            
        if mode in ['python-new','python-jhilam']:
            listOfDfsByA = []
            listOfDfsByZ = []
            
            for n1Iter in range(-self.nErr,self.nErr+1):
                for z1Iter in range(-self.zErr,self.zErr+1):
                    for n2Iter in range(-self.nErr,self.nErr+1):
                        for z2Iter in range(-self.zErr,self.zErr+1):
                            nFrags = self.nFrags.copy() + np.array([n1Iter,n2Iter])
                            zFrags = self.zFrags.copy() + np.array([z1Iter,z2Iter])
                            nNeck = self.nNeck - n1Iter - n2Iter
                            zNeck = self.zNeck - z1Iter - z2Iter
                            
                            if nNeck <= 0 or zNeck <= 0:
                                warnings.warn('Negative number of neck nucleons in error bars')
                                continue
                                                    
                            probArr = self.get_raw_probs(nFrags,zFrags,nNeck,zNeck)
                            
                            dfByA = self.get_dist_by_A(nFrags,zFrags,nNeck,zNeck,probArr,sigmaA)
                            # print(dfByA)
                            # sys.exit()
                            listOfDfsByA.append(dfByA)
                            
                            dfByZ = self.get_dist_by_Z(nFrags,zFrags,nNeck,zNeck,probArr,sigmaZ)
                            listOfDfsByZ.append(dfByZ)
            
            dfByA = listOfDfsByA[0]
            for (dfIter,df) in enumerate(listOfDfsByA[1:]):
                dfByA = pd.merge(dfByA,df,how='outer',on='A',suffixes=[None,str(dfIter+1)])
                
            dfByZ = listOfDfsByZ[0]
            for (dfIter,df) in enumerate(listOfDfsByZ[1:]):
                dfByZ = pd.merge(dfByZ,df,how='outer',on='Z',suffixes=[None,str(dfIter+1)])
                
            dfByA[dfByA < 10**(-7)] = 0
            dfByZ[dfByZ < 10**(-7)] = 0
            
            dfByA = dfByA.set_index('A')
            dfByA['max'] = dfByA.max(axis=1)
            dfByA['min'] = dfByA.min(axis=1)
            # print(dfByA)
            
            dfByZ = dfByZ.set_index('Z')
            dfByZ['max'] = dfByZ.max(axis=1)
            dfByZ['min'] = dfByZ.min(axis=1)
            
            if mode == 'python-new':
                dfByA, dfByZ = self.convolution_python_new(dfByA,dfByZ,sigmaA,
                                                           sigmaZ,sigmaZFinal)
            elif mode == 'python-jhilam':
                dfByA = self.jhilam_convolution_A(dfByA,sigmaA)
                dfByZ = self.jhilam_convolution_Z(dfByZ,sigmaZ,sigmaZFinal)
            
        else:
            dfByA, dfByZ = self.jhilam_fortran()
            
        dfByA = dfByA/dfByA.sum() * 100
        dfByZ = dfByZ/dfByZ.sum() * 100
            
        return dfByA, dfByZ

def reflect_df(df):
    reflectedDf = df.copy()
    reflectedDf["r"] = -1*reflectedDf["r"]
    
    df = pd.merge(df,reflectedDf,how="outer")
    df = df.sort_values(["r","z"],ignore_index=True)
    
    return df

def plot_localization(df,col):
    assert col in ['rhoN/2', 'rhoP/2', 'localizarionN', 'localizationP']
    
    df = reflect_df(df)
    
    unVals = [np.unique(df["r"]),np.unique(df["z"])]
    shp = [len(u) for u in unVals]
    
    rr, zz = df["r"].to_numpy().reshape(shp), df["z"].to_numpy().reshape(shp)
    arrToPlot = df[col].to_numpy().reshape(shp)
    
    fig, ax = plt.subplots()
    cf = ax.contourf(rr,zz,arrToPlot,cmap="Spectral_r",levels=50)
    ax.set(xlabel=r"$r$",ylabel=r"$z$",title=col,
           xlim=xlims,ylim=ylims
           )
    
    plt.colorbar(cf,ax=ax)
    return fig, ax
    
def get_minima(df,col,plot=False):
    assert col in ['rhoN/2', 'rhoP/2', 'localizarionN', 'localizationP']
    
    minRVal = np.min(df["r"]) #Typically doesn't evaluate at r=0
    axisDf = df[df["r"]==minRVal]
    
    interp_loc = scipy.interpolate.interp1d(axisDf["z"],axisDf[col],
                                            kind='cubic')
    denseZ = np.linspace(axisDf['z'].min(),axisDf['z'].max(),500)
    denseLocalization = interp_loc(denseZ)
    idx = scipy.signal.argrelextrema(denseLocalization,np.less)
    
    #Excludes points of ~zero density that happen to be minima
    idx = idx[0][np.argwhere(denseLocalization[idx]>0.01)[:,0]]
    
    idx2 = scipy.signal.argrelextrema(denseLocalization,np.greater)
    idx2 = idx2[0][np.argwhere(denseLocalization[idx2]>0.01)[:,0]]
    
    idx = np.concatenate((idx,idx2))
    idx.sort()
    
    if plot:
        fig, ax = plt.subplots()
        ax.plot(denseZ,denseLocalization)
        for i in idx:
            ax.axvline(denseZ[i],color="black")
    
    return denseZ[idx], denseLocalization[idx]

def experimental_get_prefragment_locs(df,col,allExtremaZVals=None,close=True):
    """
    Note that, while this finds centers of prefragments without extrema values,
    they tend to be inconsistent with those from the old Fortran code (in which
    prefragment centers were selected from a list of extremal values)
    """
    fig, ax = plot_localization(df,col)
    
    df = reflect_df(df)
    
    unVals = [np.unique(df["r"]),np.unique(df["z"])]
    shp = [len(u) for u in unVals]
    
    rr, zz = df["r"].to_numpy().reshape(shp), df["z"].to_numpy().reshape(shp)
    arrToPlot = df[col].to_numpy().reshape(shp)
    
    test = ax.contour(rr,zz,arrToPlot,levels=[0.1,],colors=["black",])
    segs = test.allsegs[0][0]
    
    oneSidedSegs = segs[segs[:,0]>0]
    segs_interp = scipy.interpolate.interp1d(oneSidedSegs[:,1],oneSidedSegs[:,0],
                                             kind='cubic')
    denseZ = np.linspace(oneSidedSegs[:,1].min(),oneSidedSegs[:,1].max(),500)
    denseSegs = segs_interp(denseZ)
    
    otherInds = scipy.signal.argrelextrema(denseSegs,np.greater)[0]
    
    lessInds = np.where(denseZ<denseZ[otherInds][0])
    ax.scatter(denseSegs[lessInds],denseZ[lessInds],color="lime")
    
    if allExtremaZVals is not None:
        closestExtremaIdx = np.argmin(np.abs(allExtremaZVals - denseZ[lessInds][-1]))
        lowerZVal = allExtremaZVals[closestExtremaIdx]
    else:
        lowerZVal = denseZ[lessInds][-1]
    
    gtrInds = np.where(denseZ>denseZ[otherInds][-1])
    ax.scatter(denseSegs[gtrInds],denseZ[gtrInds],color="lime")
    
    if allExtremaZVals is not None:
        closestExtremaIdx = np.argmin(np.abs(allExtremaZVals - denseZ[gtrInds][0]))
        upperZVal = allExtremaZVals[closestExtremaIdx]
    else:
        upperZVal = denseZ[gtrInds][0]
    
    if allExtremaZVals is not None:
        for (zIter,z) in enumerate(allExtremaZVals):
            ax.axhline(z,color="black",ls="--",)
            text = '%.3f'%z
            if zIter % 2 == 0:
                ax.text(xlims[0], z, text, #transform=ax.transAxes, 
                        fontsize=6,
                        verticalalignment='center',horizontalalignment="left",
                        bbox=textboxProps,zorder=100)
            else:
                ax.text(xlims[0]+2, z, text, #transform=ax.transAxes, 
                        fontsize=6,
                        verticalalignment='center',horizontalalignment="left",
                        bbox=textboxProps,zorder=100)
        
    ax.scatter(0,lowerZVal,marker="x",color="lime")
    ax.scatter(0,upperZVal,marker="x",color="lime")
    
    if close:
        plt.close(fig)
    return upperZVal, lowerZVal

def integrate_single_prefragment(df,col,z,upDown):
    """
    Integral over density: 2*pi for all angle, x2 for fraction of fragment on
    the interior of the nucleus, x2 because the densities are given over 2 for
    some reason
    """
    df = reflect_df(df)
    
    unVals = [np.unique(df["r"]),np.unique(df["z"])]
    shp = [len(u) for u in unVals]
    arr = df[col].to_numpy().reshape(shp)
    
    interp_func = scipy.interpolate.RegularGridInterpolator(unVals,arr,method="cubic")
    
    def func_to_integrate(z,r):
        return r*interp_func(np.array((r,z)).reshape((-1,2)))
    
    if upDown == "up":
        integVal = scipy.integrate.dblquad(func_to_integrate,0,unVals[0].max(),
                                           z,unVals[1].max(),epsabs=10**(-1))
    elif upDown == "down":
        integVal = scipy.integrate.dblquad(func_to_integrate,0,unVals[0].max(),
                                           unVals[1].min(),z,epsabs=10**(-1))
    
    return 8*np.pi*integVal[0]

def get_prefragments_and_neck_numbers(df,neutronCenters,protonCenters,A,Z):
    """
    centers are the z locations of the fragments
    
    """
    N = A - Z
    
    neutronFrags = np.zeros(2,dtype=int)
    neutronFrags[0] = round(integrate_single_prefragment(df,"rhoN/2",neutronCenters[0],"up"))
    neutronFrags[1] = round(integrate_single_prefragment(df,"rhoN/2",neutronCenters[1],"down"))
    
    protonFrags = np.zeros(2,dtype=int)
    protonFrags[0] = round(integrate_single_prefragment(df,"rhoP/2",protonCenters[0],"up"))
    protonFrags[1] = round(integrate_single_prefragment(df,"rhoP/2",protonCenters[1],"down"))
    
    nNeutronNeck = N - np.sum(neutronFrags)
    nProtonNeck = Z - np.sum(protonFrags)
    
    return neutronFrags, nNeutronNeck, protonFrags, nProtonNeck

def restore_neck_nucleons(frags,neck):
    """
    When you have zero/negative neck nucleons, pulls evenly from each prefragment
    and adds to the neck. The neck will have either 2 or 3 nucleons, depending on
    whether the neck has an even or odd (nonpositive) number of nucleons
    """
    if neck <= 1:
        nToTakeFromEach = math.ceil((2-neck)/2)
        frags -= nToTakeFromEach
        neck += 2*nToTakeFromEach
    
    return frags, neck

def plot_yields(df,fig=None,ax=None,color=None,fillKWargs={}):
    if fig is None:
        fig, ax = plt.subplots()
        
    if color is None:
        lns = ax.plot(df.index,df['min'])
        color = lns[0].get_color()
    else:
        ax.plot(df.index,df['min'],color=color)
    ax.plot(df.index,df['max'],color=color)
    ax.fill_between(df.index,df['min'],df['max'],**fillKWargs)
    
    return fig, ax

def yields_agreement_test(nFrags,zFrags,nNeck,zNeck):
    """
    For demonstrating agreement between the three methods
    """
    # nFrags = np.array([76, 76])
    # zFrags = np.array([47, 47])
    # nNeck = 2
    # zNeck = 6

    # nFrags = np.array([81, 50])
    # zFrags = np.array([47, 44])
    # nNeck = 23
    # zNeck = 9

    fragYields = FragmentYields(nFrags,zFrags,nNeck,zNeck)
    dfByAToReturn, dfByZToReturn = fragYields()

    color = 'black'
    fig, ax = plot_yields(dfByAToReturn,color=color,
                          fillKWargs={'edgecolor':color,
                                      'zorder':100,'hatch':'/','facecolor':'none',
                                      'label':'New-Python'})
    figZ, axZ = plot_yields(dfByZToReturn,color=color,
                            fillKWargs={'edgecolor':color,
                                        'zorder':100,'hatch':'/','facecolor':'none',
                                        'label':'New-Python'})

    fragYields = FragmentYields(nFrags,zFrags,nNeck,zNeck)
    dfByA, dfByZ = fragYields(mode='python-jhilam')

    color = 'cyan'
    plot_yields(dfByA,fig,ax,
                fillKWargs={'edgecolor':color,
                            'zorder':100,'hatch':'-','facecolor':'none',
                            'label':'Jhilam-Python'})
    plot_yields(dfByZ,figZ,axZ,
                fillKWargs={'edgecolor':color,
                            'zorder':100,'hatch':'-','facecolor':'none',
                            'label':'Jhilam-Python'})

    fragYields = FragmentYields(nFrags,zFrags,nNeck,zNeck)
    dfByA, dfByZ = fragYields(mode='fortran-jhilam')

    color = 'magenta'
    plot_yields(dfByA,fig,ax,
                fillKWargs={'edgecolor':color,
                            'zorder':100,'hatch':'\\','facecolor':'none',
                            'label':'Jhilam-Fortran'})
    plot_yields(dfByZ,figZ,axZ,
                fillKWargs={'edgecolor':color,
                            'zorder':100,'hatch':'\\','facecolor':'none',
                            'label':'Jhilam-Fortran'})

    ax.legend()
    axZ.legend()
    return dfByAToReturn, dfByZToReturn

    
textboxProps = {"boxstyle":'round', "facecolor":'white', "alpha":1}
#Axis limits for plotting NLFs
xlims = (-20,20)
ylims = (-20,20)

mode = 'asymm'
Eshift = '0.0'
center_method = 'extent'
locFiles = [f'localization_{mode}_Eshift_{Eshift}.dat',]

# neutronLocs = [7.56,-5.15]
# protonLocs = [6.58,-5.65]
A = 258
Z = 100

locs = []

for f in locFiles:
    df = pd.read_csv(f,sep="\s+")
    
    for key in ['localizarionN','localizationP']:
        minZVals, minLocVals = get_minima(df,key,plot=True)
        print('Possible locs '+key+':')
        print(minZVals)
        if center_method == 'lines':    
            locs.append(experimental_get_prefragment_locs(df,key,minZVals,close=False))
        elif center_method == 'extent': 
            locs.append(experimental_get_prefragment_locs(df,key,close=False))
    
    print('Neutron locs: ',locs[0])
    print('Proton locs: ',locs[1])
    
    print(50*'=')
    nFrags, nNeck, zFrags, zNeck = \
        get_prefragments_and_neck_numbers(df,locs[0],locs[1],A,Z)
        
    nFrags, nNeck = restore_neck_nucleons(nFrags,nNeck)
    zFrags, zNeck = restore_neck_nucleons(zFrags,zNeck)
    
    print(nFrags, nNeck, zFrags, zNeck)
    
    if nFrags[0] < nFrags[1]:
        if zFrags[0] <= zFrags[1]:
            nFrags = np.flip(nFrags)
            zFrags = np.flip(zFrags)
        else:
            warnings.warn('Possible unusual ordering of prefragment nucleons')

    #Recommend running in this mode for a while to catch any issues that I may
    #have missed
    dfByA, dfByZ = yields_agreement_test(nFrags,zFrags,nNeck,zNeck)
    dfByA[['min','max']].to_csv(f'frag_result/mass_{mode}_Eshift_{Eshift}_{center_method}.dat',sep='\t')
    dfByZ[['min','max']].to_csv(f'frag_result/charge_{mode}_Eshift_{Eshift}_{center_method}.dat',sep='\t')
    
    with open(f'frag_result/{mode}_Eshift_{Eshift}_{center_method}_centers.txt', 'w') as f:
        f.write('Neutron\n')
        f.write(f'{locs[0]} \n')
        f.write('Proton\n')
        f.write(f'{locs[1]}')
    
    #After we trust the code, run in this mode to just use my results
    # fragYields = FragmentYields(nFrags,zFrags,nNeck,zNeck)
    # dfByA, dfByZ = fragYields()
    

