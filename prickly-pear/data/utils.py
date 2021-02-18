import pandas as pd
import re
import logging

logger = logging.getLogger('Dataloading_V1')  # Logging is fun!
"""
Column headers in database: 
starred are the ones we would like to find

shot
t1
t2
* Tepedheight(keV)
error
* nepedheight1019(m3)
error
* pepedheight(kPa)
error
TepsiN09preELM(keV)
error
nepsiN09preELM1019(m3)
error
pepsiN09preELM(kPa)
error
Tepedestalwidth(psiN)
error
Nepedestalwidth(psiN)
error
pepedestalwidth(psiN)
error
TepedestalwidthRmid(cm)
error
NepedestalwidthRmid(cm)
error
pepedestalwidthRmid(cm)
error
neseparatrixfromexpdata1019(m3)
error
neseparatrixfromfit1019(m3)
error
neposition(psiN)
error
Teposition(psiN)
error
peposition(psiN)
error
nepositionRmid(m)
error
TepositionRmid(m)
error
pepositionRmid(m)
error
separatrixpositionRmid(m)
error
neinnerslope
error
Teinnerslope
error
peinnerslope
error
beta_polped(electrons)
error
nupedestal
error
rhopedestal
error
Zeff
error
BetaN(dia)
error
BetaN(MHD)
error
Ip(MA)
error
B(T)
error
R(m)
error
a(m) <<<<<<<<<<<<<<---------------- is the psi_n param (radial polodial thing)
error
averagetriangularity
error
Meff
error
P_NBI(MW)
error
P_ICRH(MW)
error
P_TOTPNBIPohmPICRHPshi(MW)
error
plasmavolume(m3)
error
q95
error
gasflowrateofmainspecies1022(es)
error
H(HD)
error
electronratefromGIM1(es)
error
electronratefromGIM2(es)
error
electronratefromGIM3(es)
error
electronratefromGIM4(es)
error
electronratefromGIM5(es)
error
electronratefromGIM6(es)
error
electronratefromGIM7(es)
error
electronratefromGIM8(es)
error
electronratefromGIM9(es)
error
electronratefromGIM10(es)
error
electronratefromGIM11(es)
error
electronratefromGIM12(es)
error
electronratefromGIM13(es)
error
electronratefromGIM14(es)
error
electronratefromGIM15(es)
error
subdivertorpressurefrombar1(10u6nbar)
error
Atomicnumberofseededimpurity
error
flowrateofseededimpurity1022(es)
error
FLAGDEUTERIUM
FLAGHYDROGEN
FLAGHDmix
FLAGHeJETC
FLAGSeeding
FLAGKicks
FLAGRMP
FLAGpellets
FLAGHRTSdatavalidated
divertorconfiguration


"""

def convert_to_dataframe(data_loc):
    """

    :param data_loc: string
    :return: (pd dataframe)  returns the data as a dataframe.
    """
    shots = []
    with open(data_loc, 'r') as infile:
        headers = infile.readline()
        for line in infile:
            shot_n = line.strip()
            shot_n = shot_n.split(',')
            shot_n = [re.sub(r'[^a-zA-Z0-9_().]', '', word) for word in shot_n]
            shots.append(shot_n)
    headers = headers.split(',')
    headers_fin = [re.sub(r'[^a-zA-Z0-9_()]', '', word) for word in headers]
    df = pd.DataFrame(shots, columns=headers_fin, dtype='float32')
    logger.info('Converted data from {} to a pandas dataframe, moving on to cleaning'.format(data_loc))
    return df


def clean_data(dataframe):
    """
    The goal of this function is to drop all NA values
    Maybe we want to also get rid of some stuff but for the moment just NA values
    :param dataframe: (pandas dataframe) to be cleaned
    :return: cleaned dataframe
    """
    dataframe_cleaned = dataframe.dropna()
    logger.info('Cleaned data, i.e., removed all na rows, moving on to normailizing')
    return dataframe_cleaned
