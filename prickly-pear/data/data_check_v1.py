"""
Basically we read line by line, to some fancy regex filtering so that only numbers appear
then we create a big list with all shots,and finally insert it into dataframe
This is so that we can work with it a bit easier.
"""
import re
import pandas as pd


file_loc = 'andreas-data.dat'
file_comma = 'daten-comma.txt'


def convert_to_dataframe(data_loc):
    shots = []
    with open(data_loc, 'r') as infile:
        headers = infile.readline()
        for line in infile:
            shot_n = line.strip().split(',') # first split shot data into a list by comma sep. objects
            shot_n = [re.sub(r'[^a-zA-Z0-9_().]', '', word) for word in shot_n] # regex filtering of tabs and spaces
            shots.append(shot_n) # add to mega list
    headers = headers.split(',')
    headers_fin = [re.sub(r'[^a-zA-Z0-9_()]', '', word) for word in headers]

    df = pd.DataFrame(shots, columns=headers_fin)
    return df


df = convert_to_dataframe(file_comma)
target_labels = ['Tepedheight(keV)', 'nepedheight1019(m3)']

for col in df.columns:
    print(col)


input_labels = ['BetaN(MHD)', 'Ip(MA)', 'R(m)', 'B(T)', 'a(m)', 'averagetriangularity', 'plasmavolume(m3)', 'q95', 'P_TOTPNBIPohmPICRHPshi(MW)']
targets = df[target_labels]
inputs = df[input_labels]



print(targets)
print(inputs)