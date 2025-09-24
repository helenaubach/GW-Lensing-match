import numpy as np

from pycbc.waveform import get_fd_waveform
from pycbc.filter import sigma, match

from pycbc.psd.analytical import aLIGOAdVO4T1800545 # O4 noise curve
from pycbc.psd.analytical import EinsteinTelescopeP1600143 # ET noise curve

import lens_library as lf


tlen = 16
f_low = 15.
nyquist = 2048


ML_arr = np.logspace(-2,4,1000) # mass of the lens, in solar masses
y_arr = np.logspace(-2, 1,1000) # source position y, dimensionless

z = 0. # redshift
ML_arr = ML_arr*(1+z) # redshifted lens mass


#LVK waveform (unlensed)
wfshort, _ = get_fd_waveform(approximant='IMRPhenomD',
                        mass1=10*(1+z), mass2=10*(1+z), # redshifted source mass
                        delta_f=1/tlen, f_lower=f_low,
                        distance=100000.) # 100 Gpc, but does not have effect
freqs = wfshort.sample_frequencies.data

wf = wfshort.copy()
wf.resize(1+nyquist*tlen)

# noise curve: 
# use aLIGOAdVO4T1800545 FOR LVK (O4)
# use EinsteinTelescopeP1600143 for ET:
psd = EinsteinTelescopeP1600143(1+nyquist*tlen, delta_f=1./tlen,
                                    low_freq_cutoff=10.) # noise curve


SNR_0 = sigma(wf, psd, low_frequency_cutoff=f_low)


def phs(arr):
    return np.unwrap(np.angle(arr))

# we apply the lensing effect to the unlensed waveform
def lensed_prediction(wf, ML, y):
    tM = 2e-5*ML # ML in solar masses
    factor = lf.hybrid_factor(freqs, tM, y) # transmission factor
    # Remove the overall time shift
    a, b = np.polyfit(freqs, phs(factor), 1)
    factor *= np.exp(1.j*a*freqs)
    # the lensed waveform in time domain is the product
    # of the unlensed waveform and the transmission factor:
    wflensed = wfshort*factor # lensed waveform
    wflensed.resize(1+nyquist*tlen)

    # one value of SNR
    SNR_lensed = sigma(wflensed, psd=psd, low_frequency_cutoff=f_low)
    mtch = 1-match(wflensed, wf, psd=psd, low_frequency_cutoff=f_low)[0] # mismatch
    return SNR_0, SNR_lensed, mtch


# we put the results in a file
result = []
for idx, ML in enumerate(ML_arr):
    print(idx)
    tmp = []
    for y in y_arr:
        pred = lensed_prediction(wf, ML, y)
        tmp.append((ML, y, pred[1]/pred[0], pred[2]))
    result.append(tmp)
grid_result = np.array(result)

data_struct = {'psd': 'EinsteinTelescopeP1600143',
                'wf_params': (10., 10., 0., 0.),
                'tlen': tlen,
                'f_low': f_low,
                'nyquist': nyquist,
                'ML_arr': ML_arr,
                'y_arr': y_arr,
                'result': grid_result
              }

np.savez('ET_10_10_ymax10_100Gpc_redshift_0_highresnew.npz', **data_struct)