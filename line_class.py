import numpy as np

from scipy.interpolate import splrep, splev
from scipy.signal import gaussian

def _gauss_model(x, amp, sig2, wvl):
    return amp * np.exp(-0.5 * (x - wvl) ** 2 / sig2)

class AbsorptionLine:
    def __init__(self, element=None, wvl_center=None, wvl_step=None, wvl_width=None,
                        read=False, file_path='', filename=None):
        self.C0 = 299792458.  # speed of light in m/s ~3e8
        if read and filename is not None:
            # TODO - parse settings from filename
            self.filename = file_path + filename
            filename_split = '.'.join(filename.split('.')[:-1]).split('_')
            self.element = filename_split[0]
            self.wvl_center = float(filename_split[filename_split.index('center')+1])
            self.wvl_step = float(filename_split[filename_split.index('step') + 1])
            self.wvl_width = float(filename_split[filename_split.index('width') + 1])
            # read data
            self.data = self.read_data()
        else:
            self.data = None
            # TODO - checks for input parameters and its validity
            self.element = element
            self.wvl_center = wvl_center
            self.wvl_step = wvl_step
            self.wvl_width = wvl_width
            # determine name of text file
            self.filename = file_path + element+'_center_{:4.3f}_step_{:0.3f}_width_{:3.3f}.txt'\
                                        .format(wvl_center, wvl_step, wvl_width)
            # create empty file that will store all data about spectral lines
            self._empty_txt()
        self.wvl_steps = np.round(self.wvl_width / self.wvl_step)
        self.wvl_target = self._calculate_wvl_target()
        self.wvl_start = np.min(self.wvl_target)
        self.wvl_end = np.max(self.wvl_target)
        # RV conversion
        self.rv_target = self._derive_rv()
        # 
        self.nan_row = np.zeros(shape=(len(self.wvl_target)), dtype='float16')
        self.nan_row.fill(np.NAN)
        #
        self.wvl_target_convolve = None
        self.rv_target_convolve = None

    def _calculate_wvl_target(self):
        # symmetric width including line center
        return self.wvl_center + np.arange(-self.wvl_steps, self.wvl_steps+1)*self.wvl_step

    def _derive_rv(self):
        return (self.wvl_target - self.wvl_center)/self.wvl_target * self.C0 / 1000.  # conversion into km/s

    def resample_and_add_line(self, spectral_input, wavelength_input):
        if len(spectral_input) > 1:
            # determine correct band for this line
            wvl_mean_dist = [np.nanmean(wvls) for wvls in wavelength_input]
            use_band = np.argmin(np.abs(wvl_mean_dist - self.wvl_center))
        else:
            use_band = 0
        try:
            # xb and xe omitted as it returns: error (xb<=x[0]) failed for 1st keyword xb
            bspline = splrep(wavelength_input[use_band], spectral_input[use_band])
            new_spectral_data = splev(self.wvl_target, bspline)
        except:
            print '  Problem resampling spectra at ____'
            new_spectral_data = self.nan_row
        self.write_line(new_spectral_data)

    def write_line(self, spectra):
        txt = open(self.filename, 'a')
        txt.write(','.join([str(f) for f in spectra]) + '\n')
        txt.close()

    def _empty_txt(self):
        txt = open(self.filename, 'w')
        txt.close()

    def read_data(self):
        print 'Reading data from file: ' + self.filename
        return np.loadtxt(self.filename, delimiter=',')

    def write_data(self, data=None, path=None):
        if path is not None:
            if data is None:
                np.savetxt(path, data, delimiter=',')
            else:
                np.savetxt(path, self.data, delimiter=',')

    def convolve_data(self, n_out=2, n_step=3):
        # n_out - number of output points on every side of line center
        # n_step - output every n index point
        gauss_width_use = 10
        idx_center = np.int32(len(self.wvl_target)/2.)
        gauss_curve = _gauss_model(self.wvl_target, 1., 0.05, self.wvl_center)
        gauss_curve_sub = gauss_curve[idx_center-gauss_width_use:idx_center+gauss_width_use+1]
        #
        idx_out = np.int32(idx_center + (np.arange(n_out*2+1)-n_out)*n_step)
        # create output array
        n_rows = len(self.data)
        n_cols = len(idx_out)
        data_out = np.ndarray((n_rows, n_cols))
        for i_row in range(n_rows):
            temp = np.convolve(self.data[i_row], gauss_curve_sub, mode='same')
            data_out[i_row] = temp[idx_out]
        self.wvl_target_convolve = self.wvl_target[idx_out]
        self.rv_target_convolve = self.rv_target[idx_out]
        # normalize data
        data_norm_fact = np.sum(gauss_curve_sub)
        return data_out / data_norm_fact


# # ONLY for test purpose
# def __line_class_test__():
#     import matplotlib.pyplot as plt
#     s = AbsorptionLine(read=True, filename='O_center_7775.388_step_0.040_width_2.000_test.txt', file_path='Absorption_lines_data/')
#     data_convol = s.convolve_data(n_out=3, n_step=3)
#     for i_r in range(len(data_convol)):
#         plt.plot(s.wvl_target, s.data[i_r], c='red')
#         plt.plot(s.wvl_target_convolve, data_convol[i_r], c='black')
#     plt.savefig('temp.png')