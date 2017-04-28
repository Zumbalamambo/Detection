import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates
import datetime
from smoother import smooth

date = '20170415'
id_count_front = np.loadtxt('./outputs/id_accumulated_counter_{}_front_SSD512x1024_sort_ma10_mh3.txt'.format(date),
                            dtype=np.int32, delimiter=',')
# id_count_side = np.loadtxt('./outputs/id_accumulated_counter_{}_side_SSD512x1024_sort_ma10_mh3.txt'.format(date),
#                            dtype=np.int32, delimiter=',')

id_count_front_flat = id_count_front.ravel()
# id_count_side_flat = id_count_side.ravel()

id_count_per_min_front = np.append(id_count_front_flat[0], np.diff(id_count_front_flat))
# id_count_per_min_side = np.append(id_count_side_flat[0], np.diff(id_count_side_flat))

# smoothing count per minute
window_len_half = 5
window_len = window_len_half*2 + 1
# keep the length of the time series
id_count_per_min_front_smoothed = smooth(id_count_per_min_front, window_len=window_len, window='blackman')[window_len_half:-window_len_half]
# id_count_per_min_side_smoothed = smooth(id_count_per_min_side, window_len=window_len, window='blackman')[window_len_half:-window_len_half]

# prepare x-axis
mins = np.arange(720, dtype=np.int) + 10*60     # from 10 am
times = np.array([datetime.datetime(2017, 3, 4, int(p/60), p%60) for p in mins])
plt.plot(times, id_count_per_min_front_smoothed, '-b')
# plt.plot(times, id_count_per_min_side_smoothed, '-.r')
plt.legend(['Front'], loc='best')
# plt.legend(['Front', 'Side'], loc='best')
plt.title('Count per Minute on {}'.format(date))

# generate a formatter, using the fields required
fmtr = dates.DateFormatter("%H:%M")
# need a handle to the current axes to manipulate it
ax = plt.gca()
# set this formatter to the axis
ax.xaxis.set_major_formatter(fmtr)

plt.show()