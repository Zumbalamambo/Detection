import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
from matplotlib import dates
import datetime
from smoother import smooth

id_count_0304_side = np.loadtxt('./outputs/id_accumulated_counter_20170304_side_SSD512x1024_sort_ma10_mh1.txt',
                                dtype=np.int32, delimiter=',')
id_count_0310_front = np.loadtxt('./outputs/id_accumulated_counter_20170310_front_SSD512x1024_sort_ma10_mh1.txt',
                                 dtype=np.int32, delimiter=',')

# time_range = pd.timedelta_range("10:00:00", "21:59:00", freq="1min")

id_count_0304_side_1dim = id_count_0304_side.ravel()
id_count_0310_front_1dim = id_count_0310_front.ravel()

id_count_per_min_0304_side = np.append(id_count_0304_side_1dim[0], np.diff(id_count_0304_side_1dim))
id_count_per_min_0310_front = np.append(id_count_0310_front_1dim[0], np.diff(id_count_0310_front_1dim))

# smoothing count per minute
window_len_half = 5
window_len = window_len_half*2 + 1
id_count_per_min_0304_side_smoothed = smooth(id_count_per_min_0304_side, window_len=window_len, window='blackman')[window_len_half:-window_len_half]
id_count_per_min_0310_front_smoothed = smooth(id_count_per_min_0310_front, window_len=window_len, window='blackman')[window_len_half:-window_len_half]

# print id_count_per_min_0304_side_smoothed.shape
# print id_count_per_min_0310_front_smoothed.shape


# # in Data Frame
# id_count_0304_side_in_ts = pd.DataFrame(data={'time': time_range, 'p_count': id_count_0304_side_1dim},
#                                         columns=['time', 'p_count'])
# id_count_0310_front_in_ts = pd.DataFrame(data={'time': time_range, 'p_count': id_count_0310_front_1dim},
#                                          columns=['time', 'p_count'])

# # in Time Series
# id_count_0304_side_in_ts = pd.Series(id_count_0304_side_1dim, index=time_range, dtype=np.int32)
# id_count_0310_front_in_ts = pd.Series(id_count_0310_front_1dim, index=time_range, dtype=np.int32)

# work around MemoryError for pandas Time Series by the following
# # plot 1: total count
# mins = np.arange(720, dtype=np.int) + 10*60     # from 10 am
# times = np.array([datetime.datetime(2017, 3, 4, int(p/60), p%60) for p in mins])
# # and plot for every 5 samples:
# plt.plot(times[1::5], id_count_0310_front_1dim[1::5], '-b')     # one value per 5 min
# plt.plot(times[1::5], id_count_0304_side_1dim[1::5], '-.r')
# plt.legend(['20170310 Fri. Front', '20170304 Sat. Side'], loc='best')
# plt.title('Total Count')
# plot 2: count per min
mins = np.arange(720, dtype=np.int) + 10*60     # from 10 am
times = np.array([datetime.datetime(2017, 3, 4, int(p/60), p%60) for p in mins])
# and plot for every 5 samples:
plt.plot(times, id_count_per_min_0310_front_smoothed, '-b')     # one value per 5 min
plt.plot(times, id_count_per_min_0304_side_smoothed, '-.r')
plt.legend(['20170310 Fri. Front', '20170304 Sat. Side'], loc='best')
plt.title('Count per Minute')

# generate a formatter, using the fields required
fmtr = dates.DateFormatter("%H:%M")
# need a handle to the current axes to manipulate it
ax = plt.gca()
# set this formatter to the axis
ax.xaxis.set_major_formatter(fmtr)

# plt.figure()
# plt.subplot(121)
# plt.plot(np.arange(12*60), id_count_0310_front.ravel(), '-b')
# plt.plot(np.arange(12*60), id_count_0304_side.ravel(), '-.r')
# plt.legend(['20170310 Fri. Front', '20170304 Sat. Side'])
# plt.title('Total Count')
#
# plt.subplot(122)
# id_count_per_min_0310_front = np.append(id_count_0310_front[0][0], np.diff(id_count_0310_front.ravel()))
# id_count_per_min_0304_side = np.append(id_count_0304_side[0][0], np.diff(id_count_0304_side.ravel()))
# plt.plot(np.arange(12*60), id_count_per_min_0310_front.ravel(), '-b')
# plt.plot(np.arange(12*60), id_count_per_min_0304_side.ravel(), '-.r')
# plt.legend(['20170310 Fri. Front', '20170304 Sat. Side'])
# plt.title('Count per Minute')

plt.show()