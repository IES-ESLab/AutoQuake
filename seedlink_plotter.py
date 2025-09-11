#%%
#!/usr/bin/env python
from __future__ import print_function
from matplotlib.dates import date2num

from obspy import Stream, Trace, read
# from obspy import __version__ as OBSPY_VERSION
from obspy.core import UTCDateTime
from argparse import ArgumentParser,ArgumentDefaultsHelpFormatter
import threading
import time
import warnings
import os
import sys
from urllib.request import URLError
import logging
import numpy as np


import matplotlib as mpl
# mpl.use('Agg')  # Commented out for interactive plotting in Jupyter
import matplotlib.pyplot as plt
from matplotlib.dates import date2num,DateFormatter
from matplotlib.ticker import MaxNLocator
# OBSPY_VERSION = [int(x) for x in OBSPY_VERSION.split(".")[:2]]

from obspy.clients.seedlink.slpacket import SLPacket
from obspy.clients.seedlink import SLClient
from obspy.clients.fdsn import Client
#%%
def _parse_time_with_suffix_to_seconds(timestring):
    """
    Parse a string to seconds as float.

    If string can be directly converted to a float it is interpreted as
    seconds. Otherwise the following suffixes can be appended, case
    insensitive: "s" for seconds, "m" for minutes, "h" for hours, "d" for days.

    >>> _parse_time_with_suffix_to_seconds("12.6")
    12.6
    >>> _parse_time_with_suffix_to_seconds("12.6s")
    12.6
    >>> _parse_time_with_suffix_to_minutes("12.6m")
    756.0
    >>> _parse_time_with_suffix_to_seconds("12.6h")
    45360.0

    :type timestring: str
    :param timestring: "s" for seconds, "m" for minutes, "h" for hours, "d" for
        days.
    :rtype: float
    """
    try:
        return float(timestring)
    except:
        timestring, suffix = timestring[:-1], timestring[-1].lower()
        mult = {'s': 1.0, 'm': 60.0, 'h': 3600.0, 'd': 3600.0 * 24}[suffix]
        return float(timestring) * mult


def _parse_time_with_suffix_to_minutes(timestring):
    """
    Parse a string to minutes as float.

    If string can be directly converted to a float it is interpreted as
    minutes. Otherwise the following suffixes can be appended, case
    insensitive: "s" for seconds, "m" for minutes, "h" for hours, "d" for days.

    >>> _parse_time_with_suffix_to_minutes("12.6")
    12.6
    >>> _parse_time_with_suffix_to_minutes("12.6s")
    0.21
    >>> _parse_time_with_suffix_to_minutes("12.6m")
    12.6
    >>> _parse_time_with_suffix_to_minutes("12.6h")
    756.0

    :type timestring: str
    :param timestring: "s" for seconds, "m" for minutes, "h" for hours, "d" for
        days.
    :rtype: float
    """
    try:
        return float(timestring)
    except:
        seconds = _parse_time_with_suffix_to_seconds(timestring)
    return seconds / 60.0

seedlink_streams = "TW_LT01:00HHZ,TW_LT03:00HHZ,TW_LT04:00HHZ,TW_LT05:00HHZ,TW_LT06:00HHZ,TW_LT07:00HHZ,TW_LT08:00HHZ,TW_LT09:00HHZ,TW_LT10:00HHZ"
seedlink_streams = "TW_LT10:00HHN,TW_LT10:00HHE,TW_LT10:00HHZ"
backtrace_time = _parse_time_with_suffix_to_seconds("5m")
seedlink_server = "140.109.81.158:18000"
update_time = _parse_time_with_suffix_to_seconds("2s")

class SeedlinkTrimmer():
    def __init__(
        self,
        stream=None,
        myargs=None,
        lock=None,
        trace_ids=None,
        *args,
        **kwargs
        ):
        # args = myargs
        self.lock = lock
        self.backtrace = backtrace_time
        # self.args = args
        self.stream = stream
        self.ids = trace_ids
        self.trimmer()

    def trimmer(self):
        now = UTCDateTime()
        self.start_time = now - self.backtrace
        self.stop_time = now

        with self.lock:
            # leave some data left of our start for possible processing
            self.stream.trim(
                starttime=self.start_time - 120, nearest_sample=False)
            stream = self.stream.copy()
        try:
            logging.info(str(stream.split()))
            if not stream:
                raise Exception("Empty stream for plotting")

            stream.merge(-1)
            stream.trim(starttime=self.start_time, endtime=self.stop_time)

            for tr in stream:
                tr.stats.processing = []
            # self.plot_lines(stream)
            # print(self.stream)
            return stream
        except Exception as e:
            logging.error(e)
            pass

    # def plot_lines(self, stream):
    #     for id_ in self.ids:
    #         if not any([tr.id == id_ for tr in stream]):
    #             net, sta, loc, cha = id_.split(".")
    #             header = {'network': net, 'station': sta, 'location': loc,
    #                       'channel': cha, 'starttime': self.start_time}
    #             data = np.zeros(2)
    #             stream.append(Trace(data=data, header=header))
    #     # stream.sort()
    #     for tr in stream:
    #         tr.stats.processing = []
    #     print(stream[0].stats)
        # return stream

class SeedlinkUpdater(SLClient):

    def __init__(self, stream, myargs=None, lock=None):
        # loglevel NOTSET delegates messages to parent logger
        super(SeedlinkUpdater, self).__init__(loglevel="NOTSET")
        self.stream = stream
        self.lock = lock
        self.args = myargs


    def packet_handler(self, count, slpack):
        """
        for compatibility with obspy 0.10.3 renaming
        """
        self.packetHandler(count, slpack)

    def packetHandler(self, count, slpack):
        """
        Processes each packet received from the SeedLinkConnection.
        :type count: int
        :param count:  Packet counter.
        :type slpack: :class:`~obspy.seedlink.SLPacket`
        :param slpack: packet to process.
        :return: Boolean true if connection to SeedLink server should be
            closed and session terminated, false otherwise.
        """

        # check if not a complete packet
        if slpack is None or (slpack == SLPacket.SLNOPACKET) or \
                (slpack == SLPacket.SLERROR):
            return False

        # get basic packet info
        type = slpack.get_type()
        # print(slpack.get_string_payload())
        # process INFO packets here
        if type == SLPacket.TYPE_SLINF:
            return False
        if type == SLPacket.TYPE_SLINFT:
            logging.info("Complete INFO:" + self.slconn.getInfoString())
            if self.infolevel is not None:
                return True
            else:
                return False

        # process packet data
        trace = slpack.get_trace()
        if trace is None:
            logging.info(
                self.__class__.__name__ + ": blockette contains no trace")
            return False

        # new samples add to the main stream which is then trimmed
        with self.lock:
            self.stream += trace
            self.stream.merge(-1)
            for tr in self.stream:
                tr.stats.processing = []
        return False

    def getTraceIDs(self):
        """
        Return a list of SEED style Trace IDs that the SLClient is trying to
        fetch data for.
        """
        ids = []
        streams = self.slconn.get_streams()
        # if OBSPY_VERSION < [1, 0]:
        #     streams = self.slconn.getStreams()
        # else:
        #     streams = self.slconn.get_streams()
        for stream in streams:
            net = stream.net
            sta = stream.station
            selectors = stream.get_selectors()
            # if OBSPY_VERSION < [1, 0]:
            #     selectors = stream.getSelectors()
            # else:
            #     selectors = stream.get_selectors()
            for selector in selectors:
                if len(selector) == 3:
                    loc = ""
                else:
                    loc = selector[:2]
                cha = selector[-3:]
                ids.append(".".join((net, sta, loc, cha)))
        ids.sort()
        return ids


def _resampling(st):
    # st.taper(max_percentage=0.001, type='cosine', max_length=2)
    st.decimate(2, no_filter=True)
    # need_resampling = [tr for tr in st ]
    # for indx, tr in enumerate(need_resampling):
        # tr.decimate(2)
    #     if tr.stats.delta < 0.01:
    #         tr.filter('lowpass',freq=45,zerophase=True)
        # tr.resample(50)
        # tr.stats.sampling_rate = 50
        # tr.stats.delta = 0.02
        # tr.data.dtype = 'int32'
        # st.remove(tr)
        # st.append(tr)
    return st

def updator(streams):
    # print(streams.trimmer()[0])
    while 1:
        stream = streams.trimmer()
        time.sleep(2)
        yield stream

if __name__ == '__main__':
    global streams

    # args.seedlink_streams = "TW_LT06:00HHE,TW_LT05:00HHE"


    loglevel = logging.CRITICAL
    logging.basicConfig(level=loglevel)

    now = UTCDateTime()
    stream = Stream()
    lock = threading.Lock()
    test_time = UTCDateTime("2025-09-05T01:00:00")
    # cl is the seedlink client
    seedlink_client = SeedlinkUpdater(stream, lock=lock)
    seedlink_client.slconn.set_sl_address(seedlink_server)
    seedlink_client.multiselect = seedlink_streams
    seedlink_client.begin_time = (now - backtrace_time).format_seedlink()
    # seedlink_client.end_time = (test_time + 60).format_seedlink()

    seedlink_client.initialize()
    ids = seedlink_client.getTraceIDs()
    # start cl in a thread
    thread = threading.Thread(target=seedlink_client.run)
    thread.setDaemon(True)
    thread.start()
    # Wait few seconds to get data for the first plot
    print('Initialize the first stream......')
    time.sleep(2)

    streams = SeedlinkTrimmer(stream=stream, lock=lock, trace_ids=ids)
    # updator(streams)
    # print( streams.trimmer()[0] )

    OutFigPath = "/raid1/For_webserver/"

    Station = ['LT01','LT03','LT04','LT05','LT06','LT07','LT08','LT09','LT10']

    # while True:
    #     time.sleep(update_time)
    #     # WF=streams.trimmer()
    #     WF=_resampling(streams.trimmer())
    #     subtitle=str(WF[0].stats.starttime.year)+'/'+str(WF[0].stats.starttime.month).zfill(2)
    #     # bound_list=[]
    #     # for idx in range(len(WF)):
    #     #     bound_list.append( max(WF[idx].data) )
    #     #     bound_list.append( min(WF[idx].data) )
    #     # bound_value=max(bound_list)*1.5

    #     stn_list=[tr.stats.station for tr in WF]

    #     fig, axs = plt.subplots(len(Station),sharex=True,figsize=(14,14))
    #     fig.autofmt_xdate()
    #     myFmt = DateFormatter('%H:%M:%S')

    #     for idx,stn in enumerate(Station):
    #         if stn in stn_list:
    #             tr_idx=stn_list.index(stn)
    #             axs[idx].plot(WF[tr_idx].times("matplotlib"), WF[tr_idx],color='k',linewidth=0.2)
    #             # axs[i].yaxis.set_ticklabels([])
    #             # axs[i].set_ylim(-1.0*bound_value,bound_value)
    #             # axs[i].xaxis_date()
    #             axs[idx].xaxis.set_major_formatter(myFmt)
    #             axs[idx].tick_params(axis='both', left='off')
    #             axs[idx].yaxis.set_label_position("right")
    #             axs[idx].set_ylabel(stn,rotation=0, labelpad=20)
    #             axs[idx].grid(axis='x')
    #         else:
    #             axs[idx].yaxis.set_label_position("right")
    #             axs[idx].set_ylabel(stn,rotation=0, labelpad=20)
    #             axs[idx].grid(axis='x')


    #     # for i in range(len(WF)):
    #     #     y_index=Station.index(WF[i].stats.station)
    #     #     axs[y_index].plot(WF[i].times("matplotlib"), WF[i],color='k',linewidth=0.2)
    #     #     # axs[i].yaxis.set_ticklabels([])
    #     #     # axs[i].set_ylim(-1.0*bound_value,bound_value)
    #     #     # axs[i].xaxis_date()
    #     #     axs[y_index].xaxis.set_major_formatter(myFmt)
    #     #     axs[y_index].tick_params(axis='both', left='off')
    #     #     axs[y_index].yaxis.set_label_position("right")
    #     #     axs[y_index].set_ylabel(WF[i].stats.station,rotation=0, labelpad=20)
    #     #     axs[y_index].grid(axis='x')

    #     fig.tight_layout(pad=0.5)
    #     fig.suptitle(subtitle+" Lantai network Raw Waveform, Z component", fontsize=16)
    #     fig.subplots_adjust(top=0.96)
    #     plt.locator_params(axis="x", nbins=10)
    #     plt.savefig(OutFigPath + 'Lantai_stream' +'.png',dpi=100, pad_inches=0, bbox_inches='tight')
    #     plt.close(fig)
    #     # plt.show()
    #     # break

# %%
