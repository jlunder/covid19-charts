#!/usr/bin/python3
import sys, os, time, csv, re, datetime, math, itertools
import matplotlib.pyplot as plt

from coviddata import *


# Main script begins here

class XAxis:
    axis_label = ''
    _data_start_func = lambda self, r: 0

    def __init__(self, axis_label, data_start_func, lazy_xaxis_func=None):
        self.axis_label = axis_label
        self._data_start_func = data_start_func
        self._lazy_xaxis_func = lazy_xaxis_func

    def set_yaxis(self, yaxis):
        if self._lazy_xaxis_func:
            xaxis = self._lazy_xaxis_func(yaxis)
            self.axis_label = xaxis.axis_label
            self._data_start_func = xaxis._data_start_func

    def data_start(self, r):
        return self._data_start_func(r)

    @staticmethod
    def since_day(day_num):
        global dates
        return XAxis(str(dates[day_num]), lambda r: day_num)

    @staticmethod
    def since_threshold(data_name, val, data_func):
        return XAxis('Days Since %g %s' % (val, data_name,),
                     lambda r: threshold(data_func(r), val))

    @staticmethod
    def since_yaxis_threshold(val):
        return XAxis(None, None,
                     lambda yaxis: XAxis('Days Since %g %s' % (val, yaxis.axis_label,),
                                   lambda r: threshold(yaxis.data_values(r), val)))


    @staticmethod
    def since_confirmed_total_10_per_million():
        return XAxis.since_threshold(
            'Cases Per 1M', 10,
            lambda r: [c * 1e6 for c in r.percapita_confirmed_total()])

    @staticmethod
    def since_deaths_total_smoothed_1_per_million():
        return XAxis.since_threshold(
            'Deaths Per 1M', 1,
            lambda r: convert_smooth([c * 1e6 for c in r.percapita_deaths_total()]))

class YAxis:
    axis_label = ''
    title_part = ''
    scale = 'log'
    _data_label_func = lambda self, r: ''
    _data_values_func = lambda self, r: []

    def __init__(self, title_part, axis_label, scale, data_label_func, data_values_func):
        self.title_part = title_part
        self.axis_label = axis_label
        self.scale = scale
        self._data_label_func = data_label_func
        self._data_values_func = data_values_func

    def data_label(self, r):
        return self._data_label_func(r)

    def data_values(self, r):
        return self._data_values_func(r)

    @staticmethod
    def log_confirmed_total_per_million():
        return YAxis('Total Confirmed COVID-19 Cases', 'Cases Per 1M', 'log',
                     lambda r: r.prettyname,
                     lambda r: [c * 1e6 for c in r.percapita_confirmed_total()])

    @staticmethod
    def log_confirmed_new_per_million():
        return YAxis('Daily New Confirmed COVID-19 Cases', 'Cases Per 1M', 'log',
                     lambda r: r.prettyname,
                     lambda r: [c * 1e6 for c in r.percapita_confirmed_new()])

    @staticmethod
    def log_confirmed_new_per_million_smoothed():
        return YAxis('Daily New Confirmed COVID-19 Cases (Smoothed)', 'Cases Per 1M', 'log',
                     lambda r: r.prettyname,
                     lambda r: [c * 1e6 for c in convert_smooth(r.percapita_confirmed_new())])

    @staticmethod
    def log_estimated_total_per_million():
        return YAxis('Total Estimated COVID-19 Cases', 'Cases Per 1M', 'log',
                     lambda r: '%s (%.2g%%)' %(r.prettyname, 100 / r.estimation_factor(),),
                     lambda r: [c * 1e6 for c in r.percapita_estimated_total()])

    @staticmethod
    def log_deaths_total_per_million():
        return YAxis('Total Deaths Attributed To COVID-19', 'Deaths Per 1M', 'log',
                     lambda r: r.prettyname,
                     lambda r: [c * 1e6 for c in r.percapita_deaths_total()])

    @staticmethod
    def log_deaths_new_per_million_smoothed():
        return YAxis('New Deaths Attributed To COVID-19', 'Deaths Per 1M', 'log',
                     lambda r: r.prettyname,
                     lambda r: [c * 1e6 for c in convert_smooth(r.percapita_deaths_new())])

    @staticmethod
    def log_confirmed_ongoing_per_million():
        return YAxis('Unresolved Confirmed COVID-19 Cases', 'Unresolved Cases Per 1M', 'log',
                     lambda r: r.prettyname,
                     lambda r: [c * 1e6 for c in r.percapita_confirmed_ongoing()])


class RegionSet:
    title_part = ''
    regions = []

    def __init__(self, title_part, regions):
        self.title_part = title_part
        self.regions = regions


def plot_srs(xaxis, yaxis, regions):
    fig, ax = plt.subplots()
    xaxis.set_yaxis(yaxis)
    ax.set_title(yaxis.title_part + ' ' + regions.title_part)
    ax.set_xlabel(xaxis.axis_label)
    ax.set_ylabel(yaxis.axis_label)
    ax.set_yscale(yaxis.scale)

    min_len = 3

    plots_to_do = []
    for r in regions.regions:
        start = xaxis.data_start(r)
        if start != None:
            data = yaxis.data_values(r)[start:]
            if len(data) >= min_len:
                plots_to_do.append((list(range(len(data))), data, yaxis.data_label(r),))

    if len(plots_to_do) < 1:
        print('No data for chart "%s"' % (title,))
        return

    # compute max_len: finds the longest series but rejecting outliers
    # threshold for being an outlier is, >4/3x as long as our 75% longest series
    # (if you order them by length -- bottom of the top quartile)
    series_lengths = sorted([len(p[0]) for p in plots_to_do])
    bottom_of_top_quartile = (len(series_lengths) - 1) * 3 // 4
    max_len = max([series_lengths[i] for i in range(bottom_of_top_quartile, len(series_lengths))
        if series_lengths[i] * 0.75 <= series_lengths[bottom_of_top_quartile]])

    for p in plots_to_do:
        ax.plot(p[0][:max_len], p[1][:max_len], label=p[2])
    ax.legend()
    plt.show()


cases_data_format, subregion_cases = \
    read_data('COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
deaths_data_format, subregion_deaths = \
    read_data('COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
recovered_data_format, subregion_recovered = \
    read_data('COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
us_cases_data_format, us_adminregion_cases = \
    read_data('COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv')
us_deaths_data_format, us_adminregion_deaths = \
    read_data('COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv')
subregions, regions = merge_data(subregion_cases + us_adminregion_cases,
                                 subregion_deaths + us_adminregion_deaths,
                                 subregion_recovered)
canada_subregions = [r for r in regions['canada'].child_regions.values() if r.population]
us_subregions = [r for r in regions['us'].child_regions.values() if r.population]
us_adminregions = sum([[ar for ar in sr.child_regions.values() if ar.population]
                       for sr in regions['us'].child_regions.values() if sr.child_regions], start=[])
#for r in us_adminregions:
#    print(r.name)
#print(sum((r.population for r in regions['us'].child_regions.values() if r.population)))

reference_region = RegionData.from_source(
    RegionSourceData(subregion='33% Daily Growth', region='',
        data=[nominal_growth ** x for x in range(len(cases_data_format.dates))]),
    RegionSourceData(subregion='33% Daily Growth', region='',
        data=[nominal_death_rate * nominal_growth ** (x - nominal_confirm_to_death) for x in range(len(cases_data_format.dates))]),
    RegionSourceData(subregion='33% Daily Growth', region='',
        data=[(1 - nominal_death_rate) * nominal_growth ** (x - 14) for x in range(len(cases_data_format.dates))]),
    {}
    )
reference_region.population = 1e6

dates = cases_data_format.dates

significant_canada_subregions = RegionSet('In Canada',
                                          [subregions[n] for n in ['canada/british columbia', 'canada/alberta',
                                                                   'canada/ontario', 'canada/quebec']])
international_subregions = RegionSet('Internationally',
                                     [subregions[n] for n in ['canada/british columbia', 'china/hubei']]
                                     + [regions[n] for n in ['canada', 'korea, south', 'japan', 'italy',
                                                             'france', 'us', 'germany', 'sweden', 'brazil',
                                                             ]])

us_hotspot_states = RegionSet('In US Hotspot States',
                              [subregions['canada/british columbia']]
                              + sorted([r for r in us_subregions if r.population > 100000],
                                       key=lambda r: r.percapita_confirmed_total()[-1])[-10:])
us_hotspot_cities = RegionSet('In US Hotspot Cities',
                              [subregions['canada/british columbia']]
                              + sorted([r for r in us_adminregions if r.population > 100000],
                                       key=lambda r: r.percapita_confirmed_total()[-1])[-10:])



if False:
    plot_srs(xaxis=XAxis.since_yaxis_threshold(10),
             yaxis=YAxis.log_confirmed_total_per_million(),
             regions=significant_canada_subregions)

if False:
    plot_srs(xaxis=XAxis.since_yaxis_threshold(10),
             yaxis=YAxis.log_confirmed_total_per_million(),
             regions=international_subregions)

if False:
    plot_srs(xaxis=XAxis.since_yaxis_threshold(10),
             yaxis=YAxis.log_confirmed_total_per_million(),
             regions=us_hotspot_states,) 

if False:
    plot_srs(xaxis=XAxis.since_yaxis_threshold(10),
             yaxis=YAxis.log_confirmed_total_per_million(),
             regions=us_hotspot_cities,) 

if False:
    plot_srs(xaxis=XAxis.since_confirmed_total_10_per_million(),
             yaxis=YAxis.log_confirmed_new_per_million_smoothed(),
             regions=significant_canada_subregions,)

if False:
    plot_srs(xaxis=XAxis.since_confirmed_total_10_per_million(),
             yaxis=YAxis.log_confirmed_new_per_million_smoothed(),
             regions=international_subregions,)

if False:
    plot_srs(xaxis=XAxis.since_confirmed_total_10_per_million(),
             yaxis=YAxis.log_confirmed_new_per_million_smoothed(),
             regions=us_hotspot_states,)

if False:
    plot_srs(xaxis=XAxis.since_confirmed_total_10_per_million(),
             yaxis=YAxis.log_confirmed_new_per_million_smoothed(),
             regions=us_hotspot_cities,)

if False:
    plot_srs(xaxis=XAxis.since_deaths_total_smoothed_1_per_million(),
             yaxis=YAxis.log_deaths_new_per_million_smoothed(),
             regions=international_subregions,)

if False:
    plot_srs(xaxis=XAxis.since_yaxis_threshold(10),
             yaxis=YAxis.log_confirmed_ongoing_per_million(),
             regions=significant_canada_subregions,)

if False:
    plot_srs(xaxis=XAxis.since_yaxis_threshold(10),
             yaxis=YAxis.log_confirmed_ongoing_per_million(),
             regions=international_subregions,)

if False:
    plot_srs(xaxis=XAxis.since_day(40), yaxis=YAxis.log_deaths_per_million(),
             regions=significant_canada_subregions,)

if False:
    plot_srs(xaxis=XAxis.since_day(40), yaxis=YAxis.log_deaths_per_million(),
             regions=international_subregions,)

if False:
    plot_srs(xaxis=XAxis.since_yaxis_threshold(100),
             yaxis=YAxis.log_estimated_total_per_million(),
             regions=significant_canada_subregions,)

if False:
    plot_srs(xaxis=XAxis.since_yaxis_threshold(100),
             yaxis=YAxis.log_estimated_total_per_million(),
             regions=us_hotspot_states,) 

if False:
    plot_srs(xaxis=XAxis.since_yaxis_threshold(100),
             yaxis=YAxis.log_estimated_total_per_million(),
             regions=us_hotspot_cities,) 

if False:
    plot_srs(xaxis=XAxis.since_yaxis_threshold(100),
             yaxis=YAxis.log_estimated_total_per_million(),
             regions=international_subregions,)

if False:
    fig, ax = plt.subplots()
    ax.set_xlabel('Difference In Days Between Average Confirmation And Average Death')
    ax.set_ylabel('RMS Of Normalized Difference')
    ax.set_yscale('linear')
    ax.set_title('Fit Of Daily New Confirmed Cases To New Deaths')
    for sr in international_subregions.regions:
        start = next((i for i in range(len(sr.deaths)) if sr.deaths[i] >= 10), -1)
        if start < 0:
            continue

        start = max(0, start - (default_smooth // 2))
        confirmed = convert_smooth(sr.confirmed_new(), default_smooth)
        deaths = convert_smooth(sr.deaths_new(), default_smooth)
        data_len = min(len(confirmed), len(deaths))

        xs = []
        ys = []
        for i in range(-5, nominal_death + 5):
            cs, ce = start, data_len
            ds, de = start + i, data_len + i
            if ds < 0:
                cs -= ds
                ds -= ds
            if de > len(deaths):
                ce -= de - len(deaths)
                de -= de - len(deaths)
            n = ce - cs
            if n < 4:
                continue
            confirmed_slice = confirmed[cs:ce]
            deaths_slice = deaths[ds:de]
            d_norm = sum(deaths_slice) / n
            c_norm = sum(confirmed_slice) / n
            xs.append(i)
            ys.append(math.sqrt(sum(map(lambda d: (d[0] / c_norm - d[1] / d_norm) ** 2,
                zip(confirmed_slice, deaths_slice))) / (n - 1)))
        ax.plot(xs, [y / sum(ys) for y in ys], label=sr.prettyname,)
    ax.legend()
    plt.show()

if False:
    fig, ax = plt.subplots()
    ax.set_xlabel('Ongoing Confirmed Cases Per 1M')
    ax.set_xscale('log')
    ax.set_ylabel('New Confirmed Cases \u22C5 Recovery Time Per 1M')
    ax.set_yscale('log')
    ax.set_title('New Cases For Existing Cases In Canada')
    max_new = 0
    max_ongoing = 0
    for sr in significant_canada_subregions.regions:
        cases = [c * 1e6 for c in convert_smooth(sr.percapita_confirmed_total())]
        start = threshold(cases, 10)
        if start != None and start < len(cases):
            new_cases = [d * nominal_recovery for d in convert_delta(cases)[start:]]
            ongoing_cases = [c * 1e6 for c in convert_smooth(sr.percapita_confirmed_ongoing())[start:]]
            max_new = max(max_new, max(new_cases))
            max_ongoing = max(max_ongoing, max(ongoing_cases))
            ax.plot(ongoing_cases, new_cases, label=sr.prettyname,)
    max_series = [1, max(max_new, max_ongoing)]
    ax.plot(max_series, max_series, label='Danger <-> Relax')
    ax.legend()
    plt.show()

if False:
    fig, ax = plt.subplots()
    ax.set_xlabel('Confirmed Cases Per 1M')
    ax.set_xscale('log')
    ax.set_ylabel('New Cases \u22C5 Recovery Time Per 1M')
    ax.set_yscale('log')
    ax.set_title('New Cases For Existing Cases Internationally')
    for sr in international_subregions.regions:
        cases = [c * 1e6 for c in convert_smooth(sr.percapita_confirmed_total())]
        start = threshold(cases, 10)
        if start != None and start < len(cases):
            new_cases = [d * nominal_recovery for d in convert_delta(cases)[start:]]
            ax.plot(cases[start:], new_cases, label=sr.prettyname,)
    ax.legend()
    plt.show()


if True:
    plot_srs(xaxis=XAxis.since_day(1),
             yaxis=YAxis.log_confirmed_new_per_million_smoothed(),
             regions=significant_canada_subregions,)

if True:
    plot_srs(xaxis=XAxis.since_day(1),
             yaxis=YAxis.log_confirmed_new_per_million_smoothed(),
             regions=us_hotspot_states,) 

if True:
    plot_srs(xaxis=XAxis.since_day(1),
             yaxis=YAxis.log_confirmed_new_per_million_smoothed(),
             regions=international_subregions,)
 