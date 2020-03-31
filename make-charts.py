#!/usr/bin/python3
import sys, os, time, csv, re, datetime
import matplotlib.pyplot as plt

one_day = datetime.timedelta(days=1)

region_populations = {
    'canada/alberta': 4413146,
    'canada/british columbia': 5110917,
    'canada/manitoba': 1377517,
    'canada/new brunswick': 779993,
    'canada/newfoundland and labrador': 521365,
    'canada/nova scotia': 977457,
    'canada/ontario': 14711827,
    'canada/prince edward island': 158158,
    'canada/quebec': 8537674,
    'canada/saskatchewan': 1181666,
    'canada/yukon': 41078,
    'canada/northwest territories': 44904,
    'canada/nunavut': 39097,

    'china/hubei': 59050000,

    'us': 331002651,
    'italy': 60484989,
    'korea, south': 51269185,
    'china': 1408526449,
    'japan': 126476461,
    'germany': 83783942,
    'singapore': 5850342,
    'france': 65270000,
    }

class DataFormat:
    subregion_col = 0
    region_col = 1
    lat_col = 2
    long_col = 3
    date_cols = {}

    def __init__(self, row):
        date_cols = {}

        date_re = re.compile('([0-9]+)/([0-9]+)/([0-9]+)')

        col = 0
        start_date = None
        end_date = None

        for header in row:
            header_strip = header.strip().lower()
            if header_strip.startswith('province') or header_strip.startswith('state'):
                self.subregion_col = col
            elif header_strip.startswith('country'):
                self.region_col = col
            elif header_strip.startswith('lat'):
                self.lat_col = col
            elif header_strip.startswith('long'):
                self.long_col = col
            else:
                date_match = date_re.match(header_strip)
                if date_match:
                    date_month = int(date_match.group(1))
                    date_day = int(date_match.group(2))
                    date_year = int(date_match.group(3)) + 2000
                    date = datetime.date(date_year, date_month, date_day)
                    if not start_date:
                        start_date = date
                        end_date = date + one_day
                    else:
                        start_date = min(start_date, date)
                        end_date = max(end_date, date + one_day)
                    date_cols[date] = col
            col += 1

        dates = []
        date = start_date
        while date < end_date:
            dates.append(date)
            if date not in date_cols:
                date_cols[date] = None
            date = date + one_day

        self.start_date = start_date
        self.end_date = end_date
        self.dates = dates
        self.date_cols = date_cols

    def region(self, row):
        return row[self.region_col].strip()

    def subregion(self, row):
        return row[self.subregion_col].strip()

    def latitude(self, row):
        return row[self.lat_col].strip()

    def longitude(self, row):
        return row[self.long_col].strip()

    def data(self, row):
        data = []
        for date in self.dates:
            col = self.date_cols[date]
            data.append(int(row[col]))
        return data


class RegionSourceData:
    @staticmethod
    def from_row(data_format, row):
        return RegionSourceData(
            data_format = data_format,
            region = data_format.region(row),
            subregion = data_format.subregion(row),
            latitude = data_format.latitude(row),
            longitude = data_format.longitude(row),
            data = data_format.data(row)
        )

    def __init__(self, data_format=None, region=None, subregion=None, latitude=None, longitude=None, data=None):
        self.data_format = data_format
        self.region = region
        self.subregion = subregion
        self.latitude = latitude
        self.longitude = longitude
        self.data = data


class SubregionData:
    def __init__(self, source_cases, population_data):
        self.name = '%s/%s' % (source_cases.region.lower(), source_cases.subregion.lower(),)
        self.region = source_cases.region
        self.subregion = source_cases.subregion
        self.latitude = source_cases.latitude
        self.longitude = source_cases.longitude
        self.cases = source_cases.data
        fq_subregion = self.name.lower()
        if fq_subregion in population_data:
            self.population = population_data[fq_subregion]
        else:
            self.population = None


class RegionData:
    def __init__(self, region, subregions, population_data):
        self.name = region.lower()
        self.region = region
        self.subregion = region
        self.subregions = dict([(sr.name, sr,) for sr in subregions])
        self.cases = [sum([(rs.cases[i] or 0) for rs in subregions]) for i in range(len(subregions[0].cases))]
        fq_region = self.region.lower()
        if fq_region in population_data:
            self.population = population_data[fq_region]
        else:
            self.population = sum([(rs.population or 0) for rs in subregions])
            if self.population == 0:
                self.population = None


def read_data(data_path):
    with open(data_path) as csv_file:
        data_reader = csv.reader(csv_file)
        data_format = None
        region_data = []
        for row in data_reader:
            if not data_format:
                data_format = DataFormat(row)
            else:
                region_data.append(RegionSourceData.from_row(data_format, row))
        return (data_format, region_data,)


def extract_regions(subregions):
    region_groups = {}
    regions = {}
    for sr in subregions:
        region_name = sr.region.lower()
        if not region_name in region_groups:
            region_groups[region_name] = []
        region_groups[region_name].append(sr)
    for g in region_groups.values():
        region = RegionData(g[0].region, g, region_populations)
        regions[region.name] = region
    return regions


def merge_data(subregion_cases):
    subregions = {}
    for source_cases in subregion_cases:
        sr = SubregionData(source_cases, region_populations)
        subregions[sr.name] = sr
    return subregions, extract_regions(subregions.values())


# Main script begins here

cases_data_format, subregion_cases = \
    read_data('COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
subregions, regions = merge_data(subregion_cases)
canada_subregions = [sr for sr in regions['canada'].subregions.values() if sr.population]
reference_region = SubregionData(
    RegionSourceData(subregion='33% Daily Growth', region='',
        data=[1.33 ** x for x in range(len(cases_data_format.dates))]), {})
reference_region.population = 100000

dates = cases_data_format.dates

def plot_srs(title, xlabel, ylabel, yscale, srs, threshold_func, data_func, label_func):
    fig, ax = plt.subplots()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale(yscale)
    ax.set_title(title)

    min_len = 3

    plots_to_do = []
    for sr in srs:
        start = threshold_func(sr)
        if start != None:
            data = data_func(sr)[start:]
            if len(data) >= min_len:
                plots_to_do.append((list(range(len(data))), data, label_func(sr),))

    if len(plots_to_do) < 1:
        print('No data for chart "%s"' % (title,))

    # compute max_len: finds the longest series but rejecting outliers
    # threshold for being an outlier is, >2x as long as our 75% longest series
    # (if you order them by length -- bottom of the top quartile)
    series_lengths = sorted([len(p[0]) for p in plots_to_do])
    bottom_of_top_quartile = (len(series_lengths) - 1) * 3 // 4
    max_len = max([series_lengths[i] for i in range(bottom_of_top_quartile, len(series_lengths))
        if series_lengths[i] * 0.5 <= series_lengths[bottom_of_top_quartile]])

    for p in plots_to_do:
        ax.plot(p[0][:max_len], p[1][:max_len], label=p[2])
    ax.legend()
    plt.show()

def threshold_cases_per_100k(sr, threshold):
    cases = sr.cases
    for i in range(len(cases)):
        per_100k = cases[i] * 100000 / sr.population
        if per_100k >= threshold:
            print('Data for %s begins at %s, with %d infected (%g per 100,000)'
                % (sr.subregion, dates[i], cases[i], per_100k))
            return i
    else:
        return None

def get_cases(sr):
    return sr.cases

def get_delta_cases(sr):
    return [(sr.cases[i] - (sr.cases[i - 1] if i > 0 else 0)) for i in range(len(sr.cases))]

def get_ongoing_cases(sr):
    new_cases = get_delta_cases(sr)
    return [sum(new_cases[max(0, i - 13):i + 1]) for i in range(len(new_cases))]

def convert_per_100k(sr, data):
    return [d * 100000 / sr.population for d in data]


significant_canada_subregions = [subregions[n] for n in ['canada/british columbia', 'canada/alberta', 'canada/ontario', 'canada/quebec']]
international_subregions = [subregions[n] for n in ['canada/british columbia', 'china/hubei']] + [regions[n] for n in ['canada', 'korea, south', 'japan', 'italy', 'france', 'us', 'germany']]

plot_srs(
    title='COVID-19 Confirmed Cases In Canada, Normalized By Population',
    xlabel='Days Since 1 Case Per 100,000 Confirmed',
    ylabel='Confirmed Cases Per 100,000',
    yscale='log',
    srs=[reference_region] + significant_canada_subregions,
    threshold_func=lambda sr: threshold_cases_per_100k(sr, 1),
    data_func=lambda sr: convert_per_100k(sr, get_cases(sr)),
    label_func=lambda sr: sr.subregion)

plot_srs(
    title='COVID-19 Confirmed Cases Internationally, Normalized By Population',
    xlabel='Days Since 1 Case Per 100,000 Confirmed',
    ylabel='Confirmed Cases Per 100,000',
    yscale='log',
    srs=[reference_region] + international_subregions,
    threshold_func=lambda sr: threshold_cases_per_100k(sr, 1),
    data_func=lambda sr: convert_per_100k(sr, get_cases(sr)),
    label_func=lambda sr: sr.subregion)

plot_srs(
    title='New COVID-19 Confirmed Cases In Canada, Normalized By Population',
    xlabel='Days Since 1 Case Per 100,000 Confirmed',
    ylabel='New Confirmed Cases Per 100,000',
    yscale='linear',
    srs=significant_canada_subregions,
    threshold_func=lambda sr: threshold_cases_per_100k(sr, 1),
    data_func=lambda sr: convert_per_100k(sr, get_delta_cases(sr)),
    label_func=lambda sr: sr.subregion)

plot_srs(
    title='COVID-19 Cases Presumed In Treatment In Canada, Normalized By Population',
    xlabel='Days Since Data Start',
    ylabel='Confirmed Cases Per 100,000',
    yscale='linear',
    srs=significant_canada_subregions,
    threshold_func=lambda sr: 0,
    data_func=lambda sr: convert_per_100k(sr, get_ongoing_cases(sr)),
    label_func=lambda sr: sr.subregion)

plot_srs(
    title='COVID-19 Cases Presumed In Treatment Internationally, Normalized By Population',
    xlabel='Days Since 1 Case Per 100,000 Confirmed',
    ylabel='Confirmed Cases Per 100,000',
    yscale='linear',
    srs=international_subregions,
    threshold_func=lambda sr: threshold_cases_per_100k(sr, 1),
    data_func=lambda sr: convert_per_100k(sr, get_ongoing_cases(sr)),
    label_func=lambda sr: sr.subregion)

