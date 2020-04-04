#!/usr/bin/python3
import sys, os, time, csv, re, datetime, math
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
            region = data_format.region(row).strip(),
            subregion = data_format.subregion(row).strip(),
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
    deaths = None
    recovered = None

    def __init__(self, source_cases, source_deaths, source_recovered, population_data):
        self.name = '%s/%s' % (source_cases.region.lower(), source_cases.subregion.lower(),)
        self.region = source_cases.region
        self.subregion = source_cases.subregion
        self.latitude = source_cases.latitude
        self.longitude = source_cases.longitude
        self.cases = source_cases.data
        if source_deaths:
            self.deaths = source_deaths.data
        if source_recovered:
            self.recovered = source_recovered.data
        fq_subregion = self.name.lower()
        if fq_subregion in population_data:
            self.population = population_data[fq_subregion]
        else:
            self.population = None


class RegionData:
    deaths = None
    recovered = None

    def __init__(self, region, subregions, population_data):
        self.name = region.lower()
        self.region = region
        self.subregion = region
        self.subregions = dict([(sr.name, sr,) for sr in subregions])
        self.cases = [sum(sr.cases[i] for sr in subregions) for i in range(len(subregions[0].cases))]
        if all(sr.deaths for sr in subregions):
            self.deaths = [sum(sr.deaths[i] for sr in subregions) for i in range(len(subregions[0].cases))]
        if all(sr.recovered for sr in subregions):
            self.recovered = [sum(sr.recovered[i] for sr in subregions) for i in range(len(subregions[0].cases))]
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


def merge_data(subregion_cases, subregion_deaths, subregion_recovered):
    def get_source_fq(source):
        norm_fq_region = source.region.lower()
        if source.subregion:
            norm_fq_region += '/' + source.subregion.lower()
        return norm_fq_region

    subregions = {}
    subregion_cases_names = {}

    for source in subregion_cases:
        subregion_cases_names[get_source_fq(source)] = None
    indexed_deaths = {}
    indexed_recovered = {}
    for source in subregion_deaths:
        fq = get_source_fq(source)
        if fq in subregion_cases_names:
            if fq in indexed_deaths:
                print('Dup subregion %s in deaths data' % (fq,))
            indexed_deaths[fq] = source
    for source in subregion_recovered:
        fq = get_source_fq(source)
        if fq in subregion_cases_names:
            if fq in indexed_recovered:
                print('Dup subregion %s in deaths data' % (fq,))
            indexed_recovered[fq] = source
    for source_cases in subregion_cases:
        fq = get_source_fq(source_cases)
        source_deaths = indexed_deaths[fq] if fq in indexed_deaths else None
        source_recovered = indexed_recovered[fq] if fq in indexed_recovered else None
        if fq == 'canada/british columbia':
            print('bc:', source_deaths.data)
        sr = SubregionData(source_cases, source_deaths, source_recovered, region_populations)
        subregions[sr.name] = sr
    return subregions, extract_regions(subregions.values())


# Main script begins here

def plot_srs(title, xlabel, ylabel, srs, threshold_func, data_func, label_func, yscale='log'):
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

def plot_compare(title, xlabel, ylabel, yscale, srs, threshold_func, data_funcs, label_funcs):
    fig, ax = plt.subplots()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale(yscale)
    ax.set_title(title)
    for sr in srs:
        start = threshold_func(sr)
        if start != None:
            for i in range(len(data_funcs)):
                data = data_funcs[i](sr)[start:]
                ax.plot(range(max(0, len(data) - start)), data[start:], label=label_funcs[i](sr))
    ax.legend()
    plt.show()


nominal_growth = 1.33
nominal_death_rate = 0.016
nominal_death_days = 18
nominal_confirm_days = 3
default_smooth = 6


def convert_delta(data):
    return [(data[i] - (data[i - 1] if i > 0 else 0)) for i in range(len(data))]

def convert_per_100k(sr, data):
    return [d * 100000 / sr.population for d in data]

def convert_smooth(data, days):
    if len(data) < days:
        return []
    return [sum(data[i:i + days]) / days for i in range(len(data) - days + 1)]

def threshold(data, val):
    for i in range(len(data)):
        if data[i] >= val:
            return i
    else:
        return None

def get_confirmed_cases(sr):
    return sr.cases

def get_estimated_ongoing_confirmed_cases(sr):
    new_cases = convert_delta(get_confirmed_cases(sr))
    return [sum(new_cases[max(0, i - nominal_death_days):i + 1]) for i in range(len(new_cases))]

def get_ongoing_confirmed_cases(sr):
    if sr.deaths and sr.recovered:
        return [sr.cases[i] - (sr.deaths[i] + sr.recovered[i]) for i in range(len(sr.cases))]
    else:
        return get_estimated_ongoing_confirmed_cases(sr)

def get_deaths_estimated_cases(sr, nominal_death_days=nominal_death_days, nominal_confirm_days=nominal_confirm_days):
    days_diff = nominal_death_days - nominal_confirm_days
    return [sr.deaths[i + days_diff] / nominal_death_rate for i in range(max(len(sr.deaths) - days_diff, 0))]

def get_estimated_cases(sr, nominal_death_days=nominal_death_days, nominal_confirm_days=nominal_confirm_days):
    deaths_estimated_cases = get_deaths_estimated_cases(sr, nominal_death_days, nominal_confirm_days)
    confirmed_sum = sum(get_confirmed_cases(sr)[:len(deaths_estimated_cases)])
    estimated_sum = sum(deaths_estimated_cases)
    if confirmed_sum > 0 and estimated_sum > 0:
        return [c * estimated_sum / confirmed_sum for c in sr.cases]
    else:
        return cases

def get_ongoing_estimated_cases(sr):
    new_cases = convert_delta(get_estimated_cases(sr))
    return [sum(new_cases[max(0, i - nominal_death_days):i + 1]) for i in range(len(new_cases))]

def get_deaths(sr):
    if sr.deaths:
        return sr.deaths
    else:
        new_cases = convert_delta(get_confirmed_cases(sr))
        return [sum(new_cases[max(0, i - nominal_death_days):i + 1]) for i in range(len(new_cases))]

def threshold_cases_per_100k(sr, val):
    return threshold(convert_per_100k(sr, get_confirmed_cases(sr)), val)


cases_data_format, subregion_cases = \
    read_data('COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
deaths_data_format, subregion_deaths = \
    read_data('COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
recovered_data_format, subregion_recovered = \
    read_data('COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
subregions, regions = merge_data(subregion_cases, subregion_deaths, subregion_recovered)
canada_subregions = [sr for sr in regions['canada'].subregions.values() if sr.population]

reference_region = SubregionData(
    RegionSourceData(subregion='33% Daily Growth', region='',
        data=[nominal_growth ** x for x in range(len(cases_data_format.dates))]),
    RegionSourceData(subregion='33% Daily Growth', region='',
        data=[nominal_death_rate * nominal_growth ** (x - 14) for x in range(len(cases_data_format.dates))]),
    RegionSourceData(subregion='33% Daily Growth', region='',
        data=[(1 - nominal_death_rate) * nominal_growth ** (x - 14) for x in range(len(cases_data_format.dates))]),
    {}
    )
reference_region.population = 100000

dates = cases_data_format.dates

significant_canada_subregions = [subregions[n] for n in ['canada/british columbia', 'canada/alberta', 'canada/ontario', 'canada/quebec']]
international_subregions = [subregions[n] for n in ['canada/british columbia', 'china/hubei']] + [regions[n] for n in ['canada', 'korea, south', 'japan', 'italy', 'france', 'us', 'germany']]

if True:
    plot_srs(
        title='COVID-19 Confirmed Cases In Canada',
        xlabel='Days Since 1 Case Per 100,000 Confirmed',
        ylabel='Confirmed Cases Per 100,000',
        srs=[reference_region] + significant_canada_subregions,
        threshold_func=lambda sr: threshold_cases_per_100k(sr, 1),
        data_func=lambda sr: convert_per_100k(sr, get_confirmed_cases(sr)),
        label_func=lambda sr: sr.subregion)

if False:
    plot_srs(
        title='COVID-19 Confirmed Cases Internationally',
        xlabel='Days Since 1 Case Per 100,000 Confirmed',
        ylabel='Confirmed Cases Per 100,000',
        srs=[reference_region] + international_subregions,
        threshold_func=lambda sr: threshold_cases_per_100k(sr, 1),
        data_func=lambda sr: convert_per_100k(sr, get_confirmed_cases(sr)),
        label_func=lambda sr: sr.subregion)

if True:
    plot_srs(
        title='New COVID-19 Confirmed Cases In Canada',
        xlabel='Days Since 1 Case Per 100,000 Confirmed',
        ylabel='New Confirmed Cases Per 100,000',
        srs=significant_canada_subregions,
        threshold_func=lambda sr: threshold_cases_per_100k(sr, 1),
        data_func=lambda sr: convert_per_100k(sr, convert_delta(convert_smooth(get_confirmed_cases(sr), default_smooth))),
        label_func=lambda sr: sr.subregion)

if True:
    plot_srs(
        title='COVID-19 Cases In Treatment In Canada',
        xlabel='Days Since ' + str(dates[40]),
        ylabel='Cases In Treatment Per 100,000',
        srs=significant_canada_subregions,
        threshold_func=lambda sr: 0,
        data_func=lambda sr: convert_per_100k(sr, get_ongoing_confirmed_cases(sr))[40:],
        label_func=lambda sr: sr.subregion)

if False:
    plot_srs(
        title='COVID-19 Cases In Treatment Internationally',
        xlabel='Days Since 1 Case Per 100,000 Confirmed',
        ylabel='Cases In Treatment Per 100,000',
        srs=international_subregions,
        threshold_func=lambda sr: threshold_cases_per_100k(sr, 1),
        data_func=lambda sr: convert_per_100k(sr, get_ongoing_confirmed_cases(sr)),
        label_func=lambda sr: sr.subregion)

if False:
    plot_srs(
        title='Deaths From COVID-19 In Canada',
        xlabel='Days Since ' + str(dates[40]),
        ylabel='Deaths Per 100,000',
        srs=significant_canada_subregions,
        threshold_func=lambda sr: 0,
        data_func=lambda sr: convert_per_100k(sr, get_deaths(sr))[40:],
        label_func=lambda sr: sr.subregion)

if True:
    plot_srs(
        title='Estimated Actual COVID-19 Cases In Canada',
        xlabel='Days Since 1 Case Per 100,000 Estimated',
        ylabel='Estimated Cases Per 100,000',
        srs=[reference_region] + significant_canada_subregions,
        threshold_func=lambda sr: threshold(convert_per_100k(sr, get_estimated_cases(sr)), 1),
        data_func=lambda sr: convert_per_100k(sr, get_estimated_cases(sr)),
        label_func=lambda sr: sr.subregion)

if False:
    plot_srs(
        title='Estimated Actual COVID-19 Cases Internationally',
        xlabel='Days Since 1 Case Per 100,000 Estimated',
        ylabel='Estimated Cases Per 100,000',
        srs=[reference_region] + international_subregions,
        threshold_func=lambda sr: threshold_cases_per_100k(sr, 1),
        data_func=lambda sr: convert_per_100k(sr, get_estimated_cases(sr)),
        label_func=lambda sr: sr.subregion)

if False:
    plot_srs(
        title='Estimated Actual COVID-19 Cases Internationally',
        xlabel='Days Since 1 Case Per 100,000 Estimated',
        ylabel='Estimated Cases Per 100,000',
        srs=[reference_region] + international_subregions,
        threshold_func=lambda sr: threshold_cases_per_100k(sr, 1),
        data_func=lambda sr: convert_per_100k(sr, get_estimated_cases(sr)),
        label_func=lambda sr: sr.subregion)

if False:
    plot_compare(
        title='Comparison Of Estimated Vs. Confirmed Cases',
        xlabel='Days Since 1 Case Per 100,000 Confirmed',
        ylabel='Estimated Cases Per 100,000',
        srs=[subregions['canada/british columbia']],#international_subregions,
        threshold_func=lambda sr: 0,
        data_funcs=[
            lambda sr: convert_per_100k(sr, get_confirmed_cases(sr)),
            lambda sr: convert_per_100k(sr, get_estimated_cases(sr)),
        ],
        label_funcs=[
            lambda sr: sr.subregion + ' (confirmed)',
            lambda sr: sr.subregion + ' (estimated)',
        ])

if False:
    plot_srs(
        title='Comparison Of Estimated Vs. Confirmed Cases In Canada',
        xlabel='Days Since ' + str(dates[40]),
        ylabel='Estimated Actual Cases / Confirmed Cases',
        srs=significant_canada_subregions,
        threshold_func=lambda sr: 0,
        data_func=lambda sr: list(map(lambda c, p: c / p if p else 1, get_confirmed_cases(sr), get_estimated_cases(sr)))[40:],
        label_func=lambda sr: sr.subregion)

if False:
    plot_srs(
        title='Comparison Of Estimated Vs. Confirmed Cases Internationally',
        xlabel='Days Since ' + str(dates[40]),
        ylabel='Estimated Actual Cases / Confirmed Cases',
        srs=international_subregions,
        threshold_func=lambda sr: 0,
        data_func=lambda sr: list(map(lambda c, p: p / c if c else 1, get_confirmed_cases(sr), get_estimated_cases(sr)))[40:],
        label_func=lambda sr: sr.subregion)

if True:
    fig, ax = plt.subplots()
    ax.set_xlabel('Days Between Contraction And Detection')
    ax.set_ylabel('RMS Of Overall Difference')
    ax.set_yscale('linear')
    ax.set_title('Fit Of Confirmed To Estimated Cases')
    for sr in significant_canada_subregions + international_subregions:
        base_confirmed_cases = convert_smooth(get_confirmed_cases(sr), default_smooth)

        start = -1
        for i in range(len(base_confirmed_cases)):
            if base_confirmed_cases[i] > 100:
                start = i
                break

        xs = []
        ys = []
        for i in range(21):
            base_estimated_cases = convert_smooth(get_deaths_estimated_cases(sr, nominal_confirm_days=i), default_smooth)
            if start >= 0 and start < len(base_estimated_cases):
                estimated_cases = base_estimated_cases[start:]
                confirmed_cases = base_confirmed_cases[start:len(base_estimated_cases)]
                avg_estimated = sum(estimated_cases) / len(estimated_cases) or 1
                avg_confirmed = sum(confirmed_cases) / len(confirmed_cases) or 1
                xs.append(i)
                #ys.append(avg_estimated / avg_confirmed)

                ys.append(math.sqrt(sum(map(lambda d: (d[0] - d[1]) ** 2,
                    zip((p / avg_estimated for p in estimated_cases),
                        (c / avg_confirmed for c in confirmed_cases))))))
        ax.plot(xs, ys, label=sr.subregion)
    ax.legend()
    plt.show()

if True:
    fig, ax = plt.subplots()
    ax.set_xlabel('Ongoing Cases Per 100,000')
    ax.set_xscale('log')
    ax.set_ylabel('New Cases Per 100,000')
    ax.set_yscale('log')
    ax.set_title('New Cases For Existing Cases In Canada')
    max_new = 0
    max_ongoing = 0
    for sr in significant_canada_subregions:# + international_subregions:
        estimated_cases = convert_per_100k(sr, convert_smooth(get_estimated_cases(sr), default_smooth))
        start = threshold(estimated_cases, 1)
        if start != None and start < len(estimated_cases):
            new_cases = convert_delta(estimated_cases)[start:]
            ongoing_cases = convert_per_100k(sr, convert_smooth(get_ongoing_estimated_cases(sr), default_smooth)[start:])
            max_new = max(max_new, max(new_cases))
            max_ongoing = max(max_ongoing, max(ongoing_cases))
            ax.plot(ongoing_cases, new_cases, label=sr.subregion)
    max_series = [1, max(max_new, max_ongoing)]
    ax.plot(max_series, max_series, label='Danger!')
    ax.legend()
    plt.show()
