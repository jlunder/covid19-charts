#!/usr/bin/python3
import sys, os, time, csv, re, datetime, math, itertools
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

    'canada': 37855702,
    'us': 331002651,
    'italy': 60484989,
    'korea, south': 51269185,
    'china': 1408526449,
    'japan': 126476461,
    'germany': 83783942,
    'singapore': 5850342,
    'france': 65270000,
    'sweden': 10099265,
    }

class DataFormat:
    subregion_col = -1
    adminregion_col = -1
    region_col = -1
    lat_col = -1
    long_col = -1
    population_col = -1
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
            elif header_strip.startswith('admin'):
                self.adminregion_col = col
            elif header_strip.startswith('country'):
                self.region_col = col
            elif header_strip.startswith('lat'):
                self.lat_col = col
            elif header_strip.startswith('long'):
                self.long_col = col
            elif header_strip.startswith('population'):
                self.population_col = col
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
        if self.region_col < 0:
            return ''
        return row[self.region_col].strip()

    def subregion(self, row):
        if self.subregion_col < 0:
            return ''
        return row[self.subregion_col].strip()

    def adminregion(self, row):
        if self.adminregion_col < 0:
            return ''
        return row[self.adminregion_col].strip()

    def latitude(self, row):
        if self.lat_col < 0:
            return ''
        return row[self.lat_col].strip()

    def longitude(self, row):
        if self.long_col < 0:
            return ''
        return row[self.long_col].strip()

    def population(self, row):
        if self.population_col < 0:
            return None
        return float(row[self.population_col].strip())

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
            adminregion = data_format.adminregion(row).strip(),
            latitude = data_format.latitude(row),
            longitude = data_format.longitude(row),
            population = data_format.population(row),
            data = data_format.data(row)
        )

    def __init__(self, data_format=None, region=None, subregion=None, adminregion=None,
                 latitude=None, longitude=None, population=None, data=None):
        self.data_format = data_format
        self.id = region.lower()
        if subregion:
            self.id += '/' + subregion.lower()
            if adminregion:
                self.id += '/' + adminregion.lower()
        self.region = region
        self.subregion = subregion
        self.adminregion = adminregion
        self.latitude = latitude
        self.longitude = longitude
        self.population = population
        self.data = data


nominal_growth = 1.33
nominal_death_rate = 0.0037
nominal_recovery = 14
nominal_confirm = 14
nominal_death = 18
nominal_confirm_to_death = nominal_death - nominal_confirm
nominal_confirm_to_recovery = nominal_confirm - nominal_recovery
default_smooth = 6


def convert_delta(data):
    return [(data[i] - (data[i - 1] if i > 0 else 0)) for i in range(len(data))]

def convert_smooth(data, days=default_smooth):
    if len(data) < days:
        return []
    return [sum(data[i:i + days]) / days for i in range(len(data) - days + 1)]

def threshold(data, val):
    for d, i in zip(data, itertools.count()):
        if d >= val:
            return i
    else:
        return None


class RegionData:
    deaths = None
    recovered = None
    population = None

    def __init__(self):
        self._memo = {}
        self._confirm_days = nominal_confirm
        self._death_days = nominal_death
        self._recovery_days = nominal_recovery

    def set_confirm_days(self, confirm_days):
        self._confirm_days = confirm_days
        self._memo = {}

    def confirm_days(self):
        return self._confirm_days

    def death_days(self):
        return self._death_days

    def confirm_to_death(self):
        return self._death_days - self._confirm_days

    def confirmed_total(self):
        return self.cases

    def confirmed_new(self):
        return self.memoize('confirmed_new', lambda: convert_delta(self.confirmed_total()))

    def confirmed_ongoing(self):
        def data_fun():
            if self.deaths and self.recovered:
                return [self.cases[i] - (self.deaths[i] + self.recovered[i])
                        for i in range(len(self.cases))]
            else:
                new_cases = self.confirmed_new()
                return [sum(new_cases[max(0, i - nominal_recovery):i + 1])
                        for i in range(len(new_cases))]
        return self.memoize('confirmed_ongoing', data_fun)

    def deaths_total(self):
        if self.deaths:
            return self.deaths
        def data_fun():
            return [c * nominal_death_rate for c in self.cases[nominal_confirm_to_death:]]
        return self.memoize('deaths_total', data_fun)

    def deaths_new(self):
        return self.memoize('deaths_new', lambda: convert_delta(self.deaths_total()))

    def estimated_total(self):
        def estimated_fun():
            return [d / nominal_death_rate for d in self.deaths]
        return self.deaths_estimated('estimated_total', self.confirmed_total, estimated_fun)

    def estimated_new(self):
        def estimated_fun():
            return convert_delta(self.estimated_total())
        return self.deaths_estimated('estimated_new', self.confirmed_new, estimated_fun)

    def estimated_ongoing(self):
        def estimated_fun():
            new_cases = self.estimated_new()
            return [sum(new_cases[max(0, i - nominal_recovery):i + 1])
                    for i in range(len(new_cases))]
        return self.deaths_estimated('estimated_ongoing', self.confirmed_ongoing, estimated_fun)

    def estimation_factor(self):
        def estimated_fun():
            estimated_total = self.estimated_total()
            ce = len(estimated_total) - self.confirm_to_death()
            cs = max(0, ce - default_smooth)
            c_sum = sum(self.confirmed_total()[cs:ce])
            e_sum = sum(self.estimated_total()[-(ce - cs):])
            return e_sum / c_sum if (c_sum > 0 and e_sum > 0) else 1
        return self.deaths_estimated('estimation_factor', lambda: 1, estimated_fun)

    def percapita_confirmed_total(self):
        return self.percapita('percapita_confirmed_total', self.confirmed_total)

    def percapita_confirmed_new(self):
        return self.percapita('percapita_confirmed_new', self.confirmed_new)

    def percapita_confirmed_ongoing(self):
        return self.percapita('percapita_confirmed_ongoing', self.confirmed_ongoing)

    def percapita_deaths_total(self):
        return self.percapita('percapita_deaths_total', self.deaths_total)

    def percapita_deaths_new(self):
        return self.percapita('percapita_deaths_new', self.deaths_new)

    def percapita_estimated_total(self):
        return self.percapita('percapita_estimated_total', self.estimated_total)

    def percapita_estimated_new(self):
        return self.percapita('percapita_estimated_new', self.estimated_new)

    def percapita_estimated_ongoing(self):
        return self.percapita('percapita_estimated_ongoing', self.estimated_ongoing)

    def memoize(self, name, data_fun):
        if not name in self._memo:
            self._memo[name] = data_fun()
        return self._memo[name]

    def percapita(self, name, data_fun):
        return self.memoize(name, (lambda: [x / self.population for x in data_fun()]) if self.population else data_fun)

    def deaths_estimated(self, name, confirmed_fun, estimated_fun):
        return self.memoize(name, estimated_fun if self.deaths else confirmed_fun)

    @staticmethod
    def from_source(source_cases, source_deaths, source_recovered, population_data):
        self = RegionData()
        if source_cases.adminregion:
            self.name = '%s/%s/%s' % (source_cases.region.lower(), source_cases.subregion.lower(), source_cases.adminregion.lower(),)
        elif source_cases.subregion:
            self.name = '%s/%s' % (source_cases.region.lower(), source_cases.subregion.lower(),)
        else:
            self.name = source_cases.region.lower()
        self.id = self.name.lower()
        self.prettyname = source_cases.adminregion or source_cases.subregion or source_cases.region
        self.region = source_cases.region
        self.subregion = source_cases.subregion
        self.adminregion = source_cases.adminregion
        self.child_regions = None
        self.latitude = source_cases.latitude
        self.longitude = source_cases.longitude
        self.cases = source_cases.data
        if source_deaths:
            self.deaths = source_deaths.data
        if source_recovered:
            self.recovered = source_recovered.data
        if self.id in population_data:
            self.population = population_data[self.id]
        else:
            self.population = (source_cases.population
                               or (source_deaths.population if source_deaths else None)
                               or (source_recovered.population if source_recovered else None))
        return self

    @staticmethod
    def from_child_regions(region, subregion, child_regions, population_data):
        self = RegionData()
        self.name = '%s/%s' % (region, subregion,) if subregion else region
        self.id = self.name.lower()
        self.prettyname = subregion or region
        self.region = region
        self.subregion = subregion
        self.adminregion = ''
        self.child_regions = dict(((r.id, r,) for r in child_regions))
        self.cases = [sum(r.cases[i] for r in child_regions) for i in range(len(child_regions[0].cases))]
        if all(r.deaths for r in child_regions):
            self.deaths = [sum(r.deaths[i] for r in child_regions) for i in range(len(child_regions[0].cases))]
        if all(r.recovered for r in child_regions):
            self.recovered = [sum(r.recovered[i] for r in child_regions) for i in range(len(child_regions[0].cases))]
        if self.id in population_data:
            self.population = population_data[self.id]
        if all((r.population for r in child_regions)):
            self.population = sum([r.population for r in child_regions])
        return self


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


def group_regions(child_regions):
    subregion_groups = {}
    region_groups = {}
    regions = {}
    for r in child_regions:
        region_name = r.region.lower()
        if r.adminregion:
            subregion_name = '%s/%s' % (region_name, r.subregion.lower(),)
            if not subregion_name in subregion_groups:
                subregion_groups[subregion_name] = []
            subregion_groups[subregion_name].append(r)
        elif r.subregion:
            if not region_name in region_groups:
                region_groups[region_name] = []
            region_groups[region_name].append(r)
        else:
            if region_name in regions:
                print('region %s appears multiple times in data?' % (region_name,))
            regions[region_name] = r
    for g in subregion_groups.values():
        region_name = g[0].region.lower()
        subregion_name = g[0].subregion.lower()
        subregion = RegionData.from_child_regions(g[0].region, g[0].subregion, g, region_populations)
        if not region_name in region_groups:
            region_groups[region_name] = []
        region_groups[region_name].append(subregion)
    for g in region_groups.values():
        region_name = g[0].region.lower()
        region = RegionData.from_child_regions(g[0].region, '', g, region_populations)
        if region_name in regions:
            composite_region = regions[region_name]
            if composite_region.child_regions:
                print('region %s already has child regions?' % (region_name,))
            composite_region.child_regions = dict(((r.id, r) for r in g))
            if region.population and not composite_region.population:
                composite_region.population = region.population
            if (region.deaths and not composite_region.deaths):
                composite_region.cases = region.cases
                composite_region.deaths = region.deaths
                composite_region.recovered = region.recovered
        else:
            regions[region_name] = region
    return regions


def merge_data(subregion_cases, subregion_deaths, subregion_recovered):
    subregions = {}
    subregion_cases_names = {}

    for source in subregion_cases:
        subregion_cases_names[source.id] = None
    indexed_deaths = {}
    indexed_recovered = {}
    for source in subregion_deaths:
        if source.id in subregion_cases_names:
            if source.id in indexed_deaths:
                print('Dup subregion %s in deaths data' % (source.id,))
            indexed_deaths[source.id] = source
    for source in subregion_recovered:
        if source.id in subregion_cases_names:
            if source.id in indexed_recovered:
                print('Dup subregion %s in deaths data' % (source.id,))
            indexed_recovered[source.id] = source
    for source_cases in subregion_cases:
        source_deaths = indexed_deaths[source_cases.id] if source_cases.id in indexed_deaths else None
        source_recovered = indexed_recovered[source_cases.id] if source_cases.id in indexed_recovered else None
        sr = RegionData.from_source(source_cases, source_deaths, source_recovered, region_populations)
        subregions[sr.id] = sr
    return subregions, group_regions(subregions.values())


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
        return XAxis(str(dates[day_num]), lambda r, y: day_num)

    @staticmethod
    def since_threshold(data_name, val, data_func):
        return XAxis('Days Since %g %s' % (val, data_name,),
                     lambda r: threshold(data_func(r), val))

    @staticmethod
    def since_yaxis_threshold(val):
        return XAxis(None, None,
                     lambda yaxis: XAxis('Days Since %g %s' % (val, yaxis.axis_label,),
                                   lambda r: threshold(yaxis.data_values(r), val)))


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
        data=[nominal_death_rate * nominal_growth ** (x - 14) for x in range(len(cases_data_format.dates))]),
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
                                                             'france', 'us', 'germany', 'sweden']])

if False:
    plot_srs(xaxis=XAxis.since_yaxis_threshold(10),
             yaxis=YAxis.log_confirmed_total_per_million(),
             regions=significant_canada_subregions)

if False:
    plot_srs(xaxis=XAxis.since_yaxis_threshold(10),
             yaxis=YAxis.log_confirmed_total_per_million(),
             regions=international_subregions)

if True:
    plot_srs(xaxis=XAxis.since_threshold('Cases Per 1M', 10, lambda r: [c * 1e6 for c in r.percapita_confirmed_total()]),
             yaxis=YAxis.log_confirmed_new_per_million_smoothed(),
             regions=significant_canada_subregions,)

if True:
    plot_srs(xaxis=XAxis.since_threshold('Cases Per 1M', 10, lambda r: [c * 1e6 for c in r.percapita_confirmed_total()]),
             yaxis=YAxis.log_confirmed_new_per_million_smoothed(),
             regions=international_subregions,)

if True:
    plot_srs(xaxis=XAxis.since_threshold('Deaths Per 1M', 1, lambda r: convert_smooth([c * 1e6 for c in r.percapita_deaths_total()])),
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

