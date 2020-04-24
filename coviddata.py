import csv, re, datetime, itertools

one_day = datetime.timedelta(days=1)

nominal_growth = 1.33
nominal_death_rate = 0.005
nominal_recovery = 14
nominal_confirm = 14
nominal_death = 18
nominal_confirm_to_death = nominal_death - nominal_confirm
nominal_confirm_to_recovery = nominal_confirm - nominal_recovery

default_smooth = 6

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

    'us/california': 39937489,
    'us/texas': 29472295,
    'us/florida': 21992985,
    'us/new york': 19440469,
    'us/pennsylvania': 12820878,
    'us/illinois': 12659682,
    'us/ohio': 11747694,
    'us/georgia': 10736059,
    'us/north carolina': 10611862,
    'us/michigan': 10045029,
    'us/new jersey': 8936574,
    'us/virginia': 8626207,
    'us/washington': 7797095,
    'us/arizona': 7378494,
    'us/massachusetts': 6976597,
    'us/tennessee': 6897576,
    'us/indiana': 6745354,
    'us/missouri': 6169270,
    'us/maryland': 6083116,
    'us/wisconsin': 5851754,
    'us/colorado': 5845526,
    'us/minnesota': 5700671,
    'us/south carolina': 5210095,
    'us/alabama': 4908621,
    'us/louisiana': 4645184,
    'us/kentucky': 4499692,
    'us/oregon': 4301089,
    'us/oklahoma': 3954821,
    'us/connecticut': 3563077,
    'us/utah': 3282115,
    'us/iowa': 3179849,
    'us/nevada': 3139658,
    'us/arkansas': 3038999,
    'us/puerto rico': 3032165,
    'us/mississippi': 2989260,
    'us/kansas': 2910357,
    'us/new mexico': 2096640,
    'us/nebraska': 1952570,
    'us/idaho': 1826156,
    'us/west virginia': 1778070,
    'us/hawaii': 1412687,
    'us/new hampshire': 1371246,
    'us/maine': 1345790,
    'us/montana': 1086759,
    'us/rhode island': 1056161,
    'us/delaware': 982895,
    'us/south dakota': 903027,
    'us/north dakota': 761723,
    'us/alaska': 734002,
    'us/district of columbia': 720687,
    'us/vermont': 628061,
    'us/wyoming': 567025,

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
    'brazil': 209500000,
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
        self._death_rate = nominal_death_rate
        self._confirm_days = nominal_confirm
        self._death_days = nominal_death
        self._recovery_days = nominal_recovery

    def death_rate(self):
        return self._death_rate

    def set_confirm_days(self, confirm_days):
        self._confirm_days = confirm_days
        self._memo = {}

    def confirm_days(self):
        return self._confirm_days

    def death_days(self):
        return self._death_days

    def recovery_days(self):
        return self._recovery_days

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
                return [sum(new_cases[max(0, i - self.recovery_days()):i + 1])
                        for i in range(len(new_cases))]
        return self.memoize('confirmed_ongoing', data_fun)

    def deaths_total(self):
        if self.deaths:
            return self.deaths
        def data_fun():
            return [c * self.death_rate() for c in self.cases[self.confirm_to_death_days():]]
        return self.memoize('deaths_total', data_fun)

    def deaths_new(self):
        return self.memoize('deaths_new', lambda: convert_delta(self.deaths_total()))

    def estimated_total(self):
        def estimated_fun():
            return [d / self.death_rate() for d in self.deaths]
        return self.deaths_estimated('estimated_total', self.confirmed_total, estimated_fun)

    def estimated_new(self):
        def estimated_fun():
            return convert_delta(self.estimated_total())
        return self.deaths_estimated('estimated_new', self.confirmed_new, estimated_fun)

    def estimated_ongoing(self):
        def estimated_fun():
            new_cases = self.estimated_new()
            return [sum(new_cases[max(0, i - self.recovery_days()):i + 1])
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
        if source_cases.adminregion:
            self.prettyname = '%s, %s' % (source_cases.adminregion, source_cases.subregion,)
        else:
            self.prettyname = source_cases.subregion or source_cases.region
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

