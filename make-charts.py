#!/usr/bin/python3
import sys, os, csv, re, datetime
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
    def __init__(self, data_format, row):
        self.data_format = data_format
        self.region = data_format.region(row)
        self.subregion = data_format.subregion(row)
        self.latitude = data_format.latitude(row)
        self.longitude = data_format.longitude(row)
        self.data = data_format.data(row)


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
                region_data.append(RegionSourceData(data_format, row))
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
canada_subregions = [sr.name for sr in regions['canada'].subregions.values() if sr.population]

