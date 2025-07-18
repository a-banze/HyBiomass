#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Timothée Stassin (stassin@gfz-potsdam.de)
# @License : (C)Copyright 2025, GFZ Potsdam
# @Desc    : Functions to process GEDI data
# %%
from datetime import datetime
import pandas as pd
import time


def timestamp_to_datetime(timestamp_id):
    dt = datetime.strptime(timestamp_id, "%Y%m%dT%H%M%SZ")
    return dt


def toYearFraction(date):
    def sinceEpoch(date):  # returns seconds since epoch
        return time.mktime(date.timetuple())

    s = sinceEpoch

    year = date.year
    startOfThisYear = datetime(year=year, month=1, day=1)
    startOfNextYear = datetime(year=year + 1, month=1, day=1)

    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed / yearDuration

    return date.year + fraction


def add_delta_time(df, dt):
    df["date"] = pd.to_datetime(df["date"])
    df["delta_t"] = df["date"] - dt
    df["delta_t_seconds"] = df["delta_t"].dt.total_seconds()
    df["delta_t_days"] = df["delta_t_seconds"] / (24 * 3600)
    df["abs_delta_t_days"] = df["delta_t_days"].abs()
    df["enmap_date"] = dt.strftime("%Y-%m-%d")
    df["enmap_date_dec"] = toYearFraction(dt)
    df["enmap_doy"] = int(dt.strftime("%-j"))
    return df
